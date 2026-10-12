import bisect
import hashlib
import json
import math
import os
import random
import time
from .request import Request
from .algorithms.fairness import (
    FairnessTracker,
    fairness_debt,
    fairness_target,
    fairness_urgency,
)
from .algorithms.fairroute import (
    choose_fairroute_candidate,
    fairroute_gate,
    iteration_time,
    predict_latency_seconds,
)
from .algorithms.locality import locality_score
from .algorithms.prediction import predict_queue, replica_is_safe
from .algorithms.utils import kv_utilization, reactive_load
from .logger import get_logger


class Router:
    def __init__(
            self,
            num_instances,
            schedulers, req_num,
            routing_policy="RR",
            seed=42,
            custom_routing_fn=None,
            fairroute_n0=0.0,
    ):
        self.schedulers = schedulers
        self.num_instances = num_instances
        self.prefill_schedulers = [s for s in schedulers if s.pd_type != "decode"]
        self.prefill_instances = len(self.prefill_schedulers)
        self.decode_schedulers = [s for s in schedulers if s.pd_type == "decode"]
        self.decode_instances = len(self.decode_schedulers)
        self.req_num = req_num
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self._rnd = random.Random(seed) if seed is not None else random
        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0
        self.fairness_tracker = FairnessTracker()
        self._request_clients = {}
        self._request_service = {}
        self._request_arrivals = {}
        self._request_instances = {}
        self._request_output_estimates = {}
        self._client_output_ewma = {}
        self._client_credit = {}
        self._client_delay_ewma = {}
        self._client_delay_samples = {}
        self._client_credit_refill = {}
        self._replica_service = {}
        self._known_clients = set()
        self._total_service = 0.0
        self.custom_routing_fn = custom_routing_fn
        self.fairroute_n0 = max(0.0, float(fairroute_n0))
        self._global_output_ewma = 1.0
        self._global_delay_ewma = 0.0
        self._global_output_samples = []
        self._policy_params = {
            "prism_p": 4.0,
            "prism_beta": 2.0,
            "h_min": 16,
            "fairroute_h_min": 256,
            "fairroute_gamma": 1.0,
            "fairroute_eps_max": 0.100,
            "fairroute_rho_slack": 0.25,
            "fairroute_k": 10.0,
            "credit_w_e": 1.0,
            "credit_w_q": 2.0,
            "credit_kappa": 1.0,
        }
        self._decision_times_ns = []
        self._routed_scores = []
        self._routed_margins = []
        self._routing_stage_counts = {}
        self._decision_records = []
        self._routing_policy_names = {
            "FAIRROUTE", "FAIRROUTE_V2", "FAIRROUTE_LINEAR",
            "FAIRROUTE_MULTIPLICATIVE", "FAIRROUTE_GATED",
            "FAIRROUTE_LEXICOGRAPHIC", "FAIRROUTE_SLACK",
            "FAIRROUTE_LEARNED", "PRISM", "D2LPM",
        }

        # Pending requests (loaded but not yet routed)
        self._pending_requests = []
        self._pending_idx = 0
        self._enable_prefix_caching = False
        self._is_init = True

        # Agentic session dependency tracking
        self._deferred_sessions = {}     # session_id -> session state dict
        self._request_to_session = {}    # request_id -> (session_id, sub_request_index)
        self._next_request_id = 0        # monotonic counter for unique request IDs

        if self.routing_policy == "RR":
            self._select_instance = self._rr_select
        elif self.routing_policy == "RAND":
            self._select_instance = self._rand_select
        elif self.routing_policy == "LOAD":
            self._select_instance = self._least_load_select
        elif self.routing_policy in self._routing_policy_names:
            self._select_instance = self._fair_route
        elif self.routing_policy == "CUSTOM":
            self._select_instance = self._rr_select
        else:
            raise ValueError(f"Unknown routing_policy '{routing_policy}'. "
                             "Supported: RR, RAND, LOAD, and FairRoute policies")
        self.logger = get_logger(self.__class__)

    # -----------------------------------------------------------------------
    # Instance selection policies
    # -----------------------------------------------------------------------

    def _get_counter(self, role):
        return self.decode_rr_counter if role == "decode" else self.prefill_rr_counter

    def _set_counter(self, role, value):
        if role == "decode":
            self.decode_rr_counter = value
        else:
            self.prefill_rr_counter = value

    def _rr_select(self, schedulers, role):
        num_instances = len(schedulers)
        idx = self._get_counter(role) % num_instances
        self._set_counter(role, idx + 1)
        return idx

    def _rand_select(self, schedulers, role):
        return self._rnd.randrange(len(schedulers))

    def _least_load_select(self, schedulers, role):
        """vLLM-style least-loaded routing, normalized by instance capacity."""
        best_idx = 0
        best_score = float('inf')
        num_instances = len(schedulers)
        start = self._get_counter(role) % num_instances
        for offset in range(num_instances):
            idx = (start + offset) % num_instances
            sched = schedulers[idx]
            waiting = len(sched.waiting)
            running = len(sched.running)
            raw_score = waiting * 4 + running
            capacity = getattr(sched, "max_num_seqs", 0)
            score = raw_score
            if capacity not in (0, float('inf')):
                score = raw_score / capacity
            if score < best_score:
                best_score = score
                best_idx = idx
        self._set_counter(role, (best_idx + 1) % num_instances)
        return best_idx

    def _fair_route(self, schedulers, role, req_data=None):
        if req_data is None:
            raise RuntimeError("FairRoute selection requires request data")
        if self.routing_policy == "PRISM":
            return self._select_prism(schedulers, req_data)
        if self.routing_policy == "D2LPM":
            return self._select_d2lpm(schedulers, req_data)
        if self.routing_policy in {"FAIRROUTE", "FAIRROUTE_V2"}:
            return self._select_fairroute_v2(schedulers, req_data)
        return self._select_fairroute_candidate(schedulers, req_data)

    def _select_fairroute_v2(self, schedulers, req_data):
        candidates, estimate = self._candidate_set(schedulers, req_data)
        gamma = self._policy_params["fairroute_gamma"]
        gated = [
            c for c in candidates
            if c["free_kv"] >= c["kv"]
            and c["ttft"] <= gamma * c["slo"]
        ]
        slo_risk = not gated
        if not gated:
            gated = candidates
        best_latency = min(c["predicted_latency"] for c in gated)
        client_id = req_data.get("client_id", "default")
        self._known_clients.add(client_id)

        # --- Client-level fairness: service-based debt urgency ----------------
        # Computes egalitarian debt relative to mean service across all known
        # clients and maps it to an urgency in [0, 1) via exponential saturation.
        target = fairness_target(self._total_service, len(self._known_clients))
        service_debt = fairness_debt(
            self.fairness_tracker.get_service(client_id), target
        )
        service_urgency = fairness_urgency(service_debt)

        # --- Client-level fairness: delay-based urgency (optional) ------------
        # When fairroute_n0 > 0, also compare client EWMA delay to the cluster
        # median; a client experiencing disproportionate latency gets extra pull.
        delays = [
            value for owner, value in self._client_delay_ewma.items()
            if owner in self._known_clients
        ]
        median_delay = sorted(delays)[len(delays) // 2] if delays else 0.0
        client_delay = self._client_delay_ewma.get(client_id, 0.0)
        delay_urgency = 0.0
        if self.fairroute_n0 and median_delay > 0 and client_delay / median_delay > 1:
            delay_urgency = min(1.0, client_delay / median_delay - 1.0)

        # Combined: respond to whichever fairness dimension is most violated.
        urgency = max(service_urgency, delay_urgency)

        epsilon = min(
            self._policy_params["fairroute_eps_max"],
            self._policy_params["fairroute_rho_slack"] * best_latency,
        ) * (1.0 - urgency)
        slack = [c for c in gated if c["predicted_latency"] <= best_latency + epsilon]
        mean_service = (
            sum(self._replica_service.get(i, 0.0) for i in range(len(schedulers)))
            / max(1, len(schedulers))
        )

        # Normalise queued work across the slack set to get per-candidate load
        # headroom [0, 1].  Headroom 1 = least-loaded, 0 = most-loaded.
        queued_works = [c["W"] for c in slack]
        w_max, w_min = max(queued_works), min(queued_works)
        w_span = max(w_max - w_min, 1.0)

        for candidate in slack:
            candidate["locality_key"] = (
                candidate["hit"]
                if candidate["hit"] >= self._policy_params["fairroute_h_min"]
                and candidate["credit"] > 0
                else 0
            )
            candidate["deficit"] = (
                mean_service - candidate["service"]
            ) / max(mean_service, 1.0)
            # Client-level fairness signal: steer underserved clients (high
            # urgency) toward less-loaded replicas (high headroom).  This is
            # candidate-varying, so it creates a genuine spread across replicas
            # rather than a uniform offset that would cancel in argmin.
            candidate["client_fairness"] = (
                urgency * (w_max - candidate["W"]) / w_span
            )

        selected = min(
            slack,
            key=lambda c: (
                -c["locality_key"],
                -c["client_fairness"],  # client-level fairness (new)
                -c["deficit"],          # replica-level fairness
                c["predicted_latency"],
                c["index"],
            ),
        )

        client_id = selected["client_id"]
        key = (client_id, selected["index"])
        selected["credit"] -= self._policy_params["credit_w_e"] * selected["n_ext"]
        self._client_credit[key] = selected["credit"]
        if all(value <= 0 for (owner, _), value in self._client_credit.items() if owner == client_id):
            for k in list(self._client_credit):
                if k[0] == client_id:
                    self._client_credit[k] += self._credit_quantum(schedulers[k[1]])

        gated_latencies = sorted(c["predicted_latency"] for c in gated)
        margin = (gated_latencies[1] - gated_latencies[0]) if len(gated_latencies) > 1 else 0.0
        stage = "fallback" if slo_risk else ("slack" if epsilon > 0 else "gate")
        stage = f"{stage}:lexicographic"
        self._routing_stage_counts[stage] = self._routing_stage_counts.get(stage, 0) + 1
        credit_exhausted = self._client_credit.get(key, 0.0) <= 0
        score = -selected["predicted_latency"]
        if slo_risk:
            score -= 0.001
        self._routed_scores.append(score)
        self._routed_margins.append(margin)

        self._replica_service[selected["index"]] = (
            self._replica_service.get(selected["index"], 0.0)
            + self._policy_params["credit_w_e"] * selected["n_ext"]
            + self._policy_params["credit_w_q"] * estimate
        )

        route_metrics = (
            service_debt,
            urgency,
            selected["hit"],
            selected["predicted_latency"],
            selected["hit"],
            margin,
            len(gated),
            len(slack),
            stage,
            int(credit_exhausted),
            epsilon,
        )
        return selected["index"], score, route_metrics

    def _request_to_route_data(self, req):
        """Build the same candidate input for routed and handed-off requests."""
        return {
            'index': req.id,
            'input_toks': req.original_input,
            'output_toks': req.output,
            'arrival_time_ns': req.arrival,
            'client_id': req.client_id,
            'input_hash_ids': req.input_hash_ids or [],
            'output_hash_ids': req.output_hash_ids or [],
        }

    def _custom_select(self, schedulers, role, req_data):
        if self.custom_routing_fn is None:
            raise RuntimeError(
                "CUSTOM routing requires custom_routing_fn=(schedulers, role, req_data) "
                "when constructing Router"
            )
        result = self.custom_routing_fn(schedulers, role, req_data)
        if isinstance(result, int):
            return result, 0.0, (0.0, 0.0, 0.0, 0.0, 0, 0.0)
        if len(result) != 3:
            raise ValueError(
                "custom_routing_fn must return (instance_index, score, metrics)"
            )
        return result

    def _replica_locality(self, sched, req_data):
        """Return prefix reuse without mutating the request being routed."""
        input_hash_ids = req_data.get('input_hash_ids', [])
        if not sched.enable_prefix_caching or not input_hash_ids:
            return 0
        probe = Request(
            req_data['index'], sched.model, req_data['input_toks'],
            req_data['output_toks'], req_data['arrival_time_ns'],
            sched.instance_id, req_data.get('client_id'), input_hash_ids,
            req_data.get('output_hash_ids', []), is_init=self._is_init,
        )
        _, npu_hit, lower_hit = sched.memory.kv.get_computed_blocks(probe)
        return min(req_data['input_toks'], npu_hit + lower_hit)

    def _replica_prediction(self, sched):
        waiting = len(sched.waiting)
        running = len(sched.running)
        capacity = getattr(sched, 'max_num_seqs', 0)
        load = reactive_load(waiting, running, capacity)
        predicted_queue = predict_queue(waiting, 0.0, 0.0, 1.0)
        pool = getattr(sched.memory, 'npu_pool', None)
        used_blocks = getattr(pool, 'used_blocks', 0) if pool else 0
        total_blocks = getattr(pool, 'num_blocks', 0) if pool else 0
        kv_util = kv_utilization(used_blocks, total_blocks)
        return load, predicted_queue, kv_util

    def _queued_tokens(self, sched):
        """Approximate W_i from pending prompts and running remainders."""
        pending = sum(max(0, int(getattr(req, "num_tokens", req.input)))
                      for req in sched.waiting)
        running = sum(max(0, int(getattr(req, "num_tokens", req.input)
                                  - getattr(req, "num_computed_tokens", 0)))
                      for req in sched.running)
        return pending + running

    def _free_kv_tokens(self, sched):
        pool = getattr(getattr(sched, "memory", None), "npu_pool", None)
        if pool is None:
            return float("inf")
        return max(0, pool.num_blocks - pool.used_blocks) * pool.block_size

    def _predictor_terms(self, sched, req_data, hit, output_estimate):
        n_in = float(req_data["input_toks"])
        n_ext = max(0.0, n_in - hit)
        b = len(sched.running) + len(sched.waiting)
        d = sum(1 for req in sched.running
                if getattr(req, "num_tokens", req.input)
                <= getattr(req, "num_computed_tokens", 0) + 1)
        s_max = max(1, int(getattr(sched, "max_num_batched_tokens", 1)))
        w = self._queued_tokens(sched)
        capacity = max(1, s_max - d)
        t_base = iteration_time(b + 1, s_max, p_max=s_max)
        ttft = math.ceil((w + n_ext) / capacity) * t_base
        decode = output_estimate * iteration_time(d + 1, d + 1, p_max=s_max)
        total = ttft + decode
        slo = self._policy_params["fairroute_k"] * math.ceil(n_in / s_max) * iteration_time(1, s_max, p_max=s_max)
        return {
            "hit": hit,
            "n_ext": n_ext,
            "W": w,
            "B": b,
            "D": d,
            "S_max": s_max,
            "ttft": ttft,
            "decode": decode,
            "predicted_latency": total,
            "slo": max(slo, 1e-9),
            "kv": n_in + output_estimate,
            "free_kv": self._free_kv_tokens(sched),
            "requests": b + len(sched.waiting),
        }

    def _client_output_estimate(self, client_id):
        value = self._client_output_ewma.get(client_id, self._global_output_ewma)
        if self.fairroute_n0:
            value = (
                value + self.fairroute_n0 * self._global_output_ewma
            ) / (1.0 + self.fairroute_n0)
        return max(1.0, value)

    def _credit_quantum(self, sched):
        config = getattr(sched, "config", {})
        max_in = config.get("max_position_embeddings", 2048)
        max_out = 1024
        return self._policy_params["credit_kappa"] * (
            self._policy_params["credit_w_e"] * max_in
            + self._policy_params["credit_w_q"] * max_out
        )

    def _credit(self, client_id, index, schedulers):
        key = (client_id, index)
        if key not in self._client_credit:
            self._client_credit[key] = self._credit_quantum(schedulers[index])
        owner_values = [
            self._client_credit.setdefault(
                (client_id, replica), self._credit_quantum(schedulers[replica])
            )
            for replica in range(len(schedulers))
        ]
        if not any(value > 0 for value in owner_values):
            for replica in range(len(schedulers)):
                self._client_credit[(client_id, replica)] += self._credit_quantum(schedulers[replica])
        return self._client_credit[key]

    def _candidate_set(self, schedulers, req_data):
        client_id = req_data.get("client_id", "default")
        estimate = self._client_output_estimate(client_id)
        candidates = []
        for index, sched in enumerate(schedulers):
            hit = self._replica_locality(sched, req_data)
            terms = self._predictor_terms(sched, req_data, hit, estimate)
            terms.update({
                "index": index,
                "client_id": client_id,
                "credit": self._credit(client_id, index, schedulers),
                "service": self._replica_service.get(index, 0.0),
                "safe": terms["free_kv"] >= terms["kv"],
            })
            candidates.append(terms)
        return candidates, estimate

    def _select_prism(self, schedulers, req_data):
        candidates, _ = self._candidate_set(schedulers, req_data)
        p = self._policy_params["prism_p"]
        beta = self._policy_params["prism_beta"]
        selected = None
        best = -1.0
        for candidate in candidates:
            s_lat = 1.0 / (1.0 + (candidate["predicted_latency"] / candidate["slo"]) ** p)
            rho = candidate["W"] / max(1.0, candidate["S_max"])
            s_load = 1.0 / (1.0 + (rho / candidate["slo"]) ** p)
            s_cap = max(0.0, min(1.0, (candidate["free_kv"] - candidate["kv"]) / max(candidate["kv"], 1e-9)))
            affinity = beta if candidate["hit"] >= self._policy_params["h_min"] else 1.0
            score = s_lat * s_load * s_cap * affinity
            if score > best or (score == best and (selected is None or candidate["requests"] < selected["requests"])):
                selected, best = candidate, score
        return self._finish_policy_choice(selected, best, "prism", candidates)

    def _select_d2lpm(self, schedulers, req_data):
        candidates, _ = self._candidate_set(schedulers, req_data)
        matched = [c for c in candidates if c["hit"] >= self._policy_params["h_min"]]
        max_hit = max((c["hit"] for c in candidates), default=0)
        group = [c for c in matched if c["hit"] == max_hit] if max_hit >= self._policy_params["h_min"] else candidates
        active = [c for c in candidates if c["credit"] > 0]
        eligible = [c for c in group if c["credit"] > 0]
        pool = eligible or active
        if not pool:
            self._refill_credits(req_data.get("client_id", "default"), schedulers)
            pool = candidates
        selected = min(pool, key=lambda c: (c["requests"], c["index"]))
        return self._finish_policy_choice(selected, -selected["requests"], "d2lpm", candidates)

    def _refill_credits(self, client_id, schedulers):
        for index, sched in enumerate(schedulers):
            self._client_credit[(client_id, index)] = self._credit_quantum(sched)

    def _finish_policy_choice(self, selected, score, stage, candidates):
        client_id = selected["client_id"]
        key = (client_id, selected["index"])
        selected["credit"] -= self._policy_params["credit_w_e"] * selected["n_ext"]
        self._client_credit[key] = selected["credit"]
        self._routing_stage_counts[stage] = self._routing_stage_counts.get(stage, 0) + 1
        self._routed_scores.append(score)
        self._routed_margins.append(0.0)
        return selected["index"], score, (
            0.0, 0.0, selected["hit"], selected["predicted_latency"],
            selected["hit"], 0.0, len(candidates), len(candidates),
            stage, int(selected["credit"] <= 0), 0.0,
        )

    @staticmethod
    def _tiebreak_jitter(request_index, candidate_index):
        """Deterministic hash-based jitter to break ties without systematic bias.

        Using -candidate["index"] as tie-break always selects the lowest index,
        causing pathological hotspotting when scores are equal. This hashes the
        (request, candidate) pair to produce a stable but uniformly scattered
        value, so ties distribute evenly across replicas.
        """
        h = hashlib.md5(f"{request_index}:{candidate_index}".encode()).digest()
        return int.from_bytes(h[:4], 'little') / 0xFFFFFFFF

    def _select_fairroute_candidate(self, schedulers, req_data):
        """Route with FairRoute's gate -> slack -> lexicographic pipeline."""
        client_id = req_data.get("client_id", "default")
        request_index = req_data.get("index", 0)
        self._known_clients.add(client_id)
        target = fairness_target(self._total_service, len(self._known_clients))
        debt = fairness_debt(self.fairness_tracker.get_service(client_id), target)
        urgency = fairness_urgency(debt)
        client_output = self._client_output_ewma.get(client_id, self._global_output_ewma)
        if self.fairroute_n0:
            output_estimate = (
                client_output + self.fairroute_n0 * self._global_output_ewma
            ) / (1.0 + self.fairroute_n0)
        else:
            output_estimate = client_output
        max_replica_service = max(self._replica_service.values(), default=0.0)
        candidates = []

        for index, sched in enumerate(schedulers):
            hit = self._replica_locality(sched, req_data)
            locality = locality_score(hit, req_data["input_toks"])
            waiting = len(sched.waiting)
            running = len(sched.running)
            capacity = getattr(sched, "max_num_seqs", 0)
            load = reactive_load(waiting, running, capacity)
            predicted_queue = predict_queue(waiting, 0.0, 0.0, 1.0)
            pool = getattr(sched.memory, "npu_pool", None)
            used_blocks = getattr(pool, "used_blocks", 0) if pool else 0
            total_blocks = getattr(pool, "num_blocks", 0) if pool else 0
            kv_util = kv_utilization(used_blocks, total_blocks)
            max_queue = max(1, capacity * 4) if capacity not in (0, float("inf")) else float("inf")
            safe = replica_is_safe(predicted_queue, max_queue, kv_util)
            credit_key = (client_id, index)
            credit = self._client_credit.setdefault(credit_key, 8192.0)
            replica_service = self._replica_service.get(index, 0.0)
            candidates.append({
                "index": index,
                "safe": safe,
                "hit": hit,
                "locality": locality,
                "future_locality": locality,
                "fairness_credit": credit,
                "replica_deficit": max_replica_service - replica_service,
                "load": load,
                "predicted_queue": predicted_queue,
                "kv_util": kv_util,
                "predicted_latency": predict_latency_seconds(
                    load, req_data["input_toks"] - hit, output_estimate),
            })

        feasible, slo_risk = fairroute_gate(candidates, ttft_slo=0.1, gamma=1.0)
        gate_size = len(feasible)
        mode_by_policy = {
            "FAIRROUTE_LINEAR": "LINEAR",
            "FAIRROUTE_MULTIPLICATIVE": "MULTIPLICATIVE",
            "FAIRROUTE_GATED": "GATED",
            "FAIRROUTE_LEXICOGRAPHIC": "LEXICOGRAPHIC",
            "FAIRROUTE_SLACK": "SLACK_LEXICOGRAPHIC",
            "FAIRROUTE_LEARNED": "LEARNED",
            "FAIRROUTE": "SLACK_LEXICOGRAPHIC",
            "FAIRROUTE_V2": "SLACK_LEXICOGRAPHIC",
            "PRISM": "SLACK_LEXICOGRAPHIC",
            "D2LPM": "LEARNED",
        }
        selected, epsilon = choose_fairroute_candidate(
            feasible, mode=mode_by_policy[self.routing_policy], urgency=urgency)
        slack_size = len(feasible)
        stage = "fallback" if slo_risk else "gate"
        if epsilon > 0:
            stage = "slack"
        if self.routing_policy in {"FAIRROUTE", "FAIRROUTE_V2"}:
            stage = f"{stage}:lexicographic"
        self._routing_stage_counts[stage] = self._routing_stage_counts.get(stage, 0) + 1
        selected_index = selected["index"]
        extend_tokens = max(0, req_data["input_toks"] - selected["hit"])
        self._client_credit[(client_id, selected_index)] -= extend_tokens
        if all(value <= 0 for (owner, _), value in self._client_credit.items() if owner == client_id):
            for key in list(self._client_credit):
                if key[0] == client_id:
                    self._client_credit[key] += 8192.0

        alternatives = sorted(candidate["predicted_latency"] for candidate in feasible)
        margin = (alternatives[1] - alternatives[0]) if len(alternatives) > 1 else 0.0
        score = -selected["predicted_latency"]
        if slo_risk:
            score -= 0.001
        self._routed_scores.append(score)
        self._routed_margins.append(margin)
        credit_exhausted = self._client_credit[(client_id, selected_index)] <= 0
        return selected_index, score, (
            debt, urgency, selected["locality"],
            selected["predicted_latency"],
            selected["hit"], margin, gate_size, slack_size, stage,
            int(credit_exhausted), epsilon,
        )

    def get_score_stats(self):
        """Return aggregate statistics of routing scores and margins for monitoring."""
        if not self._routed_scores:
            return {"min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0, "margin_mean": 0.0, "margin_std": 0.0, "count": 0}
        scores = self._routed_scores
        margins = self._routed_margins
        n = len(scores)
        mean_score = sum(scores) / n
        var_score = sum((x - mean_score) ** 2 for x in scores) / n
        std_score = math.sqrt(var_score)
        mean_margin = sum(margins) / n
        var_margin = sum((x - mean_margin) ** 2 for x in margins) / n
        std_margin = math.sqrt(var_margin)
        return {
            "min": min(scores),
            "mean": mean_score,
            "max": max(scores),
            "std": std_score,
            "margin_mean": mean_margin,
            "margin_std": std_margin,
            "count": n,
        }

    def get_decision_diagnostics(self):
        """Return aggregate routing timing and stage data for a run."""
        values = sorted(self._decision_times_ns)
        if not values:
            percentiles = {"p50": 0, "p95": 0, "p99": 0}
        else:
            def percentile(fraction):
                return values[min(len(values) - 1, int((len(values) - 1) * fraction))]
            percentiles = {
                "p50": percentile(0.50),
                "p95": percentile(0.95),
                "p99": percentile(0.99),
            }
        diag = {
            "count": len(values),
            "decision_ns": percentiles,
            "stage_counts": dict(self._routing_stage_counts),
            "policy": self.routing_policy,
            "fairroute_n0": self.fairroute_n0,
        }
        if self._decision_records:
            n = len(self._decision_records)
            diag["top1_agreement_rate"] = sum(1 for r in self._decision_records if r["top1_agreement"]) / n
            diag["locality_sacrificed_rate"] = sum(1 for r in self._decision_records if r["locality_sacrificed"]) / n
            diag["mean_gate_size"] = sum(r["gate_size"] for r in self._decision_records) / n
            diag["mean_slack_size"] = sum(r["slack_size"] for r in self._decision_records) / n
            diag["credit_exhaustion_rate"] = sum(1 for r in self._decision_records if r["credit_exhausted"]) / n
            diag["mean_urgency"] = sum(r["urgency"] for r in self._decision_records) / n
        return diag

    def save_diagnostics(self, output_file):
        """Write aggregate routing diagnostics beside a result CSV."""
        if not output_file:
            return
        target = output_file if os.path.isabs(output_file) else f"../{output_file}"
        target_dir = os.path.dirname(target)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        with open(f"{target}.routing.json", "w", encoding="utf-8") as handle:
            json.dump(self.get_decision_diagnostics(), handle, indent=2)

    def save_decision_sidecar(self, output_file):
        """Write per-request decision sidecar beside the output CSV."""
        if not output_file or not self._decision_records:
            return
        target = output_file if os.path.isabs(output_file) else f"../{output_file}"
        paths = [f"{target}.sidecar.jsonl"]
        if target.endswith('.csv'):
            paths.append(target[:-4] + '_sidecar.jsonl')
        for path in set(paths):
            out_dir = os.path.dirname(path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                for rec in self._decision_records:
                    handle.write(json.dumps(rec) + "\n")

    # -----------------------------------------------------------------------
    # Request loading and real-time routing
    # -----------------------------------------------------------------------

    def load_requests(self, path, enable_prefix_caching=False, is_init=True):
        """Load requests from dataset into pending queue (not yet routed).

        Supports two JSONL formats:
        - Flat: {"input_toks", "output_toks", "arrival_time_ns", ...}
        - Agentic session: {"session_id", "arrival_time_ns", "sub_requests": [...]}

        For agentic sessions, only the first sub-request is added to the
        pending queue. Subsequent sub-requests are released dynamically
        via notify_request_completed() when predecessors finish.
        """
        if not os.path.exists(path) and os.path.exists(f'../{path}'):
            path = f'../{path}'
        self._enable_prefix_caching = enable_prefix_caching
        self._is_init = is_init
        loaded_lines = 0

        with open(path) as f:
            for line in f:
                if self.req_num > 0 and loaded_lines >= self.req_num:
                    break
                row = json.loads(line)
                if 'sub_requests' in row:
                    self._load_agentic_session(row, enable_prefix_caching)
                else:
                    self._load_flat_request(row, enable_prefix_caching)
                loaded_lines += 1

        # Sort pending requests by arrival time (agentic first sub-requests
        # may interleave with flat requests)
        self._pending_requests.sort(key=lambda r: r['arrival_time_ns'])

        self.logger.info("Loaded %d requests into pending queue "
                         "(%d agentic sessions deferred)",
                         len(self._pending_requests),
                         len(self._deferred_sessions))

    def _load_flat_request(self, row, enable_prefix_caching):
        """Load a single flat request into pending queue."""
        req_id = self._next_request_id
        self._next_request_id += 1
        req_data = {
            'index': req_id,
            'input_toks': int(row['input_toks']),
            'output_toks': int(row['input_toks'] + row['output_toks']),
            'arrival_time_ns': int(row['arrival_time_ns']),
            'client_id': row.get('client_id', 'default'),
            'session_id': row.get('session_id', ''),
            'sub_request_index': row.get('sub_request_index', 0),
        }
        if enable_prefix_caching:
            req_data['input_hash_ids'] = row.get('input_tok_ids', [])
            req_data['output_hash_ids'] = row.get('output_tok_ids', [])
        self._pending_requests.append(req_data)

    def _load_agentic_session(self, row, enable_prefix_caching):
        """Load an agentic session: first sub-request to pending, rest deferred."""
        sub_reqs = row['sub_requests']
        if not sub_reqs:
            return 0
        session_id = row.get('session_id', f'session_{self._next_request_id}')
        base_id = self._next_request_id
        self._next_request_id += len(sub_reqs)
        arrival_ns = int(row['arrival_time_ns'])

        # Store session state for dependency chain
        self._deferred_sessions[session_id] = {
            'sub_requests': sub_reqs,
            'next_index': 1,  # index 0 is being queued now
            'id_base': base_id,
            'client_id': row.get('client_id', 'default'),
        }

        # Queue the first sub-request
        first = sub_reqs[0]
        req_data = {
            'index': base_id,
            'input_toks': int(first['input_toks']),
            'output_toks': int(first['input_toks'] + first['output_toks']),
            'arrival_time_ns': arrival_ns,
            'client_id': row.get('client_id', 'default'),
            'session_id': session_id,
            'sub_request_index': 0,
        }
        if enable_prefix_caching:
            req_data['input_hash_ids'] = first.get('input_tok_ids', [])
            req_data['output_hash_ids'] = first.get('output_tok_ids', [])
        self._pending_requests.append(req_data)
        self._request_to_session[base_id] = (session_id, 0)

        return len(sub_reqs)

    def route_arrived_requests(self, current_time_ns):
        """Route requests that have arrived by current_time_ns to instances.

        Called at the start of each iteration in the main simulation loop.
        Returns the number of newly routed requests.
        """
        routed = 0
        while self._pending_idx < len(self._pending_requests):
            req_data = self._pending_requests[self._pending_idx]
            if req_data['arrival_time_ns'] > current_time_ns:
                break

            started_ns = time.perf_counter_ns()
            candidates, _ = self._candidate_set(self.prefill_schedulers, req_data)
            slack_used = 0.0

            if self.routing_policy == "CUSTOM":
                instance_id, route_score, route_metrics = self._custom_select(
                    self.prefill_schedulers, "prefill", req_data)
            elif self.routing_policy in self._routing_policy_names:
                instance_id, route_score, route_metrics = self._select_instance(
                    self.prefill_schedulers, "prefill", req_data)
                if len(route_metrics) > 10:
                    slack_used = float(route_metrics[10])
            elif self.routing_policy == "RR":
                instance_id = self._rr_select(self.prefill_schedulers, "prefill")
                stage = "round_robin"
                self._routing_stage_counts[stage] = self._routing_stage_counts.get(stage, 0) + 1
                selected = candidates[instance_id]
                route_score = -selected["predicted_latency"]
                self._routed_scores.append(route_score)
                self._routed_margins.append(0.0)
                route_metrics = (
                    0.0, 0.0, selected["hit"], selected["predicted_latency"],
                    selected["hit"], 0.0, len(candidates), len(candidates),
                    stage, 0, 0.0
                )
            elif self.routing_policy == "RAND":
                instance_id = self._rand_select(self.prefill_schedulers, "prefill")
                stage = "random"
                self._routing_stage_counts[stage] = self._routing_stage_counts.get(stage, 0) + 1
                selected = candidates[instance_id]
                route_score = -selected["predicted_latency"]
                self._routed_scores.append(route_score)
                self._routed_margins.append(0.0)
                route_metrics = (
                    0.0, 0.0, selected["hit"], selected["predicted_latency"],
                    selected["hit"], 0.0, len(candidates), len(candidates),
                    stage, 0, 0.0
                )
            elif self.routing_policy == "LOAD":
                instance_id = self._least_load_select(self.prefill_schedulers, "prefill")
                stage = "least_load"
                self._routing_stage_counts[stage] = self._routing_stage_counts.get(stage, 0) + 1
                selected = candidates[instance_id]
                route_score = -selected["predicted_latency"]
                self._routed_scores.append(route_score)
                self._routed_margins.append(0.0)
                route_metrics = (
                    0.0, 0.0, selected["hit"], selected["predicted_latency"],
                    selected["hit"], 0.0, len(candidates), len(candidates),
                    stage, 0, 0.0
                )
            else:
                instance_id = self._select_instance(self.prefill_schedulers, "prefill")
                route_score = 0.0
                route_metrics = (0.0, 0.0, 0.0, 0.0, 0, 0.0, len(candidates), len(candidates), "default", 0, 0.0)

            decision_ns = time.perf_counter_ns() - started_ns
            self._decision_times_ns.append(decision_ns)
            sched = self.prefill_schedulers[instance_id]

            if sched.enable_prefix_caching:
                new_req = sched.add_request([
                    req_data['index'], sched.model,
                    req_data['input_toks'], req_data['output_toks'],
                    req_data['arrival_time_ns'], sched.instance_id,
                    req_data.get('client_id'),
                    req_data.get('input_hash_ids', []), req_data.get('output_hash_ids', []),
                ], is_init=self._is_init)
            else:
                new_req = sched.add_request([
                    req_data['index'], sched.model,
                    req_data['input_toks'], req_data['output_toks'],
                    req_data['arrival_time_ns'], sched.instance_id,
                    req_data.get('client_id'),
                ], is_init=self._is_init)

            new_req.routing_policy = self.routing_policy
            new_req.session_id = req_data.get('session_id')
            new_req.sub_request_index = req_data.get('sub_request_index')
            new_req.routing_score = route_score
            (new_req.routing_fairness_debt,
             new_req.routing_fairness_urgency,
             new_req.routing_locality,
             new_req.routing_prediction,
             _,
             new_req.routing_score_margin,
             *routing_details) = route_metrics if len(route_metrics) >= 6 else (*route_metrics, 0.0)[:6]
            new_req.routing_decision_ns = decision_ns
            if len(routing_details) >= 4:
                new_req.routing_gate_size = int(routing_details[0])
                new_req.routing_slack_size = int(routing_details[1])
                new_req.routing_stage = str(routing_details[2])
                new_req.routing_credit_exhausted = bool(routing_details[3])

            cand_lats = [c["predicted_latency"] for c in candidates]
            cand_ttfts = [c.get("ttft", c["predicted_latency"]) for c in candidates]
            cand_hits = [c["hit"] for c in candidates]
            cand_loads = [c.get("W", c.get("requests", 0)) for c in candidates]
            best_pred = min(range(len(candidates)), key=lambda i: cand_lats[i]) if cand_lats else 0
            max_hit_idx = max(range(len(candidates)), key=lambda i: cand_hits[i]) if cand_hits else 0
            max_hit_val = cand_hits[max_hit_idx] if cand_hits else 0
            loc_sacrificed = bool(instance_id != max_hit_idx and max_hit_val > 0)
            loc_tax = float(cand_lats[instance_id] - cand_lats[max_hit_idx]) if cand_lats else 0.0

            decision_record = {
                "request_id": int(new_req.id),
                "client_id": new_req.client_id,
                "arrival_time_ns": int(new_req.arrival),
                "policy": self.routing_policy,
                "chosen_replica": int(instance_id),
                "decision_ns": int(decision_ns),
                "stage": getattr(new_req, 'routing_stage', ''),
                "gate_size": int(getattr(new_req, 'routing_gate_size', len(candidates))),
                "slack_size": int(getattr(new_req, 'routing_slack_size', len(candidates))),
                "total_candidates": len(candidates),
                "urgency": float(getattr(new_req, 'routing_fairness_urgency', 0.0)),
                "fairness_debt": float(getattr(new_req, 'routing_fairness_debt', 0.0)),
                "slack_used": slack_used,
                "credit_exhausted": bool(getattr(new_req, 'routing_credit_exhausted', False)),
                "per_replica_predicted_latency": [float(x) for x in cand_lats],
                "per_replica_predicted_ttft": [float(x) for x in cand_ttfts],
                "per_replica_hit_tokens": [int(x) for x in cand_hits],
                "per_replica_load": [float(x) for x in cand_loads],
                "best_predicted_replica": int(best_pred),
                "max_locality_replica": int(max_hit_idx),
                "locality_sacrificed": loc_sacrificed,
                "locality_tax_s": loc_tax,
                "top1_agreement": bool(instance_id == best_pred),
            }
            self._decision_records.append(decision_record)

            self._request_clients[new_req.id] = new_req.client_id
            self._request_service[new_req.id] = new_req.output - new_req.input
            self._request_arrivals[new_req.id] = new_req.arrival
            self._request_instances[new_req.id] = instance_id
            self._request_output_estimates[new_req.id] = self._client_output_estimate(
                new_req.client_id
            )

            self._pending_idx += 1
            routed += 1

        return routed

    def has_pending_requests(self):
        """Check if there are unrouted requests remaining."""
        return self._pending_idx < len(self._pending_requests)

    def get_first_arrival_time(self):
        """Return the first request's arrival time in ns, or 1 if no requests."""
        if self._pending_requests:
            return max(1, self._pending_requests[0]['arrival_time_ns'])
        return 1

    # -----------------------------------------------------------------------
    # Agentic dependency chain management
    # -----------------------------------------------------------------------

    def notify_request_completed(self, request_id, completion_time_ns):
        """Called when a request finishes. Releases the next sub-request in
        the session chain after the tool_call duration elapses.

        For flat requests (not in a session), this is a no-op.
        """
        session_info = self._request_to_session.pop(request_id, None)
        client_id = self._request_clients.pop(request_id, 'default')
        service = self._request_service.pop(request_id, 0)
        arrival = self._request_arrivals.pop(request_id, completion_time_ns)
        instance_id = self._request_instances.pop(request_id, None)
        expected_output = self._request_output_estimates.pop(
            request_id, self._global_output_ewma
        )
        self.fairness_tracker.record_service(client_id, service)
        self._total_service += service
        previous_output = self._client_output_ewma.get(client_id, float(service))
        self._client_output_ewma[client_id] = 0.1 * float(service) + 0.9 * previous_output
        self._global_output_samples.append(float(service))
        self._global_output_samples = self._global_output_samples[-101:]
        ordered = sorted(self._global_output_samples)
        self._global_output_ewma = ordered[len(ordered) // 2]
        delay = float(completion_time_ns - arrival)
        previous_delay = self._client_delay_ewma.get(client_id, delay)
        self._client_delay_ewma[client_id] = 0.05 * delay + 0.95 * previous_delay
        self._global_delay_ewma = 0.05 * delay + 0.95 * self._global_delay_ewma
        if instance_id is not None and self.routing_policy in {"D2LPM", "FAIRROUTE", "FAIRROUTE_V2"}:
            key = (client_id, instance_id)
            if self.routing_policy == "D2LPM":
                self._client_credit[key] = self._client_credit.get(key, 0.0) - (
                    self._policy_params["credit_w_q"] * service
                )
            elif self.routing_policy in {"FAIRROUTE", "FAIRROUTE_V2"}:
                self._replica_service[instance_id] = (
                    self._replica_service.get(instance_id, 0.0)
                    + self._policy_params["credit_w_q"] * (service - expected_output)
                )
        if instance_id is not None:
            if self.routing_policy not in {"FAIRROUTE", "FAIRROUTE_V2"}:
                self._replica_service[instance_id] = self._replica_service.get(instance_id, 0.0) + service
        self.fairness_tracker.record_latency(client_id, float(completion_time_ns - arrival))
        if session_info is None:
            return
        session_id, completed_idx = session_info
        session = self._deferred_sessions.get(session_id)
        if session is None:
            return

        sub_reqs = session['sub_requests']
        next_idx = session['next_index']
        base_id = session['id_base']

        # Get tool duration from the completed sub-request
        tool_duration_ns = int(sub_reqs[completed_idx].get('tool_duration_ns', 0))
        release_time_ns = completion_time_ns + tool_duration_ns

        if next_idx < len(sub_reqs):
            # Release next sub-request
            next_sub = sub_reqs[next_idx]
            next_id = base_id + next_idx
            req_data = {
                'index': next_id,
                'input_toks': int(next_sub['input_toks']),
                'output_toks': int(next_sub['input_toks'] + next_sub['output_toks']),
                'arrival_time_ns': release_time_ns,
                'client_id': session['client_id'],
                'session_id': session_id,
                'sub_request_index': next_idx,
            }
            if self._enable_prefix_caching:
                req_data['input_hash_ids'] = next_sub.get('input_tok_ids', [])
                req_data['output_hash_ids'] = next_sub.get('output_tok_ids', [])
            # Insert in sorted position after _pending_idx
            self._insert_pending_sorted(req_data)
            self._request_to_session[next_id] = (session_id, next_idx)
            session['next_index'] = next_idx + 1
        else:
            # Session complete — all sub-requests have been released
            del self._deferred_sessions[session_id]

    def _insert_pending_sorted(self, req_data):
        """Insert a request into _pending_requests maintaining arrival-time
        sort order for the not-yet-consumed portion (from _pending_idx onward)."""
        arrival = req_data['arrival_time_ns']
        # Binary search in the unconsumed portion
        lo = self._pending_idx
        hi = len(self._pending_requests)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._pending_requests[mid]['arrival_time_ns'] <= arrival:
                lo = mid + 1
            else:
                hi = mid
        self._pending_requests.insert(lo, req_data)

    def has_deferred_sessions(self):
        """Check if there are agentic sessions with unreleased sub-requests."""
        return bool(self._deferred_sessions)

    def get_next_pending_arrival(self):
        """Return the next pending request's arrival time, or None."""
        if self._pending_idx < len(self._pending_requests):
            return self._pending_requests[self._pending_idx]['arrival_time_ns']
        return None

    # -----------------------------------------------------------------------
    # Legacy: upfront routing (kept for backward compat)
    # -----------------------------------------------------------------------

    def generate(self, path, enable_prefix_caching=False, is_init=True):
        """Load and immediately route all requests (legacy behavior)."""
        self.load_requests(path, enable_prefix_caching, is_init)
        # Route all at once (arrival time ignored)
        self.route_arrived_requests(float('inf'))
        for scheduler in self.schedulers:
            self.logger.info(
                "Added %d requests to scheduler[%d] (%s type)",
                len(scheduler.waiting),
                scheduler.instance_id,
                scheduler.pd_type
            )

    def transfer_prefill_request(self, requests):
        for req in requests:
            if self.routing_policy in self._routing_policy_names:
                instance_id, _, _ = self._select_instance(
                    self.decode_schedulers,
                    "decode",
                    self._request_to_route_data(req),
                )
            else:
                instance_id = self._select_instance(self.decode_schedulers, "decode")
            self.decode_schedulers[instance_id].add_decode(req)
