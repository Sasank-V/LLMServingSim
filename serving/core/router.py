import bisect
import json
import random
from .request import Request
from .algorithms.fairness import (
    FairnessTracker,
    fairness_debt,
    fairness_target,
    fairness_urgency,
)
from .algorithms.fairroute import (
    fairroute_score,
    score_aflc,
    score_fairness_gated,
    score_linear,
    score_multiplicative,
)
from .algorithms.locality import locality_score
from .algorithms.prediction import (
    predict_queue,
    prediction_benefit,
    prediction_risk,
    replica_is_safe,
)
from .algorithms.utils import kv_utilization, load_benefit, reactive_load
from .logger import get_logger


class Router:
    def __init__(
            self,
            num_instances,
            schedulers, req_num,
            routing_policy="RR",
            seed=42,
            custom_routing_fn=None,
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
        self.custom_routing_fn = custom_routing_fn
        self._routing_policy_names = {"H0", "H1", "H2", "H3", "H4", "FAIRROUTE"}

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
                             "Supported: RR, RAND, LOAD, H0, H1, H2, H3, H4, FAIRROUTE")
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
        return self._select_fairroute_candidate(schedulers, req_data)

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
            return result, 0.0, (0.0, 0.0, 0.0, 0.0, 0)
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

    def _select_fairroute_candidate(self, schedulers, req_data):
        client_id = req_data.get('client_id', 'default')
        total_service = sum(self.fairness_tracker.service.values())
        clients = set(self.fairness_tracker.service)
        clients.add(client_id)
        target = fairness_target(total_service, len(clients))
        debt = fairness_debt(self.fairness_tracker.get_service(client_id), target)
        urgency = fairness_urgency(debt)
        candidates = []

        for index, sched in enumerate(schedulers):
            hit = self._replica_locality(sched, req_data)
            locality = locality_score(hit, req_data['input_toks'])
            load, predicted_queue, kv_util = self._replica_prediction(sched)
            capacity = getattr(sched, 'max_num_seqs', 0)
            max_queue = max(1, capacity * 4) if capacity not in (0, float('inf')) else float('inf')
            safe = replica_is_safe(predicted_queue, max_queue, kv_util)
            reactive_prediction = load_benefit(load)
            risk = prediction_risk(predicted_queue, kv_util)
            predictive = prediction_benefit(risk, max(1.0, capacity or 1.0))

            if self.routing_policy == "H0":
                score = score_linear(urgency, locality, reactive_prediction, 1.0, 1.0, 1.0)
                prediction = reactive_prediction
            elif self.routing_policy == "H1":
                score = score_multiplicative(urgency, locality, reactive_prediction, 1.0, 1.0, 1.0)
                prediction = reactive_prediction
            elif self.routing_policy == "H2":
                score = score_fairness_gated(urgency, 0.5, locality, reactive_prediction)
                prediction = reactive_prediction
            elif self.routing_policy == "H3":
                score = score_aflc(urgency, locality, reactive_prediction, 1.0, 1.0, 1.0, debt)
                prediction = reactive_prediction
            else:
                score = fairroute_score(
                    debt, urgency, locality, risk,
                    prediction_temperature=max(1.0, capacity or 1.0),
                )
                prediction = predictive

            candidates.append((index, score, safe, hit, locality, prediction))

        safe_candidates = [candidate for candidate in candidates if candidate[2]]
        selected = max(safe_candidates or candidates, key=lambda candidate: (candidate[1], -candidate[0]))
        return selected[0], selected[1], (debt, urgency, selected[4], selected[5], selected[3])

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

            if self.routing_policy == "CUSTOM":
                instance_id, route_score, route_metrics = self._custom_select(
                    self.prefill_schedulers, "prefill", req_data)
            elif self.routing_policy in self._routing_policy_names:
                instance_id, route_score, route_metrics = self._select_instance(
                    self.prefill_schedulers, "prefill", req_data)
            else:
                instance_id = self._select_instance(self.prefill_schedulers, "prefill")
                route_score = 0.0
                route_metrics = (0.0, 0.0, 0.0, 0.0, 0)
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
            new_req.routing_score = route_score
            (new_req.routing_fairness_debt,
             new_req.routing_fairness_urgency,
             new_req.routing_locality,
             new_req.routing_prediction,
             _) = route_metrics
            self._request_clients[new_req.id] = new_req.client_id
            self._request_service[new_req.id] = new_req.output - new_req.input

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
        self.fairness_tracker.record_service(client_id, service)
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
