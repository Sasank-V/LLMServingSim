"""Calibrated, token- and slot-aware FairRoute policy.

The ``fp_*`` router parameters are deliberately explicit calibration knobs.
Fit ``fp_prefill_tps`` and ``fp_tpot`` from low-load RR measurements before
using this policy for a comparison.
"""
import hashlib
import math


DEFAULTS = {
    "prefill_tps": 22000.0, "tpot": 0.0113, "t0": 0.016,
    "lam": 2.0, "eps_max": 0.15, "rho_s": 0.25, "h_min": 256,
    "k": 10.0, "gamma": 1.0, "q_mult": 2.0, "w_q": 2.0,
    "lam_debt": 3.0, "slot_extra_s": 0.05,
}


def _jitter(request_index, replica):
    digest = hashlib.md5(f"{request_index}:{replica}".encode()).digest()
    return int.from_bytes(digest[:4], "little") / 0xFFFFFFFF


def _params(router):
    params = dict(DEFAULTS)
    params.update({key[3:]: value for key, value in router._policy_params.items()
                   if key.startswith("fp_")})
    return params


def select_fairroute_p(router, schedulers, req_data):
    """Choose a replica and return Router's standard choice tuple."""
    params = _params(router)
    client = req_data.get("client_id")
    client = "default" if client is None else client
    n_in = float(req_data["input_toks"])
    output_estimate = router._client_output_estimate(client)
    router._known_clients.add(client)

    candidates = []
    for index, sched in enumerate(schedulers):
        hit = router._replica_locality(sched, req_data)
        slots = max(1, int(getattr(sched, "max_num_seqs", 1) or 1))
        running, waiting = len(sched.running), len(sched.waiting)
        queued_tokens = router._queued_tokens(sched)
        uncached = max(0.0, n_in - hit)
        prefill = (queued_tokens + (1.0 + params["lam"]) * uncached) / params["prefill_tps"]
        slot_occupancy = output_estimate * params["tpot"] + params["slot_extra_s"]
        slot_wait = max(0, running + waiting + 1 - slots) * slot_occupancy / slots
        ttft = prefill + slot_wait
        candidates.append({
            "index": index, "hit": hit, "n_ext": uncached, "W": queued_tokens,
            "ttft": ttft, "T": ttft + output_estimate * params["tpot"],
            "free_kv": router._free_kv_tokens(sched), "kv": n_in + output_estimate,
            # Keep Router's sidecar schema uniform across policies.
            "predicted_latency": ttft + output_estimate * params["tpot"],
        })

    service = router.__dict__.setdefault("_fp_client_service", {})
    cost_ewma = router.__dict__.get("_fp_cost_ewma", 8192.0)
    quantum = params["q_mult"] * cost_ewma
    target = sum(service.values()) / max(1, len(router._known_clients))
    debt = max(0.0, (target - service.get(client, 0.0)) / target) if target else 0.0
    urgency = 1.0 - math.exp(-params["lam_debt"] * debt)

    slo = params["k"] * (n_in / params["prefill_tps"] + params["t0"])
    gated = [candidate for candidate in candidates
             if candidate["free_kv"] >= candidate["kv"]
             and candidate["ttft"] <= params["gamma"] * slo]
    slo_risk = not gated
    gated = gated or candidates
    best_time = min(candidate["T"] for candidate in gated)
    epsilon = min(params["eps_max"], params["rho_s"] * best_time) * (1.0 - urgency)
    band = [candidate for candidate in gated if candidate["T"] <= best_time + epsilon]

    for candidate in candidates:
        key = (client, candidate["index"])
        router._client_credit.setdefault(key, quantum)
        candidate["credit"] = router._client_credit[key]
    mean_service = sum(router._replica_service.get(candidate["index"], 0.0)
                       for candidate in candidates) / len(candidates)
    max_work = max(candidate["W"] for candidate in band)
    work_span = max(1.0, max_work - min(candidate["W"] for candidate in band))

    def choice_key(candidate):
        locality = candidate["hit"] if (candidate["hit"] >= params["h_min"]
                                         and candidate["credit"] > 0) else 0
        deficit = ((mean_service - router._replica_service.get(candidate["index"], 0.0))
                   / max(mean_service, 1.0))
        fairness = urgency * (max_work - candidate["W"]) / work_span
        return (-locality, candidate["T"], -fairness, -deficit,
                _jitter(req_data.get("index", 0), candidate["index"]))

    selected = min(band, key=choice_key)
    selected_key = (client, selected["index"])
    router._client_credit[selected_key] -= selected["n_ext"]
    refilled = False
    if all(router._client_credit.get((client, candidate["index"]), 0.0) <= 0
           for candidate in candidates):
        for candidate in candidates:
            key = (client, candidate["index"])
            router._client_credit[key] = router._client_credit.get(key, 0.0) + quantum
        refilled = True
    cost = selected["n_ext"] + params["w_q"] * output_estimate
    service[client] = service.get(client, 0.0) + cost
    router._fp_cost_ewma = 0.95 * cost_ewma + 0.05 * (n_in + params["w_q"] * output_estimate)
    router._replica_service[selected["index"]] = router._replica_service.get(selected["index"], 0.0) + cost

    stage = ("fallback" if slo_risk else ("slack" if epsilon else "gate")) + ":fp"
    router._routing_stage_counts[stage] = router._routing_stage_counts.get(stage, 0) + 1
    alternatives = sorted(candidate["T"] for candidate in gated)
    margin = alternatives[1] - alternatives[0] if len(alternatives) > 1 else 0.0
    score = -selected["T"] - (0.001 if slo_risk else 0.0)
    router._routed_scores.append(score)
    router._routed_margins.append(margin)
    router._last_policy_candidates = candidates
    router._last_credit_refilled = refilled
    return selected["index"], score, (
        debt, urgency, selected["hit"], selected["T"], selected["hit"], margin,
        len(gated), len(band), stage, int(router._client_credit[selected_key] <= 0), epsilon,
    )
