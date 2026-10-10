from types import SimpleNamespace

from serving.core.router import Router


class _Pool:
    num_blocks = 1000
    used_blocks = 0
    block_size = 16


class _KV:
    def get_computed_blocks(self, request):
        return [], 0, 0


class _Memory:
    npu_pool = _Pool()
    kv = _KV()


def _scheduler(index):
    return SimpleNamespace(
        instance_id=index,
        pd_type="prefill",
        waiting=[],
        running=[],
        max_num_seqs=128,
        max_num_batched_tokens=2048,
        enable_prefix_caching=False,
        memory=_Memory(),
        config={"max_position_embeddings": 8192},
    )


def _router(policy):
    schedulers = [_scheduler(i) for i in range(4)]
    return Router(4, schedulers, 0, routing_policy=policy)


def _request():
    return {
        "index": 1,
        "input_toks": 512,
        "arrival_time_ns": 0,
        "client_id": 7,
        "input_hash_ids": [],
        "output_hash_ids": [],
    }


def test_fairroute_zero_slack_picks_minimum_prediction():
    router = _router("FAIRROUTE")
    router._policy_params["fairroute_eps_max"] = 0.0
    candidates, _ = router._candidate_set(router.prefill_schedulers, _request())
    expected = min(candidates, key=lambda item: item["predicted_latency"])["index"]
    chosen, _, _ = router._select_fairroute_v2(router.prefill_schedulers, _request())
    assert chosen == expected


def test_rr_and_load_do_not_read_client_id():
    for policy in ("RR", "LOAD"):
        router = _router(policy)
        schedulers = router.prefill_schedulers
        first = router._select_instance(schedulers, "prefill")
        second = router._select_instance(schedulers, "prefill")
        assert first != second
        assert router._least_load_select(schedulers, "prefill") in range(4)


def test_prism_and_d2lpm_are_registered():
    assert _router("PRISM").routing_policy == "PRISM"
    assert _router("D2LPM").routing_policy == "D2LPM"
