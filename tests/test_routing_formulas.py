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


# ---------------------------------------------------------------------------
# Client-level fairness tests
# ---------------------------------------------------------------------------

def _fairroute_candidate(index, load, latency):
    """Minimal candidate dict for choose_fairroute_candidate tests."""
    return {
        "index": index,
        "safe": True,
        "hit": 0,
        "locality": 0.0,
        "future_locality": 0.0,
        "fairness_credit": 100.0,
        "replica_deficit": 0.0,
        "load": load,
        "predicted_latency": latency,
    }


def test_high_urgency_routes_to_less_loaded_replica():
    """With equal predicted latency, high urgency must prefer the less-loaded replica."""
    from serving.core.algorithms.fairroute import choose_fairroute_candidate

    candidates = [
        _fairroute_candidate(index=0, load=0.9, latency=0.5),  # heavy
        _fairroute_candidate(index=1, load=0.1, latency=0.5),  # light
    ]

    for mode in ("GATED", "LINEAR", "LEARNED", "SLACK_LEXICOGRAPHIC"):
        selected, _ = choose_fairroute_candidate(
            candidates, mode=mode, urgency=1.0, eps_max=0.5
        )
        assert selected["index"] == 1, (
            f"mode={mode}: urgency=1 should route to less-loaded replica (index 1), "
            f"got {selected['index']}"
        )


def test_zero_urgency_selects_on_latency_alone():
    """With urgency=0, the client-level term is zero and latency dominates."""
    from serving.core.algorithms.fairroute import choose_fairroute_candidate

    candidates = [
        _fairroute_candidate(index=0, load=0.9, latency=0.3),  # heavy but fast
        _fairroute_candidate(index=1, load=0.1, latency=0.7),  # light but slow
    ]

    for mode in ("GATED", "LINEAR", "SLACK_LEXICOGRAPHIC"):
        selected, _ = choose_fairroute_candidate(
            candidates, mode=mode, urgency=0.0, eps_max=0.0
        )
        assert selected["index"] == 0, (
            f"mode={mode}: urgency=0 with eps_max=0 should pick fastest replica (index 0), "
            f"got {selected['index']}"
        )


def test_fairroute_v2_service_urgency_steers_starved_client():
    """A client with zero service (maximally starved) should be steered to the
    less-loaded replica in FAIRROUTE / FAIRROUTE_V2 via service-based urgency."""
    router = _router("FAIRROUTE_V2")

    # Prime the router so one client has received service but the test client has not.
    router._total_service = 500.0
    router._known_clients.add("other_client")
    router.fairness_tracker.record_service("other_client", 500.0)
    # Test client has zero service recorded — debt is at maximum.

    # Set up two schedulers: index 0 is heavy (many waiting), index 1 is idle.
    schedulers = router.prefill_schedulers

    # Make scheduler 0 appear busy
    class _FakeReq:
        num_tokens = 512
        num_computed_tokens = 0
        input = 512

    schedulers[0].waiting = [_FakeReq(), _FakeReq(), _FakeReq()]
    schedulers[0].running = [_FakeReq(), _FakeReq()]
    schedulers[1].waiting = []
    schedulers[1].running = []

    req = {
        "index": 99,
        "input_toks": 256,
        "output_toks": 512,
        "arrival_time_ns": 0,
        "client_id": "starved_client",
        "input_hash_ids": [],
        "output_hash_ids": [],
    }
    chosen, _, _ = router._select_fairroute_v2(schedulers, req)
    assert chosen == 1, (
        "A starved client (zero service) should be routed to the idle replica (index 1)"
    )


def test_decision_sidecar_recording_and_persistence(tmp_path):
    """Test that Router records decisions into sidecar and writes valid JSONL."""
    import json

    class _FullScheduler:
        def __init__(self, idx):
            self.instance_id = idx
            self.pd_type = "prefill"
            self.waiting = []
            self.running = []
            self.max_num_seqs = 128
            self.max_num_batched_tokens = 2048
            self.enable_prefix_caching = False
            self.memory = _Memory()
            self.model = "meta-llama/Llama-3.1-8B"
            self.config = {"max_position_embeddings": 8192}

        def add_request(self, args, is_init=True):
            from serving.core.request import Request
            req = Request(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
            self.waiting.append(req)
            return req

    for pol in ("FAIRROUTE", "LOAD", "RR"):
        schedulers = [_FullScheduler(i) for i in range(4)]
        router = Router(4, schedulers, 0, routing_policy=pol)
        router._pending_requests = [
            {
                "index": i,
                "input_toks": 256 + i * 32,
                "output_toks": 512,
                "arrival_time_ns": i * 1000,
                "client_id": f"client_{i % 3}",
            }
            for i in range(5)
        ]
        routed = router.route_arrived_requests(100000)
        assert routed == 5
        assert len(router._decision_records) == 5

        for rec in router._decision_records:
            assert "request_id" in rec
            assert "chosen_replica" in rec
            assert "per_replica_predicted_latency" in rec
            assert len(rec["per_replica_predicted_latency"]) == 4
            assert "stage" in rec
            assert "gate_size" in rec
            assert "slack_size" in rec
            assert "urgency" in rec
            assert "top1_agreement" in rec
            assert "locality_sacrificed" in rec

        # Test sidecar persistence
        csv_file = tmp_path / f"test_{pol}.csv"
        csv_file.write_text("header\n")
        router.save_decision_sidecar(str(csv_file))

        sidecar_file = tmp_path / f"test_{pol}.csv.sidecar.jsonl"
        assert sidecar_file.exists()
        lines = sidecar_file.read_text().strip().split("\n")
        assert len(lines) == 5
        loaded = json.loads(lines[0])
        assert loaded["policy"] == pol

        # Check diagnostics
        diag = router.get_decision_diagnostics()
        assert diag["count"] == 5
        assert "stage_counts" in diag
        assert "top1_agreement_rate" in diag

