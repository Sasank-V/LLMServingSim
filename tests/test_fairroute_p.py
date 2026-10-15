"""Regression checks for calibrated FairRoute-P's routing invariants."""
from serving.core.algorithms.fairroute_p import select_fairroute_p


class _Scheduler:
    def __init__(self, run=0, wait=0, slots=4, hit=0, queued=0, free_kv=25000):
        self.running = [object()] * run
        self.waiting = [object()] * wait
        self.max_num_seqs = slots
        self.hit = hit
        self.queued = queued
        self.free_kv = free_kv


class _Router:
    def __init__(self):
        self._policy_params = {}
        self._known_clients = set()
        self._client_credit = {}
        self._replica_service = {}
        self._routing_stage_counts = {}
        self._routed_scores = []
        self._routed_margins = []

    def _client_output_estimate(self, client):
        return 32.0

    def _replica_locality(self, scheduler, request):
        return scheduler.hit

    def _queued_tokens(self, scheduler):
        return scheduler.queued

    def _free_kv_tokens(self, scheduler):
        return scheduler.free_kv


def _request(index=0, client=1):
    return {"index": index, "input_toks": 3200, "client_id": client}


def _pick(schedulers, request=None, router=None):
    router = router or _Router()
    choice = select_fairroute_p(router, schedulers, request or _request())
    return choice[0], router


def test_idle_misses_spread_by_deterministic_jitter():
    router, counts = _Router(), [0] * 4
    for index in range(400):
        selected, _ = _pick([_Scheduler() for _ in range(4)], _request(index, index % 8), router)
        counts[selected] += 1
    assert counts == [100, 100, 100, 100]


def test_locality_yields_to_queue_and_slot_pressure():
    selected, _ = _pick([_Scheduler(), _Scheduler(hit=3072, queued=200), _Scheduler(), _Scheduler()])
    assert selected == 1
    selected, _ = _pick([_Scheduler(), _Scheduler(hit=3072, queued=20000, run=4, wait=3),
                         _Scheduler(), _Scheduler()])
    assert selected != 1
    selected, _ = _pick([_Scheduler(run=4, wait=2, queued=100), _Scheduler(run=1, queued=100),
                         _Scheduler(run=4, wait=2, queued=100), _Scheduler(run=4, wait=2, queued=100)])
    assert selected == 1


def test_fallback_is_recorded_for_infeasible_candidates():
    _, router = _pick([_Scheduler(queued=10**7)] * 4)
    assert router._routing_stage_counts == {"fallback:fp": 1}
