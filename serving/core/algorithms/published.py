"""Stateful reference policy primitives for VTC, DLPM, D²LPM and Preble.

The local policies are deliberately independent of Scheduler so accounting and
queue selection can be called at the simulator's admission/completion hooks.
"""
from collections import defaultdict, deque
import math


def iteration_time(batch_size, sequence_count, *, tau0=0.001, w0=0.001,
                   ws=0.00001, p_max=2048.0, k_batch=0.05,
                   k_sequence=0.05, tau_batch=0.00001,
                   tau_sequence=0.00001):
    """NexusSched saturation model used by simulator latency predictions."""
    batch = max(0.0, float(batch_size))
    sequences = max(0.0, float(sequence_count))
    throughput = p_max * (1.0-math.exp(-k_batch*batch)) * (1.0-math.exp(-k_sequence*sequences))
    if throughput <= 0:
        return tau0 + tau_batch*batch + tau_sequence*sequences
    return tau0 + (w0 + ws*sequences)/throughput + tau_batch*batch + tau_sequence*sequences


class VTC:
    """Virtual-time fairness, charged for prompt at admission and output per step."""
    def __init__(self, w_p=1.0, w_q=2.0):
        self.w_p, self.w_q = float(w_p), float(w_q)
        self.counters = defaultdict(float)
        self.last_left = None

    def on_arrival(self, req, waiting):
        client = req.client_id or "default"
        if any((r.client_id or "default") == client for r in waiting):
            return
        peers = {(r.client_id or "default") for r in waiting}
        if peers:
            self.counters[client] = max(self.counters[client], min(self.counters[p] for p in peers))
        elif self.last_left is not None:
            self.counters[client] = max(self.counters[client], self.counters[self.last_left])

    def pick(self, waiting):
        if not waiting:
            return None
        clients = {(r.client_id or "default") for r in waiting}
        client = min(clients, key=lambda c: (self.counters[c], str(c)))
        return min((r for r in waiting if (r.client_id or "default") == client), key=lambda r: (r.arrival, r.id))

    def on_admit(self, req, waiting):
        client = req.client_id or "default"
        if req.num_preemptions == 0:
            self.counters[client] += self.w_p * req.original_input
        if not waiting:
            self.last_left = client

    def on_output_tokens(self, req, count):
        self.counters[req.client_id or "default"] += self.w_q * count


class DLPM:
    """Deficit-based local policy with prefix-aware waiting order."""
    def __init__(self, quantum=2000.0, w_e=1.0, w_q=2.0):
        self.quantum, self.w_e, self.w_q = float(quantum), float(w_e), float(w_q)
        self.deficits = defaultdict(float)
        self.active = set()

    def on_arrival(self, req):
        client = req.client_id or "default"
        self.active.add(client)
        self.deficits.setdefault(client, 0.0)

    def on_output_tokens(self, req, count):
        self.deficits[req.client_id or "default"] -= self.w_q * count

    def refill_for(self, waiting):
        queued = {(r.client_id or "default") for r in waiting}
        if queued and not any(self.deficits[c] > 0 for c in queued):
            for client in self.active:
                if self.deficits[client] <= 0:
                    self.deficits[client] += self.quantum

    def on_admit(self, req, hit_tokens):
        client = req.client_id or "default"
        self.deficits[client] -= self.w_e * max(0, req.original_input - hit_tokens)

    @staticmethod
    def order(waiting, hit_tokens):
        return sorted(waiting, key=lambda r: (-hit_tokens(r), r.arrival, r.id))


class D2LPMRouter:
    """Per-client/per-worker deficits and a dispatch-time prefix shadow index."""
    def __init__(self, workers, quantum=4000.0, w_e=1.0, w_q=2.0, charge="input"):
        self.workers = tuple(range(workers))
        self.quantum, self.w_e, self.w_q = float(quantum), float(w_e), float(w_q)
        if charge not in {"input", "extend"}:
            raise ValueError("charge must be 'input' or 'extend'")
        self.charge = charge
        self.deficits = defaultdict(lambda: defaultdict(float))
        self.queues = [0] * workers
        self.prefix_workers = defaultdict(set)
        self.inflight = {}

    def route(self, client, hashes, input_tokens, block_size, queue_sizes=None):
        matches = {}
        for worker in self.workers:
            n = 0
            for h in hashes:
                if worker not in self.prefix_workers.get(h, ()):
                    break
                n += 1
            matches[worker] = n
        longest = max(matches.values(), default=0)
        locality = {w for w, n in matches.items() if n == longest} if longest else set(self.workers)
        active = {w for w in self.workers if self.deficits[client][w] > 0}
        if not active:
            for w in self.workers:
                self.deficits[client][w] += self.quantum
            active = {w for w in self.workers if self.deficits[client][w] > 0}
        eligible = locality & active
        pool = eligible or active
        loads = queue_sizes if queue_sizes is not None else self.queues
        worker = min(pool, key=lambda w: (loads[w], w))
        cost = input_tokens if self.charge == "input" else max(0, input_tokens - longest * block_size)
        self.deficits[client][worker] -= self.w_e * cost
        self.queues[worker] += 1
        for h in hashes:
            self.prefix_workers[h].add(worker)
        return worker

    def on_finish(self, request_id, output_tokens):
        client, worker = self.inflight.pop(request_id)
        self.deficits[client][worker] -= self.w_q * output_tokens
        self.queues[worker] = max(0, self.queues[worker] - 1)


class PrebleE2:
    """Preble's exploit/explore router with configurable profiled PT/DT curves."""
    def __init__(self, workers, prefill_time, decode_time, match_tokens, imbalance=0.7):
        self.workers = tuple(workers)
        self.PT, self.DT, self.match_tokens = prefill_time, decode_time, match_tokens
        self.imbalance = float(imbalance)
        self.history = {w: deque(maxlen=4096) for w in self.workers}

    def route(self, req, now_ns, queue_load, decode_ratio=None):
        hits = {w: self.match_tokens(w, req) for w in self.workers}
        longest = max(hits.values(), default=0)
        missed = req["input_toks"] - longest
        if longest and missed < longest:
            eligible = [w for w in self.workers if hits[w] == longest]
        else:
            ratios = decode_ratio or {w: 0.0 for w in self.workers}
            heavy = max(self.workers, key=lambda w: ratios[w])
            eligible = [heavy] if ratios[heavy] > self.imbalance else list(self.workers)
        def cost(w):
            hist = self.history[w]
            avg_out = sum(x[2] for x in hist) / len(hist) if hist else 0.0
            queued = sum(self.PT(m) + self.DT(avg_out) for _, m, _ in hist)
            return queued + self.PT(req["input_toks"] - hits[w])
        chosen = min(eligible, key=lambda w: (cost(w), queue_load[w], w))
        self.history[chosen].append((now_ns, req["input_toks"] - hits[chosen], req["output_toks"] - req["input_toks"]))
        return chosen


class PerfModel:
    """NexusSched saturation performance model; observe/fitting is explicit."""
    def __init__(self, tau0=0.002, ws=1.0, tau_b=0.0, tau_s=0.0, pmax=1e5, kb=0.05, ks=1e-3):
        self.tau0, self.ws, self.tau_b, self.tau_s = tau0, ws, tau_b, tau_s
        self.pmax, self.kb, self.ks = pmax, kb, ks
        self.samples = deque(maxlen=20000)

    def predict(self, batch, tokens):
        if batch <= 0 or tokens <= 0:
            return self.tau0
        throughput = self.pmax * (1-math.exp(-self.kb*batch)) * (1-math.exp(-self.ks*tokens))
        return self.tau0 + self.ws*tokens/max(throughput, 1e-12) + self.tau_b*batch + self.tau_s*tokens

    def observe(self, batch, tokens, seconds):
        self.samples.append((batch, tokens, seconds))


def lens_search(model, n_running, n_waiting, max_tokens, max_batch, target, allocate, iterations=10):
    """Choose the feasible batch/token target closest to LENS's adaptive latency."""
    best = None
    for batch in range(max(1, n_running), min(n_running+n_waiting, max_batch)+1):
        lo, hi, selected = batch, max(batch, max_tokens), batch
        for _ in range(iterations):
            mid = (lo+hi)//2
            if model.predict(batch, mid) <= target:
                selected, lo = mid, mid+1
            else:
                hi = mid-1
        plan, actual_batch, actual_tokens = allocate(batch, selected)
        error = abs(model.predict(actual_batch, actual_tokens)-target)
        if best is None or error < best[0]:
            best = (error, plan, batch, selected)
    return None if best is None else best[1:]
