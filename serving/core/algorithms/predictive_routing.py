"""Predictive load-routing algorithms from recent literature.

Five stateful routing algorithms that predict future system load and route
requests to the replica predicted to have the least load. Each algorithm
maintains internal state (completed-request history, trained models, etc.)
that is updated via ``on_request_completed()``.

Algorithms
----------
1. LodestarRouter      -- Online-learning MLP reward predictor (Lodestar, 2026)
2. PastFutureRouter    -- Histogram-based memory prediction (Past-Future, 2025)
3. BFIORouter          -- Balance-Future Integer Optimization (BF-IO, 2026)
4. BalanceRouteHRouter -- Termination classifier + mean regressor (BalanceRoute-H, 2026)
5. QuartzPredRouter    -- Quantile-aware routing (QUARTZ, 2026)

Each class exposes:
    route(req_data, schedulers) -> (instance_idx, score, metrics_tuple)
    on_request_completed(req, instance_idx, current_time_ns)
"""

import math
import random
from collections import deque


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _remaining_tokens(req):
    """Tokens remaining for a running request (output target - reached)."""
    return max(0, req.output - req.num_tokens_reached)


def _instance_load_tokens(sched):
    """Total remaining decode tokens across running + waiting requests."""
    running = sum(_remaining_tokens(r) for r in sched.running)
    waiting = sum(max(0, r.output - r.input) for r in sched.waiting)
    return running + waiting


def _instance_kv_util(sched):
    """KV cache utilization fraction [0, 1]."""
    pool = getattr(sched.memory, 'npu_pool', None)
    if pool is None:
        return 0.0
    total = getattr(pool, 'num_blocks', 0)
    if total <= 0:
        return 0.0
    used = getattr(pool, 'used_blocks', 0)
    return min(1.0, max(0.0, used / total))


def _instance_free_slots(sched):
    """Number of free request slots."""
    cap = getattr(sched, 'max_num_seqs', 0)
    return max(0, cap - len(sched.running))


# ===========================================================================================
# 1. LODESTAR — Online-learning MLP reward predictor (Lim et al., 2026)
#
# Paper: "Lodestar: An Online-Learning LLM Inference Router"
# Algorithm 4: The routing procedure
#
# Core idea: Train a small MLP online to predict reward = -TTFT for each
# (request, instance) pair, then route to argmax predicted reward.
#
# The MLP uses features: [input_toks, kv_hit_ratio, num_running, num_waiting,
#                         inflight_prefill_tokens, inflight_decode_tokens, kv_util]
# Architecture: 3 hidden layers of 128 units, ReLU, dropout=0.1
# Training: retrain every 1000 completed samples using FIFO+replay buffer
# ===========================================================================================

class _SimpleMLP:
    """Minimal 3-layer MLP with ReLU activations (no external dependency).

    Architecture from Lodestar paper: 3 hidden layers, 128 units each,
    ReLU activation, scalar output.
    Trained with SGD on MSE loss.
    """

    def __init__(self, input_dim, hidden=128, lr=1e-3, seed=42):
        self._rng = random.Random(seed)
        self.lr = lr
        self.layers = []
        dims = [input_dim, hidden, hidden, hidden, 1]
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            fan_out = dims[i + 1]
            # Xavier init
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            W = [[self._rng.uniform(-limit, limit) for _ in range(fan_out)]
                 for _ in range(fan_in)]
            b = [0.0] * fan_out
            self.layers.append((W, b))

    def forward(self, x):
        """Forward pass, returns scalar prediction and cached activations."""
        activations = [list(x)]
        h = list(x)
        for idx, (W, b) in enumerate(self.layers):
            fan_in = len(W)
            fan_out = len(b)
            out = list(b)  # copy bias
            for j in range(fan_out):
                s = b[j]
                for i in range(fan_in):
                    s += h[i] * W[i][j]
                out[j] = s
            if idx < len(self.layers) - 1:  # ReLU on hidden layers
                out = [max(0.0, v) for v in out]
            h = out
            activations.append(list(h))
        return h[0], activations

    def train_step(self, x, target):
        """Single SGD step on MSE loss. Returns loss value."""
        pred, acts = self.forward(x)
        loss = (pred - target) ** 2
        # Backprop through layers
        grad_out = [2.0 * (pred - target)]  # dL/d_output

        for idx in range(len(self.layers) - 1, -1, -1):
            W, b = self.layers[idx]
            fan_in = len(W)
            fan_out = len(b)
            h_prev = acts[idx]
            h_cur = acts[idx + 1]

            grad_h = grad_out
            # ReLU derivative for hidden layers
            if idx < len(self.layers) - 1:
                grad_h = [g * (1.0 if h_cur[j] > 0 else 0.0)
                          for j, g in enumerate(grad_h)]

            # Gradient for weights and biases
            grad_input = [0.0] * fan_in
            for j in range(fan_out):
                for i in range(fan_in):
                    dw = h_prev[i] * grad_h[j]
                    W[i][j] -= self.lr * dw
                    grad_input[i] += W[i][j] * grad_h[j]
                b[j] -= self.lr * grad_h[j]

            grad_out = grad_input
        return loss

    def batch_train(self, samples, epochs=5):
        """Train on a batch of (features, target) pairs."""
        total_loss = 0.0
        for _ in range(epochs):
            for x, y in samples:
                total_loss += self.train_step(x, y)
        return total_loss / max(1, len(samples) * epochs)


class LodestarRouter:
    """Lodestar: Online-learning MLP reward predictor.

    Paper: Lim et al., "Lodestar: An Online-Learning LLM Inference Router", 2026.
    Algorithm 4.

    Reward = -TTFT. The MLP predicts expected -TTFT for routing a request to
    each instance. Route to argmax(predicted_reward).

    Features per (request, instance) pair:
        [input_toks_norm, kv_hit_ratio, num_running_norm, num_waiting_norm,
         prefill_tokens_norm, decode_tokens_norm, kv_util]
    """

    INPUT_DIM = 7  # Feature vector size

    def __init__(self, num_instances, seed=42,
                 retrain_interval=1000, fifo_size=5000,
                 replay_size=5000, epsilon=0.05, n_min=50,
                 delta=0.05, hidden=128, lr=1e-3):
        self.num_instances = num_instances
        self.seed = seed
        self._rng = random.Random(seed)
        self.retrain_interval = retrain_interval
        self.epsilon = epsilon
        self.n_min = n_min
        self.delta = delta

        # Per-instance MLP (shared architecture, separate weights per paper)
        self._mlp = _SimpleMLP(self.INPUT_DIM, hidden=hidden, lr=lr, seed=seed)

        # Training buffers (recency + diversity, per paper Section 4.3)
        self._fifo_buffer = deque(maxlen=fifo_size)
        self._replay_buffer = deque(maxlen=replay_size)
        self._sample_count = 0
        self._since_last_train = 0

        # Feature normalization stats (running min/max)
        self._feat_min = [float('inf')] * self.INPUT_DIM
        self._feat_max = [float('-inf')] * self.INPUT_DIM

        # Tracking: which instance each request was routed to, and snapshot
        self._pending = {}  # req_id -> (instance_idx, features_snapshot)

    def _extract_features(self, req_data, sched):
        """Build the 7-dim feature vector for (request, instance)."""
        input_toks = req_data['input_toks']
        num_running = len(sched.running)
        num_waiting = len(sched.waiting)
        prefill_toks = sum(r.original_input - r.num_computed_tokens
                          for r in sched.running if r.is_init)
        decode_toks = sum(1 for r in sched.running if not r.is_init)
        kv_util = _instance_kv_util(sched)

        # KV hit ratio (prefix reuse)
        kv_hit = 0.0
        if hasattr(sched, 'enable_prefix_caching') and sched.enable_prefix_caching:
            input_hash_ids = req_data.get('input_hash_ids', [])
            if input_hash_ids and input_toks > 0:
                kv_hit = min(1.0, len(input_hash_ids) / max(1, input_toks)) * (1.0 - kv_util)

        return [
            float(input_toks) / 4096.0,   # normalize by typical max
            kv_hit,
            float(num_running) / max(1.0, float(sched.max_num_seqs)),
            float(num_waiting) / max(1.0, float(sched.max_num_seqs)),
            float(prefill_toks) / 4096.0,
            float(decode_toks) / max(1.0, float(sched.max_num_seqs)),
            kv_util,
        ]

    def _normalize(self, features):
        """Min-max normalize features to [0, 1]."""
        normed = []
        for i, v in enumerate(features):
            if self._feat_max[i] - self._feat_min[i] > 1e-8:
                normed.append((v - self._feat_min[i]) / (self._feat_max[i] - self._feat_min[i]))
            else:
                normed.append(0.5)
        return normed

    def _update_norm_stats(self, features):
        """Update running feature normalization stats."""
        for i, v in enumerate(features):
            self._feat_min[i] = min(self._feat_min[i], v)
            self._feat_max[i] = max(self._feat_max[i], v)

    def route(self, req_data, schedulers):
        """Route request to instance with highest predicted reward.

        Algorithm 4 from Lodestar paper:
        1. Cold-start: use round-robin if insufficient samples
        2. Epsilon-greedy exploration
        3. Argmax of MLP predictions
        4. Near-best randomization (delta-tolerance)
        """
        n = len(schedulers)
        req_id = req_data.get('index', 0)

        # Step 1: Cold start — round-robin
        if self._sample_count < self.n_min:
            idx = req_id % n
            score = 0.0
            features = self._extract_features(req_data, schedulers[idx])
            self._pending[req_id] = (idx, features)
            return idx, score, (0.0, 0.0, 0.0, score, 0, 0.0)

        # Step 6: Epsilon-greedy exploration
        if self._rng.random() < self.epsilon:
            idx = self._rng.randrange(n)
            features = self._extract_features(req_data, schedulers[idx])
            score, _ = self._mlp.forward(self._normalize(features))
            self._pending[req_id] = (idx, features)
            return idx, score, (0.0, 0.0, 0.0, score, 0, 0.0)

        # Steps 9-11: Predict reward for each instance, argmax
        predictions = []
        all_features = []
        for i, sched in enumerate(schedulers):
            feat = self._extract_features(req_data, sched)
            self._update_norm_stats(feat)
            normed = self._normalize(feat)
            pred, _ = self._mlp.forward(normed)
            predictions.append(pred)
            all_features.append(feat)

        best_pred = max(predictions)

        # Step 18-19: Near-best randomization (delta-tolerance)
        threshold = (1.0 - self.delta) * best_pred if best_pred > 0 else best_pred * (1.0 + self.delta)
        near_best = [i for i, p in enumerate(predictions) if p >= threshold]
        if len(near_best) > 1:
            idx = self._rng.choice(near_best)
        else:
            idx = predictions.index(best_pred)

        score = predictions[idx]
        self._pending[req_id] = (idx, all_features[idx])
        margin = (best_pred - sorted(predictions, reverse=True)[1]) if n > 1 else 0.0
        return idx, score, (0.0, 0.0, 0.0, score, 0, margin)

    def on_request_completed(self, req, instance_idx, current_time_ns):
        """Record completed request and retrain if needed.

        Reward = -TTFT (paper Section 3.1).
        """
        req_id = req.id
        pending_info = self._pending.pop(req_id, None)
        if pending_info is None:
            return

        routed_idx, features = pending_info
        self._update_norm_stats(features)

        # Reward = -TTFT (in microseconds for numerical stability)
        ttft_ns = req.ttft if req.ttft > 0 else (current_time_ns - req.arrival)
        reward = -float(ttft_ns) / 1e6  # Convert to microseconds

        sample = (self._normalize(features), reward)
        self._fifo_buffer.append(sample)
        self._sample_count += 1
        self._since_last_train += 1

        # Retrain periodically (paper: every 1000 samples)
        if self._since_last_train >= self.retrain_interval:
            self._retrain()
            self._since_last_train = 0

    def _retrain(self):
        """Retrain MLP on FIFO + replay buffer (paper Section 4.3)."""
        samples = list(self._fifo_buffer)
        if self._replay_buffer:
            # Mix in replay samples (50/50 as per paper)
            replay_n = min(len(self._replay_buffer), len(samples))
            replay_sample = self._rng.sample(list(self._replay_buffer), replay_n)
            samples.extend(replay_sample)

        if len(samples) < 10:
            return

        # Move oldest FIFO entries to replay (gradient-coreset: keep high-residual)
        while len(self._fifo_buffer) > self._fifo_buffer.maxlen // 2:
            old = self._fifo_buffer.popleft()
            pred, _ = self._mlp.forward(old[0])
            residual = abs(pred - old[1])
            if residual > 0.01 or self._rng.random() < 0.3:
                self._replay_buffer.append(old)

        self._rng.shuffle(samples)
        self._mlp.batch_train(samples, epochs=3)


# ===========================================================================================
# 2. PAST-FUTURE — Histogram-based memory prediction (ASPLOS 2025)
#
# Paper: "Past-Future: An LLM Scheduler with Effective KV-Cache Scheduling"
# Algorithm 1: Past-Future scheduling
#
# Core idea: Build a histogram of past output lengths. For running requests,
# sample from the conditional distribution P(l > l_generated). Calculate peak
# future memory across the batch. Route to instance with lowest peak memory.
# ===========================================================================================

class PastFutureRouter:
    """Past-Future: Histogram-based future memory prediction.

    Paper: Past-Future scheduler, ASPLOS 2025, Algorithm 1.

    Uses historical output length distribution to predict future memory
    occupancy for each instance. Routes to the instance with lowest
    predicted peak memory usage.
    """

    def __init__(self, num_instances, seed=42, window_size=500, block_size=16):
        self.num_instances = num_instances
        self._rng = random.Random(seed)
        self.window_size = window_size
        self.block_size = block_size

        # Sliding window of completed output lengths
        self._output_lengths = deque(maxlen=window_size)

    def _sample_output_length(self):
        """Sample from P(l) — the unconditional output length distribution."""
        if not self._output_lengths:
            return 128  # Default when no history
        return self._rng.choice(list(self._output_lengths))

    def _sample_conditional(self, generated_so_far):
        """Sample from P(l | l > generated_so_far) — conditional distribution.

        Per Algorithm 1, line 4: sample from P(l > l_t^i).
        """
        candidates = [l for l in self._output_lengths if l > generated_so_far]
        if not candidates:
            # All historical outputs are shorter; predict will finish soon
            return generated_so_far + 1
        return self._rng.choice(candidates)

    def _predict_peak_memory(self, sched, new_request_input_toks):
        """Calculate predicted peak future memory for an instance.

        Algorithm 1:
        - For each running request, sample remaining output from P(l > l_t^i)
        - For each waiting + new request, sample total output from P(l)
        - Compute M* = max(M_i) across all future timesteps
        """
        # Collect (prompt_len, generated_so_far, predicted_remaining)
        request_predictions = []

        # Running requests: sample from conditional P(l > generated)
        for req in sched.running:
            generated = req.num_tokens_reached - req.original_input
            if generated < 0:
                generated = 0
            predicted_total = self._sample_conditional(generated)
            remaining = max(0, predicted_total - generated)
            prompt_len = req.original_input
            request_predictions.append((prompt_len, generated, remaining))

        # Waiting requests: sample from unconditional P(l)
        for req in sched.waiting:
            predicted_output = self._sample_output_length()
            prompt_len = req.original_input
            request_predictions.append((prompt_len, 0, predicted_output))

        # New request being considered
        new_output = self._sample_output_length()
        request_predictions.append((new_request_input_toks, 0, new_output))

        # Sort by remaining generation descending (paper: longest-remaining first)
        request_predictions.sort(key=lambda x: x[2], reverse=True)

        # Calculate M* = max(M_i) — peak memory across future steps
        n = len(request_predictions)
        if n == 0:
            return 0.0

        peak_memory = 0.0
        for i in range(n):
            # Memory when request i finishes: all j<=i still alive
            total_kv_tokens = 0
            step_remaining_i = request_predictions[i][2]
            for j in range(n):
                prompt_j, gen_j, rem_j = request_predictions[j]
                if j <= i:
                    actual_gen = min(gen_j + rem_j, gen_j + step_remaining_i)
                    total_kv_tokens += prompt_j + actual_gen
            total_kv_blocks = math.ceil(total_kv_tokens / self.block_size)
            peak_memory = max(peak_memory, total_kv_blocks)

        return peak_memory

    def route(self, req_data, schedulers):
        """Route to instance with lowest predicted peak future memory."""
        input_toks = req_data['input_toks']
        best_idx = 0
        best_score = float('inf')
        scores = []

        for i, sched in enumerate(schedulers):
            predicted_peak = self._predict_peak_memory(sched, input_toks)

            # Normalize by instance capacity
            pool = getattr(sched.memory, 'npu_pool', None)
            total_blocks = getattr(pool, 'num_blocks', 1) if pool else 1
            normalized_load = predicted_peak / max(1, total_blocks)
            scores.append(normalized_load)

            if normalized_load < best_score:
                best_score = normalized_load
                best_idx = i

        score = -best_score  # Higher is better for the framework
        margin = 0.0
        if len(scores) > 1:
            sorted_s = sorted(scores)
            margin = sorted_s[1] - sorted_s[0]  # Gap between best and second
        return best_idx, score, (0.0, 0.0, 0.0, score, 0, margin)

    def on_request_completed(self, req, instance_idx, current_time_ns):
        """Record the actual output length for histogram."""
        actual_output = max(0, req.output - req.original_input)
        self._output_lengths.append(actual_output)


# ===========================================================================================
# 3. BF-IO — Balance-Future Integer Optimization (2026)
#
# Paper: "A Universal Load Balancing Principle and Its Application to LLM Serving"
# Algorithm 1: Balanced-Future I/O scheduling
#
# Core idea: Predict per-GPU load trajectory over a horizon H. Select the
# assignment that minimizes total future load imbalance across GPUs.
# Imbalance = G * max_g(L_g) - sum_g(L_g)
# ===========================================================================================

class BFIORouter:
    """BF-IO: Balance-Future with Integer Optimization.

    Paper: "A Universal Load Balancing Principle", 2026, Algorithm 1.

    Predicts per-GPU load trajectory over H steps. Routes new request to
    the GPU that minimizes cumulative load imbalance over the horizon.
    """

    def __init__(self, num_instances, seed=42, horizon=80, discount=0.95):
        self.num_instances = num_instances
        self._rng = random.Random(seed)
        self.horizon = horizon
        self.discount = discount

        # History for output length estimation
        self._completed_lengths = deque(maxlen=1000)
        self._mean_output = 128.0  # Default estimate

    def _estimate_remaining(self, req):
        """Estimate remaining tokens for a running request.

        Uses oracle output length from trace (req.output) in simulator.
        """
        return max(0, req.output - req.num_tokens_reached)

    def _compute_load_trajectory(self, sched, new_req_remaining=0):
        """Compute load trajectory L_g(k+h) for h=0..H-1.

        Per-step load = number of active decode tokens at step h.
        A request contributes 1 token/step until its remaining count is 0.
        """
        # Gather remaining token counts for all active requests
        remaining = []
        for req in sched.running:
            rem = self._estimate_remaining(req)
            if rem > 0:
                remaining.append(rem)
        for req in sched.waiting:
            # Waiting requests haven't started; estimate their decode length
            est_output = max(0, req.output - req.original_input)
            remaining.append(est_output)

        # Add the new request's contribution
        if new_req_remaining > 0:
            remaining.append(new_req_remaining)

        # Compute load at each step h
        trajectory = []
        for h in range(self.horizon):
            load_h = sum(1 for r in remaining if r > h)
            trajectory.append(load_h)
        return trajectory

    def _compute_imbalance(self, trajectories):
        """Compute total discounted imbalance over horizon.

        Imbalance(k+h) = G * max_g(L_g(k+h)) - sum_g(L_g(k+h))
        J = sum_{h=0}^{H-1} discount^h * Imbalance(k+h)
        """
        G = len(trajectories)
        total = 0.0
        for h in range(self.horizon):
            loads_h = [traj[h] for traj in trajectories]
            imbalance_h = G * max(loads_h) - sum(loads_h)
            total += (self.discount ** h) * imbalance_h
        return total

    def route(self, req_data, schedulers):
        """Route to instance minimizing future load imbalance.

        Algorithm 1: For each candidate instance, compute the load trajectory
        with the new request assigned there, then pick the one with minimum
        total discounted imbalance.
        """
        input_toks = req_data['input_toks']
        output_toks = req_data['output_toks']
        est_decode = max(1, output_toks - input_toks)

        best_idx = 0
        best_imbalance = float('inf')
        scores = []

        for candidate in range(len(schedulers)):
            # Compute trajectories for all instances
            trajectories = []
            for i, sched in enumerate(schedulers):
                if i == candidate:
                    traj = self._compute_load_trajectory(sched, new_req_remaining=est_decode)
                else:
                    traj = self._compute_load_trajectory(sched, new_req_remaining=0)
                trajectories.append(traj)

            imbalance = self._compute_imbalance(trajectories)
            scores.append(imbalance)

            if imbalance < best_imbalance:
                best_imbalance = imbalance
                best_idx = candidate

        score = -best_imbalance  # Higher is better
        margin = 0.0
        if len(scores) > 1:
            sorted_s = sorted(scores)
            margin = sorted_s[1] - sorted_s[0]
        return best_idx, score, (0.0, 0.0, 0.0, score, 0, margin)

    def on_request_completed(self, req, instance_idx, current_time_ns):
        """Track completed output lengths for future estimation."""
        actual = max(0, req.output - req.original_input)
        self._completed_lengths.append(actual)
        if self._completed_lengths:
            self._mean_output = sum(self._completed_lengths) / len(self._completed_lengths)


# ===========================================================================================
# 4. BALANCEROUTE-H — Termination classifier + conditional mean regressor (2026)
#
# Paper: "Online Routing at Scale for LLM Serving"
# Stage 2 subset-selection algorithm
#
# Core idea: Train two lightweight models on completed requests:
#   (a) Termination classifier: P(request finishes within H steps | progress)
#   (b) Conditional mean regressor: E[remaining | not finished in H steps]
# Expected contribution: c_i = (1 - p_fin) * H + p_fin * mu_rem
# Route to GPU that minimizes horizon-discounted load imbalance.
# ===========================================================================================

class _OnlineLogistic:
    """Simple online logistic regression (single feature: progress ratio)."""

    def __init__(self, lr=0.01):
        self.w = 0.0
        self.b = 0.0
        self.lr = lr

    def predict(self, x):
        z = self.w * x + self.b
        z = max(-20.0, min(20.0, z))  # Clamp for numerical stability
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, x, y):
        pred = self.predict(x)
        error = pred - y
        self.w -= self.lr * error * x
        self.b -= self.lr * error


class _OnlineMeanRegressor:
    """Online regressor for E[remaining_tokens | not_finished]."""

    def __init__(self, lr=0.01):
        self.w = 1.0
        self.b = 50.0
        self.lr = lr

    def predict(self, x):
        return max(1.0, self.w * x + self.b)

    def update(self, x, y):
        pred = self.predict(x)
        error = pred - y
        self.w -= self.lr * error * x
        self.b -= self.lr * error


class BalanceRouteHRouter:
    """BalanceRoute-H: Termination classifier + conditional mean regressor.

    Paper: "Online Routing at Scale for LLM Serving", 2026.

    Two online models predict:
    - p_fin: probability request finishes within H steps
    - mu_rem: expected remaining tokens if it doesn't finish
    Expected contribution: c_i = (1 - p_fin) * H + p_fin * mu_rem
    Route to minimize horizon-discounted load imbalance.
    """

    def __init__(self, num_instances, seed=42, horizon=80,
                 discount=0.64, alpha=0.95, beta=14.67):
        self.num_instances = num_instances
        self._rng = random.Random(seed)
        self.horizon = horizon
        self.discount = discount
        self.alpha = alpha
        self.beta = beta

        # Online models (paper: lightweight, trained on completed requests)
        self._termination_clf = _OnlineLogistic(lr=0.01)
        self._remaining_reg = _OnlineMeanRegressor(lr=0.001)

        # Statistics
        self._completed_count = 0
        self._mean_output = 128.0

    def _expected_contribution(self, req):
        """Compute expected in-window contribution c_i(k) per paper Eq. (4).

        c_i = (1 - p_fin) * H + p_fin * mu_rem
        Feature: progress = tokens_generated / estimated_total
        """
        generated = max(0, req.num_tokens_reached - req.original_input)
        est_total = max(1, req.output - req.original_input)
        progress = min(1.0, generated / est_total)

        p_fin = self._termination_clf.predict(progress)
        mu_rem = self._remaining_reg.predict(float(generated))

        # Expected tokens this request contributes over next H steps
        c_i = (1.0 - p_fin) * self.horizon + p_fin * mu_rem
        return min(c_i, self.horizon)  # Cap at horizon

    def _compute_gpu_load(self, sched, new_req_contribution=0):
        """Per-GPU decode load projected over horizon.

        Returns load trajectory [L_g(h)]_{h=0}^{H-1}.
        """
        contributions = []
        for req in sched.running:
            c = self._expected_contribution(req)
            contributions.append(c)

        if new_req_contribution > 0:
            contributions.append(new_req_contribution)

        trajectory = []
        for h in range(self.horizon):
            active = sum(1 for c in contributions if c > h)
            trajectory.append(active)
        return trajectory

    def _horizon_fscore(self, trajectories, candidate_idx):
        """Compute the horizon F-score for a candidate assignment.

        F_g(Q) = alpha * (1^T * d) * Delta_s - beta * (Delta_s * 1 - m_g)_+^T * d

        Simplified: penalize overload above mean across horizon.
        """
        G = len(trajectories)
        total_score = 0.0

        for h in range(self.horizon):
            d_h = self.discount ** h
            loads_h = [traj[h] for traj in trajectories]
            mean_load = sum(loads_h) / max(1, G)
            # Reward: total throughput (proportional to sum of loads)
            throughput = sum(loads_h) * d_h
            # Penalty: overload above mean
            overload = max(0, loads_h[candidate_idx] - mean_load)
            penalty = overload * d_h
            total_score += self.alpha * throughput - self.beta * penalty

        return total_score

    def route(self, req_data, schedulers):
        """Route using the horizon F-score objective."""
        input_toks = req_data['input_toks']
        output_toks = req_data['output_toks']
        est_decode = max(1, output_toks - input_toks)
        new_contribution = min(est_decode, self.horizon)

        best_idx = 0
        best_score = float('-inf')
        scores = []

        for candidate in range(len(schedulers)):
            trajectories = []
            for i, sched in enumerate(schedulers):
                if i == candidate:
                    traj = self._compute_gpu_load(sched, new_contribution)
                else:
                    traj = self._compute_gpu_load(sched)
                trajectories.append(traj)

            fscore = self._horizon_fscore(trajectories, candidate)
            scores.append(fscore)

            if fscore > best_score:
                best_score = fscore
                best_idx = candidate

        margin = 0.0
        if len(scores) > 1:
            sorted_s = sorted(scores, reverse=True)
            margin = sorted_s[0] - sorted_s[1]
        return best_idx, best_score, (0.0, 0.0, 0.0, best_score, 0, margin)

    def on_request_completed(self, req, instance_idx, current_time_ns):
        """Update termination classifier and remaining regressor."""
        actual_output = max(0, req.output - req.original_input)
        self._completed_count += 1

        # Update mean
        self._mean_output += (actual_output - self._mean_output) / self._completed_count

        # Train termination classifier:
        # For various "checkpoints" of progress, did it finish within H?
        for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            tokens_at_checkpoint = int(frac * actual_output)
            remaining_from_checkpoint = actual_output - tokens_at_checkpoint
            finished_in_H = 1.0 if remaining_from_checkpoint <= self.horizon else 0.0
            self._termination_clf.update(frac, finished_in_H)

            # Train remaining regressor on samples where it didn't finish
            if remaining_from_checkpoint > self.horizon:
                self._remaining_reg.update(float(tokens_at_checkpoint),
                                           float(remaining_from_checkpoint))


# ===========================================================================================
# 5. QUARTZ — Quantile-Aware Routing (2026)
#
# Paper: "QUARTZ: Quantile-Aware Routing for LLM Serving"
#
# Core idea: Predict alpha-quantile of prefill service time using request and
# worker features. Add estimated waiting time from queueing theory
# (Pollaczek-Khinchin formula). Route to argmin of total predicted TTFT.
#
# Score: S_alpha(r,w) = W_hat(w) + Q_alpha_prefill(r,w) + psi(r,w)
# where W_hat = waiting time, Q_alpha = quantile prefill time, psi = locality
# ===========================================================================================

class _QuantileEstimator:
    """Online quantile estimator.

    Tracks the alpha-quantile of a stream of observations.
    """

    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self._sorted = []
        self._count = 0

    def update(self, value):
        import bisect
        bisect.insort(self._sorted, value)
        self._count += 1
        # Keep only last 2000 observations for memory
        if self._count > 2000:
            self._sorted = self._sorted[-2000:]
            self._count = 2000

    def quantile(self):
        if not self._sorted:
            return 100.0  # Default high value
        idx = int(self.alpha * (len(self._sorted) - 1))
        return self._sorted[idx]


class _ServiceTimePredictor:
    """Predicts alpha-quantile of prefill service time.

    Uses bucketed quantile estimation based on prompt length.
    Per paper: trained using pinball loss on features (prompt_len, max_tokens,
    decoding_params, locality_proxy, worker_telemetry).

    In LLMServingSim, we approximate with: prompt_len -> service_time_quantile.
    """

    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self._observations = deque(maxlen=2000)
        # Simple linear model: service_time = a * prompt_len + b
        self._a = 1.0
        self._b = 0.0
        self._count = 0
        self._quantile_estimators = {}  # bucket -> QuantileEstimator

    def _bucket(self, prompt_len):
        """Bucket prompt lengths for quantile estimation."""
        if prompt_len < 128:
            return 'short'
        elif prompt_len < 512:
            return 'medium'
        elif prompt_len < 2048:
            return 'long'
        else:
            return 'very_long'

    def predict(self, prompt_len):
        """Predict alpha-quantile of prefill service time (in ns)."""
        bucket = self._bucket(prompt_len)
        if bucket in self._quantile_estimators:
            return self._quantile_estimators[bucket].quantile()
        # Default: linear estimate
        return max(1.0, self._a * prompt_len + self._b)

    def update(self, prompt_len, actual_service_time):
        """Update quantile estimate with observed service time."""
        bucket = self._bucket(prompt_len)
        if bucket not in self._quantile_estimators:
            self._quantile_estimators[bucket] = _QuantileEstimator(self.alpha)
        self._quantile_estimators[bucket].update(actual_service_time)
        self._count += 1

        # Update linear model coefficients
        self._observations.append((prompt_len, actual_service_time))
        if len(self._observations) >= 20:
            xs = [o[0] for o in self._observations]
            ys = [o[1] for o in self._observations]
            n = len(xs)
            x_mean = sum(xs) / n
            y_mean = sum(ys) / n
            cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
            var = sum((x - x_mean) ** 2 for x in xs)
            if var > 1e-8:
                self._a = cov / var
                self._b = y_mean - self._a * x_mean


class QuartzPredRouter:
    """QUARTZ: Quantile-aware routing for LLM serving.

    Paper: QUARTZ, 2026.

    Score: S_alpha(r, w) = W_hat(w) + Q_alpha_prefill(r, w) + psi(r, w)
    - W_hat(w): estimated waiting time from Pollaczek-Khinchin formula
    - Q_alpha_prefill(r, w): alpha-quantile predicted prefill time
    - psi(r, w): locality/prefix bonus

    Route to argmin(S_alpha).
    Hyperparameters from paper: alpha=0.95, lambda=0.02, beta=0.10.
    """

    def __init__(self, num_instances, seed=42, alpha=0.95,
                 fairness_lambda=0.02, fairness_beta=0.10):
        self.num_instances = num_instances
        self._rng = random.Random(seed)
        self.alpha = alpha
        self.fairness_lambda = fairness_lambda
        self.fairness_beta = fairness_beta

        # Per-worker service time quantile estimator
        self._service_predictor = _ServiceTimePredictor(alpha=alpha)

        # Per-worker backlog tracking
        self._service_rate_estimator = deque(maxlen=1000)
        self._mean_service = 100.0  # Initial estimate (ns)
        self._var_service = 1000.0  # Initial variance

        # Client service counters (for fairness)
        self._client_counters = {}

    def _waiting_time_pk(self, sched):
        """Pollaczek-Khinchin waiting time estimate.

        W = rho * E[S] * (1 + C_s^2) / (2 * (1 - rho))
        where rho = utilization, C_s = CoV of service time.
        """
        queue_len = len(sched.waiting)
        active = len(sched.running)
        capacity = max(1, getattr(sched, 'max_num_seqs', 1))

        # Approximate utilization
        rho = min(0.99, (active + queue_len * 0.5) / capacity)

        # Coefficient of variation squared
        if self._var_service > 0 and self._mean_service > 0:
            c_s_sq = self._var_service / (self._mean_service ** 2)
        else:
            c_s_sq = 1.0

        if rho >= 0.99:
            return queue_len * self._mean_service  # Fallback

        # P-K formula
        w = rho * self._mean_service * (1.0 + c_s_sq) / (2.0 * (1.0 - rho))
        # Scale by queue depth
        return w * max(1, queue_len)

    def _locality_bonus(self, sched, req_data):
        """Prefix cache locality bonus (lower is better, so return negative)."""
        if not getattr(sched, 'enable_prefix_caching', False):
            return 0.0
        input_toks = req_data['input_toks']
        kv_util = _instance_kv_util(sched)
        return -kv_util * input_toks * 0.1

    def route(self, req_data, schedulers):
        """Route to worker with minimum predicted TTFT.

        S_alpha(r, w) = W_hat(w) + Q_alpha_prefill(r, w) + psi(r, w)
        """
        input_toks = req_data['input_toks']
        client_id = req_data.get('client_id', 'default')

        best_idx = 0
        best_cost = float('inf')
        scores = []

        for i, sched in enumerate(schedulers):
            # Component 1: Waiting time (P-K formula)
            w_hat = self._waiting_time_pk(sched)

            # Component 2: Alpha-quantile of prefill service time
            q_alpha = self._service_predictor.predict(input_toks)

            # Component 3: Locality bonus
            psi = self._locality_bonus(sched, req_data)

            # Total score (lower is better)
            s_alpha = w_hat + q_alpha + psi

            # Add fairness term (paper Eq. for k_fair)
            client_counter = self._client_counters.get(client_id, 0.0)
            s_alpha += self.fairness_beta * client_counter

            scores.append(s_alpha)
            if s_alpha < best_cost:
                best_cost = s_alpha
                best_idx = i

        score = -best_cost  # Higher is better for framework
        margin = 0.0
        if len(scores) > 1:
            sorted_s = sorted(scores)
            margin = sorted_s[1] - sorted_s[0]
        return best_idx, score, (0.0, 0.0, 0.0, score, 0, margin)

    def on_request_completed(self, req, instance_idx, current_time_ns):
        """Update service time statistics and client counters."""
        # Service time = TTFT (prefill time)
        if req.ttft > 0:
            service_time = float(req.ttft)
            self._service_predictor.update(req.original_input, service_time)
            self._service_rate_estimator.append(service_time)

            # Update mean and variance
            if self._service_rate_estimator:
                vals = list(self._service_rate_estimator)
                self._mean_service = sum(vals) / len(vals)
                if len(vals) > 1:
                    self._var_service = sum((v - self._mean_service) ** 2
                                          for v in vals) / (len(vals) - 1)

        # Update client counter
        client_id = req.client_id or 'default'
        gen_toks = max(0, req.output - req.original_input)
        self._client_counters[client_id] = self._client_counters.get(client_id, 0.0) + gen_toks


# ---------------------------------------------------------------------------
# Registry: policy name -> router class
# ---------------------------------------------------------------------------

PREDICTIVE_ROUTER_CLASSES = {
    'LODESTAR': LodestarRouter,
    'PAST_FUTURE': PastFutureRouter,
    'BF_IO': BFIORouter,
    'BALANCEROUTE_H': BalanceRouteHRouter,
    'QUARTZ_PRED': QuartzPredRouter,
}
