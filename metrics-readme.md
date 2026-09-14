Searched for "crushing"
Viewed export_crushing_plots.py:1-288
Viewed new-metric.py:1-298

The **Export Crushing Plots** script ([`scripts/export_crushing_plots.py`](file:///home/sriram/sem7/p1/LLMServingSim/scripts/export_crushing_plots.py)) generates publication-quality visualization figures for **FairRoute** in **LLMServingSim**. These plots specifically emphasize workload conditions where conventional routers (such as locality-only, fairness-only, or naive round-robin routers) suffer severe performance bottlenecks compared to FairRoute.

Here is a breakdown of all four plots, the metrics they measure, what those metrics represent, and how each metric is calculated.

---

### Summary Table

| Plot | Output File | Scenario | Metric | Representation | Key Formula |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Plot 1** | `plots/locality_speedup_adversarial.png` | `full_adversarial` | **Locality Speedup Factor (LPSF)** | Efficiency of converting KV cache hits into latency reduction under stress | $\frac{\text{HitRatio} \times 1000}{\text{TTFT}_{P99} \text{ (ms)}}$ |
| **Plot 2** | `plots/real_workload_queue_bottleneck.png` | `real_workload_1000` | **P99 Queueing Delay (ms)** | Head-of-line blocking / queue bottleneck severity under real ShareGPT replay | $\text{Quantile}_{0.99}(\text{queuing\_delay\_ms})$ |
| **Plot 3** | `plots/cross_scenario_lpe_score.png` | Geometric mean across 6 scenarios | **Locality-Performance Efficiency (LPE) Score** | Overall throughput, cache reuse, and latency trade-off aggregate | $\text{GeomMean}\left( \frac{\text{Goodput} \cdot (1 + \text{HitRatio})}{\text{TTFT}_{P99} \text{ (s)}} \right)$ |
| **Plot 4** | `plots/adversarial_ttft_p99_seconds.png` | `full_adversarial` | **Tail Latency (P99 TTFT in Sec)** | 99th percentile Time-To-First-Token latency under adversarial traffic | $\frac{\text{Quantile}_{0.99}(\text{TTFT\_ms})}{1000}$ |

---

### Detailed Breakdown of Each Plot & Metric

#### Plot 1: Locality Speedup Factor (LPSF) under Adversarial Traffic
- **File**: [`plots/locality_speedup_adversarial.png`](file:///home/sriram/sem7/p1/LLMServingSim/plots/locality_speedup_adversarial.png)
- **Function**: [`plot1_locality_speedup_adversarial()`](file:///home/sriram/sem7/p1/LLMServingSim/scripts/export_crushing_plots.py#L52-L100)
- **Scenario**: `full_adversarial` (Adversarial stress workload)

##### 1. What the Metric Represents
**LPSF** captures the trade-off between KV cache locality and tail latency responsiveness. While locality-only algorithms achieve high prefix hit ratios by sending requests with matching prefixes to the same node, they often create severe queueing bottlenecks on hot replicas. LPSF measures how effectively a router translates prefix cache hits into actual user speedups without inflating the 99th percentile Time-To-First-Token ($\text{TTFT}_{P99}$). **Higher values are better.**

##### 2. How It Is Calculated
1. **Prefix Hit Ratio per Request**:
   $$\text{prefix\_hit\_ratio}_i = \min\left(1.0, \, \frac{\text{prefix\_cache\_hit}_i}{\max(1, \text{input\_tokens}_i)}\right)$$
2. **Mean Prefix Hit Ratio**:
   $$\text{Mean Prefix Hit Ratio} = \frac{1}{N} \sum_{i=1}^N \text{prefix\_hit\_ratio}_i$$
3. **P99 TTFT (ms)**:
   $$\text{TTFT}_{P99} = \text{Quantile}_{0.99}\left(\{ \text{TTFT\_ms}_i \}_{i=1}^N \right)$$
4. **Locality Speedup Factor (LPSF)**:
   $$\text{LPSF} = \frac{\text{Mean Prefix Hit Ratio} \times 1000.0}{\text{TTFT}_{P99}}$$

---

#### Plot 2: P99 Queueing Delay in Real ShareGPT Replay
- **File**: [`plots/real_workload_queue_bottleneck.png`](file:///home/sriram/sem7/p1/LLMServingSim/plots/real_workload_queue_bottleneck.png)
- **Function**: [`plot2_real_workload_queue_bottleneck()`](file:///home/sriram/sem7/p1/LLMServingSim/scripts/export_crushing_plots.py#L103-L153)
- **Scenario**: `real_workload_1000` (1,000 real ShareGPT replay requests)

##### 1. What the Metric Represents
**P99 Queueing Delay** measures the maximum queuing delay (99th percentile) experienced by requests while waiting in replica queues before prefill execution begins. Under bursty, realistic production arrival patterns, routers that mispredict load or blindly route for locality cause massive head-of-line blocking. **Lower values are better** (plotted on a logarithmic scale).

##### 2. How It Is Calculated
1. **Per-Request Conversion**:
   $$\text{queuing\_delay\_ms}_i = \frac{\text{queuing\_delay}_i \text{ (ns)}}{10^6}$$
2. **99th Percentile Computation**:
   $$\text{P99 Queue Delay (ms)} = \text{Quantile}_{0.99}\left(\{ \text{queuing\_delay\_ms}_i \}_{i=1}^N \right)$$

---

#### Plot 3: Cross-Scenario Locality-Performance Efficiency (LPE Score)
- **File**: [`plots/cross_scenario_lpe_score.png`](file:///home/sriram/sem7/p1/LLMServingSim/plots/cross_scenario_lpe_score.png)
- **Function**: [`plot3_cross_scenario_lpe_score()`](file:///home/sriram/sem7/p1/LLMServingSim/scripts/export_crushing_plots.py#L154-L205)
- **Scenario**: Aggregated across all 6 benchmark scenarios

##### 1. What the Metric Represents
The **LPE Score** evaluates overall system throughput (**Goodput** in req/s), cache utility (**Mean Prefix Hit Ratio**), and latency responsiveness ($\text{TTFT}_{P99}$ in seconds) in a single compound metric. It rewards routers that maintain high SLO-compliant goodput and prompt reuse while penalizing latency spikes. Aggregating via a geometric mean penalizes algorithms that collapse under specific edge-case scenarios. **Higher values are better.**

##### 2. How It Is Calculated
1. **SLO Attainment**: A request satisfies the SLO if its TTFT is $\le 100\text{ ms}$ and total latency is $\le 15{,}000\text{ ms}$:
   $$\text{slo\_met}_i = (\text{TTFT\_ms}_i \le 100.0) \land (\text{latency\_ms}_i \le 15000.0)$$
   $$\text{slo\_attainment\_pct} = 100.0 \times \frac{1}{N}\sum_{i=1}^N \mathbf{1}(\text{slo\_met}_i)$$
2. **Goodput (Requests per Second)**:
   $$\text{Goodput (req/s)} = \left(\frac{\text{slo\_attainment\_pct}}{100.0}\right) \times \frac{N_{\text{requests}}}{\text{End Time (s)}}$$
3. **Per-Scenario LPE Score**:
   $$\text{LPE}_s = \frac{\text{Goodput}_s \times (1.0 + \text{Mean Prefix Hit Ratio}_s)}{\frac{\text{TTFT}_{P99, s}}{1000.0}}$$
4. **Cross-Scenario Aggregation (Geometric Mean)**:
   $$\text{LPE}_{\text{overall}} = \exp\left( \frac{1}{S} \sum_{s=1}^S \ln(\text{LPE}_s + 10^{-4}) \right) \quad \text{for } S=6 \text{ scenarios}$$

---

#### Plot 4: Tail Latency (P99 TTFT in Seconds) under Full Adversarial Traffic
- **File**: [`plots/adversarial_ttft_p99_seconds.png`](file:///home/sriram/sem7/p1/LLMServingSim/plots/adversarial_ttft_p99_seconds.png)
- **Function**: [`plot4_adversarial_ttft_p99_seconds()`](file:///home/sriram/sem7/p1/LLMServingSim/scripts/export_crushing_plots.py#L207-L254)
- **Scenario**: `full_adversarial`

##### 1. What the Metric Represents
**P99 TTFT in Seconds** measures the tail Time-To-First-Token latency under stressful adversarial traffic (where client prompt prefix patterns conflict with balanced node loads). It shows how fast the system responds to 99% of requests when pushed to its limits. **Lower values are better** (plotted on a log scale in seconds).

##### 2. How It Is Calculated
1. **Per-Request TTFT in ms**:
   $$\text{TTFT\_ms}_i = \frac{\text{TTFT}_i \text{ (ns)}}{10^6}$$
2. **Quantile and Second Conversion**:
   $$\text{P99 TTFT (Seconds)} = \frac{\text{Quantile}_{0.99}\left(\{ \text{TTFT\_ms}_i \}_{i=1}^N \right)}{1000.0}$$