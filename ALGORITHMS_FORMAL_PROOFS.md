# Formal Proofs for the FairRoute Algorithms (`serving/core/algorithms/` + `router.py`, LLMServingSim v2)

This document proves the mathematical properties of every algorithm described in `ALGORITHMS_ANALYSIS.md` and `algorithms_deep_analysis.md`.

---

## 0. Scope, Conventions, and Honest Limits

**What is proved.** Every statement below is a theorem about the *mathematical specification* given in the two analysis documents. I did not have the `.py` sources. Where a formula's behaviour contradicts a prose claim in those documents, I prove what the formula actually does and flag the discrepancy (collected in §9).

**Conventions.**

* All arithmetic is exact real arithmetic. Floating-point effects are mentioned only where they matter.
* $\epsilon = 10^{-8} > 0$. Unless stated otherwise, every model parameter is strictly positive.
* A *candidate set* $\mathcal C$ is finite. Each candidate $c$ carries a real predicted latency $\hat T_c > 0$ and boolean `safe`$_c$.
* $\operatorname{clip}(x,a,b)=\min(b,\max(a,x))$.
* "Nondecreasing/increasing" are used in the weak/strict sense respectively.

**Standing assumptions** (each is used only where cited).

| Id | Assumption |
|---|---|
| **A1** | Service amounts passed to `record_service` and latencies passed to `record_latency` are $\ge 0$. |
| **A2** | $S_{\text{total}} = \sum_{c\in\mathcal C} S_c$ over the *same* client set whose size is $N_{\text{clients}}$. |
| **A3** | $0 \le H \le N_{\text{in}}$ (a prefix hit cannot exceed the prompt). |
| **A4** | $0 \le M_{\text{used}} \le M_{\text{total}}$ for KV accounting. |
| **A5** | Replica-level `locality` $h_c \in [0,1]$ and `replica_deficit` $d_c \in [0,1]$ (needed only for bounded-trade-off statements in §6.5). |

**Not proved.** `normalize`, `normalize_inverse`, `load_benefit` (formulas not given in the analysis documents), and MD5's statistical quality (§7.5 is conditional on a random-oracle model, which is a modelling assumption, not a theorem).

---

## 1. `fairness.py`

### 1.1 `FairnessTracker.record_service` and `get_service`

**Proposition 1.1.** Under A1, after $m$ calls with amounts $a_1,\dots,a_m$ for client $c$, $S_c=\sum_{k=1}^m a_k$. Hence $S_c\ge 0$, $S_c$ is nondecreasing in time, and $\sum_c S_c$ equals the sum of all amounts ever recorded.

*Proof.* Induct on $m$. Base: $S_c=0$ (the default of `get_service` for unseen clients $=\sum_{\emptyset}$). Step: $S_c\leftarrow S_c+a_{m+1}=\sum_{k\le m+1}a_k$. Since $a_k\ge0$, partial sums are nonnegative and nondecreasing. Summing over clients, each recorded amount is added to exactly one client, so the grand total equals the sum of all amounts. ∎

### 1.2 `record_latency` and `get_mean_latency`

**Proposition 1.2.** Let $\ell_1,\dots,\ell_{N_c}$ be the latencies recorded for $c$. Then $L^{\text{total}}_c=\sum_k\ell_k$, $N_c$ is their count, and for $N_c>0$:
$$\min_k \ell_k\;\le\;\bar L_c=\frac{L_c^{\text{total}}}{N_c}\;\le\;\max_k \ell_k .$$
Moreover the mean obeys the online recursion $\bar L_c^{\text{new}}=\bar L_c+\dfrac{\ell-\bar L_c}{N_c+1}$.

*Proof.* The sum/count invariants follow by induction as in Prop. 1.1. The mean of finitely many reals lies between their min and max. For the recursion: $\dfrac{L+\ell}{N+1}=\dfrac{N\bar L+\ell}{N+1}=\bar L+\dfrac{\ell-\bar L}{N+1}$. For $N_c\le 0$ the function returns $0$, avoiding division by zero. ∎

### 1.3 `jains_index_latency`

Let $x_1,\dots,x_n\ge0$ ($n\ge2$) be the mean latencies of active clients, $J(x)=\dfrac{(\sum x_i)^2}{n\sum x_i^2}$, and $J:=1$ if $\sum x_i^2=0$ or $n<2$.

**Theorem 1.3.**

1. **Upper bound.** $J(x)\le 1$, with equality iff $x_1=\dots=x_n$.
2. **Lower bound.** $J(x)\ge 1/n$, with equality iff at most one $x_i$ is nonzero (and not all are zero).
3. **Scale invariance.** $J(ax)=J(x)$ for all $a>0$. In particular $J$ is independent of the time unit (ns vs s).
4. **Variance form.** With $\mu=\frac1n\sum x_i$ and $\sigma^2=\frac1n\sum(x_i-\mu)^2$ (population variance), if $\mu>0$: $J(x)=\dfrac{1}{1+\sigma^2/\mu^2}=\dfrac1{1+\mathrm{CV}^2}$.
5. **Pigou–Dalton monotonicity.** Transferring $\delta$ from a larger $x_i$ to a smaller $x_j$ with $0<\delta<x_i-x_j$ strictly increases $J$.

*Proof.*

(1) Cauchy–Schwarz: $(\sum_i 1\cdot x_i)^2\le(\sum_i 1^2)(\sum_i x_i^2)=n\sum x_i^2$, so $J\le1$. Equality iff $(x_i)$ is proportional to $(1,\dots,1)$, i.e. all equal. If all are zero, the code returns $1$, consistent with "all equal."

(2) $(\sum x_i)^2=\sum x_i^2+2\sum_{i<j}x_ix_j\ge\sum x_i^2$ because $x_ix_j\ge0$. Dividing by $n\sum x_i^2>0$ gives $J\ge 1/n$. Equality iff every cross-product $x_ix_j=0$ for $i\ne j$, i.e. at most one nonzero entry.

(3) Numerator and denominator are both homogeneous of degree 2.

(4) $\sum x_i^2=n(\sigma^2+\mu^2)$ and $(\sum x_i)^2=n^2\mu^2$, so $J=\dfrac{n^2\mu^2}{n\cdot n(\sigma^2+\mu^2)}=\dfrac{\mu^2}{\mu^2+\sigma^2}$.

(5) $\sum x_i$ is unchanged and $\sum x^2$ changes by $(x_i-\delta)^2+(x_j+\delta)^2-x_i^2-x_j^2=2\delta(\delta-(x_i-x_j))<0$. With a fixed numerator and a smaller denominator, $J$ increases. ∎

*Remark.* $J=1$ is perfect *latency equality*, which equals fairness only if clients' requests are comparable in size. The analysis document's rationale (token-based $J$ is routing-invariant) is consistent with this: $J$ on latency is routing-sensitive.

### 1.4 `fairness_target`

**Proposition 1.4.** Under A2, for $N_{\text{clients}}>0$, $T=\frac1N\sum_cS_c$ is the arithmetic mean of client service, and
$$\sum_c (S_c-T)=0,\qquad \sum_c\max(0,T-S_c)=\sum_c\max(0,S_c-T)=\tfrac12\sum_c|S_c-T| .$$
Consequently, total deficit equals total surplus; if the $S_c$ are not all equal, at least one client has $S_c<T$ and at least one has $S_c>T$.

*Proof.* $\sum_c(S_c-T)=S_{\text{total}}-NT=0$. Write $x^+=\max(0,x)$; then $x=x^+-(-x)^+$ and $|x|=x^++(-x)^+$. Summing $x=S_c-T$: $0=\sum x^+-\sum(-x)^+$, so both sums are equal, and each is $\frac12\sum|x|$. If not all equal, some $x\neq0$, so $\sum|x|>0$ and both positive/negative parts are nonzero. ∎

### 1.5 `fairness_debt`

$D=\dfrac{\max(0,T-S)}{T+\epsilon}$.

**Theorem 1.5.** For $S\ge0$, $T\ge0$, $\epsilon>0$:

1. $0\le D<1$ (strictly less than 1).
2. $D=0\iff S\ge T$.
3. $D$ is nonincreasing in $S$, and strictly decreasing on $S<T$, with $\partial D/\partial S=-1/(T+\epsilon)$ there.
4. For fixed $S<T$, $D$ is strictly increasing in $T$.
5. $D(S{=}0)=\dfrac{T}{T+\epsilon}=1-\dfrac{\epsilon}{T+\epsilon}$; $D=0$ when $T=0$ (no $0/0$).
6. $\sum_c D_c=\dfrac{1}{2(T+\epsilon)}\sum_c|S_c-T|$ (under A2).

*Proof.* (1) If $S\ge T$, $D=0$. Otherwise $0<T-S\le T<T+\epsilon$ (using $S\ge0$), so $0<D<1$. (2) The numerator is zero iff $T-S\le0$. (3) On $S<T$, $D=(T-S)/(T+\epsilon)$ is affine in $S$ with slope $-1/(T+\epsilon)$; on $S\ge T$, it is constant $0$. (4) $\partial_T\frac{T-S}{T+\epsilon}=\frac{(T+\epsilon)-(T-S)}{(T+\epsilon)^2}=\frac{S+\epsilon}{(T+\epsilon)^2}>0$. (5) Direct. (6) Prop. 1.4. ∎

*Remark (correction to the analysis document).* The document states $D\in[0,1]$ and "$D\approx1.0$ when $S_c=0$." The sharp statement is $D<1$ always; this matters in §1.6/§5.4.

### 1.6 `fairness_urgency`

$U(D)=1-e^{-\lambda D^+}$ with $D^+=\max(0,D)$, $\lambda>0$.

**Theorem 1.6.**

1. $U(0)=0$; $U$ is strictly increasing and strictly concave on $[0,\infty)$; $0\le U<1$ and $U\to1$ as $D\to\infty$.
2. $U'(D)=\lambda e^{-\lambda D}$, $U'(0^+)=\lambda$, $U''(D)=-\lambda^2e^{-\lambda D}<0$.
3. $U(D)\le\min(1,\lambda D)$ (so $\lambda$ is the Lipschitz constant).
4. Inverse: $D=-\frac1\lambda\ln(1-U)$; half-urgency at $D=\ln 2/\lambda$ ($=0.3466$ for $\lambda=2$).
5. **Composite with the tracker.** For $D=D(S)$ from Thm 1.5, $S\mapsto U(D(S))$ is nonincreasing, and
$$0\le U(D(S))\le 1-e^{-\lambda T/(T+\epsilon)}<1-e^{-\lambda}.$$
For $\lambda=2$: $U<1-e^{-2}\approx0.8647$ always.

*Proof.* (1)–(2) Differentiate: $U'=\lambda e^{-\lambda D}>0$ and $U''<0$; $U(0)=0$; $e^{-\lambda D}\downarrow0$. (3) $1-e^{-x}\le x$ for $x\ge0$ (tangent line of the concave function at 0) and $1-e^{-x}<1$. (4) Solve $U=1-e^{-\lambda D}$. (5) $U\circ D$ is a nondecreasing function composed with the nonincreasing $D(S)$. The bound uses $D\le T/(T+\epsilon)<1$ (Thm 1.5(5)) and monotonicity of $U$. ∎

*Consequence.* With the tracker-based urgency and default $\lambda=2$, the slack factor $(1-u)$ in §5.4 is $>e^{-2}\approx0.135$; the slack never collapses to exactly zero through this channel. (The router's *delay-based* urgency, §7.1, can equal exactly $1$.)

---

## 2. `locality.py`

### 2.1 `locality_score`

$h(H,N)=\operatorname{clip}\!\big(\tfrac{H}{N+\epsilon},0,1\big)$ for $N>0$; $h=0$ for $N\le0$.

**Theorem 2.1.** Under A3 ($0\le H\le N$, $N>0$):

1. $h=\dfrac{H}{N+\epsilon}$ (the clip never activates) and $0\le h\le\dfrac{N}{N+\epsilon}<1$.
2. $h=0\iff H=0$; $h$ is increasing in $H$ and decreasing in $N$ (for $H>0$).
3. $\left|h-\dfrac HN\right|=\dfrac{H\epsilon}{N(N+\epsilon)}\le\dfrac{\epsilon}{N+\epsilon}<\epsilon$.
4. With $n_{\text{ext}}=\max(0,N-H)=N-H$: $n_{\text{ext}}-N(1-h)=\dfrac{-H\epsilon}{N+\epsilon}$, so $|n_{\text{ext}}-N(1-h)|<\epsilon$. Thus $n_{\text{ext}}\approx N(1-h)$ to within $10^{-8}$ tokens.

*Proof.* (1) $0\le H\le N<N+\epsilon$ gives $0\le H/(N+\epsilon)<1$, so both clip bounds are inactive. (2) Obvious from the quotient. (3) $\frac HN-\frac H{N+\epsilon}=\frac{H\epsilon}{N(N+\epsilon)}$ and $H\le N$. (4) $N(1-h)=N-\frac{NH}{N+\epsilon}$; subtract from $N-H$: $-H+\frac{NH}{N+\epsilon}=-\frac{H\epsilon}{N+\epsilon}$. ∎

*Correction.* The analysis states $h=1.0$ for a complete hit. Exactly, a full hit $H=N$ gives $h=1-\frac{\epsilon}{N+\epsilon}<1$; $h=1$ requires $H\ge N+\epsilon$, impossible under A3. The deviation is negligible numerically.

### 2.2 Companion locality formulas

**Proposition 2.2 (recompute cost).** $C_{\text{recompute}}=1-h\in(0,1]$ is strictly decreasing in $H$. *Proof.* By Thm 2.1(1), $h<1$.

**Proposition 2.3 (tier-aware cost).** Suppose $H_{\text{npu}}+H_{\text{cpu}}+H_{\text{st}}+n_{\text{ext}}=N_{\text{in}}$ and costs $c_{\text{npu}}\le c_{\text{cpu}}\le c_{\text{st}}\le c_{\text{rec}}$ ($0,0.25,0.60,1.0$). Then
1. $0\le C_{\text{tier}}\le\dfrac{N_{\text{in}}}{N_{\text{in}}+\epsilon}c_{\text{rec}}<1$;
2. moving $k$ tokens from recompute to CPU-tier changes $C_{\text{tier}}$ by $-k(c_{\text{rec}}-c_{\text{cpu}})/(N+\epsilon)=-0.75k/(N+\epsilon)<0$ (and similarly for other upgrades).

*Proof.* $C_{\text{tier}}=\sum_k \frac{n_k}{N+\epsilon}c_k$ with $\sum_kn_k=N$, i.e. a convex combination scaled by $N/(N+\epsilon)$; the bound follows from $c_k\in[0,1]$. (2) Only two terms change. ∎

**Proposition 2.4 (adaptive locality weight).** $\beta(D)=\beta_0e^{-\lambda D^+}$ satisfies $\beta(D)=\beta_0\,(1-U(D))$, is strictly decreasing in $D$, and $0<\beta\le\beta_0$ with $\beta(0)=\beta_0$.

*Proof.* $e^{-\lambda D}=1-U$ by definition of $U$. Monotonicity from $e^{-x}$. ∎

---

## 3. `prediction.py`

### 3.1 `predict_queue`

$\hat Q=\max\big(0,\,Q+(\lambda-\mu)H\big)$, $Q\ge0$, $H\ge0$, $\delta:=\lambda-\mu$.

**Theorem 3.1.**

1. $\hat Q\ge0$.
2. **Exactness.** $\hat Q$ is the exact solution at time $H$ of the reflected fluid model $\dot q=\delta$ while $q>0$, $q$ held at $0$ once it reaches $0$ with $\delta\le0$, started from $q(0)=Q$.
3. $\hat Q$ is 1-Lipschitz and nondecreasing in $Q$; nondecreasing in $\lambda$, nonincreasing in $\mu$; nondecreasing in $H$ iff $\delta\ge0$.
4. **Fixed points.** $\hat Q=Q$ if $H=0$ or $\delta=0$ (in particular the router's default $\lambda=\mu=0$, $H=1$ gives $\hat Q=Q$).
5. **Semigroup.** For constant rates and $H_1,H_2\ge0$: $\hat Q(\hat Q(Q,H_1),H_2)=\hat Q(Q,H_1+H_2)$.

*Proof.* (1) By the outer max. (2) If $\delta\ge0$ the queue never hits $0$ from $Q\ge0$: $q(t)=Q+\delta t$. If $\delta<0$, $q(t)=Q+\delta t$ until $t^*=Q/|\delta|$, then $0$; i.e. $q(t)=\max(0,Q+\delta t)$. (3) Max of affine functions with the stated slopes. (4) $\max(0,Q)=Q$ for $Q\ge0$. (5) If $\delta\ge0$ both sides equal $Q+\delta(H_1+H_2)$. If $\delta<0$: let $q_1=\max(0,Q+\delta H_1)$. If $Q+\delta H_1\ge0$, $q_1=Q+\delta H_1$ and the left side is $\max(0,Q+\delta(H_1+H_2))$. Otherwise $q_1=0$ and the left is $\max(0,\delta H_2)=0$; the right is $\max(0,Q+\delta H_1+\delta H_2)=0$ since $Q+\delta H_1<0$ and $\delta H_2\le0$. ∎

*Consequence for the system.* With the defaults used in `router.py`, the "prediction" in `predict_queue` is the identity (persistence). Genuine foresight enters only through the TTFT/decode model (§6, §7).

### 3.2 `replica_is_safe`

$\mathrm{safe}=(\hat Q<Q_{\max})\wedge(\hat U<U_{\max})$.

**Theorem 3.2.**

1. **Downward-closed.** If $(\hat Q,\hat U)$ is safe and $\hat Q'\le\hat Q$, $\hat U'\le\hat U$ then $(\hat Q',\hat U')$ is safe. The safe region is an open box.
2. **Margin.** Safe implies KV headroom $1-\hat U>1-U_{\max}=5\%$ (default) and queue slack $Q_{\max}-\hat Q>0$.
3. **Fail-closed.** If capacity $C\le0$, then $Q_{\max}=4C\le0\le\hat Q$ so the replica is unsafe; also `kv_utilization` $=1\ge0.95$ (Thm 4.2), so it is doubly unsafe.

*Proof.* (1)–(2) Strict inequalities are preserved under decreasing arguments. (3) $\hat Q\ge0\ge Q_{\max}$ violates $\hat Q<Q_{\max}$. ∎

### 3.3 Companion: EWMA, KV projection, risk

**Proposition 3.3 (EWMA).** $\hat X_t=\alpha X_t+(1-\alpha)\hat X_{t-1}$, $\alpha\in(0,1]$. Then
1. $\hat X_t=\alpha\sum_{k=0}^{t-1}(1-\alpha)^kX_{t-k}+(1-\alpha)^t\hat X_0$, weights sum to $1$;
2. $\min(\hat X_0,X_{1..t})\le\hat X_t\le\max(\hat X_0,X_{1..t})$;
3. if $X$ is constant $=x$: $|\hat X_t-x|=(1-\alpha)^t|\hat X_0-x|$;
4. for iid $X$ with variance $\sigma^2$ (asymptotically, $t\to\infty$): $\mathrm{Var}(\hat X_t)\to\dfrac{\alpha}{2-\alpha}\sigma^2$ ($\approx0.0526\sigma^2$ for $\alpha=0.1$, $\approx0.0256\sigma^2$ for $\alpha=0.05$).

*Proof.* (1) Unroll the recursion; $\alpha\sum_{k<t}(1-\alpha)^k+(1-\alpha)^t=1$. (2) Convex combination. (3) $\hat X_t-x=(1-\alpha)(\hat X_{t-1}-x)$. (4) $\mathrm{Var}=\alpha^2\sigma^2\sum_k(1-\alpha)^{2k}=\dfrac{\alpha^2\sigma^2}{1-(1-\alpha)^2}=\dfrac{\alpha\sigma^2}{2-\alpha}$. ∎

**Proposition 3.4 (KV projection).** $\hat K=\max(0,K+(\lambda_{kv}-\mu_{kv})H)$ is the same function as §3.1; Theorem 3.1 applies verbatim.

**Proposition 3.5 (risk/benefit).** With $\kappa_\sigma\ge0$, $\mathrm{Risk}=\hat W+\kappa_\sigma\sigma\ge\hat W$; $\mathrm{Benefit}=e^{-\mathrm{Risk}/\tau}\in(0,1]$ for $\mathrm{Risk}\ge0$, strictly decreasing in Risk, and $\frac{\mathrm{Benefit}(R_1)}{\mathrm{Benefit}(R_2)}=e^{(R_2-R_1)/\tau}$ (ratio depends only on the difference, so the ordering of replicas is unaffected by a common additive risk offset).

---

## 4. `utils.py`

### 4.1 `reactive_load`

$\mathcal L=\dfrac{4N_{\text{wait}}+N_{\text{run}}}{C}$ ($C>0$), $\mathcal L=\infty$ ($C\le0$).

**Theorem 4.1.** For $N_{\text{wait}},N_{\text{run}}\ge0$, $C>0$:

1. $\mathcal L\ge0$; $\mathcal L=0$ iff the replica is idle; $\mathcal L$ is nondecreasing in both counts, strictly decreasing in $C$.
2. $\mathcal L\cdot C=4N_{\text{wait}}+N_{\text{run}}$ (a weighted request count); one waiting request costs exactly as much as four running ones.
3. Moving one request from waiting to running lowers $\mathcal L$ by exactly $3/C$.
4. $\mathcal L$ is unbounded above and may exceed 1.
5. For replicas with equal $C$, $\arg\min\mathcal L=\arg\min(4N_{\text{wait}}+N_{\text{run}})$.

*Proof.* Immediate from the formula; (3): $\frac{4(N_w-1)+(N_r+1)}{C}-\frac{4N_w+N_r}{C}=-\frac3C$. ∎

*Consequence.* §5.2 clips load to $[0,1]$; all replicas with $\mathcal L\ge1$ are indistinguishable to `predict_latency_seconds` (an information-loss property, not an error).

### 4.2 `kv_utilization`

**Theorem 4.2.** For $M_{\text{total}}>0$:

1. $U\in[0,1]$; under A4, $U=\dfrac{M_{\text{used}}}{M_{\text{total}}+\epsilon}<1$ (clip inactive at the top), and $U=1$ if $M_{\text{total}}\le0$ (fail-closed).
2. $U$ is nondecreasing in $M_{\text{used}}$ and decreasing in $M_{\text{total}}$.
3. Under A4: $\left|U-\tfrac{M_{\text{used}}}{M_{\text{total}}}\right|\le\dfrac{\epsilon}{M_{\text{total}}+\epsilon}<\epsilon$.
4. Headroom $1-U\in[0,1]$, and $U<0.95\Rightarrow M_{\text{used}}<0.95\,(M_{\text{total}}+\epsilon)$.

*Proof.* Same algebra as Thm 2.1: $\frac{M}{T}-\frac M{T+\epsilon}=\frac{M\epsilon}{T(T+\epsilon)}\le\frac\epsilon{T+\epsilon}$ for $M\le T$. ∎

---

## 5. `fairroute.py`

### 5.1 `iteration_time`

Parameters $\tau_0,w_0,w_s,p_{\max},k_b,k_s,\tau_b,\tau_s>0$. For $B,S\ge0$:
$$P(B,S)=p_{\max}(1-e^{-k_bB})(1-e^{-k_sS}),\qquad
T(B,S)=\begin{cases}\tau_0+\tau_bB+\tau_sS,&P=0\\[2pt]\tau_0+\dfrac{w_0+w_sS}{P}+\tau_bB+\tau_sS,&P>0.\end{cases}$$

**Theorem 5.1.**

1. **Support of $P$.** $P=0\iff B=0$ or $S=0$; otherwise $0<P<p_{\max}$, and $P$ is strictly increasing in each argument and $P\to p_{\max}$ as $B,S\to\infty$.
2. **Positivity and floor.** $T\ge\tau_0>0$. For $B,S>0$: $T>\tau_0+\dfrac{w_0+w_sS}{p_{\max}}+\tau_bB+\tau_sS$.
3. **Unbounded growth.** $T\ge\tau_bB+\tau_sS\to\infty$; $T$ does not saturate (only $P$ does).
4. **Boundary discontinuity.** For $S>0$: $\lim_{B\to0^+}T(B,S)=+\infty$, while $T(0,S)=\tau_0+\tau_sS<\infty$. The same holds as $S\to0^+$ for fixed $B>0$.
5. **Convexity in $B$.** For fixed $S>0$, $B\mapsto T(B,S)$ is strictly convex on $(0,\infty)$, with a unique minimiser $B^*(S)$ solving
$$a(S)\,k_b\,\frac{e^{-k_bB}}{(1-e^{-k_bB})^2}=\tau_b,\qquad a(S)=\frac{w_0+w_sS}{p_{\max}(1-e^{-k_sS})}.$$
$T$ is strictly decreasing on $(0,B^*)$ and strictly increasing on $(B^*,\infty)$.

*Proof.* (1) $1-e^{-kx}$ is $0$ at $x=0$, strictly increasing, and $<1$. $p_{\max}>0$. (2) All terms nonnegative; when $P>0$ we use $P<p_{\max}$ to bound the fraction from below, and $w_0>0$ makes the fraction strictly positive. (3) Linear terms. (4) $P\to0^+$ while the numerator $\ge w_0>0$, so the fraction $\to+\infty$; at $B=0$ the first branch is used. (5) For $S>0$ write $T=\text{const}+a\,\varphi(B)+\tau_bB$ with $\varphi(B)=(1-e^{-k_bB})^{-1}=1+g(k_bB)$, $g(x)=(e^x-1)^{-1}$. Then
$g''(x)=\dfrac{e^x(e^x+1)}{(e^x-1)^3}>0$, so $\varphi$ is strictly convex and $T$ is strictly convex. $T'(B)=-a k_b\frac{e^{-k_bB}}{(1-e^{-k_bB})^2}+\tau_b$; $T'\to-\infty$ as $B\to0^+$ and $T'\to\tau_b>0$ as $B\to\infty$, and $T'$ is strictly increasing (convexity), so it has exactly one zero. ∎

**Corollary 5.1a (router domain).** The router calls $T(B_i+1,S_{\max})$ with $p_{\max}=S_{\max}$ and $T(D_i+1,D_i+1)$. All arguments are $\ge1$, so $P>0$ and the discontinuity of item 4 is never hit. But item 5 still applies: for $S_{\max}=2048$, $p_{\max}=2048$, defaults, $B^*\approx4.57$ (numerically verified). So an idle replica ($B=1$) is predicted ≈ $0.6\%$ *slower per iteration* than one with $B=5$ ($0.021705$ s vs $\approx0.021577$ s). The effect is small and bounded but formally non-monotone in load.

### 5.2 `predict_latency_seconds`

$\hat T=\dfrac{0.01+W_{\text{ext}}+W_{\text{dec}}}{\Theta(\tilde{\mathcal L})}$, $\Theta=\max(0.05,1-0.65\tilde{\mathcal L})$, $\tilde{\mathcal L}=\operatorname{clip}(\text{load},0,1)$, $W_{\text{ext}}=n_{\text{ext}}^+/2048$, $W_{\text{dec}}=\hat Y^+/512$. Let $\mathcal N=0.01+W_{\text{ext}}+W_{\text{dec}}$.

**Theorem 5.2.**

1. **The floor $0.05$ is never active.** $\Theta(\tilde{\mathcal L})=1-0.65\tilde{\mathcal L}\in[0.35,1]$.
2. **Bounds.** $\mathcal N\le\hat T\le\tfrac{1}{0.35}\mathcal N=\tfrac{20}{7}\mathcal N\approx2.857\,\mathcal N$; and $\hat T\ge0.01$.
3. **Monotonicity.** $\hat T$ is nondecreasing in load (strictly on $(0,1)$, constant outside), in $n_{\text{ext}}$ and in $\hat Y$; convex in $\tilde{\mathcal L}$.
4. **Value of a prefix hit.** Reusing $H'$ additional cached tokens reduces $\hat T$ by exactly $\dfrac{H'}{2048\,\Theta}$, which lies in $\left[\dfrac{H'}{2048},\ \dfrac{20}{7}\dfrac{H'}{2048}\right]$: *cache hits are worth up to $2.86\times$ more on a loaded replica than on an idle one.*
5. **Locality-consistency.** For two replicas with equal load and equal $\hat Y$, the one with larger hit has strictly smaller $\hat T$. For equal hit, the one with smaller $\tilde{\mathcal L}$ has $\hat T$ no larger.
6. **Pairwise preference rule.** With $\mathcal N_i=0.01+(N-H_i)/2048+\hat Y/512$, replica $i$ is predicted faster than $j$ iff $\mathcal N_i\,\Theta_j<\mathcal N_j\,\Theta_i$.

*Proof.* (1) $\tilde{\mathcal L}\in[0,1]\Rightarrow1-0.65\tilde{\mathcal L}\in[0.35,1]$ and $0.35>0.05$, so the max returns its second argument. (2) Divide $\mathcal N$ by $\Theta\in[0.35,1]$. (3) Numerator nondecreasing in $n_{\text{ext}},\hat Y$; $1/\Theta$ nondecreasing, and $x\mapsto 1/(1-0.65x)$ is convex on $[0,1]$. (4) $\hat T(n_{\text{ext}})-\hat T(n_{\text{ext}}-H')=\frac{H'/2048}{\Theta}$. (5) From (3). (6) Cross-multiply $\mathcal N_i/\Theta_i<\mathcal N_j/\Theta_j$ with $\Theta>0$. ∎

*Correction.* The analysis states the floor "caps the maximum penalty factor at $20\times$." By (1)–(2) the true maximum penalty is $20/7\approx2.857\times$; the floor is dead code under the clip.

### 5.3 `fairroute_gate`

Let $\mathcal F=\{c:\mathrm{safe}_c\wedge\hat T_c\le\gamma\,\mathrm{SLO}\}$, $\mathcal S=\{c:\mathrm{safe}_c\}$. Output: $(\mathcal F,\text{False})$ if $\mathcal F\ne\emptyset$; $(\mathcal S,\text{True})$ if $\mathcal F=\emptyset\ne\mathcal S$; $(\mathcal C,\text{True})$ otherwise.

**Theorem 5.3.** For finite $\mathcal C$:

1. **Totality.** If $\mathcal C\ne\emptyset$, the returned set $\mathcal G$ is nonempty and $\mathcal G\subseteq\mathcal C$. (If $\mathcal C=\emptyset$, the output is $(\emptyset,\text{True})$.)
2. **Case partition.** Exactly one of the three cases applies, and `slo_risk` $=\text{False}\iff\mathcal F\neq\emptyset$.
3. **Soundness.** If `slo_risk` is False, every returned candidate is safe and meets the SLO budget. If $\mathcal S\ne\emptyset$, every returned candidate is safe.
4. **Optimal latency is preserved.** Let $A=\mathcal S$ if $\mathcal S\ne\emptyset$, else $A=\mathcal C$. Then $\mathcal G\subseteq A$ and $\min_{c\in\mathcal G}\hat T_c=\min_{c\in A}\hat T_c$.
5. **Monotone in $\gamma$.** $\mathcal F(\gamma)$ is nondecreasing in $\gamma$, and `slo_risk` is False iff $\gamma\ge\gamma^*:=\dfrac{\min_{c\in\mathcal S}\hat T_c}{\mathrm{SLO}}$.
6. **Idempotence.** Applying the gate with the same $(\mathrm{SLO},\gamma)$ to its own output returns the same set and flag.

*Proof.* (1) Case 1 is nonempty by its guard; case 2 by its guard; case 3 returns $\mathcal C\ne\emptyset$. (2) The guards are mutually exclusive and exhaustive by construction. (3) Directly from the definitions of $\mathcal F,\mathcal S$. (4) Case 2/3: $\mathcal G=A$. Case 1: $\mathcal F\subseteq\mathcal S=A$, and let $c_0\in\mathcal F$ with $\hat T_{c_0}\le\gamma\mathrm{SLO}$; the safe candidate $c_m$ minimising $\hat T$ satisfies $\hat T_{c_m}\le\hat T_{c_0}\le\gamma\mathrm{SLO}$, so $c_m\in\mathcal F$; hence $\min_{\mathcal F}=\min_{\mathcal S}$. (5) The condition $\hat T_c\le\gamma\mathrm{SLO}$ is monotone in $\gamma$; $\mathcal F\ne\emptyset\iff\min_{\mathcal S}\hat T\le\gamma\mathrm{SLO}$. (6) Case 1: all elements of $\mathcal F$ pass, so $\mathcal F(\mathcal F)=\mathcal F$. Case 2: $\mathcal F(\mathcal S)=\mathcal F(\mathcal C)=\emptyset$ (an element of $\mathcal S$ in $\mathcal F$ would be in $\mathcal F(\mathcal C)$), $\mathcal S(\mathcal S)=\mathcal S$. Case 3: $\mathcal F(\mathcal C)=\mathcal S(\mathcal C)=\emptyset$ again. ∎

### 5.4 `fairroute_slack_set`

$T^*=\min_{\mathcal C}\hat T_c$, $u=\operatorname{clip}(\text{urgency},0,1)$, $\epsilon_0=\min(\epsilon_{\max},\rho T^*)$, $\varepsilon=\epsilon_0(1-u)$, $\mathcal C_{\text{slack}}=\{c:\hat T_c\le T^*+\varepsilon\}$. (Symbol $\varepsilon$ is the slack; $\epsilon=10^{-8}$ is the regulariser.)

**Theorem 5.4.** For $\mathcal C\ne\emptyset$, $\hat T_c>0$, $\rho,\epsilon_{\max}>0$:

1. **Nonempty.** $\mathcal C_{\text{slack}}\supseteq\arg\min\hat T\ne\emptyset$.
2. **Slack range.** $0\le\varepsilon\le\epsilon_0\le\min(\epsilon_{\max},\rho T^*)$; $\varepsilon$ is nonincreasing in $u$; $\varepsilon=0\iff u=1$.
3. **Nesting.** $u_1\le u_2\Rightarrow\mathcal C_{\text{slack}}(u_2)\subseteq\mathcal C_{\text{slack}}(u_1)$.
4. **Bounded suboptimality.** Every $c\in\mathcal C_{\text{slack}}$ satisfies
$$\hat T_c-T^*\le\min(\epsilon_{\max},\rho T^*)(1-u),\qquad \frac{\hat T_c}{T^*}\le1+\rho(1-u)\le1+\rho .$$
5. **Collapse.** If $u=1$: $\mathcal C_{\text{slack}}=\arg\min_c\hat T_c$ exactly. If $u=0$: $\varepsilon=\epsilon_0$.
6. **Crossover.** The additive cap binds ($\epsilon_{\max}<\rho T^*$) iff $T^*>\epsilon_{\max}/\rho$ ($=0.2$ s for defaults); below that the multiplicative bound binds.

*Proof.* (1) The argmin $c_0$ has $\hat T_{c_0}=T^*\le T^*+\varepsilon$ because $\varepsilon\ge0$ ($T^*>0$, so $\rho T^*>0$, $\epsilon_{\max}>0$, $1-u\ge0$). (2) $\min(\cdot)\ge0$ and $1-u\in[0,1]$. (3) $\varepsilon$ nonincreasing in $u$ shrinks the threshold. (4) By definition of membership. (5) $u=1\Rightarrow\varepsilon=0\Rightarrow\hat T_c\le T^*$, i.e. $\hat T_c=T^*$. (6) Compare the two arguments of the min. ∎

**Corollary 5.4a (tracker-driven slack).** If $u=U(D(S_c))$ from §1.6 with $\lambda=2$, then $\varepsilon>e^{-2}\epsilon_0\approx0.135\,\epsilon_0$ (never exactly zero via this channel).

### 5.5 `choose_fairroute_candidate`

Let $\mathcal C\neq\emptyset$ be finite (the gated set); ties in the *final* key are assumed broken by a deterministic rule (index or §7.5 jitter).

#### Mode `GATED`
**Proposition 5.5.1.** $c^*=\arg\min\hat T_c$ exists and equals a member of $\arg\min$, i.e. $\hat T_{c^*}=T^*$. *Proof.* Finite nonempty set of reals attains its minimum. ∎

#### Mode `MULTIPLICATIVE`
Score $m_c=\dfrac{\hat T_c}{\max(10^{-8},T^*)}\cdot\dfrac{1+h_c}{\max(10^{-8},1+d_c)}$, minimised.

**Proposition 5.5.2.**
1. The factor $1/\max(10^{-8},T^*)$ is common to all candidates and does not change the $\arg\min$.
2. $\ln m_c=\ln\hat T_c+\ln(1+h_c)-\ln(1+d_c)+\text{const}$ (log-additive).
3. $\partial m_c/\partial h_c>0$ and $\partial m_c/\partial d_c<0$: **under argmin, higher locality is penalised and higher deficit is rewarded.**
4. Under A5, $j$ beats $i$ only if $\dfrac{\hat T_j}{\hat T_i}<\dfrac{(1+h_i)(1+d_j)}{(1+h_j)(1+d_i)}\le4$, so the non-latency terms can override at most a $4\times$ latency gap.

*Proof.* (1) Multiplying all scores by a positive constant preserves order. (2) Take logs. (3) Differentiate; $\hat T_c>0$, $1+h_c>0$. (4) $m_j<m_i\iff\frac{\hat T_j}{\hat T_i}<\frac{(1+h_i)(1+d_j)}{(1+h_j)(1+d_i)}$; with $h,d\in[0,1]$ the right side is $\le\frac{2\cdot2}{1\cdot1}=4$. ∎

*Flag (§9-F7).* Item 3 means that if `locality` is a hit *ratio* (as in the lexicographic mode), `MULTIPLICATIVE` steers *away* from cache hits. This is correct only if the field is a *cost* in this mode.

#### Mode `LINEAR`
Score $s_c=-\hat T_c+0.02h_c+0.01d_c$, maximised.

**Proposition 5.5.3.** $s_j>s_i\iff\hat T_i-\hat T_j>0.02(h_i-h_j)+0.01(d_i-d_j)$. Under A5 the maximum latency the policy can sacrifice for locality+deficit is $0.02+0.01=0.03$ s (30 ms); on a pair with $\hat T_j-\hat T_i>0.03$ s, $i$ always wins. *Proof.* Rearrange $s_j>s_i$; the bound uses $|h_i-h_j|,|d_i-d_j|\le1$. ∎

#### Mode `LEARNED`
Score $s_c=-\hat T_c+0.01h_c+0.01d_c+0.01\,\mathrm{credit}_c$.

**Proposition 5.5.4.** $s_j>s_i\iff\hat T_i-\hat T_j>0.01\,[(h_i-h_j)+(d_i-d_j)+(\mathrm{credit}_i-\mathrm{credit}_j)]$. If credit is in raw tokens (quantum $Q_{\text{credit}}=4096$, §7.3), a credit gap of $100$ tokens already compensates $1$ s of latency, and the full range compensates up to $\approx41$ s; so the mode collapses to $\arg\max\,\mathrm{credit}$ unless credit is normalised. (Conditional statement; see §9-F8.)

#### Mode `SLACK_LEXICOGRAPHIC` / `V2`
Steps: $\mathcal C_{\text{slack}}$ from §5.4; $\mathcal E=\{c\in\mathcal C_{\text{slack}}:\mathrm{hit}_c>0\wedge\mathrm{credit}_c>0\}$; $\mathrm{Pool}=\mathcal E$ if $\mathcal E\ne\emptyset$ else $\mathcal C_{\text{slack}}$; $c^*\in\arg\max_{\text{lex}}(h_c,h^{\text{fut}}_c,d_c,-\hat T_c)$ over Pool.

**Theorem 5.5.5.**

1. **Well-defined.** Pool is nonempty and finite; the lexicographic order on $\mathbb R^4$ is total, so a maximum tuple exists; the set of maximisers is nonempty.
2. **Bounded latency cost.** $c^*\in\mathcal C_{\text{slack}}$, so $\hat T_{c^*}\le T^*+\varepsilon$ and Theorem 5.4(4) applies — *regardless of which Pool was used*.
3. **Strict priority.** If $c_1,c_2\in$ Pool and $h_{c_1}>h_{c_2}$, then $c_2$ is not selected (no amount of deficit, future-locality or latency advantage can compensate).
4. **Credit-gated locality.** If $\mathcal E\neq\emptyset$ then $c^*$ has $\mathrm{hit}_{c^*}>0$ and $\mathrm{credit}_{c^*}>0$ and $h_{c^*}=\max_{c\in\mathcal E}h_c$. A slack-set candidate with $\mathrm{credit}\le0$ is excluded from locality-driven selection while $\mathcal E\ne\emptyset$.
5. **Urgent clients are latency-greedy.** If $u=1$, then $\hat T_{c^*}=T^*$ (same as `GATED`); locality only breaks ties among latency-optimal candidates.
6. **Role of lower keys.** Keys 2–4 matter only when all earlier keys tie exactly (e.g. cold prompts with $h=0$ everywhere).

*Proof.* (1) Thm 5.4(1) and finiteness; lexicographic order on a product of totally ordered sets is total. (2) By construction Pool $\subseteq\mathcal C_{\text{slack}}$. (3) Lex-maximality means any tuple with larger first coordinate is strictly greater. (4) By the definition of $\mathcal E$ and of $\arg\max$ on the first key. (5) Thm 5.4(5) with Pool $\subseteq\mathcal C_{\text{slack}}=\arg\min\hat T$. (6) Definition of lexicographic comparison. ∎

### 5.6 `score_fairroute_v2` ($H_5$)

$w_f(D)=w_b+2D^+$, $\mathrm{FB}_i=U(1-\mathcal L_i)$, $\mathrm{Score}_i=w_f(D)\,\mathrm{FB}_i+w_lh_i+w_pP_i$, with $w_b,w_l,w_p>0$.

**Theorem 5.6.**

1. **Equivalent form.** $\mathrm{Score}_i=\text{const}-\kappa(D)\mathcal L_i+w_lh_i+w_pP_i$, where $\kappa(D)=w_f(D)\,U(D)$ and const $=\kappa(D)$ is common to all replicas; hence $\arg\max_i\mathrm{Score}_i$ is unchanged by the constant.
2. **$\kappa$ is zero at no debt and strictly increasing in debt.** $\kappa(0)=0$, $\kappa'(D)=2U(D)+w_f(D)\lambda e^{-\lambda D}>0$ for $D>0$.
3. **Pairwise rule.** $\mathrm{Score}_i>\mathrm{Score}_j\iff\kappa(D)(\mathcal L_j-\mathcal L_i)+w_l(h_i-h_j)+w_p(P_i-P_j)>0$.
4. **Locality vs load trade-off shifts monotonically toward load as debt grows.** The load-headroom advantage needed to overturn a hit advantage $\Delta h>0$ (holding $P$ equal) is $\Delta\mathcal L^*=w_l\Delta h/\kappa(D)$, which is $+\infty$ at $D=0$ and strictly decreasing in $D$.
5. **Boundedness.** For $D<1$, $\lambda=2$: $0\le\kappa(D)<(w_b+2)(1-e^{-2})$.
6. **Sign caveat.** $\mathrm{FB}_i<0$ iff $\mathcal L_i>1$ (when $U>0$); a replica above capacity is *penalised* rather than clipped at zero benefit, unless the code clips the load.

*Proof.* (1) Expand $U(1-\mathcal L_i)=U-U\mathcal L_i$. (2) $\kappa=(w_b+2D)(1-e^{-\lambda D})$ is a product of positive increasing factors for $D>0$: $\kappa'=2U+w_fU'>0$. (3) Subtract. (4) Set the left side of (3) to zero with $P_i=P_j$. (5) $w_f<w_b+2$, $U<1-e^{-2}$ via Thm 1.5/1.6. (6) Sign of $1-\mathcal L_i$. ∎

*Consequence.* The fairness signal is *candidate-varying* exactly because $\mathcal L_i$ differs across replicas (the analysis's claim), but vanishes entirely when $D=0$.

### 5.7 AFLC ($H_3$)

$\mathrm{Score}=(1+w_fU)(\epsilon+L)^{\beta(D)}(\epsilon+P)^{\gamma}$, with $\beta=\beta_0e^{-\lambda D}$.

**Proposition 5.7.**
1. The factor $(1+w_fU)>0$ is the same for every replica, so it **does not influence $\arg\max$**. Fairness acts only through $\beta(D)$.
2. For two replicas with locality $L_1>L_2>0$ (and $\epsilon$ dropped), $(L_1/L_2)^{\beta}$ is strictly decreasing in $\beta$ and tends to $1$ as $\beta\to0$. Hence as $D$ grows, $\beta(D)\downarrow$ and the locality advantage in the score ratio shrinks toward neutrality.

*Proof.* (1) Positive common multiplier preserves order. (2) $\ln\big((L_1/L_2)^\beta\big)=\beta\ln(L_1/L_2)$ with $\ln(L_1/L_2)>0$. ∎

---

## 6. Estimators and Predictors inside `router.py`

### 6.1 Queued work and capacity

$W_i=\sum_{\text{wait}}\mathrm{tokens}(r)+\sum_{\text{run}}\max(0,\mathrm{tokens}(r)-\mathrm{computed}(r))$, $D_i=\#\{r\in\text{run}:\mathrm{tokens}(r)\le\mathrm{computed}(r)+1\}$, $\mathrm{Cap}_i=\max(1,S_{\max}-D_i)$.

**Proposition 6.1.** $W_i\ge0$ and is additive over requests; $0\le D_i\le|\text{run}_i|\le B_i$; $\mathrm{Cap}_i\ge1$ (so no division by zero), $\mathrm{Cap}_i=S_{\max}-D_i$ when $D_i<S_{\max}$, and $\mathrm{Cap}_i$ is nonincreasing in $D_i$. *Proof.* Each term is nonnegative by the max; the indicator counts at most all running requests; $\max(1,\cdot)\ge1$. ∎

### 6.2 TTFT

$\mathrm{TTFT}_i=\left\lceil\dfrac{W_i+n_{\text{ext}}}{\mathrm{Cap}_i}\right\rceil T_{\text{base},i}$, $T_{\text{base},i}=T(B_i+1,S_{\max};p_{\max}{=}S_{\max})$.

**Theorem 6.2.** Let $x=W_i+n_{\text{ext}}\ge0$.

1. **Iteration count.** Under a model where each iteration retires at most $\mathrm{Cap}_i$ pending prefill tokens, FCFS, the minimal number of iterations to finish $x$ tokens is $m=\lceil x/\mathrm{Cap}_i\rceil$.
2. **Sandwich.** For $x>0$: $\dfrac{x}{\mathrm{Cap}_i}T_{\text{base}}\le\mathrm{TTFT}_i<\Big(\dfrac{x}{\mathrm{Cap}_i}+1\Big)T_{\text{base}}$; the model overestimates the fluid time by less than one iteration.
3. **Monotonicity.** $\mathrm{TTFT}_i$ is nondecreasing in $W_i$, $n_{\text{ext}}$, $D_i$.
4. **Boundary case.** $x=0\Rightarrow\mathrm{TTFT}_i=0$.
5. **Cache value (staircase).** Reducing $n_{\text{ext}}$ by $H'$ lowers TTFT by $T_{\text{base}}\cdot(\lceil x/\mathrm{Cap}\rceil-\lceil(x-H')/\mathrm{Cap}\rceil)\in\{\lfloor H'/\mathrm{Cap}\rfloor,\lceil H'/\mathrm{Cap}\rceil\}\cdot T_{\text{base}}$; savings occur in whole iterations.

*Proof.* (1) $m\,\mathrm{Cap}\ge x$ with $m$ integer $\iff m\ge x/\mathrm{Cap}$; the least such integer is the ceiling. (2) $y\le\lceil y\rceil<y+1$. (3) Ceiling is nondecreasing, $T_{\text{base}}>0$ fixed; $D_i\uparrow\Rightarrow\mathrm{Cap}\downarrow\Rightarrow x/\mathrm{Cap}\uparrow$. (4) $\lceil0\rceil=0$. (5) $\lceil a\rceil-\lceil a-b\rceil\in\{\lfloor b\rfloor,\lceil b\rceil\}$ for $b\ge0$. ∎

*Flag.* Item 4: a request that is a full prefix hit on an empty replica gets TTFT $=0$, although producing a first token needs at least one iteration. Using $\max(1,\lceil\cdot\rceil)$ would remove this (§9-F5).

### 6.3 Decode latency and TTFT SLO

$\mathrm{Decode}_i=\hat Y_c\,T(D_i+1,D_i+1;p_{\max}{=}S_{\max})$, $\mathrm{SLO}=k_{\text{SLO}}\lceil N_{\text{in}}/S_{\max}\rceil\,T(1,S_{\max};p_{\max}{=}S_{\max})$.

**Theorem 6.3.**

1. $\mathrm{Decode}_i$ is linear in $\hat Y_c$ with slope $\ge\tau_0>0$ (Thm 5.1(2)).
2. $\mathrm{SLO}$ equals $k_{\text{SLO}}$ times $\mathrm{TTFT}$ evaluated on an *idle replica with a cold cache*: $B_i=D_i=0$, $W=0$, $H=0$. Indeed then $\mathrm{Cap}=S_{\max}$, $x=N_{\text{in}}$, $T_{\text{base}}=T(1,S_{\max};S_{\max})$.
3. Hence for the TTFT term alone, $\mathrm{TTFT}_i\le\gamma\,\mathrm{SLO}$ holds iff the replica is within a $\gamma k_{\text{SLO}}$-slowdown of the ideal; with $k_{\text{SLO}}=10$, the budget is 10 ideal TTFTs.
4. If `predicted_latency` passed to the gate is $\mathrm{TTFT}_i+\mathrm{Decode}_i$ (as §3.1 item 7 states), the gate compares total latency with a TTFT-based SLO. Then every replica with $\mathrm{Decode}_i>\gamma\,\mathrm{SLO}$ is infeasible regardless of load. *Numerical instance* (verified): $S_{\max}=p_{\max}=2048$, $N_{\text{in}}=512$, $\hat Y=256$, $D=0$: $\mathrm{SLO}=0.2171$ s, $\mathrm{Decode}=0.3142$ s $>$ SLO, so *every* replica is infeasible and `slo_risk` is always True.

*Proof.* (1) Product of constant $\hat Y$ and positive iteration time. (2) Substitute. (3),(4) Algebra and the displayed numerics. ∎

### 6.4 Empirical-Bayes output length

$\hat Y_c=\dfrac{\bar Y_c+n_0\bar Y_g}{1+n_0}$, with $\bar Y_c^{(t)}=0.1\,Y_t+0.9\,\bar Y_c^{(t-1)}$ and $\bar Y_g$ the global median.

**Theorem 6.4.**

1. $\hat Y_c$ is a convex combination (weights $\tfrac1{1+n_0},\tfrac{n_0}{1+n_0}$); $\min(\bar Y_c,\bar Y_g)\le\hat Y_c\le\max(\bar Y_c,\bar Y_g)$; at $n_0=0$ the formula already equals $\bar Y_c$ (the case split is redundant).
2. $\bar Y_g=\operatorname{median}$ lies in $[\min Y,\max Y]$ and is robust to up to $\lfloor(n-1)/2\rfloor$ arbitrarily corrupted samples.
3. **When shrinkage helps.** Let $X$ be unbiased for $\theta$ with variance $\sigma^2$ and $m$ a fixed anchor with bias $b=m-\theta$. The estimator $(1-w)X+wm$ has MSE $(1-w)^2\sigma^2+w^2b^2$, minimised at $w^*=\dfrac{\sigma^2}{\sigma^2+b^2}$, and beats $X$ iff $w<\dfrac{2\sigma^2}{\sigma^2+b^2}=2w^*$. Here $w=\frac{n_0}{1+n_0}$.
4. EWMA (§3.3) has asymptotic variance $\frac{0.1}{1.9}\sigma^2\approx0.0526\sigma^2$ (≈ the variance of an average of 19 iid samples).
5. **Sparsity caveat.** The weight $w$ does not depend on the client's sample count $n_c$. The formula shrinks every client by the same amount; "sparse-history clients shrink more" holds only if $\bar Y_c$ is initialised at $\bar Y_g$ (then a short history leaves $\bar Y_c\approx\bar Y_g$ anyway).

*Proof.* (1) Convexity. (2) Standard for the median. (3) $\mathrm{MSE}=\mathrm{Var}+\mathrm{bias}^2$ with $\mathrm{Var}=(1-w)^2\sigma^2$, bias $=wb$; differentiate: $-2(1-w)\sigma^2+2wb^2=0\Rightarrow w^*$. $\mathrm{MSE}(w)<\sigma^2\iff-2w\sigma^2+w^2(\sigma^2+b^2)<0\iff w<\frac{2\sigma^2}{\sigma^2+b^2}$. (4) Prop. 3.3. (5) Direct inspection. ∎

---

## 7. Fairness Control Loop inside `router.py`

### 7.1 Delay-based urgency (when `fairroute_n0 > 0`)

$\bar\delta_c^{(t)}=0.05\,\delta_c+0.95\,\bar\delta_c^{(t-1)}$, $\delta_{\text{med}}=\operatorname{median}_{c'}\bar\delta_{c'}$, $U_c=\operatorname{clip}\!\Big(\dfrac{\bar\delta_c}{\delta_{\text{med}}}-1,\,0,\,1\Big)$ for $\delta_{\text{med}}>0$ (and $0$ otherwise).

**Theorem 7.1.**

1. $U_c\in[0,1]$; $U_c=0\iff\bar\delta_c\le\delta_{\text{med}}$; $U_c=1\iff\bar\delta_c\ge2\delta_{\text{med}}$; $U_c$ is nondecreasing in $\bar\delta_c$ and nonincreasing in $\delta_{\text{med}}$.
2. **Scale invariance.** Rescaling all delays by $a>0$ leaves every $U_c$ unchanged.
3. **At least half of clients are never urgent.** At most $\lfloor n/2\rfloor$ clients have $\bar\delta_c>\delta_{\text{med}}$, so at least $\lceil n/2\rceil$ have $U_c=0$. If all delays are equal, all $U_c=0$.
4. EWMA properties of Prop. 3.3 hold with $\alpha=0.05$ (variance factor $\frac{0.05}{1.95}\approx0.0256$).
5. Unlike the tracker-based urgency (§1.6), this $U_c$ can equal exactly $1$; so the slack set can collapse to $\arg\min\hat T$ (Thm 5.4(5)).

*Proof.* (1)–(2) Ratio and clip. (3) Definition of median (average of the two middle values for even $n$): the number of values strictly above it is at most $\lfloor n/2\rfloor$. (4) Prop. 3.3. (5) $\bar\delta_c\ge2\delta_{\text{med}}$ gives $\frac{\bar\delta_c}{\delta_{\text{med}}}-1\ge1$. ∎

### 7.2 End-to-end monotonicity of fairness control

**Theorem 7.2 (fairness-adaptive slack).** Fix a candidate set and all other quantities, and let $u=U(D(S_c))$ (tracker route). If $S_c\le S_c'$, then $\varepsilon(S_c)\le\varepsilon(S_c')$ and $\mathcal C_{\text{slack}}(S_c)\subseteq\mathcal C_{\text{slack}}(S_c')$. *In words: the more service a client has received, the more latency slack the router is willing to spend on locality for it; the less it has received, the closer to latency-greedy.*

*Proof.* $S\mapsto D$ nonincreasing (Thm 1.5), $D\mapsto U$ increasing (Thm 1.6), $U\mapsto\varepsilon=\epsilon_0(1-U)$ nonincreasing (Thm 5.4(2)); the chain is $S\uparrow\Rightarrow D\downarrow\Rightarrow U\downarrow\Rightarrow\varepsilon\uparrow$, where each arrow is a monotone step proved in the cited theorems. Nesting from Thm 5.4(3). ∎

### 7.3 Credit lifecycle

$Q_{\text{credit}}=\kappa(w_eM_{\text{in}}^{\max}+w_qM_{\text{out}}^{\max})=1\cdot(1\cdot2048+2\cdot1024)=4096$ ✓. Debit: $\mathrm{Cr}_{(c,i)}\leftarrow\mathrm{Cr}_{(c,i)}-w_en_{\text{ext}}$ when $c$ is routed to $i$. Refill: when all $\mathrm{Cr}_{(c,i)}\le0$ for client $c$, add $Q_{\text{credit}}$ to every $i$.

**Theorem 7.3.** Assume initial credit $\le Q_{\text{credit}}$.

1. **Upper invariant.** $\mathrm{Cr}_{(c,i)}\le Q_{\text{credit}}$ at all times.
2. **Bounded locality spending.** Between two consecutive refills of client $c$, the total debit $\sum d_k$ charged to $(c,i)$ by *eligible-pool selections* (i.e. made while $\mathrm{Cr}>0$) is $<Q_{\text{credit}}+w_en^{\max}_{\text{ext}}$.
3. **Lower bound is not guaranteed by the stated rules.** Selections from the fallback pool $\mathcal C_{\text{slack}}$ do not require $\mathrm{Cr}>0$ and still debit; and a single $+Q_{\text{credit}}$ refill leaves any $\mathrm{Cr}\le-Q_{\text{credit}}$ non-positive.

*Proof.* (1) Induction: debits only decrease; at a refill each term is $\le0$ before adding $Q$ so is $\le Q$ after. (2) Let eligible selections be $k=1..m$ with debits $d_k\ge0$. Before selection $m$, $0<\mathrm{Cr}\le Q-\sum_{k<m}d_k$, so $\sum_{k<m}d_k<Q$ and $\sum_{k\le m}d_k<Q+d_m\le Q+w_en^{\max}_{\text{ext}}$. (3) Exhibit: repeated fallback selections of the same replica drive its credit below $-Q$. ∎

### 7.4 Deterministic jitter

$\mathrm{Jitter}(r,i)=\dfrac{\mathrm{int32}_{LE}(\mathrm{MD5}(r\!:\!i)[0{:}4])}{2^{32}-1}$.

**Theorem 7.4.**

1. **Deterministic**: depends only on $(r,i)$.
2. **Range.** With *unsigned* 32-bit interpretation, $\mathrm{Jitter}\in[0,1]$. With *signed* int32, the range is $[-2^{31},2^{31}-1]/(2^{32}-1)\subset[-\tfrac12,\tfrac12]$ (numerically verified: observed $\approx[-0.4995,0.4994]$). Order is preserved either way, so tie-breaking works, but the doc's "$\in[0,1]$" holds only for unsigned.
3. **Fair tie-breaking (random-oracle model).** If the four bytes are modelled as iid uniform over $\{0,\dots,2^{32}-1\}$ across $(r,i)$, then among $k$ tied candidates each wins with probability $1/k$, and $\Pr[\text{two jitters equal}]=2^{-32}$ per pair, so $\Pr[\text{any collision among }k]\le\binom k2 2^{-32}$.

*Proof.* (1),(2) Definition. (3) Exchangeability of iid variables gives equal win probability $1/k$ by symmetry; the pairwise collision probability is $\sum_v2^{-64}=2^{-32}$; the union bound gives the stated bound. ∎

*Statement scope.* (3) is a model-based statement; MD5 is not a random oracle, though it is empirically adequate for load spreading.

---

## 8. Literature Baselines in `router.py`

### 8.1 PRISM score

$\mathrm{Score}=S_{\text{lat}}S_{\text{load}}S_{\text{cap}}\mathrm{Aff}$ with $S_{\text{lat}}=\frac1{1+(\hat T/\mathrm{SLO})^p}$, $S_{\text{load}}=\frac1{1+(\rho/\mathrm{SLO})^p}$, $S_{\text{cap}}=\operatorname{clip}(\frac{\mathrm{Free}-\mathrm{KV}}{\mathrm{KV}},0,1)$, $\mathrm{Aff}=\beta$ if $H\ge h_{\min}$ else $1$ ($p=4,\beta=2,h_{\min}=16$).

**Theorem 8.1.**

1. $S_{\text{lat}},S_{\text{load}}\in(0,1)$ are strictly decreasing, and $\frac1{1+x^p}=\sigma(-p\ln x)$ with $\sigma$ the logistic function; $S_{\text{lat}}=\tfrac12$ exactly at $\hat T=\mathrm{SLO}$; $S_{\text{lat}}\to1$ as $\hat T\to0$ and $\to0$ as $\hat T\to\infty$.
2. $S_{\text{cap}}>0\iff\mathrm{Free}>\mathrm{KV}_{\text{req}}$; $S_{\text{cap}}=1\iff\mathrm{Free}\ge2\mathrm{KV}_{\text{req}}$.
3. $\mathrm{Score}\in[0,\beta]$.
4. **Zero-annihilation.** $S_{\text{cap}}=0$ forces $\mathrm{Score}=0$ (a hard constraint). If all replicas are annihilated, all scores tie at $0$ and the jitter (§7.4) decides.
5. In logs, $\ln\mathrm{Score}=\ln S_{\text{lat}}+\ln S_{\text{load}}+\ln S_{\text{cap}}+\ln\mathrm{Aff}$; the prefix-affinity boost $\beta=2$ is equivalent to a $\ln2\approx0.693$ additive log-score bonus. It overturns a latency gap iff $\frac{S_{\text{lat},j}}{S_{\text{lat},i}}<2$ (other factors equal).

*Proof.* (1) $x\mapsto x^p$ increasing; $\frac1{1+x^p}=\frac1{1+e^{p\ln x}}=\sigma(-p\ln x)$; at $x=1$, $\frac12$. (2) Numerator positive iff $\mathrm{Free}>\mathrm{KV}$; the ratio $\ge1$ iff $\mathrm{Free}\ge2\mathrm{KV}$. (3) Each of first three $\in[0,1]$; $\mathrm{Aff}\le\beta$. (4),(5) Product structure. ∎

*Flag (§9-F11).* $\rho_i=W_i/\max(1,S_{\max})$ is a count of iterations, but is divided by SLO (seconds). As written the ratio is dimensionally inconsistent; it is well-defined, but its scale is arbitrary.

### 8.2 D2LPM

$\mathcal C_{\text{mh}}=\{c:\mathrm{hit}_c=\max_j\mathrm{hit}_j\ge h_{\min}\}$, $\mathcal C_{\text{act}}=\{c:\mathrm{credit}_c>0\}$, Pool $=(\mathcal C_{\text{mh}}\cap\mathcal C_{\text{act}})$ if nonempty else $\mathcal C_{\text{act}}$ if nonempty else $\mathcal C$; $c^*=\arg\min(\mathrm{requests}_c,\mathrm{index}_c)$ lexicographically.

**Theorem 8.2.** Pool $\ne\emptyset$ whenever $\mathcal C\ne\emptyset$; $c^*$ exists and is **unique** (because `index` is unique), so D2LPM is fully deterministic; $c^*$ has the minimum backlog within Pool; if $\max\mathrm{hit}<h_{\min}$, $\mathcal C_{\text{mh}}=\emptyset$ and locality is ignored. *Proof.* Fallback chain ends in $\mathcal C$; lexicographic min on a finite set with a unique second key has a unique minimiser. ∎

### 8.3 `_least_load_select`

$c^*=\arg\min_i\mathcal L_i$: exists on finite nonempty sets; equals $\arg\min(4N_w+N_r)$ for equal capacities (Thm 4.1(5)); replicas with $C_i\le0$ have $\mathcal L=\infty$ and are never chosen when any finite-load replica exists.

---

## 9. System-Level Theorem and Findings

### 9.1 Composition theorem (FairRoute V2 pipeline)

**Theorem 9.1.** Let $\mathcal C\ne\emptyset$ be finite. Let `safe`, $\hat T>0$, urgency $u\in[0,1]$ be given. Let $A=\mathcal S$ if $\mathcal S\ne\emptyset$ else $A=\mathcal C$, and $T^*_A=\min_A\hat T$. Running gate $\to$ slack set $\to$ `V2` choice returns $c^*\in\mathcal C$ with:

1. $c^*\in A$ (safe whenever any safe replica exists);
2. if some safe replica meets $\gamma\mathrm{SLO}$, then $\hat T_{c^*}\le\gamma\mathrm{SLO}$;
3. $\hat T_{c^*}\le T^*_A+\min(\epsilon_{\max},\rho T^*_A)(1-u)\le(1+\rho(1-u))\,T^*_A$;
4. if $u=1$: $\hat T_{c^*}=T^*_A$;
5. among the replicas satisfying 1–3 (the slack set restricted to credit-eligible hits when any), $c^*$ has maximal prefix-hit ratio.

*Proof.* Let $\mathcal G$ be the gate output. By Thm 5.3(1),(4): $\mathcal G\ne\emptyset$, $\mathcal G\subseteq A$, $\min_{\mathcal G}\hat T=T^*_A$. Slack set is computed on $\mathcal G$ so $T^*=T^*_A$ and Thm 5.4 gives nonemptiness and the bound; Thm 5.5.5 gives $c^*\in\mathcal C_{\text{slack}}$ and the rest. (2): when $\mathcal F\ne\emptyset$, $\mathcal G=\mathcal F$ and $c^*\in\mathcal G$. (4): Thm 5.5.5(5). ∎

**Theorem 9.2 (fairness–locality–latency triangle).** Under tracker urgency with $\lambda=2$ and Theorem 9.1's hypotheses, the latency penalty paid for locality by client $c$ satisfies
$$\hat T_{c^*}-T^*_A\;\le\;\min(\epsilon_{\max},\rho T^*_A)\,e^{-\lambda D_c},\qquad(\text{using }1-U=e^{-\lambda D}),$$
which is nondecreasing in $S_c$ (Thm 7.2), equals $\epsilon_0$ when $S_c\ge T$ ($D=0$), and is strictly smaller than $\epsilon_0$ whenever $S_c<T$. *Proof.* Thm 9.1(3) with $1-u=e^{-\lambda D}$ and Thm 1.5/1.6. ∎

### 9.2 Findings: where the formulas depart from the analysis text

| # | Finding | Proved in | Impact |
|---|---|---|---|
| F1 | $D_c<1$ strictly; tracker urgency $U<1-e^{-2}\approx0.865$, so slack never fully collapses via the tracker channel; only the delay-based urgency (§7.1) reaches exactly 1. | Thm 1.5, 1.6 | Narrative claim "$U\to1\Rightarrow\varepsilon\to0$" is not realisable via `FairnessTracker`. |
| F2 | $\Theta\ge0.35$ always; the $0.05$ floor is dead code; max penalty is $20/7\approx2.86\times$, not $20\times$. | Thm 5.2 | Doc error. |
| F3 | `locality_score`: $h<1$ for any $H\le N$ ($h=1-\epsilon/(N+\epsilon)$ at a full hit). | Thm 2.1 | Negligible numerically. |
| F4 | `iteration_time` is discontinuous at $B=0$ or $S=0$ ($T\to\infty$ vs finite) and non-monotone in $B$ with minimiser $B^*\approx4.57$ in the router's configuration. | Thm 5.1, Cor 5.1a | Router arguments are $\ge1$, so no blow-up; idle replica $\approx0.6\%$ slower per iteration than $B=5$. |
| F5 | $\mathrm{TTFT}=0$ when $W+n_{\text{ext}}=0$. | Thm 6.2(4) | Use $\max(1,\lceil\cdot\rceil)$. |
| F6 | Gate may compare total latency (TTFT+Decode) to a TTFT SLO; in the worked instance every replica is infeasible. | Thm 6.3(4) | **Highest impact if `predicted_latency` is TTFT+Decode**: `slo_risk` would be always True and the gate would degrade to the safe filter. Verify which quantity is passed. |
| F7 | `MULTIPLICATIVE` has $(1+h)$ in the numerator under argmin: it penalises locality. | Prop 5.5.2 | Sign error unless `locality` is a cost in that mode. |
| F8 | `LEARNED` adds $0.01\times$ raw credit (up to 4096) to a score measured in seconds. | Prop 5.5.4 | Credit term dominates unless normalised. |
| F9 | EB weight is independent of $n_c$; the "sparse-history shrinks more" claim holds only via initialisation. | Thm 6.4(5) | Doc wording. |
| F10 | Jitter uses *signed* int32 ⇒ range $\approx[-\tfrac12,\tfrac12]$, not $[0,1]$. | Thm 7.4 | Harmless for ordering. |
| F11 | PRISM $\rho/\mathrm{SLO}$ mixes iterations and seconds; $S_{\text{cap}}=0$ at exact fit. | Thm 8.1 | Dimensional hygiene; conservative fit rule. |
| F12 | Credit lower bound is not guaranteed; single-step refill can leave credits $\le0$. | Thm 7.3(3) | Possible long locality starvation of a client–replica pair. |
| F13 | $\mathrm{FB}_i<0$ when $\mathcal L_i>1$ (unless load is clipped). | Thm 5.6(6) | Over-capacity replicas actively penalised. |
| F14 | AFLC's $(1+w_fU)$ is a common factor: it has no effect on the choice. | Prop 5.7 | Redundant term. |
| F15 | With defaults ($\lambda=\mu=0$) `predict_queue` is the identity. | Thm 3.1(4) | "Predictive" safety is reactive; foresight enters only through §6 models. |
| F16 | `iteration_time` args: §5.1 describes $(B,S)$ as (tokens, sequences) but the router passes (requests+1, token budget) and (D+1, D+1). | Cor 5.1a | Semantic inconsistency; the proofs hold for any nonnegative inputs. |

### 9.3 Numerical verification performed

A short script (reproducible in 20 lines of Python) confirmed:

* Jain's bounds on 20,000 random vectors, with $J([0,0,5])=1/3$ and $J([2,2,2])=1$;
* $\min\Theta=0.35$ on $[0,1]$;
* $T(0,2048)=0.02148$ vs $T(10^{-6},2048)\approx209.8$ s (discontinuity), $B^*=4.570$, convexity in $B$;
* the SLO example ($0.2171$ s) vs decode ($0.3142$ s);
* the MSE-shrinkage threshold (shrinks help for $w<0.6154$, hurt for $w=0.6164$ in the test with $\sigma^2=4$, $b=3$);
* signed jitter range $[-0.4995,0.4994]$;
* $\kappa(D)=(1+2D)(1-e^{-2D})$ strictly increasing on $[0,3]$.

---

## 10. Master Summary of Proved Properties

| Algorithm | Key proved properties |
|---|---|
| `record_service` / mean latency | Additive, monotone, mean in $[\min,\max]$ |
| Jain's index | $[1/n,1]$, $=1/(1+\mathrm{CV}^2)$, scale-invariant, Pigou–Dalton monotone |
| `fairness_target` | Mean; total debt = total surplus |
| `fairness_debt` | $[0,1)$, nonincreasing in $S$, 0 iff $S\ge T$ |
| `fairness_urgency` | $[0,1-e^{-\lambda})$ on debts $<1$; strictly increasing and concave |
| `locality_score` | $\epsilon$-accurate hit ratio, monotone |
| `predict_queue` | Exact reflected-fluid solution; semigroup |
| `replica_is_safe` | Downward-closed, fail-closed |
| `reactive_load` / `kv_utilization` | Monotone, weighted count / bounded, fail-closed |
| `iteration_time` | Positive, floor, convex in $B$, discontinuous at 0 |
| `predict_latency_seconds` | $\Theta\in[0.35,1]$, $\hat T\in[\mathcal N,\tfrac{20}{7}\mathcal N]$ |
| `fairroute_gate` | Total, sound, preserves optimum, idempotent |
| `fairroute_slack_set` | Nonempty, nested, bounded suboptimality $\min(\epsilon_{\max},\rho T^*)(1-u)$ |
| V2 selection | Strict locality priority within slack; credit-gated; greedy at $u=1$ |
| $H_5$ score | $\kappa(D)$ increasing; locality→load trade-off shifts with debt |
| Router TTFT | Ceil model sandwich, monotone, SLO $=k\times$ cold ideal |
| EB estimator | Convex combination; shrinkage helps iff $w<2w^*$ |
| Delay urgency | Scale-invariant; ≥ half of clients at 0 |
| Credit | Upper invariant $\le Q$; bounded eligible spending |
| Jitter | Deterministic; uniform tie-breaking under random-oracle model |
| End-to-end | Thm 9.1, 9.2: safe, SLO-aware when possible, bounded latency regret that shrinks with client debt |
