# Critic-Form Linear Attention — Reference & Working Protocol

This document is the fixed reference for the linear-attention direction built on a
logarithmic dot-product contrastive critic. It states the mathematical model, the
design decisions already taken, the options deliberately kept open, and the working
protocol for Claude Code on this repo.

It is **theory and protocol only**. It contains **no implementation tasks** — those
arrive as separate, specific prompts. Do not implement ahead of an explicit task.

---

## 1. The critic and the identity everything rests on

The critic is a logarithmic dot product on bounded, non-negative codes:

```
g_eps(a, b) = log( <a, b> + eps ),    a, b in [0,1]^d,    eps > 0 small
```

Its defining algebraic property is that the log cancels the exponential inside a
contrastive partition function:

```
exp( g_eps(a, b) ) = <a, b> + eps
```

This single property has **two corollaries — the same object seen twice, not two
separate applications**:

1. **Contrastive denominator collapse.** In an InfoNCE objective the denominator
   `(1/K) sum_k exp(g(z_i, z_k)) = <z_i, (1/K) sum_k z_k> + eps` collapses to a dot
   product with a single mean vector: `O(Kd)` instead of `O(K^2 d)` in batch size.
   _Each sample also contributes to that mean, so there is no implicit stop-gradient
   on the denominator; the gradient must flow through the mean._

2. **Linear attention.** Attention weights are a softmax over critic scores. Removing
   the exponential the same way gives
   ```
   sum_j ( <q, k_j> + eps ) v_j = q^T ( sum_j k_j v_j^T ) + eps * sum_j v_j
   ```
   i.e. linear attention with a KV state `S = sum_j k_j v_j^T`.

The averaged-negatives vector (corollary 1) and the KV state (corollary 2) are the
**same factorized partition function**. Preserve this framing.

**What is and isn't novel.** Positive-feature-map linear attention already exists. The
new elements are: (i) the non-negative feature map comes from an MI contrastive
estimator with a log-dot-product critic, not a heuristic choice (elu+1, ReLU); and
(ii) sparsity is _learned_ and induced by that formulation. Empirically the critic
reaches ~1–10% activity without dead units, with roughly uniform per-unit activity,
where L1 tends to produce dead units before reaching usefully-sparse regimes.

---

## 2. The claim under test

Removing the softmax reduces attention's selectivity. The hypothesis is that
**learned sparsity restores selectivity (at least partially) without reintroducing an
exponential.**

Two precision points that must not be blurred:

- **Sparsity does not shrink the KV state.** `S` is `d_k x d_v` and densifies as outer
  products accumulate. Sparsity buys **compute** (`O(kappa d_v)` per token instead of
  `O(d_k d_v)`), not state memory. Any state-size claim is about **accuracy at matched
  state size**, never reduced state.
- **Primary claim = accuracy at matched state size** (interference argument, below).
  Compute at matched accuracy is a **secondary** claim on a separate throughput axis.

---

## 3. Interference / SNR model (theoretical prediction)

Read with query `q` aimed at target key `k_t`:

```
v_hat = q^T S = <q,k_t> v_t  +  sum_{j!=t} <q,k_j> v_j
              = signal        +  crosstalk
```

With unit-norm **orthogonal values**, crosstalk power `= sum_{j!=t} w_j^2` where
`w_j = <q, k_j>`. Define `SNR = w_t^2 / E[ sum_{j!=t} w_j^2 ]`.

**Code model M1 (idealized):** each key is exactly kappa-sparse binary, support
uniform at random over `d_k`, independent across keys; clean self-query `q = k_t`;
values orthogonal.

Under M1:

- Signal: `w_t = <k_t,k_t> = kappa` (exact).
- Off-target overlap mean: `E[w_j] = kappa^2 / d_k` (exact) — this is the origin of
  the `kappa^2/d_k` crosstalk scaling.
- In the sparse regime `kappa << sqrt(d_k)`: `E[ sum_{j!=t} w_j^2 ] ≈ (N-1) kappa^2/d_k`.
- **Result:**
  ```
  SNR ≈ d_k / (N - 1)
  ```
  In this clean idealized case **kappa cancels**: signal power and per-term crosstalk
  both scale as `kappa^2`.

**This cancellation is a property of the idealization, not a negative result.** kappa
re-enters — in sparsity's favor — as soon as M1 is relaxed:

1. **Imperfect queries.** A learned query recovers only a fraction of the target
   support (`rho kappa`) plus spurious active units (`sigma`). SNR then carries a
   factor `~ (rho kappa / (rho kappa + sigma))^2`: recall tracks **query precision**,
   the fraction of query mass on-target, not kappa alone.
2. **Non-uniform support (the realistic case).** True off-target overlap is
   `E[w_j] = sum_c p_c^(q) p_c^(k) >= kappa_q kappa_k / d_k`, with **equality iff
   support is uniform** (Cauchy–Schwarz). Concentrated activity (popular units, dead
   units) strictly _raises_ crosstalk. So the mechanism is not "sparse beats dense" but
   **"uniformly-distributed sparse achieves the crosstalk floor; concentrated sparse
   does not."** The critic is claimed to produce uniform support; a heuristic feature
   map or L1 need not.

**Consequences for what we plot and how we read it:**

- `SNR vs N` at fixed kappa tests the `1/(N-1)` load scaling; **split by arm**, it tests
  the support-uniformity claim (arms with equal mean kappa can differ in `sum_c p_c^2`).
- `SNR vs kappa` at fixed N tests the cancellation directly. **Deviation from flatness
  is itself the signal** — it measures how far the real encoder is from M1's regularity
  (downward slope = concentration grows with kappa; upward = codes become more uniform).
- Plot **both**. Do not pre-commit to an interpretation; the plots are diagnostics of
  code geometry, not just recall curves.

**Assumed, not derived (check by measurement, do not extend the algebra):** orthogonal
values (real MQAR values are learned embeddings; non-orthogonality adds a value-overlap
term that likely raises the floor uniformly across arms without changing the ranking,
but this is unproven); independence of key supports (learned keys are correlated — this
_is_ the non-uniform-support effect above).

---

## 4. The four-arm ablation ladder

All on MQAR, matched state size. Softmax is the ceiling; standard linear attention is
the baseline the derivation replaces.

1. **Softmax attention** — ceiling / harness check (should ~solve MQAR).
2. **Standard positive-feature-map linear attention** — the baseline arm 4 must beat.
   Include **two** feature maps here: (a) **elu+1 / ReLU** — the generic linear-attention
   baseline; (b) **2nd-order Taylor** (the "Based" feature map, already implemented in
   zoology) — the **benchmark-specific strong baseline**. MQAR's known linear-attention
   failure mode is _insufficient spikiness / high-entropy attention weights_ (zoology
   blog 2); Based recovers spikiness via the Taylor map **without sparsity**. This is the
   near-neighbor a reviewer will cite (see §5.4), so it must be beaten or matched-at-
   lower-cost, not omitted. elu+1 alone would be a strawman.
3. **Critic-form attention** — bounded `[0,1]` keys/queries, weight `<q,k> + eps`, **no
   sparsity regularizer**. Isolates the feature map alone. (Sparsity _may_ emerge here
   from the task gradient; treat achieved kappa as a measured outcome, not an assumption.)
4. **Critic-form + contrastive sparsity regularizer** — the full method.

_Framing note:_ the field describes linear-attention's MQAR failure as "high-entropy /
non-spiky attention." Our interference/SNR account (§3) is the **same axis in different
vocabulary**: uniformly-distributed sparse keys give low pairwise overlap, so a query's
weight on its target dominates off-target weights — a spikier effective attention. Our
contribution is a _mechanism_ (learned sparsity from an MI critic) with an analytic
crosstalk argument for a recognized failure mode, not a new problem.

**Kill criterion, fixed in advance:** if arm 4 does not beat arm 2 at matched state size
across seeds, the direction is dropped entirely — no "promising trend" language, no
partial inclusion. But first rule out bugs and wrong implicit assumptions (see §6
decision rule), I will be the one making the call if necessary, and only if it comes to it. I may decide to investigate further anyway, introducing other variants.

---

## 5. Arm-4 auxiliary loss and the design decisions taken

### 5.1 Auxiliary contrastive loss (arm 4)

Keys and queries are regularized **separately**, each with its own contrastive term.
Unit of contrast (matches the manuscript, empirically stable):

- **Per-position, self-positive, all-vs-all negatives.** For key `k_i^(t)` at position
  `t`, sequence `i`: positive = a lightly masked/blurred copy of itself; negatives =
  **all** other key vectors in the batch, across all positions and all sequences,
  including within the same sequence (true all-vs-all). Same construction for queries.

Using the linearized critic in an InfoNCE term, with the collapse identity in the
denominator (mean over the full all-vs-all set):

```
L_key = - mean_i log(  (<k_i, k_i_pos> + eps)
                     / ( <k_i, mean(all k)> + eps ) )
```

(analogously `L_query`), and

```
L_aux   = mean over positions of ( L_key + L_query )
L_total = L_task + beta * L_aux
```

Decisions inside this loss:

- **Gradient through the denominator mean is kept** — compute an in-batch mean and use
  it directly. Do **not** add stop-gradient, `.detach()`, or an online/tracked average.
  This is part of the method's definition, not an approximation to simplify.
- **Positive perturbation:** start cautious — the loss is stable even with unperturbed
  self-positives. Use a light, clearly-acceptable masking/blur strength; increase later
  only if results motivate it.
- **beta** (auxiliary weight): a real hyperparameter. Pilot fixes a single value and
  **reports achieved kappa**, learning the `beta -> kappa` mapping empirically before any
  sweep (same logic as eps in the manuscript). too small -> arm4 ≈ arm3; too large ->
  recall sacrificed to sparsify.

_Interpretation note for later:_ all-vs-all negatives push apart keys from different
positions in the same sequence. On MQAR this is benign/helpful (stored keys should be
separable), but it means the aux loss does some separation work _directly_, not only via
sparsity. If arm 4 beats arm 3, use the achieved-kappa and support-uniformity
diagnostics to attribute why (sparsity/uniformity vs. direct separation).

### 5.2 The eps floor in the attention read

Normalized critic-attention weight on stored pair `j`:

```
alpha_j = ( <q,k_j> + eps ) / ( <q, sum_j' k_j'> + N eps )
```

The `N eps` term makes eps a **uniform attention floor whose relative weight grows with
load N** (`~ N eps / <q, sum k>`): at low load negligible; at high load it smears
attention toward a flat average over all stored values — a load-dependent selectivity
leak. eps therefore earns its place in the **loss** (log stability + sparsity knob) but
in the **read** it only buys this leak.

The interference derivation in §3 assumes `eps = 0` in the read, which is not necessarily what we test in code.

### 5.4 Related baselines a reviewer will raise (pre-empt)

- **Based / Taylor feature map** (Arora et al., zoology blog 2; implemented in this repo).
  Recovers attention spikiness via a 2nd-order Taylor approximation of the exponential,
  _without_ sparsity. This is the closest competitor: "is the sparse critic just a
  worse/better way to recover spikiness than the Taylor map?" Must be an arm-2 baseline
  (see §4), not ignored. Our distinct claim: sparsity reduces crosstalk _and_ buys
  compute (`O(kappa d_v)` reads), and the feature map is _learned_ from an MI objective
  rather than a fixed polynomial.
- **Delta-rule family** (DeltaNet, Gated DeltaNet; in `fla`). Handles interference via
  key-conditioned _replacement_ (update rule), not code geometry. Position: sparsity
  (code geometry) and delta rules (update rule) are **orthogonal and should compose** —
  a reviewer may ask "is sparsity just a worse delta rule?" Held out of the pilot; test
  sparse-keys-inside-delta as a possible fifth arm only if sparse-alone shows something.

### 5.3 Options frozen for the pilot vs. kept open

**Frozen for the pilot** (to match the `fla` baseline and stay clean):

- **Normalized** attention read (match `fla` linear-attention normalization).
- **Small non-null eps** in the read.

**Kept open, to be swept only if the direction is alive:**

- eps = 0 vs. small eps in the read (pilot uses small eps).
- Un-normalized vs. normalized read (pilot uses normalized).
- beta, mask/blur strength.

**Held out entirely (confounds, not in the pilot):**

- Relative-position multiplicative decay `gamma^(i-j)` (factorizes into
  `S_i = gamma S_{i-1} + k_i v_i^T`, so compatible) — held out as a confound.
- A fifth arm placing sparse keys inside a delta-rule update (DeltaNet / Gated
  DeltaNet). Position: sparsity (code geometry) and delta rules (update rule) are
  orthogonal and should compose; test only if sparse-alone shows something first.

---

## 6. Minimal decisive pilot

**Purpose:** answer one binary question — _is there a real, seed-stable gap between arm 4
and arm 2 at matched state size?_ — for the least compute that answers it credibly,
while emitting enough diagnostics that a null result is _attributable_ (real vs.
bug/bad-hyperparameter), not merely observed.

**Grid:** `4 arms x 2 d_k x ~3 N values x 1 seed`, single toggle setting (normalized
read, small eps), all diagnostics on. ~24 short 2-layer runs.

- **Arms:** all four. Do not drop any — arms 1 and 3 are cheap and each is a specific
  diagnostic (1 = harness/solvability check; 3 = feature-map-alone). Dropping either
  blinds a null result.
- **d_k:** two values bracketing the interference regime — one small where interference
  bites (arms should separate), one large where interference is mild (arms should
  converge toward the softmax ceiling). Exact values set to give a clean **matched
  state-size** comparison against the `fla` baselines; the ratio matters more than the
  absolute.
- **N (stored KV pairs):** the sweep that matters (SNR degrades in N; the claim is about
  load). ~3–4 values spanning softmax-easy to softmax-stressed.
- **Seeds:** 1 for the pilot exploration pass — explicitly _not_ decisive. Only a
  comparison showing a promising single-seed gap is promoted to 3 seeds.

**Harness calibration first (cheap, de-risks the whole pilot):** before running the
other arms, report **softmax's own degradation curve over N**. If softmax is either
perfect or zero across all N with no graceful middle, the interference-limited regime
where arms can separate may not be sampled — find zoology's difficulty knob (pairs,
sequence length, vocab) before proceeding. This risk is real and cannot be resolved on
paper.

**Diagnostics emitted even in the one-seed pass** (the difference between an
interpretable null and a mysterious one):

- Achieved kappa per arm (per-unit and per-sample activity distributions).
- Dead-unit count per arm (ports the manuscript's central diagnostic).
- Negative-overlap distribution in the aux loss (are negatives trivial?).
- Retrieval SNR: `<q, k_target>` vs. aggregated off-target, as a function of N (the
  centerpiece).
- Task accuracy vs. N per arm at each d_k (the recall curve).

**Decision rule, fixed now (do not rationalize after seeing numbers):**

1. Softmax (arm 1) doesn't ~solve MQAR at these sizes -> **harness problem**; fix before
   reading anything else.
2. Arms 3/4 not actually sparse (achieved kappa ≈ dense) -> the feature
   map/regularizer isn't inducing what's needed; **diagnose the loss** before judging
   recall.
3. (1) and (2) pass and there's a visible arm-4-over-arm-2 gap at the
   interference-limited `(d_k, N)` corner -> **promote that comparison to 3 seeds**;
   direction provisionally alive.
4. (1) and (2) pass and there's no gap anywhere, arm 3 = arm 4 = arm 2 -> the critic
   feature map is **inert for this task**; kill criterion fires cleanly; drop without
   hedging.

---

## 7. Benchmark and environment

- **Benchmark:** MQAR (Multi-Query Associative Recall) via a fork of
  HazyResearch/zoology. **Keep zoology's baselines and evaluation protocol intact** — we
  add our method on top, we do not replace their harness. MQAR is purpose-built to
  discriminate architectures by recall capacity at fixed state, i.e. our accuracy-at-
  matched-state-size axis. Tasks are defined by subclassing `DataSegmentConfig`
  (`MQARConfig`); difficulty knobs include `input_seq_len`, number of key-value pairs,
  and `vocab_size`. Adding a mixer is the repo's intended extension path. Based (Taylor
  feature map) already lives in the repo and is a required baseline (§4, §5.4).
- **Reference library for linear-attention baselines:** `fla` (flash-linear-attention).
- **Models:** 2-layer, short sequences, single GPU per run.
- **Canonical plot:** accuracy vs. recurrent state size in bytes.
- **Compute:** remote SSH server, shared lab resource. **At most 2 GPUs at a time.** A
  GPU already in use by someone else is fine _only if it has enough free resources_.
  Check before launching.
- **Environment:** a `uv` virtualenv is already provided at `~/py_venvs/sdc`, you should use it; most libraries are likely already installed. Install into that env only what is missing.

---

## 8. Working protocol for Claude Code

- **Do not jump ahead.** Implement only what the current explicit task asks. No
  speculative features, no extra arms, no toggles beyond the frozen pilot setting unless
  a task says so.
- **Do not re-scan the whole codebase every time.** Maintain a concise, short, and clear running changelog (see
  below) of where things were added and what was modified, and consult it first.
  **However, the changelog does not substitute for reading the actual code when
  precision is required** — when correctness depends on exact behavior, read the code.
- **Maintain a running changelog** (e.g. `CHANGELOG_CRITIC_ATTN.md`) recording, per
  change: file(s) touched, what was added/modified, and any assumption made. Keep it
  current so future tasks need not re-derive the layout.
- **Faithfulness checkpoints** (correctness, not style):
  - Gradient flows through the aux-loss denominator mean (no stop-gradient/detach/online
    average).
  - Arm 3 uses bounded `[0,1]` keys/queries with weight `<q,k> + eps` and _no_ sparsity
    regularizer.
  - Matched state size is genuinely matched across arms in the comparison.
  - Read normalization matches the `fla` baseline it is compared against.
- **Report cleanly.** For each run/step emit the diagnostics of §6 in a consistent,
  parseable form. State clearly what was run, what the numbers are, and what was _not_
  done. Flag anything ambiguous rather than guessing.
- **Flag, don't guess.** If a task under-specifies something that affects correctness or
  interpretation, stop and ask rather than pick silently.
- **Distinguish derived / assumed / measured** in any analysis or summary, matching the
  language of this document.

---

_This document is the theory-and-protocol reference. Specific implementation tasks are
delivered separately._
