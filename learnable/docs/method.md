# AdaImprint: method

Adaptive Imprint makes the FIO memory's retention rule learnable. This note
gives the model in mathematical form; [README.md](../README.md) covers files and
commands.

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $f_t$ | the $t$-th FIO in the stream, $t = 1, 2, \dots$ |
| $\tilde e_t \in \mathbb{R}^{384}$ | unit-norm contextual embedding of $f_t$ |
| $\mathrm{ts}_t$ | timestamp of $f_t$ (seconds) |
| $\mathcal{G} = \{g_1, \dots, g_N\}$ | prototype bank, $N \le N_{\max}$ |
| $n(t)$ | prototype assigned to $f_t$ |
| $\nu_n,\ \ell_n$ | number of FIOs assigned to prototype $n$, and when it was last seen |
| $\mathcal{M}_t$ | episodic memory (retained entries) before $f_t$ |
| $o_t$ | budget occupancy $\lvert\mathcal{M}_t\rvert / B$ |
| $B$ | memory budget (number of entries) |
| $\theta$ | importance parameters, controller and policy weights |
| $\omega$ | predictor weights |
| $\sigma(\cdot)$ | logistic sigmoid |

## 2. Baseline: the heuristic rule

`FIOOptimizer` assigns $f_t$ to a prototype and computes, from the prototype's
statistics *before* they are updated,

$$
I_t \;=\; \underbrace{\log(1+\nu_{n(t)})}_{F_t}\;
\underbrace{e^{-\lambda \Delta_t}}_{R_t}\;
\underbrace{\frac{1}{1+\log(1+N)}}_{D_t},
\qquad \Delta_t = \mathrm{ts}_t - \ell_{n(t)} .
$$

$f_t$ is retained iff $I_t \ge \tau$, with $\tau = 0.2$ and
$\lambda = 1.146\times10^{-6}\,\mathrm{s}^{-1}$ (7-day half-life). A retained
$f_t$ that has an equivalent memory entry $c_t \in \mathcal{M}_t$ (same
action|object|app and $\cos(\tilde e_t, \tilde e_{c_t}) \ge 0.75$) is merged
into it; otherwise it becomes a new entry. Every quantity here is fixed by hand.

## 3. Learnable importance

In log space the rule becomes

$$
\log \hat I_t \;=\; a_F \log F_t \;-\; \lambda \Delta_t \;-\; a_D \log\!\big(1+\log(1+N)\big) \;+\; b \;+\; \delta_t ,
$$

with learned scalars $a_F, a_D, b$ and $\lambda = e^{\kappa}$ (learning
$\kappa$ keeps $\lambda > 0$), plus a context-dependent correction $\delta_t$
from the controller (§4).

At initialisation $a_F = a_D = 1$, $b = 0$, $\lambda = 1.146\times10^{-6}$ and
$\delta_t \equiv 0$, so $\hat I_t = I_t$ and the model reproduces the
heuristic exactly.

## 4. Controller

**Input.** The controller input concatenates static and memory-dependent features:

$$
z_t = \big[\,\mathrm{norm}(s_t),\; o_t,\; E_t,\; U_t,\; W_e \tilde e_t\,\big]
$$

* $s_t$: 16 static features computed from $\mathcal{G}$ before its update: $\log F_t$, $\Delta_t$, $\log(1+N)$, $\log I_t$, novelty $S_t = 1-\max_n \cos(\tilde e_t, g_n)$, a new-prototype flag, history similarity, field richness, speech, number of persons, tool/app, time of day, day, and position in the clip.
* $o_t = |\mathcal{M}_t| / B$: budget occupancy.
* $U_t = \max_{s} \cos(\tilde e_t, e_s)\,\min(1, m_s)$: redundancy against recent memory slots (§6).
* $E_t$: predictive surprise (§7).
* $W_e \tilde e_t$: a learned 16-d projection of the embedding.

**Recurrent update.**

$$
(h_t, c_t) = \mathrm{LSTM}_\theta(z_t, h_{t-1}, c_{t-1})
$$

$$
\delta_t = 2\tanh\!\big(\mathrm{MLP}^{\delta}_\theta([h_t, z_t])\big) \in [-2, 2]
$$

## 5. Soft decisions

The keep test is relaxed to a gate centred on the threshold:

$$
\gamma_t = \sigma\!\left(\frac{\log \hat I_t - \log \tau}{T}\right), \qquad T = 0.25,
$$

so $\gamma_t > 0.5 \iff \hat I_t > \tau$. Given retention, the action
distribution is

$$
\pi_t =
\begin{cases}
\mathrm{softmax}\!\big(\mathrm{MLP}^{\pi}_\theta([h_t, z_t])\big) & \text{if } c_t \text{ exists}\\[2pt]
(1, 0, 0) & \text{otherwise}
\end{cases}
\qquad \pi_t = (\pi^{w}_t, \pi^{m}_t, \pi^{u}_t),
$$

over write, merge and update, with $P(\text{discard}) = 1-\gamma_t$. The policy
bias is initialised to favour merge, which matches the heuristic.

**Learnable consolidation gate.** The heuristic only offers $c_t$ when
$\mathrm{sim}_t = \cos(\tilde e_t, \tilde e_{c_t}) \ge 0.75$. Traces now record the best
same-pattern entry down to a floor ($0.5$) together with $\mathrm{sim}_t$, and a learned
threshold $\theta_c \in [\text{floor}, 1]$ (initialised at $0.75$) decides whether it counts:

$$
q_t = \sigma\!\left(\frac{\mathrm{sim}_t - \theta_c}{T_c}\right),\quad T_c = 0.05,
\qquad
\pi_t = (1-q_t)\,(1,0,0) + q_t\,\mathrm{softmax}\big(\mathrm{MLP}^{\pi}_\theta([h_t, z_t])\big).
$$

The hard rule is $\mathrm{sim}_t \ge \theta_c$; the gradient reaches $\theta_c$ through
$q_t$ in the straight-through policy (§9). Lowering $\theta_c$ merges more and writes less,
which $\mathcal{L}_{\text{mem}}$ rewards; $\mathcal{L}_{\text{past}}$ penalises merges that
lose content. The pattern test (same action|object|app, synonym groups) stays fixed.

**Inference.** Retain iff $\hat I_t \ge \tau$; if $c_t$ exists and $\mathrm{sim}_t \ge \theta_c$
take $a_t = \arg\max \mathrm{softmax}(\cdot)$, else write.

## 6. Memory as weighted slots

The memory is a set of slots $(e_s, \mathrm{ts}_s, m_s)$: a content vector, a time and
a mass. With $\mu^{a}_t = \gamma_t \pi^{a}_t$, processing $f_t$ adds:

| Action | Slots added |
| --- | --- |
| write | $(\tilde e_t,\ \mathrm{ts}_t,\ \mu^w_t)$ |
| merge | $(\tilde e_{c_t},\ \mathrm{ts}_t,\ \mu^m_t)$: its time is kept, its content is the entry's |
| update | $(\tilde e_t,\ \mathrm{ts}_t,\ \mu^u_t)$ and $(\tilde e_t,\ \mathrm{ts}_{c_t},\ \mu^u_t\, m_{c_t})$; the entry's write slot is scaled by $1-\mu^u_t$ |
| discard | nothing |

**Read-out** at query time $q$ uses a temporal kernel ($\sigma_{k} = 300$ s, $\epsilon = 10^{-3}$):

$$
r(q) = \frac{\sum_s m_s\, k(q, \mathrm{ts}_s)\, e_s}{\sum_s m_s\, k(q, \mathrm{ts}_s) + \epsilon},
\qquad k(q, \mathrm{ts}_s) = \exp\!\left(-\frac{|q-\mathrm{ts}_s|}{\sigma_{k}}\right).
$$

Because every mass $m_s$ is a function of $\theta$, so is $r(q)$.

## 7. Predictor and surprise

The next prototype is predicted from the controller state and the compressed memory:

$$
P_\omega(\,\cdot \mid h_t, \mathcal{M}_{\le t}) = \mathrm{softmax}\!\big(W_\omega [h_t,\ r(\mathrm{ts}_t)] \big),
$$

reading only slots from the last 256 FIOs. Surprise is the normalised negative
log-likelihood of the prototype that actually arrives:

$$
E_t = -\frac{\log P_\omega\big(n(t) \mid h_{t-1}, \mathcal{M}_{\le t-1}\big)}{\log N_{\max}} .
$$

## 8. Training objective

The objective is self-supervised and computed on chronological chunks of
$K = 128$ FIOs.

**Next-event prediction:**

$$
\mathcal{L}_{\text{pred}} = -\frac{1}{K}\sum_{t} \log P_\omega\big(n(t+1) \mid h_t, \mathcal{M}_{\le t}\big)
$$

**Past reconstruction.** Sample $Q = 64$ past FIOs $\mathcal{Q}$ from the last
2048 and recover each one's prototype from the memory, read at the FIO's own time
and content ($\bar g_n$ are normalised prototype centroids, temperature $\kappa = 0.1$).
For this read the weights of §6 become
$w_{js} = m_s\, k(\mathrm{ts}_j, \mathrm{ts}_s)\, \exp\big((\tilde e_j^\top e_s - 1)/\kappa_r\big)$,
$\kappa_r = 0.1$. A time-only read cannot work here: most FIOs of a caption clip share
one timestamp, and with σ = 300 s it scored at chance on A1_JAKE whatever was kept
(`--past-read time` restores it).

$$
\mathcal{L}_{\text{past}} = -\frac{1}{Q}\sum_{j \in \mathcal{Q}}
\log \frac{\exp\!\big(r(\mathrm{ts}_j)^\top \bar g_{n(j)} / \kappa\big)}{\sum_{n} \exp\!\big(r(\mathrm{ts}_j)^\top \bar g_n / \kappa\big)}
$$

**Memory cost.** $\rho^\star$ is the target write rate (default: the heuristic's):

$$
\mathcal{L}_{\text{mem}} = \Big|\tfrac{1}{K}\textstyle\sum_t \mu^w_t - \rho^\star\Big| \;+\; 0.05\cdot\tfrac{1}{K}\textstyle\sum_t (\mu^m_t + \mu^u_t)
$$

**Redundancy** (band $1 \le j - i \le 16$, threshold $\delta = 0.75$):

$$
\mathcal{L}_{\text{red}} = \frac{1}{|\mathcal{P}|}\sum_{(i,j) \in \mathcal{P}} \mu^w_i\, \mu^w_j \,\big[\cos(\tilde e_i, \tilde e_j) - \delta\big]_+
$$

**Total:**

$$
\mathcal{L} = \mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{past}} + \beta\,\mathcal{L}_{\text{mem}} + \gamma_{r}\,\mathcal{L}_{\text{red}}
\;\big[+\, w_{e}\,\mathcal{L}_{\text{ev}}\big],
\qquad \beta = 5,\ \gamma_r = 0.1 .
$$

$\mathcal{L}_{\text{ev}} = -\frac{1}{|\mathcal{E}|}\sum_{t\in\mathcal{E}} \log \gamma_t$
is optional. $\mathcal{E}$ is the set of FIOs within 30 s of an EgoLifeQA
`target_time`, from training wearers only.

Next-event prediction alone rewards keeping routine, predictable events.
$\mathcal{L}_{\text{past}}$ counters this: dropping $f_j$ is cheap only when
retained neighbours still identify it.

## 9. Gradients through hard decisions

The memory is built from the hard decisions $\bar\gamma_t = \mathbb{1}[\hat I_t \ge \tau]$
and $\bar\pi_t = \mathrm{onehot}(a_t)$. A straight-through estimator lets the
loss see that memory while the gradient flows through the soft values:

$$
\gamma^{\text{ST}}_t = \bar\gamma_t + \gamma_t - \mathrm{sg}(\gamma_t),
\qquad
\pi^{\text{ST}}_t = \bar\pi_t + \pi_t - \mathrm{sg}(\pi_t),
$$

where $\mathrm{sg}$ is stop-gradient. Training therefore uses the same memory
as the build.

## 10. Optimisation

$$
(\theta, \omega) \leftarrow (\theta, \omega) - \eta\,\mathrm{Adam}\big(\nabla_{\theta,\omega}\, \mathcal{L}\big), \qquad \eta = 3\times10^{-3},\ \ \|\nabla\| \le 1 .
$$

* The update runs after every chunk. $(h, c)$ is carried across chunks but detached (truncated BPTT).
* Training is leave-one-wearer-out: train on the other wearers, select the checkpoint on the held-out one.
* Selection score: $\mathcal{L}_{\text{past}}$ on the validation wearer, penalised when $|\rho - \rho^\star| > 0.02$.

## 11. Hard budget at build time

When $|\mathcal{M}_t| = B$ and the decision is *write*, the entry with the
lowest decayed utility is chosen:

$$
i^\ast = \arg\min_{i \in \mathcal{M}_t}\ \hat I_i \, e^{-\lambda(\mathrm{ts}_t - \mathrm{ts}^{\text{last}}_i)} .
$$

* If $\hat I_t$ is below that utility, $f_t$ is discarded instead.
* Otherwise $i^\ast$ is evicted. If a live entry shares its action|object|app, $i^\ast$ is consolidated into it (its `action_sequence` keeps the timestamps); if not, it is dropped.

Prototype statistics ($\nu_n, \ell_n, g_n$) are updated for every $f_t$,
retained or not, and are kept separate from $\mathcal{M}$.

## 12. Summary

| Hand-set rule | AdaImprint |
| --- | --- |
| $I_t = F_t R_t D_t$ | $\log \hat I_t = a_F \log F_t - \lambda\Delta_t - a_D \log(1+\log(1+N)) + b + \delta_t$ |
| keep iff $I_t \ge \tau$ | gate $\gamma_t = \sigma\big((\log\hat I_t - \log\tau)/T\big)$ |
| always merge into $c_t$ | policy $\pi_t$ over write / merge / update |
| $c_t$ counts iff $\cos \ge 0.75$ | learned consolidation threshold $\theta_c$ |
| unbounded memory | budget $B$ with evict-or-consolidate |
| fixed $\lambda$, exponents 1 | learned $a_F, a_D, \lambda, b, \theta_c, \theta, \omega$ by minimising $\mathcal{L}$ |