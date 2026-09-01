# Novelty Ledger

Current search checkpoint: 2026-08-31. This is a living risk register, not proof
that no related work exists. Every paper draft must update it.

## 2026-08-31 completed-study verdict

The current novelty risk is **medium--high**. Reacher replication and its
separately frozen margin-calibration extension strengthen the original
single-task evidence, but they do not create a new general control algorithm.
Two direct 2026 papers already establish thermal-aware reinforcement learning
and residual thermal control on Unitree A1 hardware. OopsieVerse (RSS 2026)
already establishes accumulated mechanical/thermal/fluid health for simulated
manipulation, including RL and MuJoCo. Prior work also establishes hidden motor-
temperature estimation, reset-free robot learning, belief-space safety,
uncertainty-aware shielding, and physics-plus-residual dynamics learning.

The paper must not claim first thermal-aware robot learning, first damage-aware
manipulation benchmark, first thermal supervisor, first hidden-temperature
estimator, first uncertainty-aware safety filter, first residual controller,
first reset-free/continuing robot learning, or first action-dependent
degradation.

The defensible contribution is a narrow controlled empirical result across two
MuJoCo tasks: hidden, action-driven thermal state changes later dynamics and
persists across task resets; a frozen uncertainty-aware supervisor is tested
under prespecified shifts; and fresh-lifetime ablations isolate the robust
uncertainty effect from the unsupported residual mechanism claim. A margin
inherited from Pusher preserved positive mean Reacher utility but missed the
original frozen point gate; development-only Reacher calibration selected a
policy that passed on new lifetimes without a detectable mean-utility loss,
while the inherited policy also passed on that same new sample.

## Claim decision matrix after Reacher

| Proposed claim | Decision | Required wording |
|---|---|---|
| Damage-aware or health-aware robot learning is new | Reject | Cite OopsieVerse and maintenance/degradation POMDPs. |
| Thermal-aware robot control or residual thermal policy is new | Reject | Cite Qian 2026, Wan 2026, Kawaharazuka 2020, and Sabelhaus 2024. |
| Uncertainty-aware safety supervision is new | Reject | Cite belief CBF, adaptive shielding, UNISafe, and Gameplay Filters. |
| Selective task reset with persistent hidden health is a novel individual concept | Avoid priority claim | Say that no exact experimental combination was identified. |
| Health changes future dynamics, unlike damage-only accounting | Use as a distinction, not a first claim | OopsieVerse explicitly keeps original task dynamics intact; maintenance and hidden-dynamics work cover adjacent ideas. |
| Inherited uncertainty margin transfers safely across morphologies | Reject | The inherited Reacher margin failed the original frozen 2% gate at 2.3%; a later 1.9% on different seeds does not erase that failure. |
| Target-development calibration can select a passing policy | Support narrowly | Report the post-confirmatory Reacher extension: 1.6% and `+0.692`; disclose inherited 1.9% on the same fresh seeds and claim neither calibration necessity nor superiority. |
| Most lifetimes improve | Reject | Only 52/100 calibrated Reacher effects were positive; exact sign `p=0.764`. |
| Mean risk-sensitive lifetime utility improves | Support | Use paired lifetime bootstrap and sign-flip evidence, with reward-preference limitations. |

## Closest-work distinction that must appear in the manuscript

| Closest work | Overlap | Remaining distinction |
|---|---|---|
| OopsieVerse (RSS 2026) | Accumulated health, thermal/mechanical damage, manipulation, RL, MuJoCo | It preserves original task dynamics and uses health for observation/reward/termination; ours hides actuator health, changes subsequent dynamics, and selectively preserves it across task resets. |
| Qian et al. (2026) | Motor thermal dynamics, RL, long-duration utility | Direct motor-temperature input, calibrated whole-body model, and hardware; ours studies hidden-state inference, uncertainty, and OOD mechanism attribution. |
| Wan et al. (2026) | Frozen nominal policy plus thermal residual and safety/performance trade-off | Their residual changes motor commands; ours corrects a thermal belief model, and its matched-margin residual benefit was not established. |
| Farrahi and Mahmood (CoLLAs 2025) / CRONOS (2026) | Continuing or reset-free robot interaction | They remove/minimize embodiment or scene resets; ours resets task state while retaining only the internal health state. |
| Adaptive Shielding / UNISafe / belief CBF work | Online uncertainty-aware intervention under hidden/OOD dynamics | They offer general filters and stronger certificates; ours provides controlled persistent-health and policy-composition evidence without formal safety guarantees. |

The full bounded depth-two search tree, retained-paper matrix, source links,
claim decisions, and final writing constraints are recorded in
[`LITERATURE_AUDIT_2026-08-31.md`](LITERATURE_AUDIT_2026-08-31.md).

## Legacy benchmark thesis (2026-08-03 checkpoint)

## Defensible working thesis

Existing work separately studies action-dependent non-stationarity, externally
scheduled actuation changes, reset-free scenes, physical damage, biological
fatigue, latent dynamics inference, and cross-episode memory.  LifePhyBench asks
when learning methods succeed under their specific combination:

1. contact-rich or embodied control;
2. a slow physical-health state changed by action/contact dose;
3. task state resets while physical health selectively persists;
4. accumulated health changes future transition dynamics, not only reward;
5. inference, long-horizon control, and physical preservation are evaluated
   separately.

Safe claim: as of the search checkpoint, we did not identify a public robot
learning benchmark that makes all five properties primary axes across multiple
degradation mechanisms.  We do not claim that any individual property is new.

## Closest work and required distinction

| Work | Prior contribution | LifePhyBench must add |
|---|---|---|
| [Action-Dependent Non-Stationarity](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3bf80b34f731313b8292f4578e820c90-Abstract-Conference.html) | Formalizes how actions affect future environments; includes wear motivation | Embodied control benchmark, selective resets, learning and diagnostic evaluation; do not claim the concept itself |
| [UBADA](https://openreview.net/forum?id=9RsXowObLi) | Gymnasium wrappers for changing actuator dynamics and continual/multitask adaptation | Degradation path must arise from accumulated actions, not a prescribed schedule |
| [CRONOS](https://embodiedai-ntu.github.io/cronos/index.html) | Reset-free multi-task manipulation with persistent scenes and reset budgets | Reset task/scene state while selectively preserving hidden physical health |
| [OopsieVerse / DamageSim](https://robin-lab.cs.utexas.edu/oopsieverse/) | Large manipulation environments with force, heat, and liquid damage | Damage must feed back into subsequent transition physics across episodes |
| [MyoSuite](https://proceedings.mlr.press/v168/caggiano22a.html) | Musculoskeletal control including fatigue and physiological variation | General selective-reset benchmark beyond muscle fatigue; no first-fatigue claim |
| [Continual World](https://proceedings.neurips.cc/paper_files/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html) | Continual manipulation across externally sequenced tasks | Same task can change because the agent endogenously changes physical health |
| [DP-MDP / LILAC](https://proceedings.mlr.press/v139/xie21c.html) | Latent dynamics change across episodes under structured drift | Slow latent state evolves within/across episodes as a controlled process |
| [CARL](https://arxiv.org/abs/2110.02102) | Contextual RL benchmark over mass, friction, damping, and strength | Context is an action-driven trajectory rather than sampled configuration |
| [HiP-MDP](https://pmc.ncbi.nlm.nih.gov/articles/PMC5466173/) | Low-dimensional latent parameters for related dynamics | Latent health is controllable and evolving, not a fixed instance parameter |
| [UPOSI](https://faculty.cc.gatech.edu/~turk/paper_pages/2017_learning_universal_policy/index.html) | Online system identification from recent interaction | Strong baseline; LifePhyBench additionally tests planning under future damage |
| [RMA](https://roboticsproceedings.org/rss17/p011.pdf) | Rapid history-based adaptation to hidden conditions | Strong baseline; separate recognition from endogenous health preservation |
| [RL²](https://arxiv.org/abs/1611.02779) | Cross-episode recurrent adaptation | No first cross-episode/dual-timescale memory claim |
| [SF-RSSM](https://ojs.aaai.org/index.php/AAAI/article/view/39825) | Separate fast and slow latent dynamics | Method must encode reset structure and action-to-health causality, not just two latents |

## Claims prohibited without new evidence

- first action-dependent non-stationary RL problem;
- first persistent, reset-free, damage-aware, wear, or fatigue benchmark;
- first cross-episode memory or dual-timescale RL method;
- intrinsically non-Markovian dynamics (the augmented latent state is Markov);
- physically accurate or realistic aging without calibrated real measurements;
- continual learning when deployment changes recurrent state but not parameters;
- general physical aging from a few hand-designed scalar degradation laws.

Use “physics-inspired” or “phenomenological degradation law” until calibration
evidence exists.

## Required novelty controls

- persistent versus episode-reset health;
- endogenous versus dose-matched exogenous drift;
- dynamics feedback versus damage-in-reward only;
- hidden versus privileged health;
- observation-only, action-aware, and clock-only health estimators;
- episode-reset versus lifetime-persistent memory;
- held-out degradation rate and held-out functional family;
- current-state oracle versus lifetime-planning oracle.

## Algorithm design constraint

A two-RNN PPO is insufficient.  The proposed method must couple:

1. fast task belief that obeys task resets;
2. slow health belief that obeys maintenance/selective resets;
3. action/contact dose in the slow transition model; and
4. an objective or planner that represents future degradation cost.

The final algorithm name is deliberately undecided until strong baselines and the
planning oracle establish which technical component is actually needed.
