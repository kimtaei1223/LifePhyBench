# Literature and novelty audit — 2026-08-31

Status: final pre-writing checkpoint for the completed Pusher--Reacher study,
not a claim of bibliometric exhaustiveness or priority.

## Bottom line

The search did not identify an exact duplicate of the completed experiment:
hidden action-driven thermal state, task-state resets with thermal persistence
and dynamics feedback, uncertainty-aware supervision of a fixed low-level
policy, prespecified OOD shifts, fresh-lifetime mechanism attribution, and a
development-only margin calibrated on one morphology then evaluated once on
new lifetimes. That combination can support a narrow empirical contribution.

It did identify direct prior art for every major component. In particular, two
2026 quadruped studies already demonstrate thermal-aware reinforcement learning
and residual thermal control on Unitree A1 hardware. Other work already covers
online hidden motor-temperature estimation, thermal supervisory control,
belief-space safety, hidden-parameter inference, uncertainty-aware shielding,
and physics-plus-residual dynamics. The paper therefore must not present any of
those ingredients as individually novel.

The strongest newly verified overlap is OopsieVerse (RSS 2026): it already
introduces accumulated object/robot health, mechanical/thermal/fluid damage,
damage-aware POMDP augmentation, RL, and MuJoCo manipulation. Its authors
explicitly state that health leaves the original task dynamics intact and is
used for observations, reward, or termination. The remaining LifePhyBench
distinction is therefore not “damage-aware manipulation,” but hidden actuator
health that changes later control dynamics, persists through selective task
resets, and is managed through belief uncertainty.

The current contribution is best described as a controlled two-task empirical
study of when uncertainty-aware belief supervision improves risk-sensitive
lifetime utility under persistent hidden thermal dynamics. The zero-shot
Pusher margin improved Reacher mean utility but missed the frozen safety gate;
target-task development-only calibration restored the gate without detectable
mean-utility loss. It is not a new general control algorithm, a realistic
motor-thermal model, or a hardware-validated thermal-management method.

## Audit protocol

### Search date and scope

- checkpoint date: 2026-08-31;
- target fields: robot learning, safe RL, belief-space control, thermal-aware
  robot control, actuator degradation, hidden-parameter adaptation, residual
  dynamics, and persistent non-stationarity;
- sources preferred: official proceedings, publisher pages, author-hosted
  manuscripts, and arXiv records;
- discovery-only sources were not used as evidence when a primary source was
  available.

### Tree depth

The audit used a bounded breadth-first search to depth two.

- **Level 0:** the final v12 claim and its method components.
- **Level 1:** papers returned by direct searches for the claim or one of its
  essential components.
- **Level 2:** references cited by the closest Level-1 papers when they introduce
  the thermal model, residual-control architecture, belief/safety formalism, or
  hidden-dynamics adaptation on which the Level-1 result depends.

Depth two is a realistic stopping rule: it exposes the conceptual ancestry of
the closest work while avoiding an unbounded expansion into generic RL,
estimation, and control references. Generic optimizer, PPO, simulator, and
locomotion citations were screened but retained only when needed to establish a
specific novelty boundary.

### Search branches

1. robot motor thermal management and thermal-aware RL;
2. thermal estimation and supervisory derating;
3. uncertainty-aware safety filters and belief-space control;
4. hidden-parameter and online dynamics inference;
5. residual dynamics and physics-plus-learning control;
6. action-dependent non-stationarity and actuator degradation;
7. long-horizon health-aware decision making and maintenance.

The retained set contains 28 Level-1 papers and 19 Level-2 conceptual nodes.
Many additional discovery hits were screened out as generic safe RL, unrelated
thermal control, or applications without robot dynamics. The counts describe
this bounded audit, not bibliometric completeness.

### Final-refresh additions

The final refresh repeated all seven direct branches and added backward
citation checks from the closest thermal, damage-benchmark, reset-free, and
uncertainty-filter papers. Newly retained direct nodes were OopsieVerse,
Gameplay Filters, the unified safety-filter review, continuing SAC without
embodiment resets, CRONOS, SF-RSSM, and degradation/maintenance POMDP work.
Searches through 31 August 2026 did not return a paper combining selective
task resets, hidden action-driven actuator health with dynamics feedback, and
fresh-lifetime evaluation of transferred versus target-calibrated uncertainty
margins.

## Level-1 closest work

Risk labels refer to overlap with the v12 paper, not paper quality.

| Work | What it already establishes | Remaining distinction of v12 | Risk |
|---|---|---|---|
| [Balaji et al., *OopsieVerse: A Safety Benchmark with Damage-Aware Simulation for Robot Manipulation* (RSS 2026)](https://arxiv.org/abs/2606.31993) | Accumulating robot/object health from mechanical, thermal, and fluid damage; damage-aware POMDP, 32 manipulation tasks, RL/IL/VLA evaluation, MuJoCo and Omniverse implementations, and sim-to-real safety evidence | DamageSim explicitly keeps original task dynamics intact and exposes health through observation/reward/termination; LifePhyBench hides actuator health, feeds it back into later dynamics, selectively preserves it across task resets, and studies belief-margin transfer | **Critical** |
| [Qian et al., *Learning Thermal-Aware Locomotion Policies for an Electrically-Actuated Quadruped Robot* (2026)](https://arxiv.org/abs/2603.01631) | Whole-body motor thermal model inside RL, temperature-aware reward, simulation and Unitree A1 deployment; operation extended from about 7 to more than 26 minutes | v12 hides thermal state, studies belief uncertainty and deployment shifts, freezes the low-level policy, and uses selective task/health resets; it has no hardware or calibrated thermal model | **Critical** |
| [Wan et al., *Learning to Balance Motor Thermal Safety and Quadrupedal Locomotion Performance with Residual Policy* (2026)](https://arxiv.org/abs/2605.27046) | Frozen nominal policy plus thermal-aware residual action policy; simulation and hardware trade-off between locomotion and overheating | v12 residual estimates hidden thermal dynamics rather than adding motor commands; the independently attributable residual benefit nevertheless failed at the deployed margin | **Critical** |
| [Farrahi and Mahmood, *Learning Without Time-Based Embodiment Resets in Soft-Actor Critic* (CoLLAs 2025)](https://arxiv.org/abs/2512.06252) | Continuing SAC, explicit separation of episode termination and embodiment reset, a modified MuJoCo Reacher without time-based resets, and a real-robot continuing task | LifePhyBench resets task/arm state while preserving only hidden thermal state; its question is selective reset and health inference rather than exploration without embodiment reset | **High** |
| [Wu et al., *CRONOS: Benchmarking Multi-Task Robotic Manipulation for Reset-Free RL* (2026)](https://embodiedai-ntu.github.io/cronos/) | Long-horizon, shared-scene multi-task manipulation under a finite reset budget | LifePhyBench deliberately resets task state but preserves internal health, and evaluates a frozen controller rather than online reset-free adaptation | **Medium--high** |
| [Nguyen et al., *Gameplay Filters: Robust Zero-Shot Safety through Adversarial Imagination* (CoRL 2024; proceedings published 2025)](https://proceedings.mlr.press/v270/nguyen25a.html) | Predictive full-order safety filtering, adversarial OOD robustness, two quadruped hardware platforms, and zero-shot deployment | LifePhyBench is a low-dimensional hidden-health diagnostic with empirical utility and no formal or full-order safety filter | **High** |
| [Hsu, Hu, and Fisac, *The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems* (2024)](https://doi.org/10.1146/annurev-control-071723-102940) | Unifies modular runtime-assurance and safety-filter architectures across model-based and data-driven control | The supervisor/filter decomposition is established; only the selective-reset health experiment and empirical calibration-transfer result remain distinctive | **Medium--high** |
| [Kawaharazuka et al., *Estimation and Control of Motor Core Temperature with Online Learning of Thermal Model Parameters* (2020)](https://doi.org/10.1109/LRA.2020.2990889) | Online estimation of hidden motor-core temperature, online thermal-parameter adaptation, anomaly detection, and thermal output limiting on a humanoid | v12 evaluates learned belief uncertainty and OOD utility, but cannot claim first hidden-temperature estimation or first thermal-model adaptation | **High** |
| [Sabelhaus et al., *Safe Supervisory Control of Soft Robot Actuators* (2024)](https://doi.org/10.1089/soro.2022.0131) | A modular supervisor dynamically saturates arbitrary nominal-controller inputs to prevent thermal-actuator overheating, with proofs and hardware | v12 uses a simulated, partially observed, action-driven thermal state and evaluates uncertainty margins; it provides no formal safety guarantee | **High** |
| [Kwon et al., *Adaptive Shielding for Safe Reinforcement Learning under Hidden-Parameter Dynamics Shifts* (2025/2026)](https://arxiv.org/abs/2506.11033) | Online hidden-parameter inference, uncertainty-aware action filtering, OOD dynamics shifts, conformal prediction, and probabilistic guarantees | v12 has a within-lifetime evolving thermal health state and selective resets rather than fixed hidden parameters, but its architectural claim is substantially narrower | **Critical** |
| [Liu et al., *Action-Conditioned Risk Gating for Safety-Critical Control under Partial Observability* (2026)](https://arxiv.org/abs/2605.14246) | Uses a finite-history proxy and action-conditioned risk predictor to gate optimistic versus conservative control under partial observability | v12's gate concerns a persistent thermal state and frozen low-level modes, but lightweight risk gating is not novel | **Critical** |
| [Seo et al., *Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures* (CoRL 2025)](https://proceedings.mlr.press/v305/seo25a.html) | Uncertainty-aware safety filtering for high-dimensional learned latent dynamics under OOD observations | v12 is a transparent low-dimensional thermal diagnostic with paired lifetime inference and reward decomposition | **High** |
| [Vahs, Pek, and Tumova, *Belief Control Barrier Functions for Risk-Aware Control* (2023)](https://doi.org/10.1109/LRA.2023.3330662) | Risk-aware safety control directly over probabilistic state beliefs under sensing and motion uncertainty | v12 uses a heuristic uncertainty margin and empirical trip utility, not a belief-space safety certificate | **High** |
| [Vahs and Tumova, *Risk-aware Control for Robots with Non-Gaussian Belief Spaces* (2023)](https://arxiv.org/abs/2309.12857) | Safe-set control using particle-filter beliefs with probabilistic guarantees in simulation and hardware | v12 focuses on action-driven persistent thermal state and task boundaries, without comparable guarantees | **High** |
| [Vahs, Verhagen, and Tumova, *Safety-critical Control Under Partial Observability* (2026)](https://arxiv.org/abs/2603.10572) | A layered belief-space architecture separates goal reaching, information gathering, and conformal safety filtering with finite-horizon guarantees | v12 also separates low-level control from belief supervision but lacks active information gathering and certificates | **High** |
| [Hu, *Permissive Safety Through Trusted Inference* (2026)](https://arxiv.org/abs/2606.02562) | Certifies neural belief-space safety filters while explicitly accounting for runtime-inference error via conformal prediction | v12 evaluates empirical OOD margins without a verified inference region or high-probability certificate | **High** |
| [Feng et al., *Adaptive Shielding via Parametric Safety Proofs* (OOPSLA 2025)](https://doi.org/10.1145/3720450) | Adaptive, increasingly permissive shields with runtime inference and end-to-end probabilistic guarantees | v12 supplies an empirical learned-belief supervisor, not proof-carrying adaptive shielding | **High** |
| [Liu et al., *Robust Regression for Safe Exploration in Control* (L4DC 2020)](https://proceedings.mlr.press/v120/liu20a.html) | Learns residual robot dynamics with uncertainty bounds under covariate shift for safety certification | v12's physics-plus-recurrent-residual model is not novel by itself, and matched-margin residual attribution was inconclusive | **High** |
| [Bonzanini and Mesbah, *Learning-based Stochastic MPC with State-Dependent Uncertainty* (L4DC 2020)](https://proceedings.mlr.press/v120/bonzanini20a.html) | Corrects nominal dynamics with state-dependent uncertainty and enforces chance constraints | v12 studies selective resets and lifetime utility using a much simpler binary supervisor | **Medium–high** |
| [Zhao et al., *Uncertainty-Aware Implicit Safe Set Algorithm* (L4DC 2023)](https://proceedings.mlr.press/v211/zhao23a.html) | Safeguards a nominal RL policy using learned dynamics, uncertainty bounds, and safe-set projection | v12 thermal trips and persistence are distinct diagnostics, but nominal-policy safeguarding is established prior art | **High** |
| [Pfrommer et al., *Safe RL with Chance-constrained MPC* (L4DC 2022)](https://proceedings.mlr.press/v168/pfrommer22a.html) | Couples a nominal RL policy with a chance-constrained safety guide | v12 has hidden persistent degradation and empirical OOD testing, but no theoretical safety result | **Medium** |
| [Murillo-Gonzalez and Liu, *Situationally-aware Dynamics Learning* (IJRR 2026)](https://doi.org/10.1177/02783649261431863) | Online hidden-state representation, probabilistic dynamics segmentation, and adaptation in simulation and real robots | v12 uses a known slow-state structure and controlled thermal shifts rather than general situation discovery | **High** |
| [Chandak et al., *Off-Policy Evaluation for Action-Dependent Non-Stationary Environments* (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3bf80b34f731313b8292f4578e820c90-Abstract-Conference.html) | Formalizes active, passive, and hybrid non-stationarity caused by past interaction | v12 gives an embodied thermal diagnostic and control study; action-dependent non-stationarity itself is not new | **High** |
| [Wu et al., *Adaptive Control Strategy for Quadruped Robots in Actuator Degradation Scenarios* (2023)](https://arxiv.org/abs/2312.17606) | Teacher-student adaptation to actuator degradation/faults on Unitree A1 | degradation is abrupt/fault-oriented rather than slowly action-generated thermal health, but actuator-degradation adaptation is established | **Medium–high** |
| [La and Kaigom, *Deep Learning for Model-Free Prediction of Thermal States of Robot Joint Motors* (2025)](https://arxiv.org/abs/2509.12739) | Data-driven prediction of robot-joint motor thermal states without a detailed parameterized thermal model | v12 couples a physics belief and residual to downstream lifetime decisions, but thermal prediction itself is prior art | **Medium–high** |
| [Sootla et al., *Saute RL* (ICML 2022)](https://proceedings.mlr.press/v162/sootla22a.html) | Converts almost-sure safety budgets into state augmentation for RL | v12's thermal health state and trip cost are application-specific and partially observed; state augmentation for safety is not new | **Medium** |
| [Wang et al., *Beyond Single-Speed Reasoning: Coordinating Fast and Slow Dynamics for Efficient World Modeling* (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39825) | Separates fast and slow latent dynamics in a dual-branch recurrent state-space model | LifePhyBench's reset semantics and action-to-health causal state are experimental structure, not a claim to first multi-timescale model | **Medium** |
| [Morato et al., *Deep Reinforcement Learning Driven Inspection and Maintenance Planning under Incomplete Information and Constraints* (2020/2022)](https://arxiv.org/abs/2007.01380) | Constrained degradation POMDPs, uncertain health, inspection, and long-horizon maintenance decisions | The domain is infrastructure maintenance rather than embodied actuator control, but partially observed degradation-aware decision making is established | **Medium** |

## Level-2 conceptual ancestry

These works were reached through references of the closest Level-1 papers and
define the boundaries of claims that might otherwise appear novel.

| Conceptual node | Relevance to v12 |
|---|---|
| [Ames et al., *Control Barrier Functions: Theory and Applications* (2019)](https://doi.org/10.1109/ECC.2019.8796030) | Establishes the safety-filter/constraint formalism used by thermal-aware and belief-space controllers. |
| [Johannink et al., *Residual Reinforcement Learning for Robot Control* (2018/2019)](https://arxiv.org/abs/1812.03201) | Establishes additive learned correction on top of conventional robot controllers. |
| [Doshi-Velez and Konidaris, *Hidden Parameter MDPs* (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5466173/) | Establishes latent task/dynamics parameter inference across related MDPs. |
| [Perez et al., *Generalized Hidden Parameter MDPs* (AAAI 2020)](https://doi.org/10.1609/aaai.v34i04.5989) | Extends hidden-parameter dynamics for rapid transferable model-based RL. |
| [Kumar et al., *Rapid Motor Adaptation* (RSS 2021)](https://roboticsproceedings.org/rss17/p011.html) | Establishes history-based online adaptation to hidden physical conditions in legged robots. |
| [Yu et al., *Preparing for the Unknown: Learning a Universal Policy with Online System Identification* (2017)](https://faculty.cc.gatech.edu/~turk/paper_pages/2017_learning_universal_policy/index.html) | Establishes explicit online system identification feeding a universal robot policy. |
| [Cosner et al., *Robust Safety under Stochastic Uncertainty with Discrete-Time CBFs* (RSS 2023)](https://roboticsproceedings.org/rss19/p084.html) | Establishes finite-time safety reasoning under stochastic uncertainty. |
| [Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning* (2017)](https://arxiv.org/abs/1710.06542) | Establishes privileged training information with restricted deployment observations. |
| [Rudin et al., *Learning to Walk in Minutes* (CoRL 2022)](https://proceedings.mlr.press/v164/rudin22a.html) | Provides the massively parallel locomotion training basis used by later thermal-aware quadruped studies. |
| [Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain* (2020)](https://doi.org/10.1126/scirobotics.abc5986) | Establishes recurrent/history-based robust locomotion under uncertain terrain and dynamics. |
| [Srinivas et al., *Gaussian Process Optimization in the Bandit Setting* (2010)](https://proceedings.mlr.press/v5/srinivas09a.html) | A key source for calibrated model-confidence bounds used by uncertainty-aware safe control. |
| [Sevinchan et al., *A Review on Thermal Management Methods for Robots* (2018)](https://doi.org/10.1016/j.applthermaleng.2018.04.132) | Shows that robot thermal management and its hardware/software split predate thermal-aware RL. |
| [Venkataraman et al., *Fundamentals of a Motor Thermal Model and Its Applications in Motor Protection* (2005)](https://doi.org/10.1109/CPRE.2005.1430428) | Establishes lumped first-order motor heating/protection models underlying later robot thermal work. |
| [Lin et al., *Temperature Distribution Prediction of the Quadruped Robot Based on Lumped-Parameter Thermal Networks* (2025)](https://doi.org/10.13973/j.cnki.robot.240208) | Supplies the coupled whole-body thermal network used by the closest 2026 quadruped studies. |
| [Nakamura, Peters, and Bajcsy, *Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis* (RSS 2025)](https://doi.org/10.15607/RSS.2025.XXI.113) | Establishes latent safety filtering for complex learned failure sets before the uncertainty-aware extension. |
| [Thananjeyan et al., *Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones* (IEEE RA-L 2021)](https://doi.org/10.1109/LRA.2021.3070252) | Establishes a learned safety critic and recovery policy around a task policy. |
| [Eysenbach et al., *Leave No Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning* (2017/2018)](https://arxiv.org/abs/1711.06782) | Establishes learned reset policies and autonomous recovery for continuing robot learning. |
| [Berkenkamp et al., *Safe Model-Based Reinforcement Learning with Stability Guarantees* (NeurIPS 2017)](https://proceedings.neurips.cc/paper/2017/hash/766ebcd59621e305170616ba3d3dac32-Abstract.html) | Establishes uncertainty-aware safe model learning with formal stability structure. |
| [Ji et al., *Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark* (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html) | Establishes broad standardized safe-RL environments and cost-based evaluation; physical health feedback remains outside its scope. |

## Component-by-component novelty verdict

| Candidate claim | Verdict | Reason |
|---|---|---|
| First robot-learning treatment of motor temperature | Prohibited | Qian 2026 and Wan 2026 provide simulation and hardware results. |
| First learned policy balancing task performance and thermal safety | Prohibited | Both 2026 quadruped papers optimize this trade-off directly. |
| First thermal supervisor over a nominal controller | Prohibited | Sabelhaus 2024 and Wan 2026 already provide supervisory/residual architectures. |
| First hidden motor-temperature inference for control | Prohibited | Kawaharazuka 2020 estimates hidden motor-core temperature and controls output limits. |
| First uncertainty-aware belief safety filter | Prohibited | Belief CBFs, latent safety filters, and adaptive shielding precede v12. |
| First physics-plus-residual safe controller | Prohibited | Residual-dynamics and uncertainty-aware safe-control work predates v12. |
| First action-driven persistent degradation problem | Prohibited | Action-dependent non-stationarity and health-aware control already formalize the concept. |
| First damage/health-augmented robot-manipulation benchmark | Prohibited | OopsieVerse already accumulates mechanical, thermal, and fluid health and uses it in robot learning. |
| First study of robot learning without ordinary resets | Prohibited | Leave No Trace, continuing SAC, and CRONOS directly study reset-free robot learning. |
| New residual model that independently improves deployed performance | Unsupported | The fresh-seed residual effect at the deployed `z=1.5` margin had a confidence interval crossing zero. |
| Controlled evidence about uncertainty under hidden persistent thermal state and selective resets | Defensible but narrow | The exact combination was not found; the two-task study identifies the uncertainty effect while retaining magnitude-versus-direction cautions. |
| Zero-shot margin transfer followed by development-only morphology calibration | Defensible empirical contribution | The inherited Reacher test remained a safety-gate failure, while a separately frozen calibration restored the gate on new seeds without detectable mean-utility loss. No exact duplicate was found. |
| General thermal-management method for robots | Unsupported | Two MuJoCo tasks share one phenomenological thermal law and provide no hardware calibration. |

## Revised defensible contribution

The paper may state:

> We present a controlled two-task simulation study of fixed low-level
> controllers operating under action-driven, persistent, and partially observed
> thermal dynamics across task resets. Under prespecified shifts, uncertainty-
> aware belief supervision improved mean risk-sensitive lifetime utility. A
> margin inherited from Pusher retained a positive mean effect on Reacher but
> missed its frozen safety tolerance; development-only Reacher calibration
> restored that tolerance on new lifetimes without detectable mean-utility
> loss. Fresh-lifetime ablations attribute the robust benefit primarily to the
> uncertainty margin rather than the learned residual alone.

This wording is a contribution statement, not a priority claim. The manuscript
should use “we did not identify” rather than “the first” whenever describing the
combination.

## What the current study adds—and what it does not

### Adds

- a selective-reset protocol where task state resets but hidden thermal state
  persists;
- action-generated rather than clock-scheduled thermal evolution;
- explicit separation of physics belief, learned residual, and uncertainty
  margin on fresh lifetime seeds;
- paired lifetime-level statistics under held-out sensor, cooling, and combined
  shifts;
- component-level accounting showing that avoided trips, not immediate task
  return, produce the positive utility;
- a break-even analysis for the thermal-trip penalty;
- a second task/morphology with disjoint development, inherited-margin test,
  and post-confirmatory calibrated-margin test seeds;
- direct evidence that a useful uncertainty margin is not automatically a
  transferable safety setting, plus a frozen target-development calibration
  procedure.

### Does not add

- a calibrated motor thermal model;
- physical robot evidence;
- a formal safety guarantee or calibrated coverage guarantee;
- independent evidence that the residual causes the deployed gain;
- a new general-purpose RL or belief-space control algorithm;
- cross-simulator, cross-degradation-law, or hardware generalization;
- a claim that most individual lifetimes improve (the Reacher exact sign test
  for the calibrated margin was not significant).

## TMLR novelty risk

Current risk: **medium--high**.

The completed second morphology and independently frozen margin extension
materially improve the empirical package. The closest 2026 papers still have
much stronger physical validity, while the closest safety-filter papers have
much stronger algorithmic and theoretical contributions. OopsieVerse also
precludes a broad damage-aware benchmark claim. The package is most credible as
a diagnostic empirical study of uncertainty-margin mechanism and calibration,
not as a new safety-filter algorithm or thermal simulator.

For a competitive TMLR submission, the writing must now do the following:

1. lead with the two-task empirical finding and calibration-transfer failure,
   not algorithmic novelty;
2. contrast OopsieVerse explicitly: accumulated damage signal versus hidden
   health that changes later actuator dynamics and survives selective resets;
3. contrast Qian/Wan explicitly: observed/calibrated whole-body motor
   temperature and hardware versus hidden phenomenological state and controlled
   inference/OOD ablations;
4. report uncertainty coverage collapse, mean-versus-sign evidence, the
   inherited 2.3% gate failure, and the post-confirmatory status prominently;
5. label the privileged threshold policy as a current-state baseline rather
   than a planning oracle and bound the recurrent baseline claim to the tested
   architecture and budget;
6. position the residual as an evaluated component and the `z=2.0` result as
   task-specific calibration, never a universal margin.

## Citation placement map

### Introduction

- motivate real robot thermal limits with Qian 2026, Wan 2026, and
  Kawaharazuka 2020, while citing OopsieVerse for accumulated physical damage;
- introduce action-dependent non-stationarity with Chandak 2022;
- state the remaining gap as hidden persistent health plus uncertainty-aware
  lifetime supervision, without a “first” claim.

### Related work

Use four explicit subsections:

1. thermal-aware robot learning and thermal supervision;
2. hidden dynamics and online system identification;
3. uncertainty-aware safety filters and belief-space control;
4. persistent degradation, selective/reset-free interaction, and maintenance.

### Methods and limitations

- contrast the heuristic margin with formal chance constraints, belief CBFs,
  and conformal adaptive shields;
- explain that the thermal law is phenomenological and substantially less
  realistic than the identified whole-body models used in the 2026 quadruped
  studies;
- identify selective resets and the factorial attribution protocol as the main
  experimental design distinctions.

## Search maintenance rule

This checkpoint is current through 2026-08-31, not through a future submission
date. Repeat Level-1 searches immediately before actual submission and record
that date. Follow new references only when they change a claim boundary or
introduce a stronger baseline. The “through submission date” checklist item
must remain open until that last refresh.
