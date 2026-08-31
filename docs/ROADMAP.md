# Research Roadmap

Progress is governed by evidence gates rather than calendar dates.

## Milestone 0 — executable definition

Status: complete.

- dependency-free latent-health diagnostic;
- explicit episode and lifetime resets;
- endogenous action dose;
- deterministic tests and first reset-bias observation.

## Milestone 1 — MuJoCo phenomenon pilot

Status: learned-policy subgate complete for the frozen v10 hierarchical thermal
diagnostic; broader phenomenon validation remains in progress.

- Pusher actuator-gain wear wrapper;
- hidden/visible health and endogenous/exogenous controls; **implemented**
- persistence and worker-isolation regression tests; **implemented**
- scripted factorial pilot over degradation scales; **first pass complete**
- discretized dynamic-programming lifetime oracle on the diagnostic tier; **implemented**
- CPU clock-shortcut and action-history identifiability audit; **implemented**

Exit condition: persistence × endogeneity creates a measurable difficulty that
cannot be solved by episode index alone and remains solvable by a planning oracle.

CPU semantic checkpoint: complete for the diagnostic and scripted MuJoCo
pilots. The v10 held-out Pusher thermal-commitment study closes the narrow
learned-policy requirement for that diagnostic: lifetime-state RecurrentPPO
beat its task-reset arm over 20 independent training seeds and the static
zero-dose control effect was zero. A strong task-reactive audit also showed
that the learned task arm was underoptimized, so the result is not a general
proof of memory necessity.

## Milestone 2 — general benchmark

Status: in progress. The v10 result covers one Pusher thermal diagnostic only;
the cross-task and cross-mechanism exit condition is not complete.

- add recoverable thermal mechanism; **implemented in Pusher pilot**
- add joint aging mechanism; **implemented in Pusher pilot**
- add at least two MuJoCo morphologies/tasks;
- freeze law-family generators and train/validation/test manifests;
- implement lifetime-level logging, paired seeds, and analysis pipeline;
- validate physical-response direction for every mutated parameter.

Exit condition: the main phenomenon appears across mechanisms without relying on
one reward coefficient or simulator artifact.

CPU semantic checkpoint: Pusher-v5 and Reacher-v5, three implemented health
channels, and power/threshold/shock law families are covered. Geometric contact
friction in Pusher was rejected after a directionality audit. The exit condition
still requires learned-policy agreement beyond the single v10 Pusher thermal
study.

## Milestone 3 — baseline map

Status: in progress.

- reactive PPO development baseline; **implemented and smoke-validated**
- episode-RNN and lifetime-persistent RNN; **implemented and CPU smoke-validated**
- frame stack and Transformer memory;
- RL² and system-identification/RMA-style adaptation;
- time-only and action-blind shortcuts;
- privileged current-health and full planning oracles;
- separate frozen-adaptation and online-update tracks.

Exit condition: failures are localized to inference, planning, generalization,
or online learning rather than reported as an undifferentiated score gap.

## Milestone 4 — proposed method

Status: scoped method and mechanism study complete on the Pusher thermal
diagnostic. The implemented method combines a physics belief, a learned
recurrent residual, and an uncertainty-aware high/low-power supervisor over a
fixed low-level policy.

Exit condition: preregistered improvement over the strongest parameter-matched
baseline on held-out law families with a practically meaningful effect.

Scoped outcome: the complete v12.2 supervisor passed its frozen held-out
comparison. The fresh v12.3 factorial did not support independent residual
attribution at matched uncertainty; the uncertainty margin produced the larger
robust effect. This closes the single-task method study but not the broader
cross-task exit condition. See
[`PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md`](PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md).

## Milestone 5 — embodied and cross-engine validation

- implement representative WearPick/AgingDrawer/FatigueInsert variants;
- run state observations broadly and RGB-D selectively;
- reproduce at least one central causal conclusion in ManiSkill or Isaac Lab;
- audit asset licenses and redistribution terms.

## Milestone 6 — final study and TMLR manuscript

Status: final scoped experiments and evidence snapshot complete; manuscript,
novelty audit, and review artifact packaging remain in progress.

- freeze configurations and seeds;
- execute independent final runs;
- hierarchical bootstrap, paired effects, Holm correction, survival/Pareto
  analyses where applicable;
- release code, raw lifetime data, and reproducible figures;
- write the paper around what the experiments establish, including negative
  results.

Current result and limitations:
[`PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md`](PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md).
