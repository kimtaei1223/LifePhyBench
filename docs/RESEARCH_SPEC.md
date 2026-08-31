# LifePhyBench Research Specification (v0)

## 1. Research question

How should an agent learn when its actions change not only the current task
state, but also a hidden physical state that persists across nominal episode
resets and changes future transition dynamics?

The central object is a **lifetime**, an ordered sequence of episodes sharing a
physical system.  An episode reset is a task reset, not a factory replacement of
the robot.

### Terminology guardrail

If `(x, z)` is treated as the complete environment state and its transition law
is fixed, the process is a stationary latent-state POMDP even though the marginal
dynamics observed through `x` drift over time.  We therefore do **not** claim
mathematical non-stationarity merely because wear changes.  Non-stationarity is
reserved for protocols in which the transition law itself changes outside the
specified state.

Likewise, a frozen network whose recurrent state persists is performing online
inference or adaptation, not continual parameter learning.  The benchmark keeps
two deployment tracks separate:

1. **Frozen adaptation:** weights are fixed; only memory or belief state changes.
2. **Online learning:** parameters and/or replay state update during deployment,
   with compute, memory, and data budgets reported.

## 2. Formal model

For lifetime `l`, episode `e`, and within-episode step `t`, define

- task state: `x[l,e,t]`;
- persistent physical state: `z[l,e,t]`;
- action: `a[l,e,t]`;
- observation: `o[l,e,t]`;
- task context: `c[l,e]`.

The controlled process is

```text
x' ~ P_x(. | x, z, a, c)
z' ~ P_z(. | z, x, a, c)
o  ~ P_o(. | x, z, c)
r  = R(x, z, a, c)
```

At an episode boundary, `x` and `c` may reset, while `z` is transformed by a
boundary kernel that can preserve damage, permit partial recovery, or introduce
maintenance.  A full lifetime reset samples a new physical system and resets
`z`.  Because `z` is normally hidden, this is a partially observed controlled
process.  Because the action affects `z`, this is not adequately represented by
exogenous domain randomization alone.

More explicitly, the episode boundary is

```text
x[l,e+1,0] ~ rho(. | c[l,e+1])
z[l,e+1,0] ~ M(. | z[l,e,H], maintenance[l,e])
```

and the defining endogenous condition is that the slow-state transition differs
for at least some feasible actions:

```text
P_z(. | z, x, a) != P_z(. | z, x, a')
```

We provisionally call this a **selective-reset controlled latent-state POMDP**.
The name is descriptive and not itself a novelty claim.

## 3. Intended contributions

These are targets to test, not claims already established.

1. A precise task family for action-dependent cross-episode physical dynamics.
2. A simulator-independent lifetime API and leakage-resistant evaluation
   protocol.
3. A benchmark spanning synthetic diagnostics and physically grounded robot
   tasks.
4. An adaptation or learning method with separate episode-scale and
   lifetime-scale state, evaluated in its correctly labelled deployment track.
5. Empirical insights identifying when persistent memory helps, fails, or is
   unnecessary.

## 4. Falsifiable hypotheses

- **H1 — persistence × endogeneity:** the gain from slow cross-episode state is
  larger in persistent–endogenous conditions than in matched reset or exogenous
  controls by a preregistered practical margin.  An RNN advantage in every
  condition does not support this hypothesis.
- **H2 — action-aware inference:** at matched capacity, a slow-state estimator
  using observation and action history has lower health-estimation error and
  adaptation regret than observation-only and clock-only estimators specifically
  under endogenous degradation.
- **H3 — functional OOD:** a reset-aware, action-conditioned health model improves
  lifetime performance over the strongest recurrent/system-identification
  baseline on held-out degradation-law families, not only held-out random seeds.

The reset-bias gap remains a benchmark validity diagnostic rather than a claim of
algorithmic superiority.  A fast/slow architecture becomes a paper contribution
only if evidence shows that simpler persistent memory is insufficient.

Each hypothesis must be rejected if its preregistered primary comparison is not
statistically supported and practically meaningful.  The paper must report
negative results.

### Evidence checkpoint (2026-08-27)

The v10 hierarchical Pusher thermal study supplies partial evidence for a
narrow learned-policy contrast: lifetime-state RecurrentPPO beat its matched
task-reset training arm in the dynamic cell and not in a static zero-dose
control. It does not close H1 because it lacks matched reset-health and
dose-matched exogenous controls, and the task-reset learner failed to recover a
representable task-reactive strategy. It does not test H2 because the policy's
health-estimation error was not measured and deterministic lifetime counting is
an alternative explanation. H3 remains untested. See
[`HIERARCHICAL_THERMAL_CONFIRMATORY_V10.md`](HIERARCHICAL_THERMAL_CONFIRMATORY_V10.md).

## 5. Benchmark tiers

### Tier 0: diagnostic systems

Low-cost systems with known latent state, including actuator wear, friction
drift, recovery, shocks, and coupled failure modes.  These isolate observability,
causal coupling, and memory horizon.  They support oracle and identifiability
analysis but are not sufficient evidence alone.

### Tier 1: general continuous control

MuJoCo tasks with persistent actuator strength, damping, friction, or backlash.
The same lifetime protocol is applied across multiple morphologies and objectives.

### Tier 2: robot manipulation

ManiSkill tasks representing cumulative contact wear, joint aging, and actuator
fatigue.  State-based experiments establish scale; a smaller RGB-D experiment
checks whether the conclusion survives realistic partial observation.

## 6. Information regimes

All results must label one of the following regimes:

1. **Hidden:** physical state and parameters are unavailable.
2. **Proprioceptive:** only signals plausible on a robot are available.
3. **System-ID probe:** the agent may perform actions to infer physical state.
4. **State oracle:** true latent physical state is provided.  A myopic state
   oracle is not automatically a lifetime-performance upper bound because it may
   consume the system aggressively; a planning oracle must optimize the same
   lifetime objective as learned methods.

Oracle information must never enter non-oracle training, normalization, early
stopping, or model selection.

## 7. Non-negotiable evaluation rules

- Split complete lifetimes, robot instances, and degradation laws.
- Never shuffle steps or episodes across a lifetime for evaluation.
- Select hyperparameters using validation laws only.
- Keep final test seeds and law parameters sealed until the analysis is frozen.
- Compare methods on paired lifetime seeds and task sequences.
- Report uncertainty over lifetimes, not over correlated transitions.
- Report task performance and damage separately; do not hide trade-offs in one
  reward scalar.
- Include a nominal no-degradation control and a per-episode-reset control.
- Record simulator, driver, package, asset, and configuration versions.

## 8. Scope boundary

The initial paper will not claim a calibrated model of a specific real robot's
material fatigue.  Simulated degradation laws are controlled scientific test
families.  Physical realism will be supported through ranges and qualitative
mechanisms from primary literature, sensitivity analysis, and cross-engine
validation.  Claims about real-world deployment require hardware evidence and
are outside the current no-hardware scope.
