# CPU Semantic-Validation Checkpoint

Date: 2026-08-03

This checkpoint records software and environment evidence produced without a
GPU. It is not a learned-policy result and must not appear as evidence of method
superiority in a paper.

## Completed CPU work

| Requirement | Evidence |
|---|---|
| Selective-reset semantics | `reset()` preserves wear and aging; thermal state persists with explicit partial cooling; `reset_lifetime()` clears all channels. |
| Three health mechanisms | actuator gain wear, recoverable thermal derating, and actuated-joint damping aging. |
| Two task families | Pusher-v5 and Reacher-v5 smoke test. |
| Endogeneity controls | endogenous action and exogenous-clock versions for each health channel. |
| Functional law families | power (development), threshold and stochastic shock (held-out semantic pilots). |
| Reproducible analysis | one JSONL record per lifetime plus paired percentile bootstrap. |

## Pilot commands

```bash
python scripts/run_multitask_health_factorial.py \
  --output outputs/cpu_semantic_pilot_v1.jsonl

python scripts/analyze_paired_lifetimes.py \
  outputs/cpu_semantic_pilot_v1.jsonl \
  --metric final_health --condition-key condition \
  --reference episode_physics_reset --treatment persistent_lifetime \
  --pairing-keys environment_id mechanism degradation_mode controller seed \
  --where environment_id=Pusher-v5 mechanism=wear \
  degradation_mode=endogenous_action controller=high_constant
```

The 288-lifetime scripted pilot showed the expected persistence distinction. In
Pusher with high constant endogenous action, final wear and joint aging were
`0.10` after an episode-physics reset versus `0.30` under a persistent
lifetime. Thermal load was `0.391` versus `0.720`; it persists while cooling,
so it need not grow linearly.

## Explicit non-claims

- Rates are diagnostic, not calibrated material models.
- The Pusher geometric-contact-friction mutation was rejected after a failed
  runtime directionality audit.
- No learned agent, generalization claim, or proposed method has yet been
  evaluated.

## GPU boundary

The next required work is learned-policy training: reactive and memory-based
PPO/SAC, system-identification baselines, and finally the proposed reset-aware
method. This checkpoint deliberately stops before that work because it requires
the shared RTX 4090.
