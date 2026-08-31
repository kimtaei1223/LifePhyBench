# Reacher cross-task replication: final evidence boundary

Status: complete, including the separately frozen post-confirmatory margin
extension (2026-08-31).

## Sequence and protocol integrity

The Reacher study was executed in two evidentially distinct stages.

1. The inherited Pusher design used 100 fresh lifetime seeds (25200--25299)
   and retained the Pusher uncertainty multiplier `z=1.5` without Reacher
   outcome-based retuning. Its protocol SHA-256 is
   `eeea9e261d36b198897346ac5b9b3d3eda90b71936b2e9ba11ea12795f9ae925`.
2. After that primary gate failed, a post-confirmatory extension searched a
   prespecified 3 x 4 cutoff/margin grid using development seeds 25100--25119.
   It selected cutoff `0.06`, `z=2.0` under a 1.5% buffered-safety rule and
   evaluated that pair once on 100 new lifetime seeds (25300--25399). Its
   frozen fresh-test protocol SHA-256 is
   `bed29c0121ddf4b632173a360cd842faefde3e6ce1ac4a345bfb8af273aa049c`.

The two test seed sets are disjoint. The extension preserves the original
failure flag and is not presented as a correction or retroactive primary
success.

## Confirmatory result with inherited margin

For physics belief `z=1.5` minus physics belief `z=0`, averaged within lifetime
over the three target OOD conditions:

- mean reward difference: `+0.7205` reward/task;
- seed-bootstrap 95% CI: `[+0.4745, +0.9785]`;
- paired sign-flip `p = 9.99999e-7`;
- positive lifetime effects: 54 of 100;
- maximum treatment trip rate: `2.3%`.

The magnitude-sensitive reward criteria passed, but the frozen safety limit
was 2.0%. The inherited-margin primary endpoint therefore failed. The exact
sign test was not significant (`p = 0.484`), so this is an expected-utility
effect rather than majority-lifetime improvement.

The hybrid secondary policy passed the safety boundary (maximum trip rate
1.4%) and improved target-OOD reward by `+0.7970`, 95% CI
`[+0.5463, +1.0589]`. Its residual contribution at matched `z=1.5` was not
independently established: `+0.0765`, 95% CI `[-0.0657, +0.2254]`.

The selected monolithic RecurrentPPO baseline was strong in-domain but its
target-OOD effect relative to physics `z=0` was `-9.0678`, 95% CI
`[-9.5075, -8.6296]`; all 100 paired effects were negative. This is evidence
about the tested architecture and training/selection budget, not every possible
end-to-end recurrent policy.

## Post-confirmatory calibrated-margin result

Seven of twelve development candidates met the 1.5% buffered-safety rule. The
prespecified utility-maximizing rule selected cutoff `0.06`, `z=2.0`.

On the new extension seed set, calibrated `z=2.0` minus physics `z=0` yielded:

- mean reward difference: `+0.6920` reward/task;
- seed-bootstrap 95% CI: `[+0.4345, +0.9618]`;
- paired sign-flip `p = 2.999997e-6`;
- positive lifetime effects: 52 of 100;
- exact sign-test `p = 0.764`;
- maximum selected-policy trip rate: `1.6%`.

All frozen extension criteria passed. The calibrated policy's mean reward was
not distinguishable from the inherited policy: `-0.0081`, 95% CI
`[-0.1210, +0.1078]`. The supported extension claim is therefore that
morphology-specific development-only calibration restored the specified
safety margin without a detectable expected-utility loss. It is not evidence
that `z=2.0` is universally optimal or improves reward over `z=1.5`.

## Integrated claim

Pusher and Reacher agree that an uncertainty margin around an explicit belief
supervisor can improve mean risk-sensitive lifetime utility under hidden,
persistent thermal dynamics and prespecified shifts. The Reacher sequence adds
an important deployment result: a margin transferred across tasks need not
satisfy a fixed safety tolerance, while a margin calibrated solely on target-
task development lifetimes can recover that tolerance on a new test set.

This supports a practical workflow--retain explicit persistent-state belief,
calibrate the conservatism margin for the deployment morphology, then freeze
and test once--within this benchmark. It does not establish real-robot thermal
validity, formal safety, probability calibration under OOD shift, or universal
dominance across reward preferences and model-free architectures.

## Publication artifacts

`scripts/render_reacher_replication_artifacts.py` validates protocol hashes,
the failure/success boundary, selection identity, row counts, and disjoint test
seeds before generating tables, a summary figure, bilingual result text, and a
hash manifest under `paper_artifacts/reacher_replication`.
