# TMLR Submission Checklist

Policy checkpoint: 2026-08-03.

Official sources:

- [Submission and editorial policies](https://www.jmlr.org/tmlr/editorial-policies.html)
- [Author guidelines](https://www.jmlr.org/tmlr/author-guide.html)
- [Acceptance criteria](https://www.jmlr.org/tmlr/acceptance-criteria.html)

## Scope strategy

TMLR explicitly considers new learning-task formalizations, evaluation methods,
algorithms with sound validation, and empirical studies that reveal strengths or
weaknesses of learning systems.  The paper therefore leads with the controlled
latent-state problem and causal factorial study.  Robot manipulation is an
important validation tier, not the entire contribution.

## Evidence checklist

- [ ] Claims are frozen before final test execution.
- [ ] Closest-work ledger is updated through the submission date.
- [ ] Persistence, endogeneity, and observability are independently controlled.
- [ ] At least one analytic/diagnostic environment has a planning oracle.
- [ ] At least one general-control and one embodied-control family agree on the
      main conclusion.
- [ ] Strong recurrent, meta-RL/system-ID, and online-learning baselines are
      included in the correct deployment track.
- [ ] Hyperparameter and interaction budgets are matched and reported.
- [ ] Independent training seeds, hierarchical confidence intervals, effect
      sizes, and corrected primary tests are reported.
- [ ] Negative runs and failed seeds remain in the analysis.
- [ ] Code, generator, seed manifest, raw lifetime rows, and environment
      fingerprints reproduce every table and figure.
- [ ] Limitations distinguish phenomenological simulation from calibrated
      physical lifetime prediction.

### v10 hierarchical-thermal checkpoint (2026-08-27)

This subsection tracks only the completed single-diagnostic study; a checked
item here does not close the broader paper-level item above.

- [x] Physical design, disjoint seed sets, primary estimand, test, interval,
      and success rule were frozen in a local manifest before held-out training.
- [x] A transparent planning oracle and exhaustive task-reactive rule search
      were retained from calibration.
- [x] Twenty independent training seeds, paired seed effects, a seed-bootstrap
      interval, and both specified seed-level tests were reported.
- [x] The negative seed, zero-effect seeds, and collapsed task policies were
      retained and disclosed.
- [x] The learned-comparator claim was separated from the stronger unsupported
      claim that lifetime memory is representationally necessary.
- [ ] The plan was externally preregistered or cryptographically bound to a
      complete public commit before training.
- [ ] A learned task-reactive baseline reaches the transparent reactive anchor
      and is beaten on new held-out seeds.
- [ ] Clock/counting, latent-health inference, and action-history adaptation are
      independently controlled under stochastic variation.
- [ ] The main conclusion replicates across tasks, mechanisms, or morphologies.
- [x] The full raw-data and environment-fingerprint release reproduces the
      publication tables and figures from a clean checkout.

### v12.2/v12.3 belief-supervision checkpoint (2026-08-30)

This is the authoritative checkpoint for the current scoped paper claim.

- [x] The v12.2 checkpoint, policy specifications, OOD conditions, 100 held-out
      seeds, estimands, and success criteria were locally frozen before final
      evaluation.
- [x] The v12.3 residual-by-uncertainty factorial used 100 disjoint fresh seeds
      and retained every lifetime.
- [x] Raw lifetime rows, protocol hashes, result files, the selected residual
      checkpoint, environment metadata, and paper-artifact hashes are preserved
      in the private Git evidence snapshot.
- [x] Reward decomposition separates base-task return, throughput bonus, and
      thermal-trip penalty without treating tasks as independent samples.
- [x] Post-hoc reward sensitivity is explicitly labeled as fixed-trajectory
      accounting rather than policy retraining.
- [x] The final claim attributes the robust benefit primarily to uncertainty-
      aware trip avoidance, not to an independently established residual gain.
- [x] A depth-two literature audit was refreshed on 2026-08-30 and explicitly
      screened direct thermal-RL, thermal-supervision, hidden-dynamics,
      belief-safety, and uncertainty-aware shielding work.
- [x] The low-level controller checkpoint and a clean-checkout reproduction
      script are included in a review-compatible artifact package.
- [x] The scoped result is replicated on Reacher as a second task/morphology;
      the inherited safety-gate failure and post-confirmatory task-specific
      calibration are reported separately.
- [ ] The final novelty ledger is updated through the submission date.

### Reacher replication checkpoint (2026-08-31)

- [x] Development, inherited-margin confirmatory, and post-confirmatory fresh
      test seeds are mutually separated by role.
- [x] The inherited `z=1.5` primary result remains a failure because 2.3%
      exceeded the frozen 2.0% safety boundary.
- [x] The follow-up cutoff/margin grid, buffered selection rule, fresh seeds,
      tests, and criteria were locally frozen before extension evaluation.
- [x] Development-only calibration selected cutoff 0.06 and `z=2.0`; all fresh
      extension criteria passed with a 1.6% maximum trip rate.
- [x] Magnitude-sensitive and direction-count evidence are both disclosed: the
      positive mean effect coexists with only 52/100 positive lifetime effects.
- [x] The recurrent baseline result is bounded to the tested RecurrentPPO
      architecture, training budget, and model-selection procedure.
- [x] The final anonymous artifact is reproduced from a clean checkout; the
      tracked-tree archive is 19,092,310 bytes.
- [x] The exact CPU-only dependency lock installs in a new Python 3.11 venv;
      `pip check`, both model loads, artifact regeneration, and all 154 tests
      pass without inheriting the development `PYTHONPATH`.
- [ ] Reconfirm the review-system upload limit immediately before submission.

Current result and claim boundary:
[`PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md`](PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md)
and
[`REACHER_REPLICATION_FINAL_RESULTS.md`](REACHER_REPLICATION_FINAL_RESULTS.md).
Clean-checkout evidence:
[`CLEAN_CHECKOUT_REPRODUCTION_2026-08-31.md`](CLEAN_CHECKOUT_REPRODUCTION_2026-08-31.md).

## Submission-policy checklist

- [ ] Manuscript uses the official TMLR LaTeX template without format changes.
- [ ] Manuscript and supplement are anonymized for double-blind review.
- [ ] Every author's OpenReview profile is complete and active.
- [ ] Submission quota is available for every author.
- [ ] Work is not under review at another archival venue.
- [ ] No text, figure, or result is reused from an archival conference paper;
      TMLR does not accept ordinary conference extensions.
- [ ] Any prior workshop is explicitly non-archival; an arXiv preprint is allowed.
- [ ] Important results are in the main paper because appendix review is
      optional.
- [ ] Broader-impact discussion covers resource use, unsafe degradation
      policies, and limits of simulated health signals.
- [ ] Anonymous code/supplement is below the current 100 MB upload limit or is
      linked using a review-compatible archival mechanism.

## Writing guardrails

Use:

> selective-reset controlled latent-state POMDP with endogenous physical
> degradation

Avoid:

- “the world is intrinsically non-Markovian”;
- “continual learning” for a frozen recurrent policy;
- “physically realistic wear” without calibration;
- universal or first-of-kind claims not supported by the novelty ledger.
