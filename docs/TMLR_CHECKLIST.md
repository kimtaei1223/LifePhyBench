# TMLR submission checklist

Policy checkpoint: 2026-09-01.

Official sources:

- [Submission and editorial policies](https://www.jmlr.org/tmlr/editorial-policies.html)
- [Author guidelines](https://www.jmlr.org/tmlr/author-guide.html)
- [Acceptance criteria](https://www.jmlr.org/tmlr/acceptance-criteria.html)

## Decision rule

TMLR's primary questions are whether the paper's claims have accurate,
convincing evidence and whether some of its audience would find the clearly
communicated findings useful. Method novelty or a new state of the art is not
itself required. The submission is therefore gated by evidence for the narrow
two-task claim below, not by every benchmark extension considered earlier.

Status labels:

- **Complete**: evidence exists and was reproduced.
- **Open**: required before submission.
- **Out of scope**: would support a broader paper, but is not required for the
  present claim and must instead appear as a limitation.
- **Historical**: useful provenance from an earlier study, but not evidence for
  the current primary claim unless explicitly reused.

## Frozen paper claim

> Under hidden action-driven thermal dynamics that persist across task resets,
> an uncertainty margin around an explicit belief supervisor can improve mean
> risk-sensitive lifetime utility on two MuJoCo manipulation tasks. Whether a
> near-boundary empirical trip-rate rule passes depends on the complete policy
> and finite test sample. Development-only calibration can select a passing
> policy, but the current data do not establish its necessity or superiority.

This is a claim about expected risk-sensitive utility in a controlled,
phenomenological two-task simulation. It is not a hardware-valid thermal model,
a formal safety guarantee, a majority-lifetime improvement, or a universal
control result.

## Current scoped-paper evidence gate

| Status | Requirement | Evidence or disposition |
|---|---|---|
| **Complete** | Claims and protocols were frozen before each designated final test. | Pusher v12.2/v12.3 and the inherited Reacher test were locally frozen; the later Reacher calibration is preserved as a separately frozen post-confirmatory extension on disjoint seeds. |
| **Complete** | Persistence, endogeneity, and partial observability are explicit in the environment and evaluation. | Task state resets while hidden thermal state persists; thermal change depends on actions; deployed policies receive a noisy sensor rather than exact health. |
| **Complete** | The causal source of the main effect is separated. | The fresh Pusher 2 x 2 residual-by-uncertainty factorial attributes the robust gain primarily to the uncertainty margin, not the learned residual. |
| **Complete** | Cross-task evidence preserves failures instead of rewriting them. | The inherited Reacher margin's 2.3% trip rate remains a failure against the frozen 2.0% boundary; its later calibration result is reported separately. |
| **Complete** | Independent analysis units, fresh seeds, intervals, effect sizes, and prespecified tests are used. | Each 20-task lifetime is one unit; the principal Pusher, inherited Reacher, and calibrated Reacher tests each use 100 fresh lifetimes with paired estimates and bootstrap intervals. |
| **Complete** | Negative and ambiguous evidence is retained. | Residual-attribution failure, Reacher zero-shot gate failure, non-significant exact sign tests, negative recurrent result, and reward-preference dependence are disclosed. |
| **Complete** | Comparator claims are bounded to the actual experiment. | The privileged comparator is described as a current-state threshold baseline, not a planning oracle; the learned comparison is limited to the tested RecurrentPPO family and budget. |
| **Complete** | Artifacts regenerate the reported results. | Protocol hashes, raw lifetime rows, models, tables, figures, manifests, and dependency lock reproduce from a clean Python 3.11 environment; 154 tests pass. |
| **Complete** | Limitations distinguish simulated degradation from physical prediction. | The evidence audit explicitly excludes hardware validity, physical-unit calibration, formal safety, universal reward preferences, and cross-simulator generalization. |
| **Open** | Closest-work and novelty ledger are current on the submission date. | Depth-two audit plus a targeted refresh are current through 2026-09-01; repeat only if the actual submission is later. |
| **Complete** | The manuscript states the contribution as an empirical deployment lesson rather than component novelty. | Abstract, Introduction, Related Work, and Conclusion now exclude component-priority claims and bound calibration, residual, and safety interpretations. |
| **Complete** | All main-paper numbers and claims are traceable to immutable artifacts. | `scripts/audit_tmlr_claims.py` verifies 38 numerical and boundary checks and regenerates `docs/TMLR_CLAIM_EVIDENCE_AUDIT_2026-09-01.md`. |

## Explicitly out of scope for this paper

These are worthwhile extensions, but leaving them undone is not a hidden
submission failure because the paper does not make the corresponding claims.

| Item | Why it is not a current gate | Required manuscript treatment |
|---|---|---|
| A planning oracle or optimal privileged policy | The current privileged threshold policy is diagnostic, not an upper bound. | Do not call it an oracle; state this limitation. |
| A general-control family outside embodied MuJoCo | The claim is deliberately limited to Pusher and Reacher under one simulator and thermal law. | Do not claim general-control or cross-simulator validity. |
| Exhaustive meta-RL, system-identification, and online-adaptation baselines | One recurrent family is diagnostic evidence, not an exhaustive algorithm ranking. | Bound the comparison and identify stronger adaptive baselines as future work. |
| Real-robot thermal validation | No physical calibration or hardware experiment was performed. | Use “phenomenological thermal dynamics,” never “realistic motor wear” or a hardware safety claim. |
| External preregistration | Protocols were locally frozen and hash-bound, but not externally preregistered. | Say “prespecified” or “locally frozen,” never “preregistered.” |
| Universal reward robustness | Recorded-trajectory sensitivity shows the utility advantage depends on trip cost. | Report the break-even analysis and its fixed-trajectory limitation. |

## Completed experiment checkpoints

### Pusher v12.2/v12.3 belief supervision

- [x] Policies, OOD conditions, 100 held-out seeds, estimands, and success
      criteria were frozen before v12.2 evaluation.
- [x] A fresh 2 x 2 residual-by-uncertainty factorial used 100 disjoint seeds
      and retained every lifetime.
- [x] The physics-only uncertainty effect was `+0.965` reward/task with
      bootstrap 95% CI `[0.733, 1.196]`; that treatment's maximum trip rate was
      2.15% and failed the 2% point gate. The full hybrid treatment's margin
      effect was `+0.736`, with a 1.3% maximum trip rate.
- [x] Reward decomposition and fixed-trajectory sensitivity distinguish lower
      immediate task return from avoided thermal-trip cost.
- [x] The residual gain at matched `z=1.5` was not independently established
      and is not presented as the central mechanism.

### Reacher replication and calibration

- [x] Development, inherited-margin confirmation, and calibrated-margin test
      seeds are mutually separated by role.
- [x] Inherited `z=1.5` improved mean utility by `+0.721`, but its 2.3% maximum
      trip rate failed the frozen 2.0% boundary.
- [x] Development-only selection chose cutoff `0.06`, `z=2.0` under a buffered
      rule, then evaluated it once on 100 new lifetimes.
- [x] The calibrated setting improved mean utility by `+0.692`, bootstrap 95%
      CI `[+0.434, +0.962]`, with maximum trip rate 1.6%.
- [x] Only 52/100 calibrated lifetime effects were positive; the exact sign
      test was non-significant. The claim is about the paired mean, not a
      majority of lifetimes.
- [x] The calibrated and inherited policies had no detectable mean-utility
      difference. The inherited policy also reached 1.9% on the same fresh
      extension seeds, so calibration is presented as a feasible selection
      procedure, not as necessary, superior, or uniquely responsible for the
      passing gate.

### Artifact and clean-environment reproduction

- [x] Protocol/result/model hashes and artifact manifests are preserved.
- [x] Pusher and Reacher paper artifacts regenerate hash-identically.
- [x] The exact CPU dependency lock installs in a clean Python 3.11 venv.
- [x] `pip check`, both model loads, 154 tests, and artifact regeneration pass
      without inheriting the development `PYTHONPATH`.
- [x] The anonymous tracked-tree archive is below the current 100 MB supplement
      limit; recheck its final size immediately before submission.

Evidence:

- [`PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md`](PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md)
- [`REACHER_REPLICATION_FINAL_RESULTS.md`](REACHER_REPLICATION_FINAL_RESULTS.md)
- [`INTEGRATED_PUSHER_REACHER_AUDIT.md`](INTEGRATED_PUSHER_REACHER_AUDIT.md)
- [`CLEAN_CHECKOUT_REPRODUCTION_2026-08-31.md`](CLEAN_CHECKOUT_REPRODUCTION_2026-08-31.md)
- [`TMLR_CLAIM_EVIDENCE_AUDIT_2026-09-01.md`](TMLR_CLAIM_EVIDENCE_AUDIT_2026-09-01.md)

## Historical v10 diagnostic study

The v10 hierarchical-thermal experiment remains provenance, not a current
submission gate. It established useful workflow elements—disjoint seeds,
paired lifetime analysis, retained failures, and clean reproduction—but it
does not supply the current Pusher–Reacher primary claim. Its unfinished
aspirations (external preregistration, exhaustive clocks/action histories,
additional mechanisms, and a stronger learned anchor) must not be silently
represented as completed evidence.

## Submission-policy gate

- [x] Use the official TMLR LaTeX style and template without format changes.
- [x] Keep the manuscript and every supplement anonymous during double-blind
      review.
- [ ] Ensure every author has a complete, active OpenReview profile and all
      required conflicts and disclosures are entered.
- [ ] Confirm every author has sufficient annual submission quota.
- [ ] Confirm the work is not under review at another archival venue.
- [ ] Confirm no text, figure, or result is reused from an archival paper or a
      paper simultaneously under review; ordinary conference extensions are
      not accepted.
- [ ] Identify any prior public version only if it is a preprint or explicitly
      non-archival workshop version, and do not link the anonymous submission
      to an identified copy.
- [x] Put every result needed to understand the main claim in the paper body;
      appendix and supplement review are discretionary.
- [x] Include a concise broader-impact discussion of unsafe degradation
      policies, resource use, simulated health signals, and mitigations.
- [x] Keep anonymized supplementary material in PDF or ZIP format and at or
      below the current 100 MB limit. The audited ZIP is 18,616,087 bytes.
- [ ] Confirm that all authors accept the CC BY 4.0 license applying from
      submission onward.
- [ ] Recheck the official policy pages immediately before submission.

Machine-audited submission details and the remaining human-only gates are in
[`TMLR_SUBMISSION_READINESS_2026-09-01.md`](TMLR_SUBMISSION_READINESS_2026-09-01.md).

## Writing guardrails

Use:

> selective-reset latent-state control with endogenous, persistent thermal
> dynamics

Avoid:

- “the world is intrinsically non-Markovian”;
- “continual learning” for a frozen recurrent policy;
- “physically realistic wear” without calibration;
- “oracle” for the privileged current-state threshold baseline;
- “safe” or “safety guarantee” for an empirical trip-rate tolerance;
- universal or first-of-kind claims unsupported by the novelty ledger.
