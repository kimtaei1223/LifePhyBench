# TMLR reviewer-risk audit — 2026-09-01

## Verdict

No fatal internal contradiction was found after the final claim-to-evidence,
literature, privacy, and extracted-supplement audits. The manuscript is
defensible as a narrow empirical study. Its main acceptance risk is external
interest and breadth, not an unreported failed experiment or an unsupported
numerical headline.

TMLR does not require a new algorithm or state of the art, but it does require
accurate evidence and a clearly communicated lesson of interest to part of its
audience. The paper now leads with that lesson: persistent hidden state should
be evaluated over lifetimes, and expected utility, a point-rate decision, and
calibration evidence should remain separate claims.

## Likely reviewer objections

| Risk | Severity | Current evidence or mitigation | Residual exposure |
|---|---|---|---|
| The algorithmic components are established prior art | High | Related Work explicitly cites thermal control, safety filters, hidden-dynamics adaptation, and calibrated conservatism; the contribution is framed as empirical design and evidence | A reviewer may still judge two tasks too narrow to be interesting |
| The thermal model is not physically realistic | High | The paper repeatedly calls it phenomenological, supplies the exact law, and excludes hardware validity | No conclusion about a real motor or robot is available |
| Two tasks share one law and one canonical initialization each | High | This is now stated as a limitation | Goal, object, simulator, and degradation-law generalization remain untested |
| The 2% threshold is statistically fragile or arbitrary | High | It is labeled a frozen empirical point rule, never a confidence bound or safety certificate; original failures remain visible | Near-boundary pass/fail outcomes remain sample-sensitive by design |
| Utility gains depend on the chosen trip cost | Medium--high | Reward decomposition and break-even analysis are reported; the manuscript says the gain is not free | No universal reward-preference result is claimed |
| Calibration appears post hoc | High | The original Reacher failure is preserved; development, selection, and new-test seeds are disjoint; inherited also passes on the extension sample | Calibration necessity and superiority are explicitly unsupported |
| The learned residual is presented as the mechanism | Resolved | Fresh factorial attribution gives `+0.103`, CI `[-0.004,0.215]`; the paper attributes the robust effect to uncertainty instead | The hybrid policy may pass even though its incremental residual effect is unresolved |
| The recurrent baseline is under-described or overgeneralized | Resolved for reporting | Training/selection budgets and its `-9.068` effect with 23.6% maximum trip rate are now in the body | It remains only one RecurrentPPO architecture and budget |
| The mean result hides heterogeneous lifetimes | Medium | Positive counts and exact sign tests are reported; selected Reacher has only 52/100 positive | The claim is expected utility, not majority benefit |
| Reproduction depends on local Git history or GPU retraining | Resolved | The anonymous ZIP reproduces without Git metadata, loads retained models on CPU, regenerates artifacts, and passes 154 tests | Full historical training is expensive but not needed to verify reported analyses |
| Environmental cost is omitted | Low--medium | The paper discloses that consistent power telemetry was not retained and avoids inventing a retrospective estimate | A precise energy total cannot be supplied |

## Reviewer-style questions and bounded answers

1. **What is learned from this paper if the filter is not new?**  A
   deployment/evaluation lesson: ignoring lifetime structure can hide persistent
   dynamics, while a mean benefit and a hard point-rate decision can disagree.
2. **Did calibration solve cross-task transfer?**  No. It selected a passing
   policy on new seeds, but the inherited setting also passed there and had no
   detectably different mean utility.
3. **Is the method safe?**  No formal or hardware safety claim is made. The
   2% rule is an empirical protocol decision.
4. **Is the residual useful?**  Its independent benefit at the deployed margin
   was not established. The supported mechanism is the uncertainty margin.
5. **Does every lifetime improve?**  No. The main estimand is a paired mean;
   the paper reports sign counts to prevent a majority interpretation.
6. **Does the result generalize to real robots?**  Not from the current data.
   The study isolates one simulated mechanism on Pusher and Reacher.

## Decision on further experiments

No additional GPU experiment is required to support the already frozen narrow
claim. A third simulator, a different degradation law, varied task
initializations, or hardware telemetry would support a broader paper, but
adding one selectively after observing these results would create a new
experimental phase rather than repair a logical error in the current study.

## Submission recommendation

Proceed once the human-only OpenReview, authorship, originality, quota,
conflict, funding, and license gates in
`TMLR_SUBMISSION_READINESS_2026-09-01.md` are complete. Do not broaden the title,
abstract, or conclusion during final upload without rerunning the evidence and
novelty audits.
