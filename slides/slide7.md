# Slide 7 — Why these approve / reject cutoffs

This slide answers **how a probability \(p\) becomes APPROVED, REJECTED, or NEEDS HUMAN REVIEW**, and why the first cutoffs failed while the deployed ones hit the 95% target.

The model (calibrated random forest) only outputs \(p \approx P(\text{trustworthy})\). Cutoffs are a separate policy layer on top of that score.

---

## The target: 95% precision on automatic decisions

Two automatic actions, each with a precision bar:

| Action | Meaning of precision |
| --- | --- |
| **Approve** | Of solutions the gate approves, ≥95% are truly trustworthy |
| **Reject** | Of solutions the gate rejects, ≥95% are truly untrustworthy |

Everything between the two cutoffs goes to a **person**. The project would rather send more code to review than quietly approve bad code or reject good code.

Cutoffs are chosen on the **calibration** problems (194 problems, 7,760 rows). The **test** problems are only for the final report. They are never used to pick \(t_{\text{high}}\) or \(t_{\text{low}}\).

---

## What \(p\), approve, and reject mean in practice

After calibration:

\[
\text{decision}(p) =
\begin{cases}
\text{APPROVED} & \text{if } p \ge t_{\text{high}} \\
\text{REJECTED} & \text{if } p \le t_{\text{low}} \\
\text{NEEDS HUMAN REVIEW} & \text{otherwise}
\end{cases}
\]

- Raising \(t_{\text{high}}\) → fewer approvals, higher approval precision, more review.  
- Lowering \(t_{\text{low}}\) → fewer rejections, higher rejection precision, more review.  
- The “most lenient” cutoff that still hits 95% is the one that automates as much as possible without breaking the bar.

---

## Left panel: what failed first (pooled calibration)

### Method

On all calibration rows **pooled together**, find the most lenient scores that still show 95% precision:

- Approve if \(p \ge 0.79\)  
- Reject if \(p \le 0.47\)

“Pooled” means: pretend every calibration solution is one big bag. Ignore that they come from different problems with different difficulty.

### Outcome on unseen test problems

Approval precision fell to **92.6%** — under the 95% target.

### Why it missed

1. **Problem heterogeneity inside calibration**  
   At that same approve cutoff, the **worst fifth** of calibration problems was already only about **92%** precise. Averaging across easy and hard problems hid that weak fold. The test set then looked like that weak fold.

2. **Base-rate shift**  
   | Split | Trustworthy rate |
   | --- | --- |
   | Calibration | 72.7% |
   | Test | 64.0% |

   Calibration was **easier**. A cutoff tuned on the easier mix is too generous when problems get harder. More silent failures appear; approval precision drops.

**Lesson:** a single pooled average is not a safe estimate of precision on new problems.

---

## Right panel: deployed rule (grouped CV)

### Method

Use **grouped cross-validation on the calibration problems** (groups = `task_id`):

1. Propose a candidate cutoff.  
2. For each held-out fold of problems, measure precision of the automatic decisions.  
3. **Keep the cutoff only if every fold** still clears 95% (and has enough support that a handful of rows cannot fake the rate).  
4. Among cutoffs that survive, take the **most lenient** one (automate as much as possible).

That is stricter than “the average fold is fine.” One bad group of problems can veto a cutoff.

### Deployed cutoffs

| Rule | Cutoff | Color on the slide |
| --- | --- | --- |
| Approve | \(p \ge 0.872\) | Green |
| Reject | \(p \le 0.090\) | Red |
| Else | Human review (**26.8%** of test) | Amber |

Rejection became much more conservative than the pooled rule (\(0.090\) vs \(0.47\)): higher reject bars did not keep 95% precision in every calibration fold.

### Results on the held-out test set

| | Pooled (failed) | Grouped CV (deployed) |
| --- | --- | --- |
| Approve if | \(p \ge 0.79\) | \(p \ge 0.872\) |
| Reject if | \(p \le 0.47\) | \(p \le 0.090\) |
| Approved share | ~60% | **52.5%** |
| Review share | ~9% | **26.8%** |
| Approval precision | 92.6% | **95.1%** |
| Rejection precision | ~92% | **95.8%** |

The cost of hitting the target is **more human review**. That is intentional policy, not a model failure.

---

## What “grouped” means here (vs model selection)

| Stage | Where grouped CV is used | Purpose |
| --- | --- | --- |
| Model selection (slide 6) | **Train** problems | Pick RF vs logistic vs boosting |
| Threshold selection (this slide) | **Calibration** problems | Pick \(t_{\text{high}}\) and \(t_{\text{low}}\) |

Same idea — never split solutions from one problem across fit and score — different data slice and different decision.

---

## Conformal prediction (the alternative on the slide)

**Split conformal** (\(\alpha = 0.1\)) builds a prediction set from calibration nonconformity scores instead of two precision-targeted cutoffs.

On this test set it:

- Automates **more** (less review, ~6.4%)  
- But only reaches **90.7%** approval precision  

So it is **reported as an alternative**, not deployed. The product requirement was 95% precision on automatic actions, not maximum automation.

---

## How this connects to the rest of the system

1. Features → score (slides 5–6).  
2. **This slide** → policy cutoffs for 95% precision.  
3. Human review band absorbs uncertainty and base-rate shift.  
4. Risk–coverage curves elsewhere show the same trade-off as a continuous plot: approve only the most confident fraction → higher precision, lower coverage.

---

## Bottom line

Pooled calibration looked like 95% in-sample and **missed on new problems (92.6%)** because averages hid hard problem groups and the test mix was harder. The deployed rule requires **every held-out group of calibration problems** to clear 95%, which raised the approve bar to **0.872**, dropped the reject bar to **0.090**, put **26.8%** of test code in review, and delivered **95.1% / 95.8%** precision on the unseen test set. Conformal automates more but does not meet the bar.
