# Slide 6 — Why a random forest, and what we compared

This slide answers **which trust scorer we use, how we chose it, and why we did not use a neural net, XGBoost, or Lasso.**

The subtitle is the design choice in one line: we score **42 engineered tabular features**, not raw code tokens. That points to classical calibrated classifiers, not a deep model over source text.

---

## What problem the scorer is solving

Each AI-written solution becomes a row of numbers (visible test, sibling agreement, lint, structure, token confidence, and so on). The scorer outputs a probability \(p\) = “how likely is this solution trustworthy?”

Trustworthy means: passes every hidden test and has no Bandit medium/high finding. The scorer never sees those hidden-test outcomes as features; it only predicts them.

That probability later becomes APPROVED / NEEDS HUMAN REVIEW / REJECTED via separate cutoffs. This slide is only about choosing and calibrating the probability model.

---

## Tabular signals vs raw code tokens

| Approach | What the model sees | Fit for TrustGate |
| --- | --- | --- |
| **Tabular features (what we did)** | 42 numbers we designed (pass/fail bits, counts, complexities, log-probs) | Natural for trees and logistic regression; easy to calibrate and explain |
| **Raw code tokens (e.g. a neural net over source)** | The program text itself | Needs more data and tuning; harder to justify for a gate; wrong default when the signal is already engineered |

**Inductive bias** means: what kind of patterns a model family is built to learn easily. Tree ensembles and linear models are biased toward combinations of a modest number of numeric features. Neural nets are biased toward high-dimensional raw inputs (images, text, token sequences). With 42 columns, a neural net’s bias is a poor match: you pay for capacity and opacity without a clear gain.

---

## Baselines vs candidates

We always keep two **baselines** so a fancy model cannot look good by beating nothing:

1. **Majority class**  
   Always predict the common label (“trustworthy”).  
   **CV AUROC 0.500** — by definition, random ranking. This is the floor.

2. **Visible-test-only logistic**  
   A logistic regression on **one** bit: did it pass the example in the prompt?  
   **CV AUROC 0.857** — today’s naive gate. Anything we deploy must beat this in ranking quality.

**Candidates** (real models on all 42 features):

| Method | Idea | Train CV AUROC |
| --- | --- | --- |
| Logistic regression | Linear weighted sum of features (scaled, class-balanced) | 0.924 |
| Hist. gradient boosting | Boosted trees (sklearn’s histogram GB; same family as XGBoost-style boosting) | 0.928 |
| **Random forest (400 trees)** | Bagged trees, class-balanced | **0.940** ← selected |

---

## Grouped 5-fold cross-validation

**Cross-validation (CV)** means: split the training data into folds, train on some folds, score on the held-out fold, rotate, then average. It estimates how well a method will generalize **without touching the final test problems**.

**Grouped** means the group is the **MBPP problem (`task_id`)**. All 40 solutions for one problem stay in the same fold. Otherwise the model could memorize “problem 483 is always silent-fail” from one sample and cheat on another sample of the same problem.

We used **5 folds** on the **train** split only (580 problems). Calibration and test problems are not used to pick the model.

**Selection rule:** among non-baselines, pick the highest **grouped-CV AUROC**. That was the random forest (0.940).

---

## What AUROC means here

**AUROC** (Area Under the ROC Curve) asks: if you pick one trustworthy solution and one untrustworthy solution at random, how often does the model give the trustworthy one a **higher** score?

- **0.5** — coin flip  
- **1.0** — perfect ranking  
- **Our RF (calibrated) on held-out test: 0.941**  
- Visible-test baseline on test: **0.856**

AUROC does **not** use the approve/reject cutoffs. It only measures ranking. That is what we optimize when choosing the model, because a three-way gate needs a well-ordered score: high → approve, low → reject, middle → human.

Precision / recall / F1 at a 0.5 cutoff are reported elsewhere; they are not the selection criterion on this slide.

---

## Held-out test AUROC (right panel)

After selection, every method is fit on the full train split and scored once on the **194 unseen test problems**:

| Model | Test AUROC |
| --- | --- |
| Majority | 0.500 |
| Visible test | 0.856 |
| Logistic regression | 0.920 |
| Hist. gradient boosting | 0.931 |
| **RF (calibrated)** | **0.941** |

The useful gap is RF vs the visible-test baseline: the whole confidence story of the project rests on ranking silent failures and easy passes better than “trust the example.”

---

## Isotonic calibration

A raw random forest vote fraction is often **not** a true probability (e.g. “0.9” might really mean ~70% on some datasets).

**Isotonic calibration** learns a flexible increasing curve from the model’s raw score to a probability, using cross-validation **inside the training problems** (`CalibratedClassifierCV`). After calibration:

\[
p \approx P(\text{trustworthy} \mid \text{features})
\]

So cutoffs like “approve if \(p \ge 0.872\)” are meaningful. We use isotonic when there is enough data (here, tens of thousands of rows); with tiny data the pipeline would fall back to Platt scaling (sigmoid).

**“RF (isotonic, deployed)”** on the appendix table means: this calibrated forest is what produces \(p\) in production.

---

## Why not NN / XGBoost / Lasso?

### Neural net (NN)

- Wrong inductive bias for 42 tabular features.  
- Needs more data, architecture search, and regularization to match a tuned forest.  
- Harder to calibrate and explain when the product is a **trust gate** that must justify approve/reject.  
- A net over **raw code** would still be blind to hidden tests at score time — same information bottleneck, more complexity.

### XGBoost

- Same *idea* as histogram gradient boosting: boosted trees on tabular data.  
- We already had **HistGradientBoosting** in the grid (CV AUROC 0.928). It lost to RF (0.940).  
- Adding a second boosting library would not change the conclusion; RF won on the metric we chose.

### Lasso

- Lasso is logistic/linear regression with **L1** penalty (sparse weights).  
- We already trained **logistic regression** on all features (CV AUROC 0.924), which trailed RF by about **1.6 AUROC points**.  
- A Lasso variant is unlikely to leapfrog a forest that is already using non-linear interactions among the strong signals (visible test × sibling agreement, etc.).

---

## How this slide connects to the rest of the deck

1. **Features (slide 5)** define the 42 columns.  
2. **This slide** picks the function that maps columns → \(p\).  
3. **Thresholds slide** turns \(p\) into approve / review / reject for 95% precision.  
4. Calibration is what makes those thresholds interpretable; grouped CV is what keeps model choice honest across problems.

---

## Bottom line

We compared honest baselines and three real tabular learners with **grouped 5-fold CV**. The **random forest** won on AUROC, was **isotonic-calibrated**, and is the deployed scorer (**test AUROC 0.941** vs **0.856** for trusting the visible example). NN, XGBoost, and Lasso were either a poor fit for tabular features, already represented in the grid, or already beaten by logistic regression’s cousin.
