# Titanic — EDA Findings Log

Cleaned-up conclusions from exploring `train.csv` (891 rows). This is the "what did we learn" record — see the Glossary for "what do these terms mean."

---

## Dataset shape
- 891 rows, 12 columns. Test set (418 rows) has the same columns minus `Survived`.
- Missing values: `Age` 177 (19.9%), `Cabin` 687 (77.1%), `Embarked` 2 (0.2%). Everything else complete.

## Sex — strong, robust effect
- Female: 74.2% survived (n=314). Male: 18.9% survived (n=577).
- ~55 point gap, both groups large → high confidence this is real.
- Likely cause: "women and children first" evacuation policy, not biological — worth remembering as historical/procedural, not a general law.

## Pclass — strong, robust, graded effect
- 1st: 63.0% | 2nd: 47.3% | 3rd: 24.2%. Clear downward gradient, large n in all three groups.
- Likely cause: cabin/deck proximity to lifeboats, boarding priority.

## Embarked — real but likely a confound, not independent
- Raw rates: C 55.4% (n=168) | Q 39.0% (n=77) | S 33.7% (n=644).
- Q's group is small → SE ≈ ±5.6 pts, true rate plausibly 28–50%. Treat Q's number cautiously.
- Checked: C and Q have notably higher % female (43.5%, 46.8%) than S (31.5%) — the Sex effect likely explains much of Embarked's apparent pattern. C also skews toward 1st/2nd class (mean Pclass 1.89 vs S's 2.35).
- Conclusion: keep as a candidate feature, but expect it's partly redundant with Sex/Pclass.

## Age — moderate effect, missing-data trap identified
- Mean 29.7, std 14.53 (sample std, N−1). Min 0.42, max 80.0 — std describes typical spread, not a hard bound.
- `Age < 12` (child) comparison: naive code silently classifies all 177 missing-Age rows as "not a child" (NaN < x → False in Pandas). Real risk of quiet misclassification — must use `.notna()` filtering or an explicit Unknown category when this matters for real preprocessing.
- With that caveat: children under 12 survived at 57.4% vs 36.8% for the rest (cutoff=18 gives 54.0% vs 36.1%). Real gap, but built on a small child group (n=68 for cutoff 12).

## SibSp / Parch — individually noisy at the tails
- Both heavily skewed toward 0 (SibSp: 608/891 at 0; Parch: 678/891 at 0), long thin tail of rare larger values.
- Some tail groups (SibSp=5, n=5; SibSp=8, n=7) show exactly 0% survival — likely noise, not a real effect (SE formula breaks down at p=0 with tiny n).

## FamilySize (engineered: SibSp + Parch + 1)
- Rise-then-collapse shape: alone (30.4%, n=537) < family of 2 (55.3%) / 3 (57.8%) / 4 (72.4%, n=29) > 5+ (mostly crashing, but n=6–22 per group — treat as noisy, not confirmed).
- Best-supported takeaway: traveling completely alone is worse than a small family. Large-family collapse is suggestive but not statistically solid yet.

## Title (engineered: extracted from Name)
- Common: Mr (517), Miss (182), Mrs (125), Master (40). Long tail of rare titles (Dr, Rev, Major, etc.), each n≤7 — candidate to bucket as "Rare."
- Confirms hypothesis: Master mean age 4.6 (max 12) — near-perfect young-boy marker. Miss (21.8) younger than Mrs (35.9) on average, though Miss's range extends to 63.
- Practical use: much better basis for imputing missing Age than a single flat average (e.g., missing-age Masters should get a young estimate, not ~29.7 overall mean).

## Cabin / HasCabin (engineered: Cabin present or not)
- Raw: HasCabin=True 66.7% survival (n=204) vs False 30.0% (n=687) — large, robust gap.
- Confound check: HasCabin=True group has mean Pclass 1.20 — overwhelmingly 1st class. Strongly suggests HasCabin is largely a proxy for class.
- Split by class: within Pclass 1 (n=40 vs 176, both reasonably sized), gap persists — 47.5% vs 66.5%. Real signal beyond class, at least for 1st class.
- Within Pclass 2 and 3, the HasCabin=True group is too small (n=16, n=12) to trust either way.
- Cabin deck letters (A–G) follow a real historical Titanic deck layout; extracting deck letter is possible but likely highly redundant with Pclass — not yet tested directly.
- Tried: recovering missing cabins via shared Ticket number. Assumption holds fairly well (87.3% of tickets with a known cabin agree across passengers), but only 11 of 687 missing cabins are actually recoverable this way — not worth implementing for the payoff.

## Ticket
- 891 rows, only 681 unique values → 210 rows share a ticket with someone else (group/family travel).
- Prefix codes (PC, STON/O2, A/5, etc.) look inconsistent, not a clean encoding scheme like Cabin's deck letters — real meaning not verified, avoid assuming a story here.
- Most solid extractable idea: group size via ticket-sharing (similar to FamilySize but not identical, since ticket-mates aren't always blood relatives).

## Stage 5 — First baseline model (Logistic Regression)
- Features: IsFemale, Pclass, FamilySize, Title_Mr/Miss/Mrs/Master/Rare (8 total). Target: Survived.
- Stratified 80/20 train/val split, random_state=42 → **82.7% validation accuracy** (baseline "everyone dies" = 61.62%).
- Stability check across random_state=1,7,42,55: accuracy ranged **81.5%–84.3%**. Conclusion: true performance is a range, not a single number — a ~2.8 point difference between runs is likely just split noise, not a real model difference.
- Model's learned weights (coef_) matched EDA findings directionally: IsFemale +1.57, Pclass −0.99, Title_Mr −1.39 — confirms earlier manual analysis was sound.

## Stage 6 — Preprocessing (Age, Cabin, Embarked, Fare)
- **Age**: filled 177 missing values using **median** Age per Title group (Master→3.5, Miss→21.0, Mr→30.0, Mrs→35.0, Rare→44.5), not a flat overall average — chosen because Title strongly predicts age range (see Stage 3 findings) and a flat mean would misrepresent groups like Master.
- **Cabin**: kept as `HasCabin` only (built in Stage 3); dropped raw Cabin. Rejected filling the 77% missing (would be mostly fabricated data) and rejected extracting deck letter (same missing-data wall, unproven added value over HasCabin/Pclass).
- **Embarked**: 2 missing values filled via **mode imputation** (filled as "S"). Checked whether the 2 missing rows shared a ticket with a known-Embarked passenger first (they shared ticket 113572 with each other, but neither had a known value) — mode imputation was the only real option, and with only 2/891 affected, risk of distortion is negligible.
- **Fare**: added as a feature. Checked first whether it's a confound: mean fare by Pclass (1st $84.15, 2nd $20.66, 3rd $13.68) shows Fare is very tightly bound to Pclass — likely largely redundant, but included so the model's own weight can tell us if it adds anything independent, rather than pre-judging it.
- **Ticket**: still excluded — raw values too inconsistent to trust an extracted pattern without more work; only the ticket-sharing idea has real support, and hasn't been implemented as a feature yet.

## Stage 6 — Second model (13 features) + scaling experiment
- Features: IsFemale, Pclass, Age, Fare, FamilySize, HasCabin, Title_* (5), Embarked_* (3) = 13 total.
- Unscaled logistic regression (random_state=55): **84.36% accuracy**, but solver hit `max_iter=100` and printed a convergence warning ("scale the data") before finishing.
- Scaled version (StandardScaler, fit on X_train only, applied to X_val via `.transform()` to avoid leaking validation distribution into the scaler): convergence warning disappeared, but **accuracy identical (84.36%)** — scaling fixed the optimizer's convergence issue but didn't change the final decision boundary this model landed on.
- **Important caution**: 84.36% is *within* the 81.5–84.3% noise range already observed for the simpler 8-feature model. Cannot yet claim the added features (Age, Fare, HasCabin, Embarked) meaningfully improved performance — the apparent gain is smaller than known split-to-split variance. Real cross-validation (Stage 9) is needed to check this properly rather than comparing single numbers.

## Stage 6 — Rigorous comparison: 8 features vs 13 features
- Compared the two models properly (not just one split each) by looping over 5 different random_states (1, 7, 42, 55, 99), retraining fresh each time, and taking the mean + standard deviation of accuracy — rather than trusting a single number.
- 8-feature model: scores [83.80%, 81.56%, 82.68%, 84.36%, 84.92%] → **mean 83.46%, stdev 1.35 pts**.
- 13-feature model: scores [82.68%, 81.01%, 82.68%, 84.36%, 86.03%] → **mean 83.35%, stdev 1.91 pts**.
- Conclusion: adding Age, Fare, HasCabin, and Embarked did **not** meaningfully improve accuracy — the 13-feature mean is actually very slightly lower, and its spread across runs is wider, not tighter. A 0.11-point difference in means is not a real result given this much run-to-run noise.
- Possible explanations (not yet tested): the new features may be too redundant with IsFemale/Pclass/Title to add independent signal; logistic regression's linear weights can't capture non-straight-line patterns we found by hand (e.g. FamilySize's rise-then-collapse shape); or more features without more independent information can slightly increase noise. Open question for Stage 7/8, not resolved here.

## Working hypothesis heading into Stage 4
Sex and Pclass appear strongest (largest effect sizes + most robust sample sizes). Title, HasCabin, and FamilySize are promising engineered candidates. Embarked is likely partly redundant with Sex/Pclass. This is our own manual, one-at-a-time analysis — not yet confirmed by an actual model, which may reveal redundancies or interactions this approach can't.
