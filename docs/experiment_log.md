# Experiment Log

Tracks every model/feature combination actually tried, with real results —
nothing here is estimated or assumed.

**Defaults used throughout, unless a row says otherwise:** stratified
train/validation split (`stratify=y`), 80/20 split. "Accuracy" below is the
**mean across 5 different random_states (1, 7, 42, 55, 99)**, not a single
run, since we found single-split accuracy varies by 2-3 points just from
random chance. Range and standard deviation are included so a difference
between rows can be judged against normal run-to-run noise, not just eyeballed.

Baseline to beat (predicting "everyone dies" for every passenger): **61.62%**.

| # | Model | Features | Accuracy (mean / range / stdev) | Notes | Next step |
|---|---|---|---|---|---|
| 1 | Logistic Regression | IsFemale, Pclass, FamilySize, Title_* (Mr/Miss/Mrs/Master/Rare, one-hot) — 8 total | 83.46% / 81.56–84.92% / ±1.35 pts | First baseline. Learned weights matched EDA directionally (IsFemale +1.57, Pclass −0.99, Title_Mr −1.39). Beats the 61.62% baseline by ~22 points. | Add more features (Age, Fare, Cabin, Embarked) and see if accuracy improves. |
| 2 | Logistic Regression | Adds Age (median-by-Title imputed), Fare, HasCabin, Embarked_* (one-hot) — 13 total | 83.35% / 81.01–86.03% / ±1.91 pts | **No real improvement** over row 1 — mean is actually slightly lower, spread is wider. Tested feature scaling separately (StandardScaler): fixed a solver convergence warning, did not change accuracy. | Logistic regression may not be able to represent FamilySize's non-linear (rise-then-collapse) survival pattern found in EDA. Try a model type that can — decision tree next. |
