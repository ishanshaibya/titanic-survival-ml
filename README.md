# Titanic Survival Prediction — ML Learning Project

Kaggle competition: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

## Why

This is my first real Machine Learning project. I'm using it to learn the ML
workflow from the ground up — not just to produce a model, but to understand
*why* each step in that workflow exists and what decisions go into it.

## What

The goal is to predict whether a given Titanic passenger survived, based on
information like their sex, ticket class, age, and family size. This is a
binary classification problem, evaluated on accuracy (per Kaggle's rules).

Beyond the prediction itself, the actual point of this project is the
process: exploring the data properly before modeling, reasoning about which
features are trustworthy versus noisy, building a simple baseline before
anything complex, and being honest about what the evidence does and doesn't
support at each step.

## Where — project structure

```
titanic-survival-ml/
├── docs/
│   ├── glossary.md         # ML/stats concepts explained as they came up
│   └── eda_findings.md     # Data exploration findings and reasoning
├── notebooks/
│   └── explore.ipynb       # Main EDA + modeling notebook
├── data/                   # Raw Kaggle CSVs (not tracked — see .gitignore)
└── .gitignore
```

## Current status

- **EDA**: done for every column — `Sex`, `Pclass`, `Age`, `Fare`, `SibSp`,
  `Parch`, `Embarked`, `Cabin`, `Name`, `Ticket`. Checked for confounds
  (e.g. `Embarked` and `HasCabin` both turned out to be partly redundant
  with `Sex`/`Pclass`), and flagged small-sample groups as unreliable rather
  than trusting every raw percentage. See `docs/eda_findings.md`.
- **Feature engineering**: `FamilySize` (from `SibSp`+`Parch`), `Title`
  (extracted from `Name`, grouped into Mr/Miss/Mrs/Master/Rare), `IsFemale`,
  `HasCabin`.
- **Baseline model**: Logistic Regression on 8 features
  (`IsFemale`, `Pclass`, `FamilySize`, `Title_*`) — roughly **81.5–84.3%**
  validation accuracy across different random splits (baseline of predicting
  "everyone dies" is 61.62%).
- **Preprocessing**: filled missing `Age` using median-by-`Title`, filled
  missing `Embarked` via mode imputation, dropped raw `Cabin` in favor of
  `HasCabin`, tested feature scaling (fixed a solver convergence warning,
  didn't change accuracy).
- **13-feature model**: added `Age`, `Fare`, `HasCabin`, `Embarked_*`. A
  controlled comparison (same 5 random splits, both models) found **no real
  improvement** over the 8-feature baseline — the new features didn't add
  measurable signal, at least not for logistic regression. Recorded as a
  genuine finding, not a failure.

## Next steps

- Try other model types (e.g. decision trees) that might capture non-linear
  patterns the current model can't (e.g. `FamilySize`'s rise-then-collapse
  shape).
- Proper cross-validation instead of manual multi-split comparisons.
- Revisit `Ticket` as a possible feature.
- Generate a first real Kaggle submission.

## Note

This is a learning project — the code and decisions here reflect where I
was in understanding ML at the time, not necessarily best practice. Findings
and reasoning are logged in `docs/` as the project progresses.
