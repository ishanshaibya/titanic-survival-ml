# Titanic ML Project — Glossary

A running reference of concepts as we build the project. Updated as we go — check back anytime.

---

## Machine Learning (ML)
- **Intuition:** Instead of writing explicit rules, we show a computer many examples and let it work out the patterns itself.
- **Technical:** A method of building systems that improve at a task by learning patterns from data, rather than following explicit instructions.
- **Titanic:** We show the computer past passengers with known outcomes, and let it learn what factors relate to survival.

## Dataset
- **Intuition:** A big table of examples.
- **Technical:** A structured collection of data, organized as rows (examples) and columns (attributes).
- **Titanic:** `train.csv` and `test.csv` — each row is one passenger.

## Feature
- **Intuition:** A clue we know *before* trying to predict the outcome.
- **Technical:** An input variable used by the model to make a prediction.
- **Titanic:** `Pclass`, `Sex`, `Age`, `Fare`, etc. Note: not every column is automatically a usable feature (e.g. `PassengerId` is just a row identifier, not a meaningful clue).

## Target / Label
- **Intuition:** The answer we're trying to predict.
- **Technical:** The output variable a supervised model is trained to predict.
- **Titanic:** `Survived` (0 or 1).

## Model
- **Intuition:** The "thing" that has learned patterns from examples and can make guesses on new examples.
- **Technical:** A mathematical structure with adjustable internal settings (parameters) that maps input features to an output prediction.
- **Titanic:** Takes a passenger's features → outputs a survival prediction.

## Training
- **Intuition:** Showing the model many examples with known answers so it can adjust itself.
- **Technical:** The process of adjusting a model's parameters using data, typically by minimizing prediction error.
- **Titanic:** Feeding it `train.csv`, where we know the real outcome.

## Prediction / Inference
- **Intuition:** Asking a trained model to guess the answer for something it hasn't seen.
- **Technical:** Using a trained model to produce an output for new, unseen input data.
- **Titanic:** Running the trained model on `test.csv`, whose `Survived` values are hidden.

## Classification
- **Intuition:** A prediction task where the answer is one of a small set of categories, not a number on a continuous scale.
- **Technical:** A supervised learning problem where the target is categorical, as opposed to **regression**, where the target is continuous.
- **Titanic:** `Survived` is 0 or 1 — a category, not a number to be measured.

## Binary Classification
- **Intuition:** Classification where there are exactly two possible categories.
- **Technical:** A special case of classification with two classes (often labeled 0/1).
- **Titanic:** Survived vs. did not survive — exactly two outcomes.

## Important nuance: Classification vs. Probability
- Most classification models don't output a hard "yes/no" directly. They first estimate a **probability** (e.g., "78% chance this passenger survived"), then apply a rule (e.g., "if >50%, predict survived") to turn that into a category.
- We'll come back to this when we discuss thresholds and evaluation metrics — not needed in depth yet.

## Standard Error (of a proportion)
- **Intuition:** if you only measure a rate from a small group, your estimate wobbles more than if you measured it from a large group. Standard error is a rough number for "how much wobble to expect."
- **Technical:** for a proportion, SE ≈ √(p(1−p)/n), where p is the observed rate and n is the group size. A rough "very likely range" for the true rate is roughly ±2×SE around the observed rate. Breaks down (gives falsely 0 error) when p is exactly 0 or 1 with small n.
- **Titanic:** used to judge whether a survival rate from a small group (e.g. Embarked=Q, n=77, or SibSp=5, n=5) is trustworthy or likely just noise. Small-n groups showing exactly 0% or 100% survival are especially suspect.

## Data leakage from missing values (silent misclassification)
- **Intuition:** comparing a "don't know" value (like a missing Age) with `<` or `>` doesn't raise a flag — it silently picks one side, usually `False`, without telling you.
- **Technical:** `NaN < x` evaluates to `False` in Pandas, so missing rows get silently folded into whichever group corresponds to "False" in a boolean condition, contaminating that group with unknowns.
- **Titanic:** `df["Age"] < 12` puts all 177 missing-age passengers into "not a child," even though we don't actually know that. Fix requires either filtering with `.notna()`/`.isna()` first, or an explicit three-way category (Child / Adult / Unknown) rather than a silent binary.

## Feature Engineering (first example: FamilySize)
- **Intuition:** sometimes two separate columns are really both measuring one bigger underlying idea; combining them can reveal a clearer pattern than either shows alone.
- **Technical:** constructing a new feature from existing ones, based on a reasoned hypothesis about what might matter, then testing it against the target.
- **Titanic:** `FamilySize = SibSp + Parch + 1` (the `+1` accounts for the passenger themselves, since neither SibSp nor Parch count the passenger). Revealed a rise-then-collapse pattern in survival rate: alone (30.4%) < small families of 2–4 (55–72%) > large families of 5+ (mostly crashing, though based on small, noisy samples).

## Overfitting vs. Underfitting (concrete version, via Decision Tree depth)
- **Intuition:** underfitting is a model too simple to capture even the real pattern; overfitting is a model so flexible it starts memorizing noise/coincidences specific to the training data, which then hurts it on new data.
- **Technical:** as model complexity increases (e.g. Decision Tree `max_depth`), training performance keeps improving, but validation performance rises then falls — the peak is the sweet spot; before it is underfitting, after it is overfitting.
- **Titanic:** swept `max_depth` ∈ {2,3,4,5,6,8}. Validation accuracy rose from depth 2 (79.2%) to a peak at depth 3-4 (~81.9%), then declined at depth 5+ (down to ~80.0%) — a clean, real example of this exact curve, not just a theoretical shape.

## Decision Tree
- **Intuition:** instead of computing one weighted sum like logistic regression, a tree asks a sequence of yes/no questions about the data, splitting passengers into smaller groups at each step, ending in a prediction.
- **Technical:** recursively splits training data on feature values, choosing splits that best separate the classes, forming a tree of if/else conditions. `max_depth` limits how many nested splits are allowed (controls overfitting).
- **Titanic:** even at its best depth, underperformed logistic regression (81.90% vs ~83.4%). Checking `feature_importances_` revealed why: the tree relied almost entirely on `Title_Mr` (0.637) and never used `IsFemale` (0.000) at all — once it split on Title_Mr, IsFemale had nothing left to add, since the two are highly redundant.

## Random Forest (Ensemble)
- **Intuition:** build many different decision trees, each seeing a random subset of the data and features, then average their predictions — so no single feature can dominate every tree the way it dominated one single tree.
- **Technical:** an ensemble method; `n_estimators` = number of trees, each trained on a bootstrap sample with a random feature subset per split. Reduces overfitting risk that a single tree has, generally with lower variance for a given depth (though not always — see below).
- **Titanic:** confirmed the hypothesis partially — `IsFemale`'s importance jumped from 0.000 (single tree) to 0.220 (forest), since many trees didn't have Title_Mr available and used IsFemale instead. Best mean accuracy (84.25%, depth=5/200 trees) was *not* reliably better than logistic regression once its own run-to-run noise (stdev ±2.45, the widest of any model tried) was accounted for.

## Feature Importance
- **Intuition:** after training a tree-based model, you can ask it "how much did each feature actually matter to your decisions" — not a guess, a direct readout of the trained model.
- **Technical:** `.feature_importances_` on a fitted tree/forest gives a score per feature, roughly proportional to how much it reduced prediction error across all splits.
- **Titanic:** used to discover the Title_Mr / IsFemale redundancy in a single tree, and to confirm Random Forest partially fixes it. Caution: only meaningful for the *specific* trained model checked — always confirm which hyperparameters that exact model object actually has (e.g. `model.max_depth`) before trusting its importances, since a leftover object from a loop may not be the one you think it is.

## `statistics` module (mean, stdev on plain lists)
- **Intuition:** a built-in Python toolkit for basic stats on a plain list of numbers — no Pandas needed.
- **Technical:** `import statistics as st`; `st.mean(list)`, `st.stdev(list)` (sample standard deviation), plus built-in `min()`/`max()` (no import needed).
- **Titanic:** used to summarize accuracy scores collected across multiple random_state loops, instead of manually computing mean/stdev by hand each time.

## `.map()` (Series lookup)
- **Intuition:** for every row, look up a value in a separate small table and write the result into a new column.
- **Technical:** `series.map(lookup)` where `lookup` is typically another Series (e.g. the output of `.value_counts()`) — matches each value to its corresponding entry.
- **Titanic:** `df["Ticket"].map(df["Ticket"].value_counts())` — gives each row the total count of passengers sharing its exact ticket number, without a groupby.

## pd.concat()
- Intuition: stack two DataFrames (or columns) on top of each other into one combined set.
- Technical: pd.concat([series1, series2]) combines rows from multiple objects into one, preserving all values from both.
- Titanic: combined Ticket values from train.csv and test.csv before counting group sizes, so families split across both files (like the Sage family) get an accurate group size instead of being undercounted.

## Data leakage boundary (what's actually forbidden)
- Intuition: not all use of test.csv is leakage — only using its Survived values (or anything derived from knowing them) is a problem, since that's the one thing genuinely unavailable at real prediction time.
- Technical: structural information (like Ticket, Pclass, Name) is legitimately available in both train and test sets simultaneously in real life — combining it across files is safe. Combining or deriving anything from the target column across the train/test boundary is not.
- Titanic: combining Ticket columns from both files to fix TicketGroupSize was safe, since test.csv has no Survived column at all to leak in the first place.

## Shuffling in cross-validation (why cv=5 alone isn't enough)
- Intuition: k-fold cross-validation slices data into folds in whatever order it's already in — if that order isn't random (e.g. sorted by PassengerId/booking order), the folds themselves can end up unevenly mixed, inflating measured variance.
- Technical: sklearn's default (Stratified)KFold from an integer cv is instantiated with shuffle=False. Passing an explicit `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` shuffles once before slicing into folds, rather than using the data's original row order.
- Titanic: unshuffled cross_val_score gave Logistic Regression a higher-std, lower-mean result (82.94%/±2.63) than the same model with explicit shuffling (83.16%/±1.44) — confirmed the row order itself was adding measurement noise, not the model being genuinely less stable.

---

*(This glossary will grow as we introduce new concepts — preprocessing, validation, overfitting, etc.)*
