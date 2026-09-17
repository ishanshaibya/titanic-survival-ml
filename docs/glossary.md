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

---

*(This glossary will grow as we introduce new concepts — preprocessing, validation, overfitting, etc.)*
