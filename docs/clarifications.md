# Clarifications & Recurring Confusions

Points that came up more than once, or caused real confusion, written down
precisely so they don't need re-explaining from scratch each time.

---

## Where do the lookup tables actually live?

**`src/preprocessing.py` contains ONLY the `preprocess()` function itself** —
no lookup tables are built inside it. It only *receives* them as arguments
and looks values up in them.

**The notebook** builds the lookup tables (`age_medians_by_title`,
`fare_medians_by_pclass`, `combined_ticket_counts`) — always computed from
**raw_train only** (or train+test combined, for tickets, since that's safe —
see leakage note below) — *before* `preprocess()` is ever called. Then those
already-built tables get handed INTO the function as arguments, once for
train, once for test, so both use the exact same frozen values.

Two separate files, two separate jobs: notebook builds the tables,
`preprocessing.py`'s function only uses them.

## Overfitting vs. Data Leakage — different concepts, often confused

- **Overfitting**: a model learns patterns too specific to its own training
  data (including noise), hurting performance on new data. This is about
  *the model's learning process*.
- **Leakage**: information that wouldn't really be available at real
  prediction time (e.g. computing a median from test data) sneaks into
  training. This is about *the preprocessing/setup process*, not the model
  itself. Even a trivially simple model can leak, even if it can't overfit.
- Test: could a dead-simple model still have this problem? If yes, it's
  leakage, not overfitting.

## What data leakage actually forbids (the real boundary)

Not all use of test.csv is leakage. Only using the **target** (`Survived`)
— or anything derived from knowing it — is the real danger, since that's
the one thing genuinely unavailable at real prediction time.

- **Safe**: combining `Ticket` values from train+test to compute
  `TicketGroupSize` (test.csv has no `Survived` column to leak in the first
  place — pure structural information, legitimately available).
- **Not safe**: computing `age_medians_by_title` from test data, or any
  statistic that would need test's `Survived` values.

## `.map()` vs `.transform()` — different tools for different jobs

- **`.map()`**: looks up each row's value in an **already-built, separate**
  table (e.g. output of `.value_counts()`, or a groupby `.median()` result).
  `df["Title"].map(age_medians_by_title)` — the table already exists;
  `.map()` just looks things up in it.
- **`.transform()`**: **computes** a group statistic directly from the data
  at hand, then broadcasts it back to every row in that group, same length
  as the original. `df.groupby("Title")["Age"].transform("median")` — no
  separate table needed, it computes on the spot.
- `.map()` only works with ONE lookup key at a time — it cannot look up a
  value using two columns together (e.g. Title AND Sex simultaneously).
  That would need a different approach (e.g. building a combined key first,
  or a multi-index lookup) — this came up when investigating Age-by-Title+Sex.

## Why `d` instead of `df` inside the function (variable scope)

Parameters inside a function are **local** — they only exist inside that
function, completely separate from any variable of the same name outside
it. Using a different name (`d`) instead of `df` makes it visually obvious
that this is a distinct, self-contained variable, not necessarily the same
`df` sitting in the notebook — especially since the function gets called
with different DataFrames (`raw_train`, `raw_test`) at different times, and
shouldn't need to "know" which one it's currently holding.

## Editing a `.py` file doesn't update a running notebook automatically

Python caches imported modules. If you edit `preprocessing.py` after
already running `from src.preprocessing import preprocess` in a notebook,
the notebook keeps using the OLD version in memory until you **restart the
kernel and re-run the import** (or re-run everything from the top). This
caused a real bug once: an old, cached 3-argument version of `preprocess()`
kept getting called even after the file was updated to take 4 arguments.

## `ModuleNotFoundError: No module named 'src'`

Happens because a notebook's working directory is wherever the `.ipynb`
file itself lives (e.g. `notebooks/`), and `src/` is a sibling folder, not
inside it. Fix: add `sys.path.append("..")` before importing, which tells
Python to also search one directory up (the project root) for modules.

## Cell execution order matters — a function call needs its inputs defined FIRST

A notebook cell run out of order (or moved without re-running everything)
can call a function using a variable that isn't defined yet, even if that
variable's definition exists lower down in the notebook. Always check that
inputs are computed in cells ABOVE where they're used, and when in doubt,
Restart Kernel + Run All rather than trust partial reruns.
