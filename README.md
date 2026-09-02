# Improved Collaborative Filtering Method

Research-oriented Python reproduction project for the User-Based Collaborative
Filtering similarity model proposed by:

Feng, J., Fengs, X., Zhang, N., and Peng, J. (2018), "An improved collaborative
filtering method based on similarity", PLOS ONE 13(9): e0204003.
https://doi.org/10.1371/journal.pone.0204003

The project is currently at:

**Step 3.1 - Packaging and Reproducibility Cleanup**

No recommendation algorithm has been implemented yet.

## Research Workflow

The intended future research pipeline is:

Ratings -> preprocessing -> train/test split -> sparsity analysis -> S1/S2/S3 -> final similarity
-> KNN -> rating prediction -> Top-N recommendation -> evaluation

This pipeline is documented for traceability only. Step 3.1 keeps the random
split unchanged and focuses on packaging, reproducibility, and documentation.

Future mathematical code must preserve the original paper definitions exactly,
with equation references in implementation and tests. Research extensions will
be implemented separately from the original paper method so they can be compared
fairly.

## Architecture

The project separates physical storage from recommendation logic:

Data Source -> Repository -> Algorithm -> Experiments

The repository layer returns domain `Rating` objects and does not expose storage
details such as SQL, sessions, table names, connection strings, or dataframe
concepts. This lets future adapters such as CSV, MovieLens, PostgreSQL, or MySQL
repositories implement the same contract without changing algorithm modules.

## Current Modules

- `recommender.domain`: immutable domain models and identifier type aliases.
- `recommender.data`: technology-agnostic repository protocol and in-memory
  repository implementation.
- `recommender.preprocessing`: sparse indexed rating representation, direct
  rating lookup, `I_u`, co-rated item intersection, and rated-item union.
- `recommender.config`: minimal reproducible research settings.
- `recommender.logging_config`: standard-library logging setup.
- `recommender.similarity`, `neighbors`, `prediction`, `recommendation`,
  `evaluation`, and `experiments`: documented placeholders for later steps.

## Implemented in Step 3.1

- sparse indexed rating representation
- user and item indexing
- direct `r_ui` rating lookup
- `I_u`, the rated-item set for a user
- users who rated an item
- co-rated item intersection
- rated-item union
- duplicate user-item rating detection
- reproducible random 80/20 train/test splitting

The splitter uses a plain rating-level random shuffle with a local
`random.Random(seed)` instance. It does not stratify by user or item, and it
does not force cold-start protection. The next step will compute training-set
sparsity from the split train dataset.

## Reproduction Decisions

### Explicitly stated by the paper

- random division of ratings
- 80% training
- 20% testing

### Reproduction implementation decisions

- deterministic configurable random seed
- local `random.Random(seed)`
- floor-based training-size calculation
- preservation of initial rating ordering before shuffle
- no additional cold-start correction because none is specified by the paper

The same seed reproduces the same split only when the incoming ratings arrive
in the same initial order. Future data adapters should supply a stable ordering
if exact replay across repeated loads is required.

The current plain random split can place users or items only in test, so future
evaluation code must explicitly account for total test ratings, predictable
ratings, and non-predictable ratings.

## Usage

Requires Python 3.11 or newer.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
ruff check .
```

## Not Implemented Yet

The following are intentionally out of scope for Step 3.1:

- rating matrix
- dataset sparsity
- user mean
- user standard deviation
- S1
- S2
- S3
- final similarity
- KNN
- rating prediction
- Top-N recommendation
- MAE
- RMSE
- threshold experiment
- baseline algorithms
