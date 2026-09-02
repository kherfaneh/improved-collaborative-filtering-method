# Improved Collaborative Filtering Method

Research-oriented Python reproduction project for the User-Based Collaborative
Filtering similarity model proposed by:

Feng, J., Fengs, X., Zhang, N., and Peng, J. (2018), "An improved collaborative
filtering method based on similarity", PLOS ONE 13(9): e0204003.
https://doi.org/10.1371/journal.pone.0204003

The project is currently at:

**Step 1 - Project Infrastructure**

No recommendation algorithm has been implemented yet.

## Research Workflow

The intended future research pipeline is:

Ratings -> preprocessing -> sparsity analysis -> S1/S2/S3 -> final similarity
-> KNN -> rating prediction -> Top-N recommendation -> evaluation

This pipeline is documented for traceability only. Step 1 only creates the
foundation needed to implement and test those stages later.

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
- `recommender.config`: minimal reproducible research settings.
- `recommender.logging_config`: standard-library logging setup.
- `recommender.preprocessing`, `similarity`, `neighbors`, `prediction`,
  `recommendation`, `evaluation`, and `experiments`: documented placeholders for
  later steps.

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
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Not Implemented Yet

The following are intentionally out of scope for Step 1:

- rating matrix
- train/test split
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
