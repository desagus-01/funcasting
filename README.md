# Funcasting

**Flexible probability scenario modeling, forecasting, and portfolio construction toolkit (research-focused).**

`funcasting` is a Python codebase for building scenario distributions, encoding views, and running univariate-to-multivariate forecast workflows with probability-weighted sampling.

The long-term ambition is to provide an end-to-end portfolio construction pipeline where forecasting, scenario generation, and risk management are integrated into one coherent workflow.

> **Disclaimer**: Research and educational use only. Not investment advice.

---

## What is in the repository now

The project has moved beyond notebooks-only experiments and now contains a modular `src/` architecture:

- `src/pipelines/`
  - End-to-end orchestration (`preprocess`, `model_selection`, `forecasting`).
- `src/time_series/`
  - Diagnostics, stationarity/seasonality tests, transforms, and univariate model wrappers.
- `src/scenarios/`
  - Scenario distribution types, entropy pooling, copula-marginal updates.
- `src/simulation/`
  - Simulation state objects and path engines.
- `src/probability/`
  - Probability distributions and sampling utilities.
- `src/utils/`
  - Shared helpers and data-source connectors (e.g., Tiingo).

---

## Project ambition

Beyond forecasting, the project is being developed toward a full portfolio construction stack with a risk-first design:

1. Build robust, probability-weighted scenario forecasts
2. Convert scenarios into risk diagnostics and constraints
3. Construct and stress-test portfolios under explicit risk budgets
4. Support iterative view updates and re-optimization

## Core forecasting entrypoint

Main pipeline entrypoint:

- `src/pipelines/forecasting.py::run_n_steps_forecast(...)`

High-level flow:

1. Univariate preprocessing (white-noise screen, detrend, deseason)
2. Mean/volatility model selection
3. Innovation extraction and scenario drawing (`bootstrap`, `historical`, `cma`)
4. Path simulation
5. Optional inverse transforms back to price-like scale

---

## Data contract

Expected tabular input for pipeline functions:

- A `date` column
- One numeric column per asset
- Optional probability vector (`ProbVector`) with:
  - shape `(n_rows,)`
  - non-negative entries
  - sum approximately `1.0`

Recommended before running pipelines:

- Sort by `date`
- Ensure unique dates
- Ensure numeric asset columns only
- Resolve nulls explicitly

---

## Null handling (current behavior)

The current codebase has mixed null handling strategies (asset-level and full-row drops depending on module).

Practical recommendation for now:

- Clean data before modeling (drop/fill policy chosen explicitly)
- Re-check probability alignment after any row filtering

---

## Installation

Requires Python `>=3.13` and [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/desagus-01/funcasting.git
cd funcasting
uv sync
uv pip install -e .
```

---

## Examples

See:

- `notebooks/examples/simple_entropy_pooling.py`
- `notebooks/examples/simple_cma_example.py`
- `notebooks/examples/advanced_entropy_pooling.py`

---

## Roadmap (short)

- Harden data validation and null-policy consistency across modules
- Expand test coverage for pipeline and simulation entrypoints
- Extend multivariate forecasting and optimization workflows
- Add risk-budgeting and portfolio-construction modules on top of scenario forecasts
- Improve documentation for scenario/view APIs and portfolio workflow design
