#  Funcasting

**Flexible Probability Scenario Modeling & Portfolio Construction Toolkit**

*funcasting* is a research-oriented Python library for building end-to-end scenario models and constructing portfolios using flexible probabilities. Inspired by Attilio Meucci’s work, it allows users to encode subjective market views numerically and transform data-driven scenarios into probability-weighted forecasts using entropy pooling and copula marginal models.

> **Disclaimer**: This tool is strictly for research and educational purposes. It is not intended for actual investment or financial decision-making.


---

## Current Features

-  **Scenario Modeling**: Combine financial data with assigned probabilities to build custom scenario models.
-  **Flexible Probabilities**: Use **entropy pooling** to adjust base probabilities while satisfying user-defined constraints.
-  **Copula Marginals**: Apply copula and marginal distribution assumptions to generate new plausible scenarios.
-  **Forecasting Pipeline**: An automated pipeline (currently supports detrending & deseasonalizing) to prepare data for scenario analysis.
-  **Lightweight & Focused**: Only uses `numpy` and `scipy` as core dependencies.

---

## Installation
git clone https://github.com/desagus-01/funcasting.git
cd funcasting
uv sync

---

## Roadmap

See `TODO.md`.

In general:
1. Finish automated univariate forecasting pipeline.
2. Add resampling based on flexible probabilities.
3. Multivariate modelling/foreacasting using monte carlo.
4. Work on optimization approach(es) based on forecasted model
5. Add risk factor implementation
6. Add visuals etc.
 

