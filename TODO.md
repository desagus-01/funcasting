## Portfolio & Risk
- [x] Create risk metrics ✅ 2026-04-09
- [x] Create some attribution factors (Z) ✅ 2026-04-09
    - [x] Either use pre-made or just some ETFs for sectors ✅ 2026-04-09
- [x] Estimate exposures using TOP-DOWN approach ✅ 2026-04-08
- [x] Recover shift alpha and residual U ✅ 2026-04-10
- [x] Build the joint distribution of factors and residuals ✅ 2026-04-10
- [x] Build factor attribution for risk ✅ 2026-04-21
    - [x] Add in min torsion to find orthogonal factors ✅ 2026-04-21
- [x] contribution of factors to risk using Euler method ✅ 2026-05-12
    - [x] marginal ✅ 2026-05-12
    - [x] total ✅ 2026-05-12
   
## Feature-Selection
- [x] Add in backward/forward regression based on Riskfolio to find 'best' factors ✅ 2026-05-12
- [x] Add in feature importance code from sklearn ✅ 2026-05-12
- [x] Add in RFE and RFECV from sklearn for feature selection ✅ 2026-05-12


## Re-factor
- [ ] Add appropriate logs to debug (but not too many)
    - [ ] Add logging files etc as well


## Forecasting pipeline
- [ ] Add metrics to evaluate forecast:
    - [ ] Add CRPS scores 
- [ ] Change model selection pipeline to drop bad forecast models
- [ ] Add EP option for simulation probabilities to reflect forecast views
- [x] create weighted OLS ✅ 2026-03-31
- [ ] Check out with white_noise test produce different answers 
- [ ] GARCH/ARCH should reflect prob weight...
- [ ] apply multiple tests to iid pipeline?
- [ ] Add check whether final innovs are iid?
- [ ] Add 'block' resampling instead to better preserve dynamics
- [ ] Add back probabilities for forecasts that were used from MC
- [ ] Simplify seasonality as currently just noise fitting
- [ ] Check whether random walk model is being activated correctly


## Lower Priority
- [ ] Go back and find ways of fixing missing values for copulas, curretly just dropping, but possible to replace...somehow...
- [ ] Create a 'prior' object to be used on ScenarioDist 
- [ ] Finish setting up effective_ranks (helps us know whether our views are fine to use by determining rank of matrix)
- [ ] Change assigned prob plot to always start at 0 
- [ ] Write own 'estimation' for copulas?





## Completed
