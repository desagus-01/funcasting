## Portfolio
- [ ] Add basic portfolio config (equal value for all)
- [ ] Create basic portfolio value out of this config

## Re-factor
- [ ] Add appropriate logs to debug (but not too many)
    - [ ] Add logging files etc as well


## Forecasting pipeline
- [ ] Add metrics to evaluate forecast:
    - [ ] Add CRPS scores 
- [ ] Change model selection pipeline to drop bad forecast models
- [ ] Always compare mean model with RW as base, go with best BIC
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
