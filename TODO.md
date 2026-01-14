
> [!ATTENTION] Update data with ADJUSTED prices (yfinance for now)

## Break Down of Stochastic Process
- [x] Apply seasonality adjustment ✅ 2026-01-14
    - [x] Identify/test for peaks ✅ 2026-01-13
    - [x] Run harmonic regression ✅ 2026-01-14
    - [x] De-season from de-trended series ✅ 2026-01-14
- [ ] Write Decision rule between stochastic and deterministic trend removal
- [ ] Log-likelihood + FFP based log-likelihood implementation for estimating parameters 
- [ ] FFT Algorithm to calculate increment distribution? (see 45.5.1) 
- [ ] Bootstrapping methods with FFP (see 47.1.3)
- [ ] Hybrid Monte Carlo-historical (47.6.6)


## Lower Priority
- [ ] Write robust tests 
- [ ] Visualisations for checks (qualitative) 
- [ ] Create a 'prior' object to be used on ScenarioDist 
- [ ] Pairwise SC measure? Not super needed but would be fun 
- [ ] Create pretty print for ScenarioProb object 
- [ ] Finish setting up effective_ranks (helps us know whether our views are fine to use by determining rank of matrix)
- [ ] Enforce mean_ref when views of mean and std are together 
- [ ] Change assigned prob plot to always start at 0 
- [ ] Create different diagnostic plots for each constraint type 
- [ ] Change less_than plot to colour above/below 
- [ ] Write own 'estimation' for distributions? 





## Completed
