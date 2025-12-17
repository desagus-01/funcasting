
> [!ATTENTION] Update data with ADJUSTED prices (yfinance for now)

> [!TODO] Break Down of Stochastic Process
1. Pre-Processing
   > - [ ] De-trending  
   > - [ ] Stochastic de-trending (check how to) 
2. Validation (Tests)
    > - [ ] Re-factor as confusing atm 
    > - [ ] Add in auto lag for ADF test 
1. Estimation
    > - [ ] Log-likelihood + FFP based log-likelihood implementation for estimating parameters 
1. Foreacasting
    > - [ ] FFT Algorithm to calculate increment distribution? (see 45.5.1) 
    > - [ ] Bootstrapping methods with FFP (see 47.1.3)
    > - [ ] Hybrid Monte Carlo-historical (47.6.6)

> [!TODO] Lower Priority
> - [ ] Visualisations for checks (qualitative) 
> - [ ] Create a 'prior' object to be used on ScenarioDist 
> - [ ] Pairwise SC measure? Not super needed but would be fun 
> - [ ] Create pretty print for ScenarioProb object 
> - [ ] Finish setting up effective_ranks (helps us know whether our views are fine to use by determining rank of matrix)
> - [ ] Enforce mean_ref when views of mean and std are together 
> - [ ] Change assigned prob plot to always start at 0 
> - [ ] Create different diagnostic plots for each constraint type 
> - [ ] Change less_than plot to colour above/below 
> - [ ] Write own 'estimation' for distributions? 





## Completed
    > - [X] Add tests for stationarity (currently only on iid) 
    > - [X] Change HyptTestRes to Dataclass/TypedTuple instead of TypedDict? 
     > - [X] Add KPSS tests [Reference link](https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html)
        > - [X] Deterministic de-trend (copy from statsmodels) 
    > - [X] Add elippsoid tests? 
> - [X] Add tests for invariance (iid) 
    > - [X] Look at other tests of iid 
