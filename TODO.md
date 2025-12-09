> [!TODO] Higher Priority
> - [ ] Learn about stochastic processes and implement 
> - [ ] One function to return all invariant tests 

**Break Down of Stochastic Process**
1. Validation (Tests)
    > - [ ] Add tests for stationarity (currently only on iid) 
    > - [ ] De-trending and de-seasoning? 
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
    > - [X] Add elippsoid tests? 
> - [X] Add tests for invariance (iid) 
    > - [X] Look at other tests of iid 
