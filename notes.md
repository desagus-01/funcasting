# General Notes


[Excellent general resource](https://python-advanced.quantecon.org/arma.html)

[Seasonality tests performed by Belgium](https://jdemetradocumentation.github.io/JDemetra-documentation/pages/theory/Tests_peaks.html)

[Repo for spectral estimation methods](https://github.com/IvanovicM/spectral-estimation/tree/master)

## Seasonality
[Explanation on Seasonality Adjustment](https://www.le.ac.uk/economics/research/RePEc/lec/leecon/dp11-12.pdf)
- The data can be detrended by differencing. Alternatively, a trend function
can be interpolated into the data and the residual deviations can be subjected
to the analysis. In the context of the seasonal adjustment of the data, the
latter approach is preferable, since the pattern of seasonal fluctuations can
be obscured by taking difference

[Amazing intuitive explanation on FFT](https://www.youtube.com/watch?v=g8RkArhtCc4)

[Best explanation of whole process](https://www.youtube.com/watch?v=pfjiwxhqd1M)

[Stephen P. info (great explanations)](https://www.le.ac.uk/users/dsgp1/)

[Code source for periodograms](https://quanteconpy.readthedocs.io/en/latest/_modules/quantecon/_estspec.html#ar_periodogram)

[Tests for seasonality in the frequency domain](https://www.census.gov/content/dam/Census/library/working-papers/2017/adrm/rrs2017-01.pdf)

**Decision**
Apply Meucci's approach of using a simpler harmonic regression which only identifies 1 seasonal component instead of band approach by Stephen P. As this is easier to implement and quikcker. **Definetly** come back and implement latter though.


> [!IMPORTANT] To adjust seasonality using fourier (spectrum) MUST be de-trended first


## Trend
- Test both stochastic and deterministic trend IN parallel.
- Determinstic/polynomial trend takes precedence as there is a smaller chance of it affecting later seasonality removal.


## Possible pipeline
**Possible Univariate Pipeline**
1. Test Efficiency
    1. Invariance Tests on **increments**
2. If efficiency
    1. Accept RW
    2. Choose distribution for Et
3. If not efficiency
    1. [x] test for deterministic trend ✅ 2026-01-05 
    2. [x] test for stochastic trend ✅ 2026-01-06
    3. [x] test for seasonality ✅ 2026-01-14
    4. re-test efficiency
4. If still fails look at other phenomenon 
 

