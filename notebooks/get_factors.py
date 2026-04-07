from utils.tiingo import (
    clean_and_save_sample,
    get_ticker_prices,
    resolve_workflow_dates,
)

# %%
factors = ["MTUM", "QUAL", "VLUE", "SIZE", "USMV"]

target_start, target_end, start_str, end_str = resolve_workflow_dates(years_back=10)

factors_prices = get_ticker_prices(
    tickers=factors,
    start_date=start_str,
    end_date=end_str,
    frequency="daily",
    return_clean=True,  # gives columns: date, ticker, close, adj_close, adj_volume
)

# %%

clean_and_save_sample(factors_prices, "./data/tiingo_factors.csv")
