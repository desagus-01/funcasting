from polars import DataFrame

from maths.time_series.iid_tests import ljung_box_test


def assets_need_mean_modelling(data: DataFrame, assets_to_test: list[str]) -> list[str]:
    needs_mean_modelling = []
    for asset in assets_to_test:
        lj = ljung_box_test(data=data, asset=asset)
        if len(lj.rejected) != 0:
            needs_mean_modelling.append(asset)
    return needs_mean_modelling


# TODO: Create autoarima function now
