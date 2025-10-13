import numpy as np
from polars import DataFrame

from data_types.vectors import (
    ConstraintSignLike,
    ConstraintTypeLike,
    View,
)
from get_data import get_example_assets
from maths.core import (
    simple_entropy_pooling,
)
from maths.prob_vectors import uniform_probs

# set-up
tickers = ["AAPL", "MSFT", "GOOG"]
assets = get_example_assets(tickers)
increms_df = assets.increments.drop("date")


def view_on_mean(
    data: DataFrame,
    target_means: dict[str, float],
    const_type: list[ConstraintTypeLike],
    sign_type: list[ConstraintSignLike],
) -> list[View]:
    """
    Builds the constraints based on each asset's targeted mean.
    """
    # Checks we have equal amounts of data, constraints, and targets
    if not (len(target_means.keys()) == len(const_type) == len(sign_type)):
        raise ValueError("All inputs must have the same length as the number of views.")

    return [
        View(
            risk_driver=key,
            data=data[key].to_numpy().T,
            views_target=np.array(target_means[key]),
            const_type=const_type[i],
            sign_type=sign_type[i],
        )
        for i, key in enumerate(target_means.keys())
    ]


targs = {"AAPL": 0.01, "GOOG": 0.02}
consts = ["inequality", "inequality"]
signs = ["equal_less", "equal_less"]

x = view_on_mean(increms_df, targs, consts, signs)

prior = uniform_probs(increms_df.height)

print(simple_entropy_pooling(prior, x))
