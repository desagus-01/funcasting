from dataclasses import dataclass
from enum import Enum
from typing import Literal, Self, TypeAlias

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars import DataFrame
from pydantic import BaseModel, ConfigDict

from flex_probs.operations import indicator_quantile_marginal


class ConstraintSigns(str, Enum):
    equal_greater = "equal_greater"
    equal_less = "equal_less"
    equal = "equal"


ConstraintSignLike: TypeAlias = (
    ConstraintSigns | Literal["equal_greater", "equal_less", "equal"]
)


class CorrInfo(BaseModel):
    asset_pair: tuple[str, str]
    corr: float


class View(BaseModel):
    """
    Allows to create a view on a single scenario
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str
    risk_driver: tuple[str, str] | str
    data: NDArray[np.floating]
    views_target: NDArray[np.floating] | None
    sign_type: ConstraintSignLike
    mean_ref: NDArray[np.floating] | None = None


@dataclass
class ViewBuilder:
    data: DataFrame
    views: list[View]

    def quantile(self, quant: float, quant_prob: float) -> Self:
        if quant_prob > quant:
            raise ValueError(
                f"Your target prob of {quant_prob}, must be smaller or equal to your current quant of {quant}!"
            )

        quant_ind = indicator_quantile_marginal(self.data, quant)

        self.views.extend(
            View(
                type="quantile",
                risk_driver=self.data.columns[0],
                data=quant_ind.select(pl.col.quant_ind).to_numpy().T,
                views_target=np.array(quant_prob),
                sign_type="equal_less",
            )
        )

        return self

    def corr(
        self,
        corr_targets: list[CorrInfo],
        sign_type: list[ConstraintSignLike],
    ) -> Self:
        self.views.extend(
            [
                View(
                    type="corr",
                    risk_driver=corr_info.asset_pair,
                    data=self.data[list(corr_info.asset_pair)].to_numpy().T,
                    views_target=np.array(corr_info.corr),
                    sign_type=sign_type[i],
                )
                for i, corr_info in enumerate(corr_targets)
            ]
        )
        return self

    def mean(
        self,
        target_means: dict[str, float],
        sign_type: list[ConstraintSignLike],
    ) -> Self:
        """
        Builds the constraints based on each asset's targeted mean.
        """
        # Checks we have equal amounts of self.data, constraints, and targets
        if not (len(target_means.keys()) == len(sign_type)):
            raise ValueError(
                "All inputs must have the same length as the number of views."
            )

        self.views.extend(
            [
                View(
                    type="mean",
                    risk_driver=key,
                    data=self.data[key].to_numpy().T,
                    views_target=np.array(target_means[key]),
                    sign_type=sign_type[i],
                )
                for i, key in enumerate(target_means.keys())
            ]
        )

        return self

    def std(
        self,
        target_std: dict[str, float],
        sign_type: list[ConstraintSignLike],
        mean_ref: float | None = None,
    ) -> Self:
        # Checks we have equal amounts of self.data, constraints, and targets
        if not (len(target_std.keys()) == len(sign_type)):
            raise ValueError(
                "All inputs must have the same length as the number of views."
            )
        self.views.extend(
            [
                View(
                    type="std",
                    risk_driver=key,
                    data=self.data[key].to_numpy().T,
                    views_target=np.array(target_std[key]),
                    sign_type=sign_type[i],
                    mean_ref=None if mean_ref is None else np.array(mean_ref),
                )
                for i, key in enumerate(target_std.keys())
            ]
        )
        return self

    def ranking(
        self,
        asset_ranking: list[str],
    ) -> Self:
        """
        Build view based on which assets should have a higher expected return (ie A >= B >= C ...).
        """

        assets_to_iterate = range(len(asset_ranking) - 1)

        comparisons = [
            (asset_ranking[i], asset_ranking[i + 1]) for i in assets_to_iterate
        ]
        self.views.extend(
            [
                View(
                    type="sorting",
                    risk_driver=comparisons[asset],
                    data=self.data[comparisons[asset]].to_numpy().T,
                    views_target=None,
                    sign_type="equal_greater",
                )
                for asset in assets_to_iterate
            ]
        )
        return self

    def marginal(self, current_marginal: str, target_marginal: str) -> Self:
        rd_np = self.data.select(target_marginal).to_numpy()

        mean_view = self.mean(
            self.data, {current_marginal: rd_np.mean()}, sign_type=["equal"]
        )
        std_view = self.std(
            self.data,
            {current_marginal: rd_np.std()},
            sign_type=["equal"],
            mean_ref=rd_np.mean(),
        )

        self.views.extend(mean_view + std_view)
        return self
