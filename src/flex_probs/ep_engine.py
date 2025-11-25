from dataclasses import dataclass

from flex_probs.prob_vectors import entropy_pooling_probs
from models.scenarios import ScenarioDistribution
from models.views import View


@dataclass
class EntropyPooling:
    """
    Module for entropy Pooling.

    Given a ScenarioDistribution and a list of Views,
    return a new ScenarioDistribution with updated probabilities.
    """

    def update_prob(
        self,
        dist: ScenarioDistribution,
        views: list[View],
        confidence: float = 1.0,
        include_diags: bool = False,
    ) -> ScenarioDistribution:
        if not views:
            raise ValueError("Requires at least one View")

        posterior = entropy_pooling_probs(
            prior=dist.prob,
            views=views,
            confidence=confidence,
            include_diags=include_diags,
        )

        return ScenarioDistribution(scenarios=dist.scenarios, prob=posterior)
