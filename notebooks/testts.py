from maths.distributions import uniform_probs
from maths.stat_tests import ellipsoid_test
from utils.template import get_template

info_all = get_template()
increms = info_all.increms_df.drop("date")
probs = uniform_probs(increms.height)

x = ellipsoid_test(
    increms,
    lags=2,
    prob=probs,
)
print(x)
