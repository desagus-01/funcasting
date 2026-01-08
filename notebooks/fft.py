# %% imports

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq

from maths.helpers import add_detrend_columns_max, add_differenced_columns
from utils.template import get_template

# %% load once
rd = get_template().asset_info.risk_drivers

x = add_detrend_columns_max(rd, ["AAPL"], max_polynomial_order=2)
a = x.select("AAPL_detrended_p2").to_numpy()

# %%

aapl = (
    add_differenced_columns(rd, ["AAPL"]).select("AAPL_diff_1").drop_nulls().to_numpy()
)
aapl_r = a.ravel()

# %%
x = aapl_r
N = x.size
X = rfft(x)


freqs = rfftfreq(N, d=1)  # d = sampling interval (1 for daily, change if not)


Pxx = (1 / N) * np.abs(X) ** 2


nonzero = freqs > 0
periods = 1 / freqs[nonzero]
powers = Pxx[nonzero]

plt.figure(figsize=(10, 5))
plt.plot(periods, powers)
plt.xlim(0, 500)  # adjust for visibility
plt.xlabel("Period (time units per cycle)")
plt.ylabel("Power")
plt.title("Periodogram (in Period Space)")
plt.grid(True)
plt.show()
