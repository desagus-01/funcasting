# INFO: Below made with AI, cba with visuals - hate it
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import polars as pl


def _unique_dates_np(dates: pl.DataFrame) -> np.ndarray:
    return dates.select(pl.col("date")).unique(maintain_order=True).to_numpy().ravel()


def plt_prob_shift(
    prior_prob: np.ndarray,
    post_prob: np.ndarray,
    dates: pl.DataFrame,
    k_top: int = 30,
    # WHY panel:
    scenarios: pl.DataFrame | None = None,
    why_assets: list[str] | None = None,
    zscore_assets: bool = True,  # standardise so assets share y-axis
    highlight_k: int = 10,  # how many up + down dates to highlight
    highlight_alpha: float = 0.10,  # transparency of highlight spans
) -> None:
    """
    Panels:
      1) posterior prob + prior baseline, with mover markers on posterior
      2) top-k |Δp| bars (red 0-line)
      3) WHY: selected assets on one axis (optionally z-scored), with up/down mover date highlights
    """
    x = _unique_dates_np(dates)

    if len(prior_prob) != len(post_prob) or len(x) != len(post_prob):
        raise ValueError(
            f"Length mismatch: len(x)={len(x)}, len(prior)={len(prior_prob)}, len(post)={len(post_prob)}"
        )

    eps = 1e-18
    df = (
        pl.DataFrame({"date": x, "p0": prior_prob, "p1": post_prob})
        .with_columns(
            (pl.col("p1") - pl.col("p0")).alias("dp"),
            ((pl.col("p1") + eps).log() - (pl.col("p0") + eps).log()).alias(
                "log_ratio"
            ),
        )
        .with_row_index("idx")
    )

    # --- stats ---
    tv = 0.5 * float(df.select(pl.col("dp").abs().sum()).item())
    kl = float(df.select((pl.col("p1") * pl.col("log_ratio")).sum()).item())
    ess0 = 1.0 / float(df.select((pl.col("p0") ** 2).sum()).item())
    ess1 = 1.0 / float(df.select((pl.col("p1") ** 2).sum()).item())

    # --- top movers by |Δp| for bars + markers ---
    top = (
        df.select("idx", "date", "dp")
        .with_columns(pl.col("dp").abs().alias("abs_dp"))
        .sort("abs_dp", descending=True)
        .head(k_top)
        .sort("date")
    )
    top_idx = top.get_column("idx").to_numpy()
    top_dates = top.get_column("date").to_numpy()
    top_dp = top.get_column("dp").to_numpy()

    # --- select up/down highlight dates (separately) ---
    movers_sorted = (
        df.select("date", "dp")
        .with_columns(pl.col("dp").abs().alias("abs_dp"))
        .sort("abs_dp", descending=True)
    )
    up_dates = (
        movers_sorted.filter(pl.col("dp") > 0)
        .head(highlight_k)
        .get_column("date")
        .to_numpy()
    )
    down_dates = (
        movers_sorted.filter(pl.col("dp") < 0)
        .head(highlight_k)
        .get_column("date")
        .to_numpy()
    )

    # --- layout: add WHY panel only if inputs provided ---
    use_why = scenarios is not None and why_assets
    nrows = 3 if use_why else 2

    fig, axs = plt.subplots(
        nrows,
        1,
        figsize=(12, 8.5 if use_why else 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": ([1.35, 1.0, 1.25] if use_why else [1.35, 1.0])},
    )
    if nrows == 2:
        ax_prob, ax_bar = axs
        ax_why = None
    else:
        ax_prob, ax_bar, ax_why = axs

    # ---------------- Panel 1: posterior + prior baseline + mover markers ----------------
    p1 = df.get_column("p1").to_numpy()
    p0 = df.get_column("p0").to_numpy()

    ax_prob.plot(x, p1, label="posterior", linewidth=1.4)
    ax_prob.plot(x, p0, label="prior", linewidth=1.0)
    ax_prob.scatter(top_dates, p1[top_idx], s=22, zorder=3)

    ax_prob.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 3))
    ax_prob.set_ylabel("Scenario prob")
    ax_prob.legend(loc="upper right")

    stats_txt = f"TV={tv:.5f}  KL={kl:.6f}  ESS0={ess0:.1f}  ESS1={ess1:.1f}"
    ax_prob.text(
        0.01,
        0.02,
        stats_txt,
        transform=ax_prob.transAxes,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.9),
        fontsize=10,
    )

    # ---------------- Panel 2: top-k Δp bars ----------------
    colors = np.where(top_dp >= 0, "C0", "C3")
    ax_bar.bar(top_dates, top_dp, color=colors)
    ax_bar.axhline(0.0, linewidth=1.2, color="red")
    ax_bar.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 3))
    ax_bar.set_ylabel(f"Top {k_top} Δp")

    # ---------------- Panel 3: WHY (assets together) ----------------
    if use_why and ax_why is not None:
        sc = scenarios.with_columns(pl.Series("date", x))

        # highlight bands first (so lines draw on top)
        for d in up_dates:
            ax_why.axvspan(
                d, d, color="C0", alpha=highlight_alpha
            )  # very thin span (acts like vline)
        for d in down_dates:
            ax_why.axvspan(d, d, color="C3", alpha=highlight_alpha)

        # plot assets as (optionally) z-scored lines
        for a in why_assets:
            y = sc.get_column(a).to_numpy()

            if zscore_assets:
                mu = float(np.nanmean(y))
                sd = float(np.nanstd(y))
                y_plot = (y - mu) / (sd if sd > 0 else 1.0)
            else:
                y_plot = y

            ax_why.plot(x, y_plot, label=a, linewidth=1.1)

        ax_why.axhline(0.0, linewidth=1.0, color="red")
        ax_why.set_ylabel("Assets (z-score)" if zscore_assets else "Asset level")
        ax_why.legend(loc="upper left", ncols=min(3, len(why_assets)))

        # Add a tiny note for what the highlights mean
        ax_why.text(
            0.01,
            0.02,
            f"Highlights: top {highlight_k} upweighted (C0) + top {highlight_k} downweighted (C3) dates by |Δp|",
            transform=ax_why.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            alpha=0.85,
        )

    # --- date formatting ---
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    axs[-1].xaxis.set_major_locator(locator)
    axs[-1].xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.show()
