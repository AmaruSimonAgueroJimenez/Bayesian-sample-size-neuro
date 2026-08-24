#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swan_figures.py
================================================================================
Publication figures for the SWAN Visit-10 hormonal analysis (English), used in
the sample-size report. Replicates and cleans up the three original figures:

  1. fig_pca_swan.png        PCA of FSH and estradiol (variance, coefficients,
                             individual scores by menopausal status).
  2. fig_pc1_age.png         PC1 correlation with age (scatter + linear fit +
                             age-specific means).
  3. fig_joint_3d.png        Joint distribution of FSH and estradiol across
                             menopausal stages (3D densities + median/IQR).

Layout fixes vs. the originals: legends are placed in dedicated margins so they
never overlap the data; the statistics box of the age figure is a subtitle
line; panel spacing uses explicit gridspecs.

Usage:  python swan_figures.py [path/to/32961-0001-Data.dta]
Output: ./figs_report/*.png  (300 dpi)
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

DTA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("32961-0001-Data.dta")
OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

INK = "#0F172A"
GREY = "#64748B"
LIGHT = "#CBD5E1"
STAGE_ORDER = ["Premenopause", "Early perimenopause", "Late perimenopause",
               "Postmenopause", "Other / unknown"]
STAGE_COLORS = {"Premenopause": "#059669", "Early perimenopause": "#2563EB",
                "Late perimenopause": "#8B5CF6", "Postmenopause": "#E11D48",
                "Other / unknown": "#64748B"}


def style_axis(ax):
    ax.grid(color="#E2E8F0", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LIGHT)


def prepare_data():
    """Same PCA pipeline as analysis_pca_fsh_estradiol.py."""
    frame = pd.read_stata(DTA, convert_categoricals=False)
    d = frame[["AGE10", "FSH10", "E2AVE10", "STATUS10"]].dropna().copy()
    d = d[(d.FSH10 > 0) & (d.E2AVE10 > 0)].reset_index(drop=True)

    logh = np.log10(d[["FSH10", "E2AVE10"]].to_numpy(float))
    z = (logh - logh.mean(axis=0)) / logh.std(axis=0, ddof=1)
    ev, C = np.linalg.eigh(np.cov(z, rowvar=False))
    order = np.argsort(ev)[::-1]
    ev, C = ev[order], C[:, order]
    if C[0, 0] < 0:
        C[:, 0] *= -1
    if C[:, 1].sum() < 0:
        C[:, 1] *= -1
    s = z @ C
    d["PC1"], d["PC2"] = s[:, 0], s[:, 1]
    explained = ev / ev.sum()
    d["Stage"] = d.STATUS10.map({2.0: "Postmenopause", 3.0: "Late perimenopause",
                                 4.0: "Early perimenopause",
                                 5.0: "Premenopause"}).fillna("Other / unknown")
    return d, C, explained


# ============================================================== 1) PCA figure
def figure_pca(d, C, explained):
    fig = plt.figure(figsize=(16.4, 5.6), dpi=300, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.80, 1.00, 1.75],
                          left=0.05, right=0.985, top=0.80, bottom=0.20,
                          wspace=0.34)

    # --- A. variance explained ------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(["PC1", "PC2"], explained * 100, color=["#2563EB", "#94A3B8"],
                  width=0.62)
    for bar, v in zip(bars, explained * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2.2, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="#172033")
    ax.plot([0, 1], np.cumsum(explained) * 100, color="#E11D48", marker="o",
            markersize=6, linewidth=1.8, label="Cumulative")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("A. Variance explained", loc="left", pad=10,
                 fontweight="bold")
    style_axis(ax)
    ax.legend(frameon=False, loc="center right", fontsize=9)

    # --- B. component coefficients --------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(C, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
              aspect="auto")
    ax.set_xticks([0, 1], ["PC1", "PC2"])
    ax.set_yticks([0, 1], ["log10 FSH", "log10 estradiol"])
    for i in range(2):
        for j in range(2):
            v = C[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=12,
                    fontweight="bold",
                    color="white" if abs(v) > 0.55 else "#172033")
    ax.set_title("B. Component coefficients", loc="left", pad=10,
                 fontweight="bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_color(LIGHT)
    ax.text(0.5, -0.16,
            f"PC1 = {C[0, 0]:.3f}·z(log FSH) {C[1, 0]:+.3f}·z(log E2)\n"
            f"PC2 = {C[0, 1]:.3f}·z(log FSH) {C[1, 1]:+.3f}·z(log E2)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.6,
            linespacing=1.6, color="#334155")

    # --- C. individual scores by stage (legend in right margin) ---------
    ax = fig.add_subplot(gs[0, 2])
    for stage in STAGE_ORDER:
        g = d[d.Stage == stage]
        if g.empty:
            continue
        ax.scatter(g.PC1, g.PC2, s=14, alpha=0.45, color=STAGE_COLORS[stage],
                   edgecolors="none", label=f"{stage} (n={len(g):,})")
    for stage in STAGE_ORDER:            # means on top, after all points
        g = d[d.Stage == stage]
        if g.empty:
            continue
        ax.scatter(g.PC1.mean(), g.PC2.mean(), s=90, marker="D",
                   color=STAGE_COLORS[stage], edgecolor="white",
                   linewidth=1.3, zorder=5)
    ax.axhline(0, color=LIGHT, linewidth=0.8)
    ax.axvline(0, color=LIGHT, linewidth=0.8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%): higher FSH / lower estradiol")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%): both hormones move together")
    ax.set_title("C. Individual PCA scores by menopausal status", loc="left",
                 pad=10, fontweight="bold")
    style_axis(ax)
    leg = ax.legend(frameon=False, fontsize=8.8, loc="upper left",
                    bbox_to_anchor=(1.015, 1.0), borderaxespad=0,
                    handletextpad=0.35, labelspacing=0.6,
                    title="Menopausal status", title_fontsize=9)
    leg._legend_box.align = "left"
    for h in leg.legend_handles:
        h.set_alpha(1.0)
        h.set_sizes([46])

    fig.suptitle("PCA of FSH and Estradiol in the Complete SWAN Visit 10 Sample",
                 y=0.965, fontsize=16, fontweight="bold", color=INK)
    fig.text(0.5, 0.022,
             f"n = {len(d):,} complete cases · log10 transformation · variables "
             "standardized before PCA · PC signs oriented for interpretation",
             ha="center", va="bottom", fontsize=9, color=GREY)
    fig.savefig(OUT / "fig_pca_swan.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ========================================================== 2) PC1 vs age
def figure_pc1_age(d):
    age = d.AGE10.to_numpy(float)
    pc1 = d.PC1.to_numpy(float)
    r, p = stats.pearsonr(age, pc1)
    fz = np.arctanh(r)
    se = 1 / np.sqrt(len(d) - 3)
    ci = np.tanh([fz - 1.96 * se, fz + 1.96 * se])

    X = np.column_stack([np.ones_like(age), age])
    b = np.linalg.lstsq(X, pc1, rcond=None)[0]
    resid = pc1 - X @ b
    mse = np.sum(resid ** 2) / (len(age) - 2)
    grid = np.linspace(age.min(), age.max(), 240)
    fit = b[0] + b[1] * grid
    sxx = np.sum((age - age.mean()) ** 2)
    se_mean = np.sqrt(mse * (1 / len(age) + (grid - age.mean()) ** 2 / sxx))
    tcrit = stats.t.ppf(0.975, df=len(age) - 2)

    rng = np.random.default_rng(20260820)
    jit = age + rng.uniform(-0.20, 0.20, size=len(age))
    summ = d.groupby("AGE10")["PC1"].agg(["mean", "std", "count"]).reset_index()
    summ["se"] = summ["std"] / np.sqrt(summ["count"])

    fig, ax = plt.subplots(figsize=(10.6, 6.4), dpi=300, facecolor="white")
    ax.scatter(jit, pc1, s=15, color="#64748B", alpha=0.22, edgecolors="none",
               label="Individual observations")
    ax.fill_between(grid, fit - tcrit * se_mean, fit + tcrit * se_mean,
                    color="#2563EB", alpha=0.15,
                    label="95% CI for regression mean")
    ax.plot(grid, fit, color="#2563EB", linewidth=2.6, label="Linear fit")
    ax.errorbar(summ.AGE10, summ["mean"], yerr=1.96 * summ["se"], fmt="o",
                markersize=6.5, color="#E11D48", ecolor="#E11D48",
                elinewidth=1.3, capsize=3, markeredgecolor="white",
                markeredgewidth=1.0, label="Age-specific mean ± 95% CI",
                zorder=5)
    ax.axhline(0, color=LIGHT, linewidth=0.9)
    ax.set_xlabel("Age at Visit 10 (years)")
    ax.set_ylabel("PC1 score\n(higher FSH / lower estradiol)")
    ax.set_xticks(np.arange(int(age.min()), int(age.max()) + 1, 2))
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # title + statistics as stacked figure-level lines (no box over the data)
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    fig.text(0.10, 0.955, "PC1 Correlation With Age", ha="left", va="top",
             fontsize=16, fontweight="bold", color=INK)
    fig.text(0.10, 0.905, "Complete sample; PCA based on standardized log10 "
             "FSH and estradiol", ha="left", va="top", fontsize=9.5,
             color=GREY)
    fig.text(0.10, 0.868,
             f"n = {len(d):,}   ·   Pearson r = {r:.3f} "
             f"(95% CI {ci[0]:.3f} to {ci[1]:.3f})   ·   {p_text}   ·   "
             f"slope = {b[1]:.3f} PC1 units/year",
             ha="left", va="top", fontsize=10, color="#172033")
    fig.text(0.5, 0.012, "Positive PC1 values represent the higher-FSH / "
             "lower-estradiol hormonal pattern.", ha="center", va="bottom",
             fontsize=9, color=GREY)
    plt.subplots_adjust(left=0.10, right=0.975, bottom=0.125, top=0.815)
    fig.savefig(OUT / "fig_pc1_age.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return r, ci, p, b[1]


# ==================================== 3) joint FSH/E2 distribution by stage
def figure_joint_3d(d):
    stages = [("Early perimenopause", "Blues", "#2563EB"),
              ("Late perimenopause", "Purples", "#8B5CF6"),
              ("Postmenopause", "RdPu", "#E11D48")]
    fsh_ticks = [1, 10, 100, 1000]
    e2_ticks = [5, 20, 100, 500]

    fig = plt.figure(figsize=(16.6, 10.6), dpi=300, facecolor="white")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.55, 1.0],
                          left=0.045, right=0.975, top=0.865, bottom=0.075,
                          hspace=0.32, wspace=0.85)

    med = {}
    for i, (stage, cmap, color) in enumerate(stages):
        g = d[d.Stage == stage]
        lf, le = np.log10(g.FSH10.to_numpy()), np.log10(g.E2AVE10.to_numpy())
        med[stage] = (np.median(g.FSH10), np.median(g.E2AVE10),
                      np.percentile(g.FSH10, [25, 75]),
                      np.percentile(g.E2AVE10, [25, 75]), len(g))

        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2], projection="3d")
        kde = stats.gaussian_kde(np.vstack([lf, le]))
        xg = np.linspace(0.0, 3.0, 90)                 # FSH 1..1000
        yg = np.linspace(np.log10(5), np.log10(500), 90)
        XX, YY = np.meshgrid(xg, yg)
        ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        ZZ = ZZ / ZZ.max()
        ax.plot_surface(XX, YY, ZZ, cmap=cmap, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, alpha=0.92)

        # floor reference lines at the medians
        mf, me = np.log10(med[stage][0]), np.log10(med[stage][1])
        ax.plot([mf, mf], [yg[0], yg[-1]], [0, 0], color="#0D9488", lw=2.4,
                ls=(0, (6, 3)), zorder=4)
        ax.plot([xg[0], xg[-1]], [me, me], [0, 0], color="#D97706", lw=2.4,
                ls=(0, (2, 2)), zorder=4)

        ax.set_xticks(np.log10(fsh_ticks), [str(t) for t in fsh_ticks])
        ax.set_yticks(np.log10(e2_ticks), [str(t) for t in e2_ticks])
        ax.set_zticks([0, 0.5, 1.0])
        ax.set_xlabel("FSH (mIU/mL)", fontsize=9, labelpad=6)
        ax.set_ylabel("Estradiol (pg/mL)", fontsize=9, labelpad=6)
        ax.set_zlabel("Panel-relative density", fontsize=8.5, labelpad=4)
        ax.tick_params(labelsize=8, pad=1)
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect((1.15, 1.0, 0.62))
        ax.xaxis.pane.set_facecolor("white")
        ax.yaxis.pane.set_facecolor("white")
        ax.zaxis.pane.set_facecolor("white")
        title = stage if stage != "Postmenopause" else "Menopause (postmenopause)"
        ax.set_title(f"{title}  ·  $n$ = {med[stage][4]:,}", fontsize=12.5,
                     fontweight="bold", color=INK, pad=0, y=1.085)
        ax.text2D(0.0, 0.925, f"FSH median: {med[stage][0]:.1f} mIU/mL",
                  transform=ax.transAxes, fontsize=9, fontweight="bold",
                  color="#0D9488", ha="left", va="top",
                  bbox=dict(boxstyle="round,pad=0.32", fc="white",
                            ec="#0D9488", lw=0.9))
        ax.text2D(0.0, 0.845, f"Estradiol median: {med[stage][1]:.1f} pg/mL",
                  transform=ax.transAxes, fontsize=9, fontweight="bold",
                  color="#D97706", ha="left", va="top",
                  bbox=dict(boxstyle="round,pad=0.32", fc="white",
                            ec="#D97706", lw=0.9))

    # ------- bottom row: median + IQR summaries --------------------------
    labels = ["Early\nperimenopause", "Late\nperimenopause", "Postmenopause"]
    xs = np.arange(3)
    for j, (var, ylab, ttl) in enumerate(
            [(0, "FSH (mIU/mL)", "FSH median and interquartile range"),
             (1, "Estradiol (pg/mL)", "Estradiol median and interquartile range")]):
        ax = fig.add_subplot(gs[1, 3 * j:3 * j + 3])
        meds = [med[s][var] for s, _, _ in stages]
        iqrs = [med[s][2 + var] for s, _, _ in stages]
        colors = [c for _, _, c in stages]
        ax.fill_between(xs, [q[0] for q in iqrs], [q[1] for q in iqrs],
                        color="#94A3B8", alpha=0.18, zorder=1)
        ax.plot(xs, meds, color=GREY, lw=1.6, zorder=2)
        for x, m, q, c in zip(xs, meds, iqrs, colors):
            ax.plot([x, x], q, color=c, lw=7, alpha=0.45,
                    solid_capstyle="round", zorder=3)
            ax.plot(x, m, "o", ms=11, color=c, mec="white", mew=1.6, zorder=4)
            ax.annotate(f"{m:.1f}", (x, m), textcoords="offset points",
                        xytext=(0, 13), ha="center", fontsize=10.5,
                        fontweight="bold", color=INK)
        ax.set_yscale("log")
        yticks = [20, 30, 50, 100, 150] if var == 0 else [10, 20, 50, 100, 150]
        ax.set_yticks(yticks, [str(t) for t in yticks])
        ax.minorticks_off()
        ax.set_xticks(xs, labels)
        ax.set_xlim(-0.45, 2.45)
        ax.set_ylabel(ylab)
        ax.set_title(ttl, loc="left", pad=16, fontsize=12.5,
                     fontweight="bold", color=INK)
        ax.text(1.0, 1.045, "Circle = median  |  bar and band = IQR",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6,
                color=GREY)
        style_axis(ax)

    fig.suptitle("Joint Distribution of FSH and Estradiol Across Menopausal "
                 "Stages", y=0.975, fontsize=17, fontweight="bold", color=INK)
    fig.text(0.5, 0.012, "SWAN Visit 10 (ICPSR 32961) · Complete cases · "
             "Logarithmic hormone scales · 3D density normalized within each "
             "panel · Dashed teal line = FSH median · Dotted orange line = "
             "estradiol median", ha="center", va="bottom", fontsize=9,
             color=GREY)
    fig.savefig(OUT / "fig_joint_3d.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return med


def main():
    d, C, explained = prepare_data()
    figure_pca(d, C, explained)
    r, ci, p, slope = figure_pc1_age(d)
    med = figure_joint_3d(d)
    print(f"n={len(d):,} · PC1 {explained[0]*100:.1f}% · r(PC1,age)={r:.3f} "
          f"[{ci[0]:.3f}, {ci[1]:.3f}] · slope={slope:.3f}/yr")
    for s in ["Early perimenopause", "Late perimenopause", "Postmenopause"]:
        print(f"{s}: FSH med {med[s][0]:.1f}  E2 med {med[s][1]:.1f}  n={med[s][4]}")
    print(f"Figures in {OUT.resolve()}")


if __name__ == "__main__":
    main()
