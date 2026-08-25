#!/usr/bin/env python3
"""Figuras 3D (n x umbral BF_H1/H0 x probabilidad) para TODOS los análisis:
Cohorte 1 y Cohorte 2, criterio Lasso-por-componente (v3) y BF global (v4),
brazos hormonal y PM2.5.  Vértices marcados en BF = 3 y BF = 10.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

import bfda_lasso_multivariante as blm
from bfda_n_optimo import r2_critico
from bfda_v4_condicional_mvg import simular_r2hat, simular_r2hat_c2, \
    r2_critico_reg

W = Path(__file__).resolve().parent
OUT_H = W / "resultados_bfda"
OUT_P = OUT_H / "pm25"
UMBRALES = [2, 3, 4, 5, 6, 8, 10, 13, 16, 20]
REFS = [3, 10]
RSC = float(np.sqrt(2) / 4)
FCOL = ["#64748B", "#059669", "#2563EB", "#7C3AED", "#E11D48"]
BFN = "BF$_{H_1/H_0}$"


def parciales_pany(y, X, K, m, rng_unused=None):
    """P(>=1 de los K primeros predictores con BF>u), por umbral."""
    yc = y - y.mean(axis=1, keepdims=True)
    Z = np.concatenate([yc[:, :, None], X], axis=2)
    Zs = (Z - Z.mean(1, keepdims=True)) / Z.std(1, ddof=1, keepdims=True)
    C = np.einsum("snp,snq->spq", Zs, Zs) / (Z.shape[1] - 1)
    Om = np.linalg.inv(C)
    dg = np.sqrt(np.abs(np.diagonal(Om, axis1=1, axis2=2)))
    rp = -Om[:, 0, 1:K + 1] / (dg[:, 0:1] * dg[:, 1:K + 1])
    return {u: float((rp ** 2 > r2_critico(m, u)).any(1).mean())
            for u in UMBRALES}


def fig3d(P, fam, etiqueta_fam, xs, xlabel, focal_x, focal_txt, titulo,
          footer, fname, zlabel):
    lbf = np.log2(np.array(UMBRALES, float))
    Xg, Yg = np.meshgrid(xs, lbf, indexing="ij")
    fig = plt.figure(figsize=(11.6, 9.0), dpi=200, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    col = {f: FCOL[i] for i, f in enumerate(fam)}
    for f in fam:
        Zg = np.array([[P[(f, x, u)] for u in UMBRALES] for x in xs])
        ax.plot_surface(Xg, Yg, Zg, color=col[f], alpha=0.58,
                        edgecolor=col[f], linewidth=0.5, rstride=1,
                        cstride=1, antialiased=True, shade=False)
        ax.plot(xs, np.full_like(xs, lbf[-1], dtype=float), Zg[:, -1],
                color=col[f], lw=2.4)
    # plano objetivo
    Xp, Yp = np.meshgrid([xs[0], xs[-1]], [lbf[0], lbf[-1]], indexing="ij")
    ax.plot_surface(Xp, Yp, np.full_like(Xp, 0.80, dtype=float),
                    color="#0F172A", alpha=0.06)
    ax.text(xs[0], lbf[-1], 0.815, "0.80 target", color="#0F172A",
            fontsize=9)
    # plano x focal
    Yq, Zq = np.meshgrid([lbf[0], lbf[-1]], [0, 1], indexing="ij")
    ax.plot_surface(np.full_like(Yq, focal_x, dtype=float), Yq, Zq,
                    color="#E11D48", alpha=0.05)
    ax.plot([focal_x, focal_x], [lbf[0], lbf[-1]], [0, 0], color="#E11D48",
            lw=1.3, ls=":")
    ax.text(focal_x, lbf[0] - 0.35, 0.02, focal_txt, color="#E11D48",
            fontsize=9.5)
    # rebanadas y vértices en BF = 3 y BF = 10
    for j, uref in enumerate(REFS):
        yr = np.log2(uref)
        Xr, Zr = np.meshgrid([xs[0], xs[-1]], [0, 1], indexing="ij")
        ax.plot_surface(Xr, np.full_like(Xr, yr, dtype=float), Zr,
                        color="#0F172A", alpha=0.045)
        for f in fam:
            zc = [P[(f, x, uref)] for x in xs]
            ax.plot(xs, np.full_like(xs, yr, dtype=float), zc, color=col[f],
                    lw=3.0, zorder=10)
        ax.plot([focal_x, focal_x], [yr, yr], [0, 1.0], color="#0F172A",
                lw=1.1, ls="--", alpha=0.6, zorder=15)
        for f in fam:
            zv = P[(f, focal_x if not isinstance(xs, list) else focal_x,
                    uref)]
            ax.scatter([focal_x], [yr], [zv], color=col[f], s=58,
                       edgecolor="white", linewidth=1.2, depthshade=False,
                       zorder=30)
            lado = -0.045 * (xs[-1] - xs[0]) if j == 0 \
                else 0.045 * (xs[-1] - xs[0])
            ax.text(focal_x + lado, yr, zv + 0.028, f"{zv:.2f}",
                    color=col[f], fontsize=8.6, fontweight="bold",
                    ha="right" if j == 0 else "left", zorder=30)
        ax.text(focal_x, yr, 1.045, f"{BFN} = {uref}", color="#0F172A",
                fontsize=9, fontweight="bold", ha="center")
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(f"{BFN} threshold", labelpad=10)
    ax.set_zlabel(zlabel, labelpad=8)
    ax.set_yticks(np.log2([2, 3, 4, 6, 10, 16]))
    ax.set_yticklabels(["2", "3", "4", "6", "10", "16"])
    ax.set_zlim(0, 1.0)
    ax.set_xlim(xs[0], xs[-1])
    ax.view_init(elev=22, azim=-55)
    ax.xaxis.pane.set_alpha(0.0); ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.06)
    ax.legend(handles=[Patch(facecolor=col[f], alpha=0.75,
                             label=f"{etiqueta_fam} = {f:.2f}")
                       for f in reversed(fam)],
              loc="upper left", frameon=False, fontsize=10,
              bbox_to_anchor=(0.02, 0.90))
    ax.set_title(titulo, loc="left", pad=16, fontweight="bold",
                 color="#0F172A", fontsize=12.5)
    fig.text(0.5, 0.015, footer, ha="center", fontsize=8.2, color="#64748B")
    fig.savefig(fname, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", fname.name)


def lasso_c1(pm25, outdir):
    if pm25:
        blm.C_EDAD_Y, blm.C_H_AGE_POST = 0.0, 0.0
        yname = "PM$_{2.5}$ exposure index"
    else:
        blm.C_EDAD_Y, blm.C_H_AGE_POST = 0.17, -0.015
        yname = "hormonal PC1"
    rng = np.random.default_rng(20260825)
    K, S = 20, 4000
    ns = [40, 50, 60, 72, 85, 100, 120, 140, 160]
    fam = [0.2, 0.3, 0.4, 0.5]
    P = {}
    for rho in fam:
        for n in ns:
            y, X, _ = blm.generar_lote(n, rho, S, K, 1, rng)
            for u, v in parciales_pany(y, X, K, n - K).items():
                P[(rho, n, u)] = v
    fig3d(P, fam, "ρ", ns, "Sample size (n)", 60, "n = 60\n(proposal)",
          f"Cohort 1 in 3D: detection over sample size and evidence "
          f"threshold (K = 20)\nper-component JZS {BFN} on the partial "
          f"correlation, y = {yname}",
          f"Surfaces: P(≥1 brain component with {BFN} above the threshold) "
          "per true effect ρ · controls: remaining components + age "
          "(m = n - 20) · 4,000 datasets per cell · vertices marked at the "
          f"reference thresholds {BFN} = 3 and 10",
          outdir / "Fig_3D_BF_lasso_c1.png",
          f"P(≥1 brain PC with {BFN} > threshold)")


def lasso_c2(pm25, outdir):
    if pm25:
        blm.C_EDAD_Y, blm.C_H_AGE_POST = 0.0, 0.0
        hname = "PM$_{2.5}$ exposure"
    else:
        blm.C_EDAD_Y, blm.C_H_AGE_POST = 0.17, -0.015
        hname = "hormonal PC1"
    rng = np.random.default_rng(20260825)
    K, S = 5, 4000
    ns_tot = [48, 60, 72, 84, 100, 120, 160]
    fam = [0.5, 0.65, 0.70, 0.8, 1.0]
    P = {}
    for d in fam:
        for n in ns_tot:
            y, X = blm.generar_lote_c2(n, d, 0.0, S, K, rng)
            for u, v in parciales_pany(y, X, K, n - (K + 1)).items():
                P[(d, n // 2, u)] = v
    xs = [n // 2 for n in ns_tot]
    fig3d(P, fam, "d", xs, "n per group", 36, "36/group",
          f"Cohort 2 in 3D: MCI-HC detection over n and evidence threshold "
          f"(K = 5)\nper-component JZS {BFN}, design: group ~ brain PCs + "
          f"{hname} + age",
          f"Surfaces: P(≥1 brain component with {BFN} above the threshold) "
          "per MCI-HC separation d · controls: remaining components + "
          f"{hname} + age (m = n - 6) · 4,000 datasets per cell · vertices "
          f"at {BFN} = 3 and 10",
          outdir / "Fig_3D_BF_lasso_c2.png",
          f"P(≥1 brain PC with {BFN} > threshold)")


def v4_c1(pm25, outdir):
    base = W
    ev = pd.read_csv(base / "calibracion_dkt_eigenvalues.csv"
                     )["eigenvalue"].to_numpy()
    if pm25:
        y_pool = pd.read_csv(base / "calibracion_pm25.csv"
                             )["z_exposicion"].to_numpy()
        ydesc = "cohort residential PM$_{2.5}$ (model surface)"
    else:
        pc1 = pd.read_csv(base / "calibracion_swan_pc1.csv")
        pool = pc1.loc[pc1.stage == "early_peri", "PC1"]
        y_pool = ((pool - pool.mean()) / pool.std(ddof=1)).to_numpy()
        ydesc = "SWAN early-perimenopausal hormonal PC1"
    rng = np.random.default_rng(42)
    K, S = 20, 4000
    w = ev[:K] / ev[:K].sum()
    ns = [50, 60, 72, 100, 150, 200, 250, 300]
    fam = [0.04, 0.09, 0.16, 0.25]
    P = {}
    for R2 in fam:
        for N in ns:
            r2h = simular_r2hat(N, R2, S, w, y_pool, rng)
            for u in UMBRALES:
                P[(R2, N, u)] = float(
                    (r2h > r2_critico_reg(N, K, u, RSC)).mean())
    fig3d(P, fam, "$R^2$", ns, "Sample size (N)", 60, "N = 60\n(proposal)",
          f"Cohort 1 in 3D: global-model {BFN} over sample size and "
          "threshold (K = 20)\npermutation-marginalized conditional MVG, "
          f"Y = {ydesc}",
          f"Surfaces: P({BFN} above the threshold) per global effect $R^2$ "
          "· JZS regression BF vs the intercept-only null (rscale = √2/4) · "
          f"4,000 datasets per cell · vertices at {BFN} = 3 and 10",
          outdir / "Fig_3D_BF_v4_c1.png",
          f"P(global {BFN} > threshold)")


def v4_c2(pm25, outdir):
    ev = pd.read_csv(W / "calibracion_dkt_eigenvalues.csv"
                     )["eigenvalue"].to_numpy()
    hname = "PM$_{2.5}$ exposure" if pm25 else "hormonal PC1"
    rng = np.random.default_rng(42)
    K, S = 5, 4000
    w = ev[:K] / ev[:K].sum()
    ns_tot = [40, 48, 60, 72, 84, 100, 120, 160]
    fam = [0.5, 0.65, 0.70, 0.8, 1.0]
    P = {}
    for D in fam:
        R2 = D * D / (4 + D * D)
        for N in ns_tot:
            r2h = simular_r2hat_c2(N, R2, S, w, rng, 0.0)
            for u in UMBRALES:
                P[(D, N // 2, u)] = float(
                    (r2h > r2_critico_reg(N, K + 1, u, RSC)).mean())
    xs = [n // 2 for n in ns_tot]
    fig3d(P, fam, "D", xs, "n per group", 36, "36/group",
          f"Cohort 2 in 3D: multivariate MCI-HC recovery over n and "
          f"threshold (K = 5)\nglobal-model {BFN}, brain PCs + {hname}, "
          "separation D (Mahalanobis)",
          f"Surfaces: P(global {BFN} above the threshold) per multivariate "
          "separation D ($R^2$ = D$^2$/(4+D$^2$)) · JZS regression BF vs "
          "null (rscale = √2/4) · 4,000 datasets per cell · vertices at "
          f"{BFN} = 3 and 10",
          outdir / "Fig_3D_BF_v4_c2.png",
          f"P(global {BFN} > threshold)")


if __name__ == "__main__":
    for pm25, outdir in [(False, OUT_H), (True, OUT_P)]:
        print("PM2.5" if pm25 else "Hormonal", "arm:")
        lasso_c1(pm25, outdir)
        lasso_c2(pm25, outdir)
        v4_c1(pm25, outdir)
        v4_c2(pm25, outdir)
    blm.C_EDAD_Y, blm.C_H_AGE_POST = 0.17, -0.015
    print("listo")
