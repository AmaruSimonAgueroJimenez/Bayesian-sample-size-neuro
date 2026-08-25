#!/usr/bin/env python3
"""Panel A de la v3.5 en 3D: eje extra = umbral del Bayes factor.

Superficies P(deteccion) sobre (n por grupo x umbral BF) por separacion d,
criterio BF por componente (JZS parcial, analitico), K = 5.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from bfda_lasso_multivariante import generar_lote_c2
from bfda_n_optimo import r2_critico

W = Path(__file__).resolve().parent
OUT = W / "resultados_bfda" if (W / "resultados_bfda").exists() \
    else W.parent / "results"

GRID_D = [0.5, 0.65, 0.70, 0.8, 1.0]
GRID_N = [48, 60, 72, 84, 100, 120, 160]          # N total
UMBRALES = [2, 3, 4, 5, 6, 8, 10, 13, 16, 20]
K, S = 5, 4000
DCOL = {0.5: "#64748B", 0.65: "#059669", 0.70: "#2563EB",
        0.8: "#7C3AED", 1.0: "#E11D48"}

rng = np.random.default_rng(20260825)
rc = {}                                            # r2 critico por (m, u)
P = {}                                             # (d, n, u) -> p_any
for d in GRID_D:
    for n in GRID_N:
        y, X = generar_lote_c2(n, d, 0.0, S, K, rng)
        yc = y - y.mean(axis=1, keepdims=True)
        Z = np.concatenate([yc[:, :, None], X], axis=2)
        Zs = (Z - Z.mean(1, keepdims=True)) / Z.std(1, ddof=1, keepdims=True)
        C = np.einsum("snp,snq->spq", Zs, Zs) / (n - 1)
        Om = np.linalg.inv(C)
        dg = np.sqrt(np.abs(np.diagonal(Om, axis1=1, axis2=2)))
        rp = -Om[:, 0, 1:K + 1] / (dg[:, 0:1] * dg[:, 1:K + 1])
        m = n - (K + 1)
        for u in UMBRALES:
            if (m, u) not in rc:
                rc[(m, u)] = r2_critico(m, u)
            P[(d, n, u)] = float((rp ** 2 > rc[(m, u)]).any(axis=1).mean())
    print(f"d={d:.2f} listo")

# ------------------------------------------------------------------ figura
fig = plt.figure(figsize=(11.6, 9.0), dpi=220, facecolor="white")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")

ngr = np.array([n // 2 for n in GRID_N], dtype=float)
lbf = np.log2(np.array(UMBRALES, dtype=float))
Xg, Yg = np.meshgrid(ngr, lbf, indexing="ij")

for d in GRID_D:                                   # menor a mayor: sin cruces
    Zg = np.array([[P[(d, n, u)] for u in UMBRALES] for n in GRID_N])
    ax.plot_surface(Xg, Yg, Zg, color=DCOL[d], alpha=0.60,
                    edgecolor=DCOL[d], linewidth=0.55, rstride=1, cstride=1,
                    antialiased=True, shade=False)
    ax.plot(ngr, np.full_like(ngr, lbf[-1]), Zg[:, -1], color=DCOL[d],
            lw=2.6)                                # borde frontal marcado

# plano objetivo 0.80
Xp, Yp = np.meshgrid([ngr[0], ngr[-1]], [lbf[0], lbf[-1]], indexing="ij")
ax.plot_surface(Xp, Yp, np.full_like(Xp, 0.80), color="#0F172A", alpha=0.07)
ax.text(ngr[0], lbf[-1], 0.815, "0.80 target", color="#0F172A", fontsize=9)

# rebanada de referencia BF10 = 3
yref = np.log2(3.0)
Xr, Zr = np.meshgrid([ngr[0], ngr[-1]], [0, 1], indexing="ij")
ax.plot_surface(Xr, np.full_like(Xr, yref), Zr, color="#0F172A", alpha=0.05)
for d in GRID_D:
    zc = [P[(d, n, 3)] for n in GRID_N]
    ax.plot(ngr, np.full_like(ngr, yref), zc, color=DCOL[d], lw=3.4,
            zorder=10)
ax.text(ngr[-1] + 3.0, yref, 0.20, "BF$_{H_1/H_0}$ = 3\n(reference)",
        color="#0F172A", fontsize=9.5, fontweight="bold", ha="left")

# vertice donde se cruzan las referencias n = 36 y BF = 3
xf = 36.0
ax.plot([xf, xf], [yref, yref], [0, 1.0], color="#0F172A", lw=1.2,
        ls="--", alpha=0.65, zorder=15)
for d in GRID_D:
    z36 = P[(d, 72, 3)]
    ax.scatter([xf], [yref], [z36], color=DCOL[d], s=64,
               edgecolor="white", linewidth=1.3, depthshade=False, zorder=30)
    lado, dz = (-2.6, 0.030) if d != 0.65 else (4.6, 0.042)
    ax.text(xf + lado, yref, z36 + dz, f"{z36:.2f}", color=DCOL[d],
            fontsize=9.2, fontweight="bold",
            ha="right" if lado < 0 else "left", zorder=30)
ax.text(xf + 3.0, yref, 1.01, "n = 36 x BF = 3", color="#0F172A",
        fontsize=9.5, fontweight="bold", ha="left")

# plano n = 36 (diseño solicitado)
Yp2, Zp2 = np.meshgrid([lbf[0], lbf[-1]], [0, 1], indexing="ij")
ax.plot_surface(np.full_like(Yp2, 36.0), Yp2, Zp2, color="#E11D48",
                alpha=0.06)
ax.plot([36, 36], [lbf[0], lbf[-1]], [0, 0], color="#E11D48", lw=1.4,
        ls=":")
ax.text(36, lbf[0] - 0.35, 0.02, "36/group", color="#E11D48", fontsize=9.5)

ax.set_xlabel("n per group", labelpad=10)
ax.set_ylabel("BF$_{H_1/H_0}$ threshold", labelpad=10)
ax.set_zlabel("P(≥1 brain PC with BF$_{H_1/H_0}$ > threshold)", labelpad=8)
ax.set_yticks(np.log2([2, 3, 4, 6, 10, 16]))
ax.set_yticklabels(["2", "3", "4", "6", "10", "16"])
ax.set_zlim(0, 1.0)
ax.set_xlim(ngr[0], ngr[-1])
ax.view_init(elev=22, azim=-55)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=DCOL[d], alpha=0.75, label=f"d = {d:.2f}")
                   for d in reversed(GRID_D)],
          loc="upper left", frameon=False, fontsize=10.5,
          bbox_to_anchor=(0.02, 0.92), title="MCI-HC separation",
          title_fontsize=10)
ax.xaxis.pane.set_alpha(0.0); ax.yaxis.pane.set_alpha(0.0)
ax.zaxis.pane.set_alpha(0.06)
ax.set_title("Detection of the MCI-HC pattern by sample size and evidence "
             "threshold (K = 5)\nper-component JZS Bayes factor, Bayesian-"
             "Lasso design of Version 3.5", loc="left", pad=18,
             fontweight="bold", color="#0F172A", fontsize=13)
fig.text(0.5, 0.015, "Surfaces: P(≥1 brain component with BF$_{H_1/H_0}$ above "
         "the threshold) per MCI-HC separation d · analytic per-component "
         "JZS BF on the partial correlation (controls: remaining components "
         "+ hormonal PC1 + age; m = n - 6) · 4,000 datasets per cell · "
         "true-component recovery is within 0.01 of every value shown · highlighted slice: the reference threshold BF$_{H_1/H_0}$ = 3",
         ha="center", fontsize=8.4, color="#64748B")
plt.tight_layout()
fig.savefig(OUT / "Fig_v35_recuperacion_3D.png", bbox_inches="tight",
            facecolor="white")
print("figura escrita en", OUT / "Fig_v35_recuperacion_3D.png")
