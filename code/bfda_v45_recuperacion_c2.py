#!/usr/bin/env python3
"""BFDA v4.5: probabilidad de RECUPERAR el patrón multivariado MCI-HC.

Extensión de la Versión 4 (MVG condicional con permutación marginalizada,
BF global de regresión) centrada en la Cohorte 2 a n fijo: ¿con qué
probabilidad el criterio global detecta una separación multivariada
equivalente a d = 0.65-0.70 con n = 36 por grupo?  Barre D (Mahalanobis),
K = {5, 10, 14, 20} y umbrales BF = {4, 6, 10}.

D se traduce a R2 cerebral global mediante R2 = D^2 / (4 + D^2)
(la inversa de D = 2 sqrt(R2 / (1 - R2))).
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bfda_v4_condicional_mvg import (simular_r2hat_c2, r2_critico_reg,
                                     COL, _estilo)

GRID_D = [0.5, 0.65, 0.70, 0.8, 1.0]
GRID_N = [40, 48, 60, 72, 84, 100, 120, 160, 200, 240, 280, 320]  # N TOTAL
KS = [5, 10, 14, 20]
TAUS = [4, 6, 10]
N_FOCAL = 72                                             # 36 por grupo


def d_a_r2(D):
    return D * D / (4.0 + D * D)


def cargar_eigen():
    base = Path(__file__).resolve().parent
    for cand in (base / "calibracion_dkt_eigenvalues.csv",
                 base.parent / "data" / "calibracion_dkt_eigenvalues.csv"):
        if cand.exists():
            return pd.read_csv(cand)["eigenvalue"].to_numpy()
    raise FileNotFoundError("calibracion_dkt_eigenvalues.csv")


def n_requerido(largo, target, filtro):
    g = largo.query(filtro).sort_values("N")
    p, nn = g.power.to_numpy(), g.N.to_numpy()
    ok = np.where(p >= target)[0]
    if len(ok) == 0:
        return np.nan
    i = ok[0]
    if i == 0:
        return float(nn[0])
    return float(np.ceil(nn[i - 1] + (target - p[i - 1])
                         * (nn[i] - nn[i - 1]) / (p[i] - p[i - 1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--rscale", type=float, default=float(np.sqrt(2) / 4))
    ap.add_argument("--outdir", type=str, default=None)
    cfg = ap.parse_args()

    base = Path(__file__).resolve().parent
    if cfg.outdir:
        outdir = Path(cfg.outdir)
    elif (base / "resultados_bfda").exists():
        outdir = base / "resultados_bfda"
    else:
        outdir = base.parent / "results"
    outdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    ev = cargar_eigen()

    print(f"[v4.5] D -> R2: " + ", ".join(
        f"{D:.2f}->{d_a_r2(D):.4f}" for D in GRID_D) +
        f" · M={cfg.nsims:,}/celda · rscale={cfg.rscale:.4f}")

    filas, t0 = [], time.time()
    for K in KS:
        w = ev[:K] / ev[:K].sum()
        p = K + 1                                    # + PC1 hormonal
        for N in GRID_N:
            if N <= p + 5:
                continue
            crits = {t: r2_critico_reg(N, p, t, cfg.rscale) for t in TAUS}
            for D in [0.0] + GRID_D:
                for dh in ([0.0, 0.3] if (N == N_FOCAL and K == 5)
                           else [0.0]):
                    R2 = d_a_r2(D)
                    r2h = simular_r2hat_c2(N, R2, cfg.nsims, w, rng, dh)
                    fila = {"K": K, "N": N, "n_por_grupo": N // 2, "D": D,
                            "R2": R2, "d_h": dh}
                    for t in TAUS:
                        fila[f"power_bf{t}"] = np.nan if np.isnan(crits[t]) \
                            else float((r2h > crits[t]).mean())
                    filas.append(fila)
        print(f"  K={K:>2} listo ({time.time()-t0:,.0f}s)", flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(outdir / "resultados_v45_grid.csv", index=False)
    largo = df.melt(id_vars=["K", "N", "n_por_grupo", "D", "R2", "d_h"],
                    value_vars=[f"power_bf{t}" for t in TAUS],
                    var_name="tau", value_name="power")
    largo["tau"] = largo.tau.str.replace("power_bf", "").astype(int)

    # ---- resumen focal y n* -----------------------------------------------
    focal = {}
    for K in KS:
        for D in GRID_D:
            for t in TAUS:
                g = largo.query(f"K=={K} and N=={N_FOCAL} and D=={D} "
                                f"and tau=={t} and d_h==0.0")
                focal[f"K{K}_D{D}_BF{t}"] = float(g.power.iloc[0])
    nreq = {}
    for K in KS:
        for D in GRID_D:
            for t in TAUS:
                v = n_requerido(largo, cfg.target,
                                f"K=={K} and D=={D} and tau=={t} and d_h==0.0")
                nreq[f"K{K}_D{D}_BF{t}"] = None if np.isnan(v) else v
    fp = {f"K{K}_BF{t}": float(largo.query(
            f"K=={K} and N=={N_FOCAL} and D==0.0 and tau=={t} and d_h==0.0"
          ).power.iloc[0]) for K in KS for t in TAUS}
    resumen = {"criterio": "BF global de regresion (JZS, rscale sqrt(2)/4) "
                           "vs nulo; y=grupo; X = K PCs cerebro (espectro "
                           "permutado) + PC1 hormonal",
               "map_D_a_R2": {str(D): round(d_a_r2(D), 4) for D in GRID_D},
               "n_focal_total": N_FOCAL, "M": cfg.nsims,
               "power_en_36_por_grupo": focal,
               "n_TOTAL_requerido": nreq,
               "fp_nulo_en_36_por_grupo": fp}
    (outdir / "resumen_v45.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False))

    # ---- figura ------------------------------------------------------------
    DCOL = {0.5: COL["gris"], 0.65: COL["verde"], 0.70: COL["azul"],
            0.8: COL["morado"], 1.0: COL["rojo"]}
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6), dpi=220,
                             facecolor="white")
    ax = axes[0, 0]
    for D in GRID_D:
        g = largo.query(f"K==5 and tau==10 and D=={D} and d_h==0.0"
                        ).sort_values("N")
        ax.plot(g.N / 2, g.power, color=DCOL[D], lw=2.4, marker="o", ms=4,
                label=f"D = {D:.2f}")
    g0 = largo.query("K==5 and tau==10 and D==0.0 and d_h==0.0").sort_values("N")
    ax.plot(g0.N / 2, g0.power, color=COL["tinta"], lw=1.4, ls=":",
            label="global null")
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.6)
    ax.axvline(36, color=COL["rojo"], lw=1.1, ls=":", alpha=0.8)
    ax.text(36.8, 0.03, "36/group", fontsize=8.5, color=COL["rojo"])
    ax.set_xlabel("n per group"); ax.set_ylabel("P(BF$_{H_1/H_0}$ ≥ 10)")
    ax.set_ylim(0, 1.02)
    ax.set_title("A. Recovery of the multivariate MCI-HC pattern (K = 5)\n"
                 "global-model BF ≥ 10, by Mahalanobis separation D",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax); ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[0, 1]
    x = np.arange(len(KS)); wbar = 0.38
    for off, D, c in [(-wbar/2, 0.65, COL["verde"]), (wbar/2, 0.70, COL["azul"])]:
        v = [focal[f"K{K}_D{D}_BF10"] for K in KS]
        ax.bar(x + off, v, wbar, color=c, alpha=0.9, label=f"D = {D:.2f}")
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", fontsize=8)
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.6)
    ax.set_xticks(x, [f"K = {K}" for K in KS])
    ax.set_ylabel("P(BF$_{H_1/H_0}$ ≥ 10) at 36/group"); ax.set_ylim(0, 1.02)
    ax.set_title("B. Recovery at n = 36 per group, by number of components\n"
                 "(BF ≥ 10, no hormonal group effect)", loc="left", pad=10,
                 fontweight="bold")
    _estilo(ax); ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    for t, c in zip(TAUS, (COL["verde"], COL["azul"], COL["morado"])):
        v = [focal[f"K5_D{D}_BF{t}"] for D in GRID_D]
        ax.plot(GRID_D, v, color=c, lw=2.4, marker="o", ms=5,
                label=f"BF$_{{H_1/H_0}}$ ≥ {t}")
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.6)
    ax.axvspan(0.65, 0.70, color=COL["rojo"], alpha=0.10)
    ax.set_xlabel("Multivariate separation D (Mahalanobis)")
    ax.set_ylabel("Power at 36/group (K = 5)"); ax.set_ylim(0, 1.02)
    ax.set_title("C. Evidence thresholds at n = 36 per group (K = 5)\n"
                 "shaded band: requested scenario D = 0.65-0.70", loc="left",
                 pad=10, fontweight="bold")
    _estilo(ax); ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[1, 1]
    kcol = {5: COL["verde"], 10: COL["azul"], 14: "#D97706", 20: COL["morado"]}
    for K in KS:
        g = largo.query(f"K=={K} and tau==4 and D==0.0 and d_h==0.0"
                        ).sort_values("N")
        ax.plot(g.N / 2, g.power, color=kcol[K], lw=2.2, label=f"K = {K}")
    ax.set_xlabel("n per group"); ax.set_ylabel("P(BF$_{H_1/H_0}$ ≥ 4 | null)")
    ax.set_ylim(0, 0.12)
    ax.set_title("D. False-positive rate under the global null (BF ≥ 4)",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax); ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Recovery of the MCI-HC multivariate pattern "
                 "(global-model Bayes factor)", y=0.995, fontsize=14.5,
                 fontweight="bold", color=COL["tinta"])
    fig.text(0.5, 0.012, "y = balanced group indicator · X = brain PCs "
             "(empirical DKT spectrum, permuted every iteration) + hormonal "
             "PC1 · R$^2$ = D$^2$/(4+D$^2$) · JZS regression BF vs null "
             f"(rscale = √2/4) · M = {cfg.nsims:,}/cell · seed {cfg.seed}",
             ha="center", fontsize=8.4, color=COL["gris"])
    plt.tight_layout()
    fig.savefig(outdir / "Fig_v45_recuperacion.png", bbox_inches="tight",
                facecolor="white")

    print("\n===== v4.5: poder en 36/grupo (d_h = 0) =====")
    for K in KS:
        for D in (0.65, 0.70):
            vals = " | ".join(f"BF{t}: {focal[f'K{K}_D{D}_BF{t}']:.3f}"
                              for t in TAUS)
            print(f"  K={K:>2} D={D:.2f}: {vals}")
    print(f"Resultados en {outdir}/")


if __name__ == "__main__":
    main()
