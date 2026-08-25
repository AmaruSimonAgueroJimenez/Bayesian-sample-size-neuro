#!/usr/bin/env python3
"""BFDA v3.5: probabilidad de RECUPERAR el patrón MCI-HC bajo el pipeline
Lasso del laboratorio, a n fijo.

Extensión de la Versión 3 centrada en la Cohorte 2: un componente cerebral
verdadero separa MCI de HC con d de Cohen (equivalente multivariado del
efecto), y se pregunta con qué probabilidad, a n = 36 por grupo, (i) algún
componente cerebral resulta significativo (detección) y (ii) el componente
VERDADERO resulta significativo (recuperación del patrón).  Barre
d = {0.5, 0.65, 0.70, 0.8, 1.0}, K = {5, 10, 14, 20} y, como criterio
complementario, el BF JZS por componente con umbrales {4, 6, 10}.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bfda_lasso_multivariante import (generar_lote_c2, evaluar_celda_c2,
                                      COL, _estilo)
from bfda_n_optimo import r2_critico

GRID_D = [0.5, 0.65, 0.70, 0.8, 1.0]
GRID_N = [48, 60, 72, 84, 100, 120, 160]        # N TOTAL (balanceado)
KS = [5, 10, 14, 20]
TAUS = (4, 6, 10)
N_FOCAL = 72                                    # 36 por grupo


def bf_criterio_c2(n, d, S, K, rng, umbrales=TAUS, d_h=0.0):
    """P(componente cerebral con BF10 > u), analitico, disenio Cohorte 2.

    BF por componente = JZS sobre la correlacion PARCIAL del componente j
    con el grupo, controlando por el resto de componentes + PC1 hormonal +
    edad (k = K + 1 controles; m = n - k)."""
    y, X = generar_lote_c2(n, d, d_h, S, K, rng)
    yc = y - y.mean(axis=1, keepdims=True)
    Z = np.concatenate([yc[:, :, None], X], axis=2)          # (S, n, K+3)
    Zs = (Z - Z.mean(1, keepdims=True)) / Z.std(1, ddof=1, keepdims=True)
    C = np.einsum("snp,snq->spq", Zs, Zs) / (n - 1)
    Om = np.linalg.inv(C)
    dg = np.sqrt(np.abs(np.diagonal(Om, axis1=1, axis2=2)))
    rp = -Om[:, 0, 1:K + 1] / (dg[:, 0:1] * dg[:, 1:K + 1])  # (S, K)
    m = n - (K + 1)
    filas = []
    for u in umbrales:
        rc = r2_critico(m, u)
        if np.isnan(rc):
            p_any = p_true = np.nan
        else:
            hit = rp ** 2 > rc
            p_any = float(hit.any(axis=1).mean())
            p_true = float(hit[:, 1].mean()) if d > 0 else np.nan
        filas.append({"K": K, "n_total": n, "n_por_grupo": n // 2, "d": d,
                      "d_h": d_h, "umbral_bf": u, "p_any": p_any,
                      "p_true": p_true})
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsims", type=int, default=400)
    ap.add_argument("--nsims-bf", type=int, default=4000)
    ap.add_argument("--n-iter", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=800)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--target", type=float, default=0.80)
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

    # ---------- parte A: criterio Lasso-CrI (Gibbs exacto) ----------
    celdas = [(72, d, 5, 0.0) for d in [0.0] + GRID_D]           # focal K=5
    celdas += [(n, d, 5, 0.0) for n in GRID_N if n != 72
               for d in [0.0] + GRID_D]                          # curvas K=5
    celdas += [(72, d, K, 0.0) for K in (10, 14, 20)
               for d in (0.0, 0.65, 0.70)]                       # K en focal
    celdas += [(72, d, 5, 0.3) for d in (0.65, 0.70)]            # d_h = 0.3

    filas, t0 = [], time.time()
    for i, (n, d, K, dh) in enumerate(celdas, 1):
        p_any, p_true, p_horm = evaluar_celda_c2(
            n, d, dh, cfg.nsims, K, rng, cfg.n_iter, cfg.burn)
        filas.append({"K": K, "n_total": n, "n_por_grupo": n // 2, "d": d,
                      "d_h": dh, "p_any": p_any, "p_true": p_true,
                      "p_horm": p_horm})
        print(f"[v3.5 {i:>2}/{len(celdas)}] K={K:>2} n={n:>3} d={d:.2f} "
              f"dh={dh:.1f}: P(≥1)={p_any:.3f} P(true)="
              f"{'nan' if np.isnan(p_true) else f'{p_true:.3f}'} "
              f"({time.time()-t0:,.0f}s)", flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(outdir / "resultados_v35_lasso.csv", index=False)

    # ---------- parte B: criterio BF por componente (analitico) ----------
    filas_bf = []
    for K in KS:
        for n in GRID_N:
            for d in [0.0] + GRID_D:
                filas_bf += bf_criterio_c2(n, d, cfg.nsims_bf, K, rng)
    df_bf = pd.DataFrame(filas_bf)
    df_bf.to_csv(outdir / "resultados_v35_bf.csv", index=False)
    print(f"[v3.5] criterio BF analitico listo ({time.time()-t0:,.0f}s)")

    # ---------- resumen ----------
    def celda(K, n, d, dh=0.0):
        g = df.query(f"K=={K} and n_total=={n} and d=={d} and d_h=={dh}")
        return g.iloc[0]

    def nreq(K, d, col="p_any"):
        g = df.query(f"K=={K} and d=={d} and d_h==0.0").sort_values("n_total")
        if len(g) < 2:
            return None
        p, nn = g[col].to_numpy(), g.n_total.to_numpy()
        ok = np.where(p >= cfg.target)[0]
        if len(ok) == 0:
            return None
        i = ok[0]
        if i == 0:
            return float(nn[0])
        return float(np.ceil(nn[i-1] + (cfg.target - p[i-1])
                             * (nn[i] - nn[i-1]) / (p[i] - p[i-1])))

    resumen = {
        "criterio": "Lasso bayesiano (y=grupo ~ K PCs cerebro + PC1 hormonal "
                    "+ edad); deteccion = >=1 PC cerebral con CrI95 "
                    "excluyendo 0; recuperacion = el PC verdadero "
                    "significativo",
        "n_focal_por_grupo": 36, "M_lasso": cfg.nsims, "M_bf": cfg.nsims_bf,
        "recuperacion_en_36_por_grupo": {
            f"K{K}_d{d}": {"p_any": float(celda(K, 72, d).p_any),
                           "p_true": float(celda(K, 72, d).p_true)}
            for K in KS for d in ((GRID_D) if K == 5 else (0.65, 0.70))},
        "fw_nulo_en_36_por_grupo": {
            f"K{K}": float(celda(K, 72, 0.0).p_any) for K in KS},
        "n_TOTAL_requerido_K5": {str(d): nreq(5, d) for d in GRID_D},
        "n_TOTAL_requerido_K5_true": {str(d): nreq(5, d, "p_true")
                                      for d in GRID_D},
        "dh03_en_36_por_grupo_K5": {
            str(d): {"p_any": float(celda(5, 72, d, 0.3).p_any),
                     "p_true": float(celda(5, 72, d, 0.3).p_true),
                     "p_horm": float(celda(5, 72, d, 0.3).p_horm)}
            for d in (0.65, 0.70)},
    }
    (outdir / "resumen_v35.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False))

    # ---------- figura ----------
    DCOL = {0.5: COL["gris"], 0.65: COL["verde"], 0.70: COL["azul"],
            0.8: COL["morado"], 1.0: COL["rojo"]}
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6), dpi=220,
                             facecolor="white")
    ax = axes[0, 0]
    for d in GRID_D:
        g = df.query(f"K==5 and d=={d} and d_h==0.0").sort_values("n_total")
        ax.plot(g.n_por_grupo, g.p_any, color=DCOL[d], lw=2.4, marker="o",
                ms=4, label=f"d = {d:.2f}")
        ax.plot(g.n_por_grupo, g.p_true, color=DCOL[d], lw=1.4, ls="--",
                alpha=0.8)
    g0 = df.query("K==5 and d==0.0 and d_h==0.0").sort_values("n_total")
    ax.plot(g0.n_por_grupo, g0.p_any, color=COL["tinta"], lw=1.3, ls=":",
            label="null (family-wise)")
    ax.plot([], [], color=COL["gris"], lw=1.4, ls="--",
            label="true component (dashed)")
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.6)
    ax.axvline(36, color=COL["rojo"], lw=1.1, ls=":", alpha=0.8)
    ax.text(36.6, 0.02, "36/group", fontsize=8.5, color=COL["rojo"])
    ax.set_xlabel("n per group")
    ax.set_ylabel("Probability"); ax.set_ylim(0, 1.02)
    ax.set_title("A. Detection (solid) and pattern recovery (dashed), K = 5\n"
                 "Bayesian Lasso, ≥1 brain PC with 95% CrI excluding 0",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax); ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    ax = axes[0, 1]
    x = np.arange(len(KS)); wbar = 0.2
    for j, (d, c) in enumerate([(0.65, COL["verde"]), (0.70, COL["azul"])]):
        va = [float(celda(K, 72, d).p_any) for K in KS]
        vt = [float(celda(K, 72, d).p_true) for K in KS]
        ax.bar(x + (2*j - 1.5) * wbar, va, wbar, color=c, alpha=0.95,
               label=f"any, d = {d:.2f}")
        ax.bar(x + (2*j - 0.5) * wbar, vt, wbar, color=c, alpha=0.45,
               label=f"true, d = {d:.2f}")
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.6)
    ax.set_xticks(x, [f"K = {K}" for K in KS])
    ax.set_ylabel("Probability at 36/group"); ax.set_ylim(0, 1.02)
    ax.set_title("B. Recovery at n = 36 per group, by number of components",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax); ax.legend(frameon=False, fontsize=8.4, ncol=2)

    ax = axes[1, 0]
    for u, c in zip(TAUS, (COL["verde"], COL["azul"], COL["morado"])):
        g = df_bf.query(f"K==5 and n_total==72 and umbral_bf=={u} and d>0"
                        ).sort_values("d")
        ax.plot(g.d, g.p_true, color=c, lw=2.4, marker="o", ms=5,
                label=f"BF$_{{H_1/H_0}}$ > {u}")
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.6)
    ax.axvspan(0.65, 0.70, color=COL["rojo"], alpha=0.10)
    ax.set_xlabel("MCI-HC separation d (true component)")
    ax.set_ylabel("P(true component BF > threshold)"); ax.set_ylim(0, 1.02)
    ax.set_title("C. Per-component Bayes-factor criterion at 36/group (K = 5)\n"
                 "shaded band: requested scenario d = 0.65-0.70", loc="left",
                 pad=10, fontweight="bold")
    _estilo(ax); ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[1, 1]
    kcol = {5: COL["verde"], 10: COL["azul"], 14: "#D97706", 20: COL["morado"]}
    for K in KS:
        g = df_bf.query(f"K=={K} and umbral_bf==4 and d==0.0"
                        ).sort_values("n_total")
        ax.plot(g.n_por_grupo, g.p_any, color=kcol[K], lw=2.0, ls="--",
                alpha=0.9, label=f"BF > 4, K = {K}")
    fw = [float(celda(K, 72, 0.0).p_any) for K in KS]
    ax.scatter([36]*len(KS), fw, color=[kcol[K] for K in KS], marker="s",
               s=42, zorder=6, label="Lasso-CrI at 36/group")
    ax.set_xlabel("n per group")
    ax.set_ylabel("Family-wise rate under the null"); ax.set_ylim(0, 0.35)
    ax.set_title("D. Specificity: false detection under the global null",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax); ax.legend(frameon=False, fontsize=8.2, ncol=2)

    fig.suptitle("Recovery of the MCI-HC pattern under the "
                 "laboratory Lasso pipeline", y=0.995, fontsize=14.5,
                 fontweight="bold", color=COL["tinta"])
    fig.text(0.5, 0.012, "y = group ~ K brain PCs + hormonal PC1 + age · "
             "one true discriminating component (Cohen d) · significance = "
             "two-sided 95% CrI · per-component JZS BF on the partial "
             f"correlation (m = n - K - 1) · {cfg.nsims} Gibbs datasets/cell, "
             f"{cfg.nsims_bf:,} for the BF criterion · seed {cfg.seed}",
             ha="center", fontsize=8.2, color=COL["gris"])
    plt.tight_layout()
    fig.savefig(outdir / "Fig_v35_recuperacion.png", bbox_inches="tight",
                facecolor="white")
    print(f"Resultados en {outdir}/")


if __name__ == "__main__":
    main()
