#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bfda_v4_condicional_mvg.py
================================================================================
VERSIÓN 4 — Bayesian Sample Size Estimation via Permutation-Marginalized
Conditional Multivariate Gaussian (paradigma de L. Luarte).

Formulación:
  * Y = PC1 hormonal empírico ESTANDARIZADO (pool SWAN Visita 10, early
    perimenopause por defecto); en cada iteración se re-muestrea con
    reemplazo (bootstrap) -> y_emp (N x 1).
  * X = p componentes principales cerebrales con espectro de autovalores
    EMPÍRICO (DKT espesor cortical, 40-55 años). Pesos normalizados
        w_i = lambda_i / sum(lambda), sum w_i = 1.
  * Marginalización de la alineación: en cada iteración m se permutan los
    pesos, w^(m) = pi(w), pi ~ Uniform(P_p)  (nuisance integrado en M).
  * Efecto global R^2 fijo: rho = sqrt(w^(m) * R^2), ||rho||^2 = R^2;
    varianza residual determinista sigma_eps^2 = 1 - R^2.
  * Generador condicional (preserva Cov(X) = I_p marginal):
        X | Y=y ~ N_p(y * rho, I_p - rho rho^T)
        X_sim = y_emp rho^T + Z L^T,  L L^T = I_p - rho rho^T (Cholesky).
  * Evidencia: BF10 del modelo de regresión completo vs nulo (solo
    intercepto), JZS/Liang et al. 2008 — equivalente exacto de
    BayesFactor::regressionBF en R, con rscale = sqrt(2)/4 ("medium"):
        BF10 = Int (1+g)^((N-p-1)/2) [1+(1-R2_hat) g]^(-(N-1)/2) pi(g) dg,
        pi(g) = (N s^2/2)^(1/2)/Gamma(1/2) g^(-3/2) exp(-N s^2/(2g)).
  * Potencia(N, R^2) = (1/M) sum I(BF10^(m) >= tau);
    n* = min N con potencia >= 0.80.

Exploración solicitada: K = p en {5, 10, 14, 20} (14 = 90% de varianza DKT)
y umbrales tau en {4, 6, 10}. Nota analítica: para el BF GLOBAL, la potencia
es invariante a la permutación de w (rotación-invariancia gaussiana: solo
importa ||rho||^2 = R^2); la permutación se ejecuta igualmente por fidelidad
a la especificación y su invariancia se verifica empíricamente.

Aceleración: BF10 es monótono en R2_hat para (N, p) fijos, así que
P(BF>=tau) = P(R2_hat > R2_crit(N, p, tau)); la raíz se busca una vez por
celda y la potencia se evalúa sobre los R2_hat simulados (OLS por lotes).

Uso:  python bfda_v4_condicional_mvg.py            (~2 min)
Requiere: calibracion_swan_pc1.csv, calibracion_dkt_eigenvalues.csv.
Implementación R de referencia: bfda_v4_condicional_mvg.R (misma lógica,
BayesFactor::regressionBF).

Autores: Luis Luarte (principal), Amaru Agüero. Agosto 2026.
================================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, special

COL = {"azul": "#2563EB", "verde": "#059669", "morado": "#8B5CF6",
       "rojo": "#E11D48", "gris": "#64748B", "gris_claro": "#CBD5E1",
       "tinta": "#0F172A"}
R2_COLORS = {0.04: "#94A3B8", 0.09: "#059669", 0.16: "#2563EB",
             0.25: "#E11D48"}
TAU_COLORS = {4: "#059669", 6: "#2563EB", 10: "#8B5CF6"}

_GLX, _GLW = np.polynomial.legendre.leggauss(400)
_U = 0.5 * 60.0 * _GLX
_WU = 0.5 * 60.0 * _GLW
_G = np.exp(_U)


def _estilo(ax):
    ax.grid(color="#E2E8F0", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COL["gris_claro"])


# ================================= BF JZS del modelo completo vs nulo (Liang)
def log_bf10_reg(r2, N, p, rscale=np.sqrt(2) / 4):
    """log BF10 de la regresión con p covariables vs modelo nulo.

    Equivalente a BayesFactor::regressionBF (linearReg.R2stat) con
    rscale='medium' = sqrt(2)/4. Vectorizado sobre r2.
    """
    r2 = np.atleast_1d(np.asarray(r2, float))[:, None]
    ns2 = N * rscale ** 2
    log_prior_jac = (0.5 * np.log(ns2 / 2.0) - special.gammaln(0.5)
                     - 0.5 * _U - ns2 / (2.0 * _G))
    logf = (log_prior_jac
            + ((N - p - 1.0) / 2.0) * np.log1p(_G)
            - ((N - 1.0) / 2.0) * np.log1p((1.0 - r2) * _G))
    return special.logsumexp(logf + np.log(_WU), axis=1)


def r2_critico_reg(N, p, tau, rscale=np.sqrt(2) / 4):
    logk = np.log(tau)
    f = lambda r2: log_bf10_reg(r2, N, p, rscale)[0] - logk
    if f(0.999) < 0:
        return np.nan
    if f(1e-12) >= 0:
        return 0.0
    return optimize.brentq(f, 1e-12, 0.999, xtol=1e-12)


# ======================================================= generador condicional
def simular_r2hat(N, R2, S, w, y_pool, rng, permutar=True, alloc="spectral"):
    """R2_hat OLS en S datasets del generador condicional MVG.

    alloc = "spectral"     -> w^(m) = pi(w), perfil espectral empirico
    alloc = "concentrated" -> todo el efecto en UN componente al azar
                              (control de la invariancia rotacional).
    """
    p = len(w)
    y = rng.choice(y_pool, size=(S, N), replace=True)          # bootstrap Y
    y = y - y.mean(axis=1, keepdims=True)

    if alloc == "concentrated":                 # todo el efecto en 1 comp.
        w_perm = np.zeros((S, p))
        w_perm[np.arange(S), rng.integers(0, p, size=S)] = 1.0
    elif permutar:                              # w^(m) = pi(w) por dataset
        w_perm = rng.permuted(np.tile(w, (S, 1)), axis=1)
    else:
        w_perm = np.tile(w, (S, 1))
    rho = np.sqrt(w_perm * R2)                                  # (S, p)

    # X = y rho^T + Z L^T con L L^T = I - rho rho^T
    # (I - rho rho^T)^{1/2} aplicado a Z via formula de rango 1:
    #   Z L^T = Z - Z rho (1 - sqrt(1-R2)) / R2 * rho^T   (si R2>0)
    Z = rng.standard_normal((S, N, p))
    if R2 > 0:
        c = (1.0 - np.sqrt(1.0 - R2)) / R2
        Zr = np.einsum("snp,sp->sn", Z, rho)
        X = Z - c * Zr[:, :, None] * rho[:, None, :]
        X = X + y[:, :, None] * rho[:, None, :]
    else:
        X = Z
    X = X - X.mean(axis=1, keepdims=True)

    XtX = np.einsum("snp,snq->spq", X, X)
    Xty = np.einsum("snp,sn->sp", X, y)
    beta = np.linalg.solve(XtX, Xty[:, :, None])[:, :, 0]
    yty = np.einsum("sn,sn->s", y, y)
    return np.einsum("sp,sp->s", beta, Xty) / yty               # R2_hat (S,)


# ============================== Cohorte 2 bajo el mismo paradigma (grupo MCI/HC)
def simular_r2hat_c2(N, R2, S, w, rng, dh=0.0, permutar=True):
    """R2_hat OLS: y = grupo estandarizado (±1 balanceado);
    X = [PCs cerebrales con espectro permutado, PC1 hormonal].

    dh = separación MCI-HC del PC1 hormonal (Cohen); su correlación
    punto-biserial es rho_h = dh / sqrt(dh^2 + 4). R2 = efecto global
    CEREBRAL; el total del modelo es R2_tot = R2 + rho_h^2.
    """
    K = len(w)
    y1 = np.concatenate([-np.ones(N // 2), np.ones(N - N // 2)])
    y1 = (y1 - y1.mean()) / y1.std()
    y = np.tile(y1, (S, 1))

    w_perm = rng.permuted(np.tile(w, (S, 1)), axis=1) if permutar \
        else np.tile(w, (S, 1))
    rho_h = dh / np.sqrt(dh ** 2 + 4.0)
    rho = np.concatenate([np.sqrt(w_perm * R2),
                          np.full((S, 1), rho_h)], axis=1)     # (S, K+1)
    R2tot = R2 + rho_h ** 2

    Z = rng.standard_normal((S, N, K + 1))
    if R2tot > 0:
        c = (1.0 - np.sqrt(1.0 - R2tot)) / R2tot
        Zr = np.einsum("snp,sp->sn", Z, rho)
        X = Z - c * Zr[:, :, None] * rho[:, None, :]
        X = X + y[:, :, None] * rho[:, None, :]
    else:
        X = Z
    X = X - X.mean(axis=1, keepdims=True)

    XtX = np.einsum("snp,snq->spq", X, X)
    Xty = np.einsum("snp,sn->sp", X, y)
    beta = np.linalg.solve(XtX, Xty[:, :, None])[:, :, 0]
    yty = np.einsum("sn,sn->s", y, y)
    return np.einsum("sp,sp->s", beta, Xty) / yty


def correr_cohorte2_v4(cfg, rng, outdir):
    if getattr(cfg, "predictor", "hormonal") == "pm25":
        hlab, hshort, hres = "PM$_{2.5}$ exposure", "exposure", \
            "exposicion PM2.5 residencial"
        harticle = "an"
    else:
        hlab, hshort, hres = "hormonal PC1", "hormonal", "PC1 hormonal"
        harticle = "a"
    ev = pd.read_csv(Path(__file__).resolve().parent /
                     "calibracion_dkt_eigenvalues.csv")["eigenvalue"].to_numpy()
    Ks = [5, 10, 14, 20]
    taus = [4, 6, 10]
    grid_N = [40, 60, 80, 100, 150, 200, 250, 300]
    grid_R2 = [0.0, 0.04, 0.09, 0.16, 0.25]

    filas = []
    for K in Ks:
        w = ev[:K] / ev[:K].sum()
        p = K + 1                                     # + PC1 hormonal
        for N in grid_N:
            if N <= p + 5:
                continue
            crits = {t: r2_critico_reg(N, p, t, cfg.rscale) for t in taus}
            for R2 in grid_R2:
                for dh in (0.0, 0.3):
                    r2h = simular_r2hat_c2(N, R2, cfg.nsims, w, rng, dh)
                    fila = {"K": K, "N": N, "n_por_grupo": N // 2,
                            "R2": R2, "d_h": dh}
                    for t in taus:
                        fila[f"power_bf{t}"] = np.nan if np.isnan(crits[t]) \
                            else float((r2h > crits[t]).mean())
                    filas.append(fila)
        print(f"  [c2] K={K:>2} listo")
    df = pd.DataFrame(filas)
    df.to_csv(outdir / "resultados_v4_cohorte2.csv", index=False)

    largo = df.melt(id_vars=["K", "N", "n_por_grupo", "R2", "d_h"],
                    value_vars=[f"power_bf{t}" for t in taus],
                    var_name="tau", value_name="power")
    largo["tau"] = largo.tau.str.replace("power_bf", "").astype(int)

    nreq = {}
    for K in Ks:
        for R2 in (0.04, 0.09, 0.16, 0.25):
            for t in taus:
                nreq[(K, R2, t)] = n_requerido(
                    largo, cfg.target,
                    f"K=={K} and R2=={R2} and tau=={t} and d_h==0.0")
    resumen = {"modelo": "y=grupo (MCI/HC) ~ PCs cerebro (espectro permutado) "
                         f"+ {hres}; BF global vs nulo",
               "map_R2_a_Mahalanobis": {"0.04": 0.408, "0.09": 0.629,
                                        "0.16": 0.873, "0.25": 1.155},
               "n_TOTAL_requerido_dh0": {f"K{K}_R2{r}_BF{t}":
                                         (None if np.isnan(v) else v)
                                         for (K, r, t), v in nreq.items()}}
    (outdir / "resumen_v4_cohorte2.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False))

    # ---------- figura ----------
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=220,
                             facecolor="white")
    ax = axes[0]
    for R2 in (0.04, 0.09, 0.16, 0.25):
        g = largo.query(f"K==5 and tau==10 and R2=={R2} and d_h==0.0"
                        ).sort_values("N")
        ax.plot(g.N / 2, g.power, color=R2_COLORS[R2], lw=2.4, marker="o",
                ms=4.5, label=f"$R^2$ = {R2:.2f} (D = "
                f"{2*np.sqrt(R2/(1-R2)):.2f})")
        v = nreq[(5, R2, 10)]
        if not np.isnan(v):
            ax.plot(v / 2, cfg.target, "o", color=R2_COLORS[R2], ms=8,
                    mec="white", mew=1.3, zorder=6)
    g0 = largo.query("K==5 and tau==10 and R2==0.0 and d_h==0.0"
                     ).sort_values("N")
    ax.plot(g0.N / 2, g0.power, color=COL["gris"], lw=1.6, ls=":",
            label="global null")
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    ax.axvline(30, color=COL["rojo"], lw=1.0, ls=":", alpha=0.7)
    ax.text(30.8, 0.03, "30/group\n(proposal)", fontsize=8.2, color=COL["rojo"])
    ax.set_xlabel("n per group")
    ax.set_ylabel("Power  P(BF$_{10}$ ≥ 10)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"A. Cohort 2, global-model BF (K = 5 + {hlab})\n"
                 "MCI vs HC multivariate separation D = 2$\\sqrt{R^2/(1-R^2)}$",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="center right")

    ax = axes[1]
    kcol = {5: COL["verde"], 10: COL["azul"], 14: "#D97706", 20: COL["morado"]}
    for K in Ks:
        g = largo.query(f"K=={K} and tau==10 and R2==0.16 and d_h==0.0"
                        ).sort_values("N")
        ax.plot(g.N / 2, g.power, color=kcol[K], lw=2.4, label=f"K = {K}")
        g2 = largo.query(f"K=={K} and tau==10 and R2==0.16 and d_h==0.3"
                         ).sort_values("N")
        ax.plot(g2.N / 2, g2.power, color=kcol[K], lw=1.4, ls="--", alpha=0.8)
    ax.plot([], [], color=COL["gris"], lw=1.4, ls="--",
            label=f"{hshort} effect d$_h$ = 0.3 (dashed)")
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    ax.axvline(30, color=COL["rojo"], lw=1.0, ls=":", alpha=0.7)
    ax.set_xlabel("n per group")
    ax.set_ylabel("Power  P(BF$_{10}$ ≥ 10)")
    ax.set_ylim(0, 1.02)
    ax.set_title("B. Number of components at $R^2$ = 0.16 (D ≈ 0.87)\n"
                 f"with and without {harticle} {hshort} MCI-HC effect",
                 loc="left",
                 pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("Version 4, Cohort 2: MCI vs HC under the permutation-"
                 "marginalized conditional MVG paradigm", y=1.0,
                 fontsize=14, fontweight="bold", color=COL["tinta"])
    fig.text(0.5, -0.02, "y = standardized balanced group indicator · X = "
             f"brain PCs (empirical DKT spectrum, permuted) + {hlab} "
             "(point-biserial ρ$_h$ = d$_h$/√(d$_h$²+4)) · JZS regression BF "
             f"vs null (rscale = √2/4) · M = {cfg.nsims:,}/cell",
             ha="center", fontsize=8.4, color=COL["gris"])
    plt.tight_layout()
    fig.savefig(outdir / "Fig_v4_cohorte2.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    print("\n===== v4 Cohorte 2: N TOTAL requerido (dh=0) =====")
    for K in Ks:
        for R2 in (0.04, 0.09, 0.16, 0.25):
            vals = " | ".join(
                f"BF{t}: " + ("n.a." if np.isnan(nreq[(K, R2, t)])
                              else f"{int(nreq[(K, R2, t)])} "
                                   f"({int(np.ceil(nreq[(K, R2, t)]/2))}/gr)")
                for t in taus)
            print(f"  K={K:>2} R2={R2:.2f}: {vals}")



# ================================================================== ejecución
def n_requerido(df, target, filtro):
    g = df.query(filtro).sort_values("N")
    pw, nn = g.power.to_numpy(), g.N.to_numpy()
    ok = np.where(pw >= target)[0]
    if len(ok) == 0:
        return np.nan
    i = ok[0]
    if i == 0:
        return float(nn[0])
    return float(np.ceil(nn[i-1] + (target - pw[i-1]) * (nn[i] - nn[i-1])
                         / (pw[i] - pw[i-1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--rscale", type=float, default=float(np.sqrt(2) / 4))
    ap.add_argument("--pool", choices=["early_peri", "transition", "all"],
                    default="early_peri")
    ap.add_argument("--outdir", type=str, default="resultados_bfda")
    ap.add_argument("--cohorte2", action="store_true",
                    help="Cohorte 2 (MCI vs HC) bajo el mismo paradigma")
    ap.add_argument("--predictor", choices=["hormonal", "pm25"],
                    default="hormonal",
                    help="pm25: pool Y = exposición residencial PM2.5")
    cfg = ap.parse_args()

    base = Path(__file__).resolve().parent
    outdir = base / cfg.outdir
    outdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    if cfg.predictor == "pm25":
        outdir = outdir / "pm25"
        outdir.mkdir(exist_ok=True)

    if cfg.cohorte2:
        correr_cohorte2_v4(cfg, rng, outdir)
        return

    # ---- insumos empíricos ------------------------------------------------
    if cfg.predictor == "pm25":
        y_pool = pd.read_csv(base / "calibracion_pm25.csv"
                             )["z_exposicion"].to_numpy()
        ydesc = "cohort residential PM2.5 (laboratory model surface 2013-2022)"
    else:
        pc1 = pd.read_csv(base / "calibracion_swan_pc1.csv")
        if cfg.pool == "early_peri":
            pool = pc1.loc[pc1.stage == "early_peri", "PC1"]
        elif cfg.pool == "transition":
            pool = pc1.loc[pc1.stage.isin(["pre", "early_peri", "late_peri"]),
                           "PC1"]
        else:
            pool = pc1["PC1"]
        y_pool = ((pool - pool.mean()) / pool.std(ddof=1)).to_numpy()
        ydesc = f"empirical SWAN {cfg.pool} PC1"

    ev = pd.read_csv(base / "calibracion_dkt_eigenvalues.csv"
                     )["eigenvalue"].to_numpy()
    Ks = [5, 10, 14, 20]
    taus = [4, 6, 10]
    grid_N = [50, 60, 72, 100, 150, 200, 250, 300]
    grid_R2 = [0.0, 0.04, 0.09, 0.16, 0.25]

    print(f"[v4] Y pool: {ydesc} (n={len(y_pool)}, "
          f"skew={pd.Series(y_pool).skew():.2f}) · autovalores DKT top-20 · "
          f"rscale={cfg.rscale:.4f} (regressionBF 'medium') · "
          f"M={cfg.nsims:,}/celda")

    filas = []
    for K in Ks:
        w = ev[:K] / ev[:K].sum()
        for N in grid_N:
            if N <= K + 5:
                continue
            crits = {t: r2_critico_reg(N, K, t, cfg.rscale) for t in taus}
            for R2 in grid_R2:
                r2h = simular_r2hat(N, R2, cfg.nsims, w, y_pool, rng)
                fila = {"K": K, "N": N, "R2": R2,
                        "r2hat_medio": float(r2h.mean())}
                for t in taus:
                    fila[f"power_bf{t}"] = np.nan if np.isnan(crits[t]) \
                        else float((r2h > crits[t]).mean())
                filas.append(fila)
        print(f"  K={K:>2} listo")

    df = pd.DataFrame(filas)
    largo = df.melt(id_vars=["K", "N", "R2"],
                    value_vars=[f"power_bf{t}" for t in taus],
                    var_name="tau", value_name="power")
    largo["tau"] = largo.tau.str.replace("power_bf", "").astype(int)
    df.to_csv(outdir / "resultados_v4_grid.csv", index=False)

    # ---- verificación de la invariancia por permutación -------------------
    w20 = ev[:20] / ev[:20].sum()
    a = simular_r2hat(100, 0.10, 4000, w20, y_pool,
                      np.random.default_rng(7), permutar=True)
    b = simular_r2hat(100, 0.10, 4000, w20, y_pool,
                      np.random.default_rng(7), permutar=False)
    print(f"[invariancia] media R2_hat permutado={a.mean():.4f} vs "
          f"fijo={b.mean():.4f} (Δ={abs(a.mean()-b.mean()):.4f})")

    # ---- n* ----------------------------------------------------------------
    nreq = {}
    for K in Ks:
        for R2 in (0.04, 0.09, 0.16, 0.25):
            for t in taus:
                nreq[(K, R2, t)] = n_requerido(
                    largo, cfg.target,
                    f"K=={K} and R2=={R2} and tau=={t}")
    resumen = {
        "paradigma": "Permutation-Marginalized Conditional MVG (Luarte)",
        "rscale": cfg.rscale, "pool_Y": cfg.pool, "M": cfg.nsims,
        "n_requerido": {f"K{K}_R2{r}_BF{t}": (None if np.isnan(v) else v)
                        for (K, r, t), v in nreq.items()},
    }
    (outdir / "resumen_v4.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False))

    # ---- figura ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.8), dpi=220,
                             facecolor="white")
    ax = axes[0, 0]                                   # curvas por R2, tau=10
    for R2 in (0.04, 0.09, 0.16, 0.25):
        g = largo.query(f"K==20 and tau==10 and R2=={R2}").sort_values("N")
        ax.plot(g.N, g.power, color=R2_COLORS[R2], lw=2.4, marker="o", ms=4.5,
                label=f"$R^2$ = {R2:.2f}")
        v = nreq[(20, R2, 10)]
        if not np.isnan(v):
            ax.plot(v, cfg.target, "o", color=R2_COLORS[R2], ms=8,
                    mec="white", mew=1.3, zorder=6)
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    for nv in (60, 72):
        ax.axvline(nv, color=COL["rojo"], lw=1.0, ls=":", alpha=0.6)
    ax.set_xlabel("Sample size (N)")
    ax.set_ylabel("Power  P(BF$_{10}$ ≥ 10)")
    ax.set_ylim(0, 1.02)
    ax.set_title("A. Power curves at BF$_{10}$ ≥ 10 (K = 20)", loc="left",
                 pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right",
              title="Global effect size")

    ax = axes[0, 1]                                   # sensibilidad a tau
    for t in taus:
        g = largo.query(f"K==20 and tau=={t} and R2==0.09").sort_values("N")
        ax.plot(g.N, g.power, color=TAU_COLORS[t], lw=2.4,
                label=f"BF$_{{10}}$ ≥ {t}, $R^2$ = 0.09")
        g2 = largo.query(f"K==20 and tau=={t} and R2==0.16").sort_values("N")
        ax.plot(g2.N, g2.power, color=TAU_COLORS[t], lw=1.6, ls="--",
                alpha=0.85)
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    ax.plot([], [], color=COL["gris"], lw=1.6, ls="--",
            label="$R^2$ = 0.16 (dashed)")
    ax.set_xlabel("Sample size (N)")
    ax.set_ylabel("Power")
    ax.set_ylim(0, 1.02)
    ax.set_title("B. Evidence thresholds BF ≥ 4 / 6 / 10 (K = 20)",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[1, 0]                                   # sensibilidad a K
    kcol = {5: COL["verde"], 10: COL["azul"], 14: "#D97706", 20: COL["morado"]}
    for K in Ks:
        g = largo.query(f"K=={K} and tau==10 and R2==0.09").sort_values("N")
        ax.plot(g.N, g.power, color=kcol[K], lw=2.4,
                label=f"K = {K}" + (" (90% var.)" if K == 14 else
                                    " (lab ICA)" if K == 20 else ""))
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    ax.set_xlabel("Sample size (N)")
    ax.set_ylabel("Power  P(BF$_{10}$ ≥ 10)")
    ax.set_ylim(0, 1.02)
    ax.set_title("C. Number of components K ($R^2$ = 0.09, BF ≥ 10)",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[1, 1]                                   # falsos positivos H0
    for t in taus:
        g = largo.query(f"K==20 and tau=={t} and R2==0.0").sort_values("N")
        ax.plot(g.N, g.power, color=TAU_COLORS[t], lw=2.4,
                label=f"BF$_{{10}}$ ≥ {t}, K = 20")
    g5 = largo.query("K==5 and tau==4 and R2==0.0").sort_values("N")
    ax.plot(g5.N, g5.power, color=TAU_COLORS[4], lw=1.6, ls="--",
            label="BF$_{10}$ ≥ 4, K = 5")
    ax.set_xlabel("Sample size (N)")
    ax.set_ylabel("P(BF$_{10}$ ≥ τ | $R^2$ = 0)")
    ax.set_ylim(0, 0.12)
    ax.set_title("D. False-positive rate under the global null", loc="left",
                 pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle("BFDA v4: Permutation-Marginalized Conditional MVG "
                 "(global-model Bayes factor)", y=0.995, fontsize=14.5,
                 fontweight="bold", color=COL["tinta"])
    fig.text(0.5, 0.012,
             f"Y bootstrapped from {ydesc} · X spectrum "
             "= empirical DKT eigenvalues · JZS regression BF vs intercept-"
             f"only null (rscale = √2/4, BayesFactor 'medium') · M = "
             f"{cfg.nsims:,}/cell · seed = {cfg.seed}",
             ha="center", fontsize=8.4, color=COL["gris"])
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(outdir / "Fig_v4_MVG.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    print("\n===== v4: N requerido (potencia ≥ 0.80) =====")
    for K in Ks:
        for R2 in (0.04, 0.09, 0.16, 0.25):
            vals = [nreq[(K, R2, t)] for t in taus]
            svals = " | ".join(f"BF{t}: " + ("n.a." if np.isnan(v)
                                             else f"{int(v)}")
                               for t, v in zip(taus, vals))
            print(f"  K={K:>2} R2={R2:.2f}: {svals}")
    print(f"\nResultados en {outdir}/")


if __name__ == "__main__":
    main()
