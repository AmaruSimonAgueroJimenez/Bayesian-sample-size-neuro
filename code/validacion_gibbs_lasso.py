#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validacion_gibbs_lasso.py
================================================================================
Validación MCMC del análisis de tamaño muestral (bfda_n_optimo.py) usando el
MODELO REAL del laboratorio: el Lasso bayesiano jerárquico que en R/JAGS se
define como `model_string_Lasso` (Park & Casella, 2008), portado a Python como
muestreador de Gibbs exacto (conjugado, sin dependencias más allá de NumPy).

Equivalencia con el modelo JAGS del lab:
    y[i]   ~ dnorm(mu[i], pre_sig2)          -> y ~ N(beta0 + X beta, sigma2)
    beta0  ~ dnorm(0, 1.0E-2)                -> beta0 ~ N(0, var=100)
    pre_sig2 ~ dgamma(0.01, 0.01)            -> sigma2 ~ InvGamma(0.01, 0.01)
    beta[j] ~ dnorm(0, 1/(sigma2*tau_sq[j])) -> beta_j | sigma2, tau2_j ~ N(0, sigma2*tau2_j)
    tau_sq[j] ~ dexp(lambda^2/2)             -> tau2_j ~ Exp(lambda^2/2)
    lambda = p*sqrt(var(resid_OLS))/sum(|beta_OLS|)   (misma fórmula empírica)

Qué valida este script (en los n candidatos del BFDA):
  1. RECUPERACIÓN: con datos simulados igual que en bfda_n_optimo.py
     (PC1+edad empíricos SWAN + covariables tipo lab: nse, IV), ¿el lasso
     bayesiano recupera el efecto de PC1? Métricas: P(CrI 95% excluye 0),
     P(signo correcto), shrinkage (media posterior / valor verdadero).
     La covariable "sexo" NO se incluye: la Cohorte 1 es solo de mujeres.
  2. SAVAGE-DICKEY: para el modelo JZS (prior Cauchy sobre el efecto
     estandarizado), el BF10 estimado por MCMC (densidad posterior en 0 vs
     prior en 0) debe coincidir con el BF analítico de correlación parcial
     (Wetzels & Wagenmakers 2012, n_eff = n - k) usado en las curvas BFDA.

Uso:  python validacion_gibbs_lasso.py            (~3-5 min)
      python validacion_gibbs_lasso.py --rapido   (menos datasets, ~1 min)
================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bfda_n_optimo import (COL, _estilo, cargar_pools, log_bf10_jzs_corr,
                           _corr_parcial_lote)


# ================================================== generación de datos (lab)
def generar_dataset(n, rho_parcial, pool, rng, c_edad=-0.15):
    """Un dataset como en bfda_n_optimo.py pero devolviendo (y, X, b_verdadero).

    X = [PC1_z, edad_z, nse, IV]  (covariables estilo laboratorio; sexo no
    entra: cohorte de mujeres). El efecto verdadero está SOLO en PC1.
    """
    pc1 = pool["PC1"].to_numpy()
    edad = pool["age"].to_numpy()
    r12 = np.corrcoef(pc1, edad)[0, 1]
    s12 = np.sqrt(1.0 - r12 ** 2)

    idx = rng.integers(0, len(pool), size=n)
    xz = (pc1[idx] - pc1.mean()) / pc1.std(ddof=1)
    az = (edad[idx] - edad.mean()) / edad.std(ddof=1)
    x_perp = (xz - r12 * az) / s12

    b = rho_parcial * np.sqrt(1.0 - c_edad ** 2)
    var_resid = 1.0 - b ** 2 - c_edad ** 2
    y = c_edad * az + b * x_perp + np.sqrt(var_resid) * rng.standard_normal(n)

    nse = rng.standard_normal(n)          # covariable tipo nse (sin efecto)
    iv = rng.standard_normal(n)           # covariable tipo IV  (sin efecto)
    X = np.column_stack([xz, az, nse, iv])
    beta_true_pc1 = b / s12               # coef verdadero sobre PC1_z dado edad_z
    return y, X, beta_true_pc1


# ========================================== Gibbs: Lasso bayesiano (JAGS port)
def lambda_empirica(y, X):
    """λ = p * sqrt(var(resid OLS)) / sum(|beta OLS|)  (fórmula del lab)."""
    p = X.shape[1]
    bols, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ bols
    return p * np.sqrt(resid.var(ddof=1)) / np.abs(bols).sum()


def gibbs_lasso(y, X, n_iter=4000, burn=1500, rng=None, lam=None):
    """Gibbs exacto para el modelo `model_string_Lasso` (Park & Casella 2008)."""
    rng = rng or np.random.default_rng()
    n, p = X.shape
    lam = lambda_empirica(y, X) if lam is None else lam
    XtX = X.T @ X

    beta0, beta = y.mean(), np.zeros(p)
    sig2, tau2 = y.var(ddof=1), np.ones(p)
    a0 = b0 = 0.01                      # prior InvGamma sobre sigma2 (como JAGS)
    keep = np.empty((n_iter - burn, p))
    keep_b0 = np.empty(n_iter - burn)

    for it in range(n_iter):
        # beta | resto  ~ N(A^-1 X'(y-beta0), sigma2 * A^-1),  A = X'X + D^-1
        A = XtX + np.diag(1.0 / tau2)
        L = np.linalg.cholesky(A)
        m = np.linalg.solve(A, X.T @ (y - beta0))
        z = rng.standard_normal(p)
        beta = m + np.sqrt(sig2) * np.linalg.solve(L.T, z)

        # beta0 | resto (prior N(0, 100))
        prec = n / sig2 + 1.0e-2
        mu0 = ((y - X @ beta).sum() / sig2) / prec
        beta0 = rng.normal(mu0, 1.0 / np.sqrt(prec))

        # 1/tau2_j | resto ~ InvGauss( sqrt(lam^2 sig2 / beta_j^2), lam^2 )
        mu_ig = np.sqrt(lam ** 2 * sig2 / np.maximum(beta ** 2, 1e-12))
        inv_tau2 = rng.wald(mu_ig, lam ** 2)
        tau2 = 1.0 / np.maximum(inv_tau2, 1e-12)

        # sigma2 | resto ~ InvGamma(a0 + (n+p)/2, b0 + RSS/2 + sum(beta^2/tau2)/2)
        rss = np.sum((y - beta0 - X @ beta) ** 2)
        shape = a0 + 0.5 * (n + p)
        scale = b0 + 0.5 * rss + 0.5 * np.sum(beta ** 2 / tau2)
        sig2 = scale / rng.gamma(shape, 1.0)

        if it >= burn:
            keep[it - burn] = beta
            keep_b0[it - burn] = beta0
    return keep, keep_b0, lam


# ============================== Gibbs: modelo JZS (Cauchy) para Savage-Dickey
def gibbs_jzs_sd(y, x, Z, n_iter=9000, burn=2000, rng=None, rscale=1.0):
    """Modelo y = a + b*x + Z*c + e, prior JZS: (b*sx/sigma) ~ Cauchy(0, rscale).

    Cauchy vía mezcla: delta | g ~ N(0, g*rscale^2), g ~ InvGamma(1/2, 1/2).
    Devuelve muestras de delta (efecto estandarizado) para Savage-Dickey.
    """
    rng = rng or np.random.default_rng()
    n = len(y)
    sx = x.std(ddof=1)
    W = np.column_stack([np.ones(n), Z])          # intercepto + covariables
    WtW = W.T @ W
    q = W.shape[1]

    b, sig2, g = 0.0, y.var(ddof=1), 1.0
    gam = np.linalg.lstsq(W, y, rcond=None)[0]
    keep = np.empty(n_iter - burn)

    xtx = float(x @ x)
    for it in range(n_iter):
        # gamma (intercepto+covariables) | resto — prior plano
        r1 = y - b * x
        mu = np.linalg.solve(WtW, W.T @ r1)
        Lw = np.linalg.cholesky(np.linalg.inv(WtW) * sig2)
        gam = mu + Lw @ rng.standard_normal(q)

        # b | resto — prior N(0, sigma2 * g * rscale^2 / sx^2)
        r2 = y - W @ gam
        prior_prec = sx ** 2 / (g * rscale ** 2)
        post_var = sig2 / (xtx + prior_prec)
        post_mu = float(x @ r2) / (xtx + prior_prec)
        b = rng.normal(post_mu, np.sqrt(post_var))

        # g | delta ~ InvGamma(1, (delta^2/rscale^2 + 1)/2)
        delta = b * sx / np.sqrt(sig2)
        g = (0.5 * (delta ** 2 / rscale ** 2 + 1.0)) / rng.gamma(1.0, 1.0)

        # sigma2 | resto ~ InvGamma((n+1)/2, (RSS + b^2*prior_prec)/2)
        rss = np.sum((y - W @ gam - b * x) ** 2)
        sig2 = (0.5 * (rss + b ** 2 * prior_prec)) / rng.gamma(0.5 * (n + 1), 1.0)

        if it >= burn:
            keep[it - burn] = b * sx / np.sqrt(sig2)
    return keep


def bf_savage_dickey(delta_samples, rscale=1.0):
    """BF10 = densidad prior en 0 / densidad posterior en 0 (Cauchy(0,rscale))."""
    kde = stats.gaussian_kde(delta_samples)
    post0 = float(kde(0.0)[0])
    prior0 = stats.cauchy.pdf(0.0, scale=rscale)
    return prior0 / max(post0, 1e-300)


# ================================================================== figura
def hacer_figura(df_lasso, df_sd, outdir):
    lr = np.corrcoef(np.log(df_sd.bf_analitico), np.log(df_sd.bf_mcmc))[0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=220,
                             facecolor="white")
    ax = axes[0]
    xs = np.arange(len(df_lasso))
    w = 0.38
    ax.bar(xs - w / 2, df_lasso.P_CrI95_excluye_0, w, color=COL["azul"],
           label="P(95% CrI excludes 0)")
    ax.bar(xs + w / 2, df_lasso.P_signo_correcto, w, color=COL["verde"],
           label="P(correct sign)")
    for i, row in df_lasso.iterrows():
        ax.text(i - w / 2, row.P_CrI95_excluye_0 + 0.015,
                f"{row.P_CrI95_excluye_0:.2f}", ha="center", fontsize=8.4)
        ax.text(i + w / 2, row.P_signo_correcto + 0.015,
                f"{row.P_signo_correcto:.2f}", ha="center", fontsize=8.4)
    ax.set_xticks(xs, [f"n={int(r.n)}\n|ρ|={r.rho_parcial}"
                       for r in df_lasso.itertuples()])
    ax.axhline(0.8, color=COL["tinta"], ls="--", lw=1.0, alpha=0.6)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("Probability across simulated datasets")
    ax.set_title("A. Lab pipeline (Bayesian Lasso, JAGS port):\n"
                 "recovery of the PC1 effect (covariates: age, nse, IV)",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.155), ncol=2, columnspacing=1.8)

    ax = axes[1]
    m0 = df_sd[df_sd.rho == 0.0]
    m1 = df_sd[df_sd.rho == 0.3]
    ax.scatter(m0.bf_analitico, m0.bf_mcmc, s=34, color=COL["gris"],
               alpha=0.8, label="H0 true (ρ = 0)")
    ax.scatter(m1.bf_analitico, m1.bf_mcmc, s=34, color=COL["azul"],
               alpha=0.8, label="H1 true (ρ = 0.3)")
    lims = [min(df_sd.bf_analitico.min(), df_sd.bf_mcmc.min()) * 0.5,
            max(df_sd.bf_analitico.max(), df_sd.bf_mcmc.max()) * 2]
    ax.plot(lims, lims, color=COL["rojo"], lw=1.4, ls="--", label="identity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Analytic partial-correlation BF$_{10}$ (W&W 2012, n−k)")
    ax.set_ylabel("MCMC Savage–Dickey BF$_{10}$")
    ax.set_title(f"B. MCMC validation of the analytic BF (n = 60)\n"
                 f"corr(log BF) = {lr:.3f}", loc="left", pad=10,
                 fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("MCMC validation of the sample-size analysis", y=1.0,
                 fontsize=14, fontweight="bold", color=COL["tinta"])
    plt.tight_layout()
    fig.savefig(outdir / "Fig_validacion_MCMC.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return lr


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--solo-figura", action="store_true",
                    help="regenera la figura desde los CSV ya guardados")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--outdir", type=str, default="results")
    cfg = ap.parse_args()

    if cfg.solo_figura:
        outdir = Path(__file__).resolve().parent.parent / cfg.outdir
        df_lasso = pd.read_csv(outdir / "validacion_lasso_recuperacion.csv")
        df_sd = pd.read_csv(outdir / "validacion_savage_dickey.csv")
        lr = hacer_figura(df_lasso, df_sd, outdir)
        print(f"Figura regenerada (corr log BF = {lr:.3f}) en {outdir}/")
        return

    ndat = 60 if cfg.rapido else 200
    ndat_sd = 40 if cfg.rapido else 80
    rng = np.random.default_rng(cfg.seed)
    raiz = Path(__file__).resolve().parent.parent
    base = raiz / "data"
    outdir = raiz / cfg.outdir
    outdir.mkdir(exist_ok=True)

    pools, _ = cargar_pools(base)
    pool = pools["early_peri"]
    k = 3 + 1  # covariables en X ademas de PC1: edad, nse, IV (+ intercepto aparte)

    # --------------------------------------------------- 1) recuperación lasso
    escenarios = [(60, 0.3), (60, 0.4), (118, 0.3), (141, 0.3)]
    filas = []
    print(f"[1/2] Lasso bayesiano (Gibbs, port del JAGS del lab) — "
          f"{ndat} datasets/escenario ...")
    for n, rho in escenarios:
        excl, signo, shrink = [], [], []
        for _ in range(ndat):
            y, X, btrue = generar_dataset(n, -rho, pool, rng)
            chain, _, lam = gibbs_lasso(y, X, rng=rng)
            b1 = chain[:, 0]                       # coeficiente de PC1
            lo, hi = np.percentile(b1, [2.5, 97.5])
            excl.append((lo > 0) or (hi < 0))
            signo.append(np.mean(b1 < 0) > 0.5)
            shrink.append(b1.mean() / btrue)
        filas.append({"n": n, "rho_parcial": rho,
                      "P_CrI95_excluye_0": np.mean(excl),
                      "P_signo_correcto": np.mean(signo),
                      "shrinkage_medio": np.mean(shrink)})
        print(f"   n={n:>3} rho={rho}: P(CrI excl 0)={np.mean(excl):.3f} | "
              f"P(signo-)={np.mean(signo):.3f} | shrink={np.mean(shrink):.2f}")
    df_lasso = pd.DataFrame(filas)
    df_lasso.to_csv(outdir / "validacion_lasso_recuperacion.csv", index=False)

    # ------------------------------------- 2) Savage-Dickey vs BF analítico
    print(f"[2/2] Savage-Dickey MCMC vs BF analítico — {ndat_sd} datasets ...")
    pares = []
    for i in range(ndat_sd):
        rho = 0.0 if i % 2 == 0 else 0.3
        n = 60
        y, X, _ = generar_dataset(n, -rho, pool, rng)
        x, Z = X[:, 0], X[:, 1:]
        delta = gibbs_jzs_sd(y, x, Z, rng=rng)
        bf_mcmc = bf_savage_dickey(delta)
        rp = _corr_parcial_lote(y[None, :], x[None, :], Z[None, :, :])[0]
        bf_ana = float(np.exp(log_bf10_jzs_corr(rp ** 2, n - k)[0]))
        pares.append({"rho": rho, "r_parcial": rp,
                      "bf_analitico": bf_ana, "bf_mcmc": bf_mcmc})
    df_sd = pd.DataFrame(pares)
    df_sd.to_csv(outdir / "validacion_savage_dickey.csv", index=False)
    lr = np.corrcoef(np.log(df_sd.bf_analitico), np.log(df_sd.bf_mcmc))[0, 1]
    med_dif = np.median(np.abs(np.log(df_sd.bf_mcmc / df_sd.bf_analitico)))
    print(f"   corr(log BF) = {lr:.4f} | mediana |Δlog BF| = {med_dif:.3f} "
          f"(~{100 * (np.exp(med_dif) - 1):.0f}% en escala BF)")

    # ------------------------------------------------------------- figura
    hacer_figura(df_lasso, df_sd, outdir)
    print(f"Listo. Resultados en {outdir}/")


if __name__ == "__main__":
    main()
