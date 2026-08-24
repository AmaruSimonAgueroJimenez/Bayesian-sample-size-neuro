#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bfda_lasso_multivariante.py
================================================================================
Determinación del n requerido para la COHORTE 1 usando el criterio real del
pipeline del laboratorio:

    Modelo de análisis = LASSO BAYESIANO (el `model_string_Lasso` de JAGS,
    Park & Casella 2008, con la fórmula empírica de lambda del laboratorio):
        y (PC1 hormonal) ~ beta0 + [PC cerebrales 1..K, edad] * beta + error
    ÉXITO en un dataset = AL MENOS UN componente principal cerebral resulta
    "significativo" de manera BIDIRECCIONAL: su intervalo de credibilidad del
    95% (equal-tailed) excluye 0, en cualquier dirección.

    n requerido = menor n tal que P(éxito) >= objetivo (0.80 por defecto).

Modelo generador (calibrado con datos reales):
    edad_z ~ N(0,1)
    PC_cerebral_1 = -0.17*edad_z + sqrt(1-0.17^2)*e1   (corr PC1-espesor/edad
                                                        en DKT 40-55: -0.171)
    PC_cerebral_j ~ N(0,1), j=2..K                      (PCs ortogonales, var 1)
    y = 0.17*edad_z + sum_j rho_j * PC_true_j + sqrt(1-0.17^2-sum rho^2)*eps
        (0.17 = corr PC1 hormonal-edad en el pool SWAN early perimenopause)
    El/los componente(s) verdadero(s) NO son el PC1 cerebral (para no confundir
    el efecto con la edad); por defecto es el componente 2 (y el 5 si hay dos).

K = 20 por defecto (n_comp = 20 del ICA del laboratorio); K = 10 como
sensibilidad. Todos los predictores (incluida la edad) reciben el prior lasso,
igual que en el JAGS del laboratorio (donde nse, sexo e IV van penalizados).

Implementación: muestreador de Gibbs EXACTO y VECTORIZADO POR LOTES (todos los
datasets simulados avanzan en paralelo), lo que permite una grilla BFDA
completa en minutos. La significancia CrI-95% se evalúa de forma equivalente
via P(beta_j > 0 | datos) > 0.975 o < 0.025 (identidad exacta con el intervalo
equal-tailed), acumulando conteos sin almacenar cadenas.

Uso:
    python bfda_lasso_multivariante.py              # grilla completa (~15 min)
    python bfda_lasso_multivariante.py --rapido     # prueba corta
    python bfda_lasso_multivariante.py --verificar  # contraste vs Gibbs escalar
================================================================================
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"azul": "#2563EB", "verde": "#059669", "morado": "#8B5CF6",
       "rojo": "#E11D48", "gris": "#64748B", "gris_claro": "#CBD5E1",
       "tinta": "#0F172A"}
EFFECT_COLORS = {0.2: "#059669", 0.3: "#2563EB", 0.4: "#8B5CF6", 0.5: "#E11D48"}

C_EDAD_Y = 0.17        # corr(predictor, edad): 0.17 hormonal (SWAN early peri)
C_EDAD_PCB1 = -0.17    # corr(PC1 cerebral espesor, edad), DKT 40-55
YNAME = "hormone PC1"            # etiqueta del predictor en figuras
YNAME_C2 = "hormonal PC1"
C2_FOOT = ("hormonal PC1 calibrated on SWAN postmenopause "
           "(corr with age = -0.015)")


def _estilo(ax):
    ax.grid(color="#E2E8F0", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COL["gris_claro"])


# ============================================================ generación
def generar_lote(n, rho, S, K, n_true, rng):
    """S datasets: y (S,n), X (S,n,K+1) = [PC cerebrales 1..K, edad]."""
    edad = rng.standard_normal((S, n))
    E = rng.standard_normal((S, n, K))
    E[:, :, 0] = C_EDAD_PCB1 * edad + np.sqrt(1 - C_EDAD_PCB1 ** 2) * E[:, :, 0]

    true_idx = [1] if n_true == 1 else [1, 4]     # nunca el PC1 (lleva edad)
    var_res = 1.0 - C_EDAD_Y ** 2 - n_true * rho ** 2
    if var_res <= 0:
        raise ValueError("rho demasiado grande para el modelo generador")
    y = C_EDAD_Y * edad + np.sqrt(var_res) * rng.standard_normal((S, n))
    for j in true_idx:
        y = y + rho * E[:, :, j]

    X = np.concatenate([E, edad[:, :, None]], axis=2)   # (S, n, K+1)
    return y, X, true_idx


# ==================================================== Gibbs lasso por lotes
def lambda_empirica_lote(y, X):
    """lambda = p*sqrt(var(resid OLS sin intercepto))/sum|beta_OLS| (regla lab)."""
    S, n, p = X.shape
    XtX = np.einsum("snp,snq->spq", X, X)
    Xty = np.einsum("snp,sn->sp", X, y)
    b = np.linalg.solve(XtX + 1e-8 * np.eye(p), Xty[:, :, None])[:, :, 0]
    rss = np.einsum("sn,sn->s", y, y) - np.einsum("sp,sp->s", b, Xty)
    var = np.maximum(rss, 1e-12) / max(n - p, 2)
    return p * np.sqrt(var) / np.abs(b).sum(axis=1), XtX, Xty


def gibbs_lasso_lote(y, X, n_iter=3000, burn=800, rng=None, lam=None,
                     track=None):
    """Gibbs exacto (priors del JAGS del lab) vectorizado sobre S datasets.

    Devuelve: frac_pos (S,p) = P(beta_j>0|datos) estimada, media posterior (S,p).
    Si `track` es una lista de columnas, devuelve además las muestras
    posteriores de esos coeficientes (S, n_kept, len(track)) para calcular
    intervalos de credibilidad del beta estandarizado.
    """
    rng = rng or np.random.default_rng()
    S, n, p = X.shape
    lam_calc, XtX, Xty = lambda_empirica_lote(y, X)
    lam = lam_calc if lam is None else lam
    Xt1 = X.sum(axis=1)                       # (S,p)
    yty = np.einsum("sn,sn->s", y, y)
    oneTy = y.sum(axis=1)
    idx = np.arange(p)

    beta = np.zeros((S, p))
    beta0 = y.mean(axis=1)
    sig2 = y.var(axis=1, ddof=1)
    tau2 = np.ones((S, p))
    a0 = b0 = 0.01
    lam2 = (lam ** 2)[:, None]

    count_pos = np.zeros((S, p))
    sum_beta = np.zeros((S, p))
    kept = 0
    if track is not None:
        track = np.asarray(track, int)
        muestras = np.empty((S, n_iter - burn, len(track)), dtype=np.float32)

    for it in range(n_iter):
        # ---- beta | resto ~ N(A^-1 rhs, sig2 A^-1), A = XtX + diag(1/tau2)
        A = XtX.copy()
        A[:, idx, idx] += 1.0 / tau2
        L = np.linalg.cholesky(A)
        rhs = Xty - beta0[:, None] * Xt1
        m = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
        z = rng.standard_normal((S, p))
        w = np.linalg.solve(np.transpose(L, (0, 2, 1)), z[:, :, None])[:, :, 0]
        beta = m + np.sqrt(sig2)[:, None] * w

        # ---- beta0 | resto (prior N(0,100))
        prec = n / sig2 + 1e-2
        mu0 = ((oneTy - np.einsum("sp,sp->s", beta, Xt1)) / sig2) / prec
        beta0 = mu0 + rng.standard_normal(S) / np.sqrt(prec)

        # ---- 1/tau2 ~ InvGauss(sqrt(lam2*sig2/beta^2), lam2)
        mu_ig = np.sqrt(lam2 * sig2[:, None] / np.maximum(beta ** 2, 1e-12))
        inv_tau2 = rng.wald(mu_ig, np.broadcast_to(lam2, mu_ig.shape))
        tau2 = 1.0 / np.maximum(inv_tau2, 1e-12)

        # ---- sig2 | resto ~ InvGamma
        bXty = np.einsum("sp,sp->s", beta, Xty)
        bXt1 = np.einsum("sp,sp->s", beta, Xt1)
        bAb = np.einsum("sp,spq,sq->s", beta, XtX, beta)
        rss = (yty - 2 * beta0 * oneTy - 2 * bXty + 2 * beta0 * bXt1
               + n * beta0 ** 2 + bAb)
        shape = a0 + 0.5 * (n + p)
        scale = b0 + 0.5 * np.maximum(rss, 0) + 0.5 * (beta ** 2 / tau2).sum(1)
        sig2 = scale / rng.gamma(shape, 1.0, size=S)

        if it >= burn:
            count_pos += beta > 0
            sum_beta += beta
            if track is not None:
                muestras[:, kept, :] = beta[:, track]
            kept += 1

    if track is not None:
        return count_pos / kept, sum_beta / kept, muestras
    return count_pos / kept, sum_beta / kept


def evaluar_celda(n, rho, S, K, n_true, rng, n_iter=3000, burn=800):
    """Devuelve métricas del criterio en una celda (n, rho)."""
    y, X, true_idx = generar_lote(n, rho, S, K, n_true, rng)
    frac_pos, _ = gibbs_lasso_lote(y, X, n_iter, burn, rng)
    sig = (frac_pos > 0.975) | (frac_pos < 0.025)       # CrI 95% excluye 0
    sig_cereb = sig[:, :K]                              # solo componentes cerebrales
    p_any = float(sig_cereb.any(axis=1).mean())
    p_true = float(sig_cereb[:, true_idx].any(axis=1).mean()) if rho > 0 else np.nan
    falsos = sig_cereb.copy()
    if rho > 0:
        falsos[:, true_idx] = False
    media_falsos = float(falsos.sum(axis=1).mean())
    return p_any, p_true, media_falsos


def n_requerido(df, target, filtro):
    g = df.query(filtro).sort_values("n")
    p, nn = g.p_any.to_numpy(), g.n.to_numpy()
    ok = np.where(p >= target)[0]
    if len(ok) == 0:
        return np.nan
    i = ok[0]
    if i == 0:
        return float(nn[0])
    return float(np.ceil(nn[i-1] + (target - p[i-1]) * (nn[i] - nn[i-1])
                         / (p[i] - p[i-1])))


# ================================================================ verificación
def verificar_vs_escalar(rng):
    """Compara el Gibbs por lotes con el Gibbs escalar ya validado."""
    from validacion_gibbs_lasso import gibbs_lasso as gibbs_escalar
    y, X, _ = generar_lote(80, 0.4, 1, 8, 1, rng)
    frac_lote, media_lote = gibbs_lasso_lote(
        np.repeat(y, 24, axis=0), np.repeat(X, 24, axis=0),
        n_iter=4000, burn=1000, rng=rng)
    chain, _, _ = gibbs_escalar(y[0], X[0], n_iter=20000, burn=4000, rng=rng)
    fp_esc = (chain > 0).mean(axis=0)
    dif_fp = np.abs(frac_lote.mean(axis=0) - fp_esc).max()
    dif_m = np.abs(media_lote.mean(axis=0) - chain.mean(axis=0)).max()
    print(f"[verificación] max|ΔP(beta>0)| = {dif_fp:.4f} · "
          f"max|Δmedia post| = {dif_m:.4f}  (mismo dataset, lote vs escalar)")
    return dif_fp, dif_m


# ===================================================================== figura
def figura(df, nreq, cfg, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6), dpi=220,
                             facecolor="white")

    # A) criterio principal: P(>=1 PC cerebral significativo), K=20, 1 true
    ax = axes[0, 0]
    base = df.query("K == 20 and n_true == 1")
    for rho in cfg.efectos:
        g = base[base.rho == rho].sort_values("n")
        ax.plot(g.n, g.p_any, color=EFFECT_COLORS[rho], lw=2.4,
                label=f"ρ = {rho:.1f}")
        nr = nreq.get((20, 1, rho), np.nan)
        if not np.isnan(nr):
            ax.plot(nr, cfg.target, "o", color=EFFECT_COLORS[rho], ms=7,
                    mec="white", mew=1.2, zorder=6)
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    for nv, lab in [(60, "n = 60"), (72, "n = 72")]:
        ax.axvline(nv, color=COL["rojo"], lw=1.1, ls=":", alpha=0.75)
        ax.text(nv + 1, 0.03, lab, fontsize=8.2, color=COL["rojo"],
                rotation=90, va="bottom")
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("P(≥1 brain PC significant)")
    ax.set_ylim(0, 1.02)
    ax.set_title("A. Detection criterion: ≥1 brain component with 95% CrI\n"
                 "excluding 0 (two-sided), Bayesian Lasso, K = 20", loc="left",
                 pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right",
              title="True effect (1 component)")

    # B) H0 global y detección correcta
    ax = axes[0, 1]
    h0 = df.query("K == 20 and n_true == 1 and rho == 0").sort_values("n")
    ax.plot(h0.n, h0.p_any, color=COL["rojo"], lw=2.4,
            label="P(≥1 significant | global null), K = 20")
    h0b = df.query("K == 10 and n_true == 1 and rho == 0").sort_values("n")
    if len(h0b):
        ax.plot(h0b.n, h0b.p_any, color=COL["rojo"], lw=1.8, ls="--",
                label="idem, K = 10")
    g3 = df.query("K == 20 and n_true == 1 and rho == 0.4").sort_values("n")
    ax.plot(g3.n, g3.p_true, color=COL["morado"], lw=2.2,
            label="P(true component detected | ρ = 0.4)")
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("B. Family-wise false detection under the global null\n"
                 "and correct-component detection", loc="left", pad=10,
                 fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=8.8, loc="center right")

    # C) sensibilidad K
    ax = axes[1, 0]
    for rho, ls in [(0.3, "-"), (0.4, "--")]:
        for K, c in [(20, COL["azul"]), (10, COL["verde"])]:
            g = df.query(f"K == {K} and n_true == 1 and rho == {rho}"
                         ).sort_values("n")
            if len(g):
                ax.plot(g.n, g.p_any, color=c, lw=2.2, ls=ls,
                        label=f"K = {K}, ρ = {rho}")
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("P(≥1 brain PC significant)")
    ax.set_ylim(0, 1.02)
    ax.set_title("C. Sensitivity to the number of brain components",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=8.8, loc="lower right")

    # D) dos componentes verdaderos
    ax = axes[1, 1]
    g1 = df.query("K == 20 and n_true == 1 and rho == 0.3").sort_values("n")
    g2 = df.query("K == 20 and n_true == 2 and rho == 0.3").sort_values("n")
    ax.plot(g1.n, g1.p_any, color=COL["azul"], lw=2.2,
            label="1 true component (ρ = 0.3)")
    if len(g2):
        ax.plot(g2.n, g2.p_any, color=COL["rojo"], lw=2.2,
                label="2 true components (ρ = 0.3 each)")
        ax.plot(g2.n, g2.p_true, color=COL["rojo"], lw=1.6, ls="--",
                label="≥1 of the 2 true detected")
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("D. One vs. two associated brain components", loc="left",
                 pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=8.8, loc="lower right")

    fig.suptitle("Required n under the laboratory criterion: Bayesian Lasso, "
                 f"{YNAME} ~ brain PCs + age",
                 y=0.995, fontsize=14.5, fontweight="bold", color=COL["tinta"])
    fig.text(0.5, 0.012,
             f"{cfg.nsims:,} simulated datasets per cell · Gibbs "
             f"{cfg.n_iter:,} iterations ({cfg.burn:,} burn-in) · priors and "
             "empirical λ rule identical to the laboratory JAGS model · "
             "significance = equal-tailed 95% CrI excludes 0 · "
             f"seed = {cfg.seed}",
             ha="center", fontsize=8.4, color=COL["gris"])
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(outdir / "Fig_lasso_n_requerido.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)



# ==================== beta posterior estandarizado (métrica del asesor neuro)
def correr_beta_posterior(cfg, rng, outdir):
    """Recuperación del BETA POSTERIOR ESTANDARIZADO del componente asociado.

    Como y (PC1 hormonal) y los componentes cerebrales están estandarizados y
    los componentes son ortogonales entre sí, el coeficiente poblacional del
    componente verdadero ES rho: beta_true = rho. Para cada dataset simulado
    se reporta, del Lasso bayesiano del laboratorio:
        - media posterior de beta (el "beta estandarizado" que se informa),
        - intervalo de credibilidad al 95% (equal-tailed),
        - shrinkage = E[beta | datos] / beta_true,
        - cobertura del beta verdadero por el CrI 95%,
        - P(CrI excluye 0)  (criterio de detección),
        - el mayor |beta| posterior entre los K componentes cerebrales
          (el que se reportaría en la práctica) y con qué frecuencia coincide
          con el componente verdadero.
    """
    K = 20
    escenarios = [(60, 0.3), (60, 0.4), (60, 0.5),
                  (72, 0.3), (72, 0.4),
                  (100, 0.3), (134, 0.3), (141, 0.3)]
    filas = []
    muestras_guardadas = {}
    t0 = time.time()
    for i, (n, rho) in enumerate(escenarios, 1):
        y, X, true_idx = generar_lote(n, rho, cfg.nsims, K, 1, rng)
        frac_pos, media, muestras = gibbs_lasso_lote(
            y, X, cfg.n_iter, cfg.burn, rng, track=[true_idx[0]])
        b = muestras[:, :, 0].astype(float)            # (S, n_kept)
        post_mean = b.mean(axis=1)
        lo = np.percentile(b, 2.5, axis=1)
        hi = np.percentile(b, 97.5, axis=1)
        sig = (lo > 0) | (hi < 0)
        cobertura = (lo <= rho) & (hi >= rho)
        cereb = media[:, :K]
        sel = np.argmax(np.abs(cereb), axis=1)
        filas.append({
            "n": n, "rho_verdadero": rho, "beta_verdadero": rho,
            "beta_post_medio": float(post_mean.mean()),
            "beta_post_sd_entre_datasets": float(post_mean.std(ddof=1)),
            "shrinkage": float(post_mean.mean() / rho),
            "CrI_lo_mediano": float(np.median(lo)),
            "CrI_hi_mediano": float(np.median(hi)),
            "CrI_ancho_mediano": float(np.median(hi - lo)),
            "cobertura_95": float(cobertura.mean()),
            "P_CrI_excluye_0": float(sig.mean()),
            "P_selecciona_verdadero": float((sel == true_idx[0]).mean()),
            "beta_max_abs_medio": float(np.abs(cereb).max(axis=1).mean()),
        })
        muestras_guardadas[(n, rho)] = post_mean
        r = filas[-1]
        print(f"[beta {i}/{len(escenarios)}] n={n:>3} rho={rho}: "
              f"beta_post={r['beta_post_medio']:.3f} "
              f"(shrink {r['shrinkage']:.2f}) · CrI mediano "
              f"[{r['CrI_lo_mediano']:.3f}, {r['CrI_hi_mediano']:.3f}] · "
              f"cobertura {r['cobertura_95']:.2f} · P(excl 0) "
              f"{r['P_CrI_excluye_0']:.3f} ({time.time()-t0:,.0f}s)", flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(outdir / "resultados_beta_posterior.csv", index=False)
    (outdir / "resumen_beta_posterior.json").write_text(
        json.dumps({"K": K, "M": cfg.nsims,
                    "nota": "beta poblacional del componente verdadero = rho "
                            "(predictores y respuesta estandarizados)",
                    "filas": filas}, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------- figura
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), dpi=220,
                             facecolor="white")

    ax = axes[0]                       # distribución del beta posterior medio
    escen_plot = [(60, 0.3), (72, 0.3), (134, 0.3),
                  (60, 0.4), (72, 0.4), (60, 0.5)]
    pos = np.arange(len(escen_plot))
    cols = [EFFECT_COLORS[r] for (_, r) in escen_plot]
    for i, ((n, rho), c) in enumerate(zip(escen_plot, cols)):
        v = muestras_guardadas[(n, rho)]
        q = np.percentile(v, [5, 25, 50, 75, 95])
        ax.plot([i, i], [q[0], q[4]], color=c, lw=1.6, alpha=0.85)
        ax.add_patch(plt.Rectangle((i - 0.19, q[1]), 0.38, q[3] - q[1],
                                   facecolor=c, alpha=0.5, edgecolor=c))
        ax.plot([i - 0.19, i + 0.19], [q[2], q[2]], color="white", lw=2.4,
                zorder=5)
        ax.plot([i - 0.19, i + 0.19], [q[2], q[2]], color=c, lw=1.1, zorder=6)
        ax.plot([i - 0.30, i + 0.30], [rho, rho], color=COL["tinta"], lw=1.4,
                ls="--", zorder=7)
    ax.axhline(0, color=COL["gris_claro"], lw=1.0)
    ax.plot([], [], color=COL["tinta"], lw=1.4, ls="--", label="true β")
    ax.set_xticks(pos, [f"n={n}\nβ={r}" for (n, r) in escen_plot])
    ax.set_ylabel("Posterior mean standardized β")
    ax.set_title("A. Posterior standardized β of the associated\n"
                 "component (median, IQR, 5–95% across studies)",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[1]                       # shrinkage y cobertura
    for rho, c in [(0.3, EFFECT_COLORS[0.3]), (0.4, EFFECT_COLORS[0.4])]:
        g = df[df.rho_verdadero == rho].sort_values("n")
        if len(g) < 2:
            continue
        ax.plot(g.n, g.shrinkage, color=c, lw=2.4, marker="o", ms=6,
                label=f"shrinkage, β = {rho}")
        ax.plot(g.n, g.cobertura_95, color=c, lw=1.6, ls="--", marker="s",
                ms=5, alpha=0.85, label=f"95% CrI coverage, β = {rho}")
    ax.axhline(1.0, color=COL["tinta"], lw=1.0, ls=":", alpha=0.6)
    ax.axhline(0.95, color=COL["gris"], lw=1.0, ls=":", alpha=0.6)
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Ratio / probability")
    ax.set_ylim(0, 1.15)
    ax.set_title("B. Shrinkage of the posterior β and\ncoverage of the true "
                 "value", loc="left", pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=8.6, loc="lower right")

    ax = axes[2]                       # CrI mediano por escenario
    ys = np.arange(len(df))
    for i, r in enumerate(df.itertuples()):
        c = EFFECT_COLORS[r.rho_verdadero]
        ax.plot([r.CrI_lo_mediano, r.CrI_hi_mediano], [i, i], color=c, lw=3.2,
                alpha=0.75, solid_capstyle="round")
        ax.plot(r.beta_post_medio, i, "o", color=c, ms=7, mec="white", mew=1.2,
                zorder=5)
        ax.plot(r.beta_verdadero, i, "|", color=COL["tinta"], ms=13, mew=2,
                zorder=6)
    ax.axvline(0, color=COL["gris_claro"], lw=1.2)
    ax.set_yticks(ys, [f"n={int(r.n)}, β={r.rho_verdadero}"
                       for r in df.itertuples()])
    ax.set_xlabel("Standardized β")
    ax.set_title("C. Median 95% credible interval\n(circle: posterior mean; "
                 "tick: true β)", loc="left", pad=10, fontweight="bold")
    _estilo(ax)

    fig.suptitle(f"Posterior standardized β of the {YNAME}-related brain "
                 "component (Bayesian Lasso, K = 20)", y=1.0, fontsize=14,
                 fontweight="bold", color=COL["tinta"])
    fig.text(0.5, -0.03, f"{cfg.nsims} simulated studies per scenario · "
             "predictors and response standardized, so the population "
             "coefficient of the associated component equals the target "
             "effect · covariates: 20 brain components + age",
             ha="center", fontsize=8.4, color=COL["gris"])
    plt.tight_layout()
    fig.savefig(outdir / "Fig_beta_posterior.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"\nListo. Resultados en {outdir}/")


# ==================================== criterio Bayes factor (analítico, JZS)
def bf_criterio_lote(n, rho, S, K, n_true, rng, umbrales=(4, 6, 10)):
    """P(>=1 componente cerebral con BF10 > u), bilateral, por umbral u.

    BF por componente = BF JZS por defecto sobre la correlación PARCIAL del
    componente j con y, controlando por el resto de componentes + edad
    (Wetzels & Wagenmakers 2012, m = n - k con k = K controles). Analítico:
    no requiere MCMC; usa la monotonicidad del BF en r^2 (r crítico por (m,u)).
    """
    from bfda_n_optimo import r2_critico
    y, X, true_idx = generar_lote(n, rho, S, K, n_true, rng)
    Z = np.concatenate([y[:, :, None], X], axis=2)          # (S, n, K+2)
    Zs = (Z - Z.mean(1, keepdims=True)) / Z.std(1, ddof=1, keepdims=True)
    C = np.einsum("snp,snq->spq", Zs, Zs) / (n - 1)
    Om = np.linalg.inv(C)
    d = np.sqrt(np.abs(np.diagonal(Om, axis1=1, axis2=2)))
    rp = -Om[:, 0, 1:K + 1] / (d[:, 0:1] * d[:, 1:K + 1])   # parciales (S,K)
    m = n - K                                               # k = K controles
    filas = []
    for u in umbrales:
        rc = r2_critico(m, u)
        if np.isnan(rc):
            p_any = p_true = np.nan
        else:
            hit = rp ** 2 > rc
            p_any = float(hit.any(axis=1).mean())
            p_true = float(hit[:, true_idx].any(axis=1).mean()) if rho > 0 \
                else np.nan
        filas.append({"K": K, "n_true": n_true, "n": n, "rho": rho,
                      "umbral_bf": u, "p_any": p_any, "p_true": p_true})
    return filas


def correr_extra(cfg, rng, outdir):
    """Escenarios adicionales: K=5 y K=14 (90% varianza) para el criterio
    lasso-CrI, y criterio por Bayes factor (umbrales 4/6/10)."""
    grid_n = [40, 50, 60, 72, 85, 100, 120, 140, 160]

    # ---- 1) lasso CrI con K=5 y K=14 --------------------------------------
    filas = []
    celdas = [(K, n, rho) for K in (5, 14)
              for n in [40, 60, 72, 100, 140]
              for rho in (0.0, 0.3, 0.4)]
    t0 = time.time()
    for i, (K, n, rho) in enumerate(celdas, 1):
        p_any, p_true, mf = evaluar_celda(n, rho, cfg.nsims, K, 1, rng,
                                          cfg.n_iter, cfg.burn)
        filas.append({"K": K, "n_true": 1, "n": n, "rho": rho,
                      "p_any": p_any, "p_true": p_true, "media_falsos": mf})
        print(f"[lasso-extra {i:>2}/{len(celdas)}] K={K:>2} n={n:>3} "
              f"rho={rho:.1f}: P(≥1)={p_any:.3f} ({time.time()-t0:,.0f}s)",
              flush=True)
    df_k = pd.DataFrame(filas)
    df_k.to_csv(outdir / "resultados_lasso_K_extra.csv", index=False)

    # ---- 2) criterio BF (analítico), S grande -----------------------------
    S_bf = 4000
    filas_bf = []
    for n in grid_n:                                   # K=20 grilla completa
        for rho in (0.0, 0.2, 0.3, 0.4, 0.5):
            filas_bf += bf_criterio_lote(n, rho, S_bf, 20, 1, rng)
    for K in (5, 10, 14):                              # sensibilidad K
        for n in grid_n:
            for rho in (0.0, 0.3, 0.4):
                filas_bf += bf_criterio_lote(n, rho, S_bf, K, 1, rng)
    df_bf = pd.DataFrame(filas_bf)
    df_bf.to_csv(outdir / "resultados_bf_criterio.csv", index=False)

    # ---- n requerido ------------------------------------------------------
    def nreq_de(df, filtro):
        g = df.query(filtro).sort_values("n")
        p, nn = g.p_any.to_numpy(), g.n.to_numpy()
        ok = np.where(p >= cfg.target)[0]
        if len(ok) == 0:
            return np.nan
        i = ok[0]
        if i == 0:
            return float(nn[0])
        return float(np.ceil(nn[i-1] + (cfg.target - p[i-1])
                             * (nn[i] - nn[i-1]) / (p[i] - p[i-1])))

    df_full = pd.read_csv(outdir / "resultados_lasso_cohorte1.csv")
    df_all_crit = pd.concat([df_full.query("n_true==1"), df_k])
    nreq_k, nreq_bf = {}, {}
    for K in (5, 10, 14, 20):
        for rho in (0.3, 0.4):
            nreq_k[(K, rho)] = nreq_de(df_all_crit,
                                       f"K=={K} and rho=={rho}")
    for u in (4, 6, 10):
        for rho in (0.2, 0.3, 0.4, 0.5):
            nreq_bf[(u, rho)] = nreq_de(
                df_bf, f"K==20 and umbral_bf=={u} and rho=={rho}")

    resumen = {
        "K90": 14,
        "n_requerido_lasso_CrI_por_K": {f"K{K}_rho{r}":
                                        (None if np.isnan(v) else v)
                                        for (K, r), v in nreq_k.items()},
        "n_requerido_BF_por_umbral_K20": {f"BF{u}_rho{r}":
                                          (None if np.isnan(v) else v)
                                          for (u, r), v in nreq_bf.items()},
    }
    (outdir / "resumen_extra.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False))

    # ---- figura -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=220,
                             facecolor="white")
    ax = axes[0]
    ucol = {4: COL["verde"], 6: COL["azul"], 10: COL["morado"]}
    for u in (4, 6, 10):
        g = df_bf.query(f"K==20 and rho==0.4 and umbral_bf=={u}"
                        ).sort_values("n")
        ax.plot(g.n, g.p_any, color=ucol[u], lw=2.4,
                label=f"BF$_{{10}}$ > {u}, ρ = 0.4")
        g3 = df_bf.query(f"K==20 and rho==0.3 and umbral_bf=={u}"
                         ).sort_values("n")
        ax.plot(g3.n, g3.p_any, color=ucol[u], lw=1.6, ls="--", alpha=0.85)
    h0 = df_bf.query("K==20 and rho==0.0 and umbral_bf==4").sort_values("n")
    ax.plot(h0.n, h0.p_any, color=COL["rojo"], lw=1.8, ls=":",
            label="global null, BF > 4")
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    for nv in (60, 72):
        ax.axvline(nv, color=COL["rojo"], lw=1.0, ls=":", alpha=0.6)
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("P(≥1 brain PC with BF$_{10}$ > threshold)")
    ax.set_ylim(0, 1.02)
    ax.set_title("A. Bayes-factor thresholds 4 / 6 / 10 (K = 20)\n"
                 "solid: ρ = 0.4 · dashed: ρ = 0.3", loc="left", pad=10,
                 fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=8.8, loc="center right")

    ax = axes[1]
    Ks = [5, 10, 14, 20]
    xs = np.arange(len(Ks))
    for rho, c, dx in [(0.3, COL["azul"], -0.12), (0.4, COL["morado"], 0.12)]:
        vals = [nreq_k[(K, rho)] for K in Ks]
        ax.plot(xs + dx, vals, "o", color=c, ms=9, mec="white", mew=1.2,
                label=f"ρ = {rho}", zorder=5)
        for x, v in zip(xs + dx, vals):
            if not np.isnan(v):
                ax.plot([x, x], [0, v], color=c, lw=2.0, alpha=0.45)
                ax.annotate(f"{v:.0f}", (x, v), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=9,
                            fontweight="bold", color=COL["tinta"])
    ax.set_xticks(xs, [f"K = {K}" + ("\n(90% var.)" if K == 14 else
                                     "\n(lab ICA)" if K == 20 else "")
                       for K in Ks])
    ax.set_ylabel("Required n (Lasso CrI criterion)")
    ax.set_ylim(0, 175)
    ax.set_title("B. Required n by number of brain components\n"
                 "(≥1 component, 95% CrI excludes 0)", loc="left", pad=10,
                 fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("Sensitivity: number of components and Bayes-factor "
                 "thresholds", y=1.0, fontsize=14, fontweight="bold",
                 color=COL["tinta"])
    fig.text(0.5, -0.02, "Left: analytic per-component JZS BF on the partial "
             "correlation of each component (controls: remaining components "
             "+ age), 4,000 datasets/cell. Right: Bayesian-Lasso CrI "
             f"criterion, {cfg.nsims} datasets/cell.",
             ha="center", fontsize=8.4, color=COL["gris"])
    plt.tight_layout()
    fig.savefig(outdir / "Fig_lasso_K_BF.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    print("\n===== n requerido, criterio lasso-CrI por K =====")
    for (K, r), v in nreq_k.items():
        print(f"  K={K:>2} rho={r}: {'n.a.' if np.isnan(v) else int(v)}")
    print("===== n requerido, criterio BF (K=20) =====")
    for (u, r), v in nreq_bf.items():
        print(f"  BF>{u:>2} rho={r}: {'n.a.' if np.isnan(v) else int(v)}")


# ==================================== Cohorte 2: misma lógica, con hormonas
C_H_AGE_POST = -0.015   # corr(PC1 hormonal, edad) en SWAN postmenopausia


def generar_lote_c2(n, d, d_h, S, K, rng):
    """Cohorte 2 (MCI vs HC, grupos balanceados): y = grupo (0/1),
    X = [K PCs cerebrales, PC1 hormonal, edad].

    d   = separación (Cohen) MCI-HC en el componente cerebral verdadero (idx 1).
    d_h = separación MCI-HC en el PC1 hormonal (0 = sin efecto hormonal).
    Grupos pareados por edad (edad ⊥ grupo, como en el diseño del estudio).
    Componentes estandarizados (var 1 marginal).
    """
    g = np.tile(np.concatenate([np.full(n // 2, -0.5),
                                np.full(n - n // 2, 0.5)]), (S, 1))
    edad = rng.standard_normal((S, n))
    E = rng.standard_normal((S, n, K))
    E[:, :, 0] = C_EDAD_PCB1 * edad + np.sqrt(1 - C_EDAD_PCB1 ** 2) * E[:, :, 0]
    delta = d / np.sqrt(1 + d * d / 4)          # var marginal = 1
    E[:, :, 1] = delta * g + np.sqrt(1 - delta ** 2 / 4) * E[:, :, 1]

    dh = d_h / np.sqrt(1 + d_h * d_h / 4)
    var_res = 1 - C_H_AGE_POST ** 2 - dh ** 2 / 4
    hp = (C_H_AGE_POST * edad + dh * g
          + np.sqrt(var_res) * rng.standard_normal((S, n)))

    X = np.concatenate([E, hp[:, :, None], edad[:, :, None]], axis=2)
    y = g + 0.5
    return y, X


def evaluar_celda_c2(n, d, d_h, S, K, rng, n_iter=3000, burn=800):
    y, X = generar_lote_c2(n, d, d_h, S, K, rng)
    frac_pos, _ = gibbs_lasso_lote(y, X, n_iter, burn, rng)
    sig = (frac_pos > 0.975) | (frac_pos < 0.025)
    sig_cereb = sig[:, :K]
    p_any = float(sig_cereb.any(axis=1).mean())
    p_true = float(sig_cereb[:, 1].mean()) if d > 0 else np.nan
    p_horm = float(sig[:, K].mean())
    return p_any, p_true, p_horm


def correr_cohorte2_lasso(cfg, rng, outdir):
    """Grilla Cohorte 2 bajo la lógica de la Cohorte 1 (lasso + criterio ≥1)."""
    grid_n = [40, 60, 80, 100, 120, 160]        # n TOTAL (grupos balanceados)
    celdas = [(n, d, 0.0) for n in grid_n for d in (0.0, 0.5, 0.65, 0.8, 1.0)]
    celdas += [(n, d, 0.3) for n in grid_n for d in (0.0, 0.65, 0.8)]

    filas, t0 = [], time.time()
    for i, (n, d, d_h) in enumerate(celdas, 1):
        p_any, p_true, p_horm = evaluar_celda_c2(n, d, d_h, cfg.nsims, 20,
                                                 rng, cfg.n_iter, cfg.burn)
        filas.append({"n_total": n, "n_por_grupo": n // 2, "d": d, "d_h": d_h,
                      "p_any": p_any, "p_true": p_true, "p_horm": p_horm})
        print(f"[c2 {i:>2}/{len(celdas)}] n={n:>3} d={d:.2f} d_h={d_h:.1f}: "
              f"P(≥1 cereb)={p_any:.3f} P(horm)={p_horm:.3f} "
              f"({time.time()-t0:,.0f}s)", flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(outdir / "resultados_lasso_cohorte2.csv", index=False)

    def nreq_de(filtro):
        g = df.query(filtro).sort_values("n_total")
        p, nn = g.p_any.to_numpy(), g.n_total.to_numpy()
        ok = np.where(p >= cfg.target)[0]
        if len(ok) == 0:
            return np.nan
        i = ok[0]
        if i == 0:
            return float(nn[0])
        return float(np.ceil(nn[i-1] + (cfg.target - p[i-1])
                             * (nn[i] - nn[i-1]) / (p[i] - p[i-1])))

    nreq = {d: nreq_de(f"d=={d} and d_h==0.0") for d in (0.5, 0.65, 0.8, 1.0)}
    resumen = {"criterio": "P(>=1 PC cerebral con CrI95 excluyendo 0) >= "
                           "target; lasso y=grupo ~ PCs cerebro + PC1 hormonal + edad",
               "n_total_requerido_dh0": {str(d): (None if np.isnan(v) else v)
                                         for d, v in nreq.items()}}
    (outdir / "resumen_lasso_cohorte2.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False))

    # -------- figura ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=220,
                             facecolor="white")
    dcol = {0.5: COL["verde"], 0.65: COL["azul"], 0.8: COL["morado"],
            1.0: COL["rojo"]}
    ax = axes[0]
    for d in (0.5, 0.65, 0.8, 1.0):
        g = df.query(f"d=={d} and d_h==0.0").sort_values("n_total")
        ax.plot(g.n_por_grupo, g.p_any, color=dcol[d], lw=2.4,
                label=f"d = {d:.2f}")
        if not np.isnan(nreq[d]):
            ax.plot(nreq[d] / 2, cfg.target, "o", color=dcol[d], ms=7,
                    mec="white", mew=1.2, zorder=6)
    g0 = df.query("d==0.0 and d_h==0.0").sort_values("n_total")
    ax.plot(g0.n_por_grupo, g0.p_any, color=COL["gris"], lw=1.8, ls=":",
            label="global null (family-wise)")
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    ax.axvline(30, color=COL["rojo"], lw=1.1, ls=":", alpha=0.75)
    ax.text(30.6, 0.03, "30/group\n(proposal)", fontsize=8.2,
            color=COL["rojo"])
    ax.set_xlabel("n per group")
    ax.set_ylabel("P(≥1 brain PC significant)")
    ax.set_ylim(0, 1.02)
    ax.set_title("A. Cohort 2 under the Cohort-1 logic: MCI vs HC,\n"
                 f"Bayesian Lasso on brain PCs + {YNAME_C2} + age (K = 20)",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="center right",
              title="MCI-HC separation (1 comp.)")

    ax = axes[1]
    for d, c, lab in [(0.0, COL["gris"], "brain null (d = 0)"),
                      (0.65, COL["azul"], "d = 0.65"),
                      (0.8, COL["morado"], "d = 0.80")]:
        g = df.query(f"d=={d} and d_h==0.3").sort_values("n_total")
        ax.plot(g.n_por_grupo, g.p_horm, color=c, lw=2.4,
                label=f"P({YNAME_C2} significant), {lab}")
    g0h = df.query("d_h==0.0 and d==0.8").sort_values("n_total")
    ax.plot(g0h.n_por_grupo, g0h.p_horm, color=COL["rojo"], lw=1.6, ls=":",
            label="no exposure effect (d$_h$ = 0)")
    ax.set_xlabel("n per group")
    ax.set_ylabel(f"P({YNAME_C2} significant)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"B. Involvement of the {YNAME_C2}\n"
                 "(MCI-HC separation d$_h$ = 0.3)", loc="left",
                 pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    fig.suptitle("Cohort 2 (MCI vs cognitively healthy controls) under the "
                 "laboratory Lasso criterion", y=1.0, fontsize=14,
                 fontweight="bold", color=COL["tinta"])
    fig.text(0.5, -0.02, f"{cfg.nsims} simulated datasets per cell · balanced "
             f"age-matched groups · {C2_FOOT} · significance = "
             "two-sided 95% CrI", ha="center", fontsize=8.4, color=COL["gris"])
    plt.tight_layout()
    fig.savefig(outdir / "Fig_lasso_cohorte2.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    print("\n===== Cohorte 2 (lasso): n TOTAL requerido (P(≥1 cereb) ≥ "
          f"{cfg.target:.2f}) =====")
    for d, v in nreq.items():
        vv = "n.a." if np.isnan(v) else f"{int(v)} ({int(np.ceil(v/2))}/grupo)"
        print(f"  d={d:.2f}: {vv}")


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsims", type=int, default=400)
    ap.add_argument("--n-iter", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=800)
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--extra", action="store_true",
                    help="escenarios K=5/K=14 y criterio BF 4/6/10")
    ap.add_argument("--predictor", choices=["hormonal", "pm25"],
                    default="hormonal",
                    help="pm25: exposición residencial PM2.5 (escalar ACAG) "
                         "en lugar del PC1 hormonal")
    ap.add_argument("--beta-posterior", action="store_true",
                    help="recuperación del beta posterior estandarizado")
    ap.add_argument("--cohorte2", action="store_true",
                    help="Cohorte 2 (MCI vs HC) bajo la lógica lasso, "
                         "incluyendo el PC1 hormonal como predictor")
    cfg = ap.parse_args()

    rng = np.random.default_rng(cfg.seed)
    outdir = Path(__file__).resolve().parent.parent / cfg.outdir
    outdir.mkdir(exist_ok=True)

    if cfg.predictor == "pm25":
        global C_EDAD_Y, C_H_AGE_POST, YNAME, YNAME_C2, C2_FOOT
        C_EDAD_Y = 0.0          # sin base para corr edad-PM2.5 en la cohorte
        C_H_AGE_POST = 0.0
        YNAME = "PM$_{2.5}$ exposure"
        YNAME_C2 = "PM$_{2.5}$ exposure"
        C2_FOOT = ("PM2.5 exposure calibrated on the cohort's residential "
                   "comunas (ACAG 2013-2022; corr with age set to 0)")
        outdir = outdir / "pm25"
        outdir.mkdir(exist_ok=True)
        print("[predictor] PM2.5 residencial (escalar ACAG) · c_edad = 0 · "
              f"salidas en {outdir}/")

    if cfg.verificar:
        verificar_vs_escalar(rng)
        return
    if cfg.extra:
        correr_extra(cfg, rng, outdir)
        return
    if cfg.beta_posterior:
        correr_beta_posterior(cfg, rng, outdir)
        return
    if cfg.cohorte2:
        correr_cohorte2_lasso(cfg, rng, outdir)
        return

    cfg.efectos = (0.2, 0.3, 0.4, 0.5)
    grid_n = [40, 50, 60, 72, 85, 100, 120, 140, 160]
    if cfg.rapido:
        cfg.nsims, grid_n = 80, [50, 72, 100, 140]

    celdas = []
    for n in grid_n:                                  # primario K=20
        for rho in (0.0,) + cfg.efectos:
            celdas.append((20, 1, n, rho))
    for n in [40, 60, 72, 100, 140]:                  # sensibilidad K=10
        for rho in (0.0, 0.3, 0.4):
            celdas.append((10, 1, n, rho))
    for n in grid_n:                                  # 2 componentes verdaderos
        celdas.append((20, 2, n, 0.3))

    filas = []
    t0 = time.time()
    for i, (K, n_true, n, rho) in enumerate(celdas, 1):
        p_any, p_true, mf = evaluar_celda(n, rho, cfg.nsims, K, n_true, rng,
                                          cfg.n_iter, cfg.burn)
        filas.append({"K": K, "n_true": n_true, "n": n, "rho": rho,
                      "p_any": p_any, "p_true": p_true,
                      "media_falsos": mf})
        print(f"[{i:>3}/{len(celdas)}] K={K} n_true={n_true} n={n:>3} "
              f"rho={rho:.1f}: P(≥1)={p_any:.3f} P(true)="
              f"{p_true if np.isnan(p_true) else round(p_true,3)} "
              f"({time.time()-t0:,.0f}s)", flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(outdir / "resultados_lasso_cohorte1.csv", index=False)

    nreq = {}
    for rho in cfg.efectos:
        nreq[(20, 1, rho)] = n_requerido(
            df, cfg.target, f"K==20 and n_true==1 and rho=={rho}")
    for rho in (0.3, 0.4):
        nreq[(10, 1, rho)] = n_requerido(
            df, cfg.target, f"K==10 and n_true==1 and rho=={rho}")
    nreq[(20, 2, 0.3)] = n_requerido(
        df, cfg.target, "K==20 and n_true==2 and rho==0.3")

    resumen = {
        "criterio": "P(>=1 PC cerebral con CrI 95% excluyendo 0, bilateral) "
                    ">= target, lasso bayesiano y=PC1 hormonal ~ PCs cerebro + edad",
        "target": cfg.target,
        "n_requerido": {f"K{K}_true{t}_rho{r}": (None if np.isnan(v) else v)
                        for (K, t, r), v in nreq.items()},
    }
    (outdir / "resumen_lasso_n_requerido.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False))

    figura(df, nreq, cfg, outdir)

    print("\n===== n REQUERIDO (P(≥1 PC cerebral significativo) ≥ "
          f"{cfg.target:.2f}) =====")
    for (K, t, r), v in nreq.items():
        print(f"  K={K:>2} comps, {t} verdadero(s), ρ={r}: n = "
              f"{'n.a.' if np.isnan(v) else int(v)}")
    print(f"\nResultados en {outdir}/")


if __name__ == "__main__":
    main()
