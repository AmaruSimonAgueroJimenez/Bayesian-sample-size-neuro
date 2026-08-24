#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bfda_n_optimo.py
================================================================================
Determinación bayesiana del tamaño muestral (BSSD / Bayes Factor Design Analysis)
para el proyecto: "The Menopausal Transition as a Probe of Neural Vulnerability".

Pregunta de diseño (Cohorte 1, mujeres 40-55 en transición menopáusica):
    ¿Qué n permite obtener, con probabilidad >= objetivo (p.ej. 0.80), un
    Bayes factor BF10 > umbral (p.ej. 3) para la asociación PARCIAL entre el
    desafío hormonal (PC1 de log-FSH y log-estradiol, calibrado con SWAN
    Visita 10) y un biomarcador cerebral (espesor cortical DKT), CONTROLANDO
    POR EDAD (y opcionalmente otras covariables del pipeline del laboratorio)?

Notas sobre covariables (modelos usados en el laboratorio):
  * La Cohorte 1 incluye SOLO mujeres, por lo que "sexo" es constante y no
    entra como covariable en este diseño (en los modelos generales del lab,
    p.ej. el Lasso JAGS con `cbind(X, nse, sexo-1, IV)`, sí; eso se cubre con
    el argumento --extra-covariables y en el script de validación Gibbs).
  * La EDAD sí entra: el efecto de interés es la correlación parcial
    PC1 -> biomarcador | edad. El BF usa la extensión del BF JZS a
    correlaciones (semi)parciales de Wetzels & Wagenmakers (2012):
    misma integral, con r = r_parcial y n_eff = n - k (k covariables).

Método (Schönbrodt & Wagenmakers, 2018, Psychon Bull Rev — BFDA, diseño n fijo):
  1. Para cada n de una grilla y cada tamaño de efecto poblacional rho_parcial:
     se simulan `nsims` datasets: (PC1, edad) re-muestreados POR PARES de la
     distribución empírica de SWAN (early perimenopause por defecto, preserva
     su dependencia real), y = c_edad*z(edad) + b*z(PC1 ortogonal a edad) +
     ruido (gaussiano o residuales empíricos DKT), con b calibrado para que la
     correlación parcial poblacional sea exactamente rho_parcial.
  2. En cada dataset se calcula el BF JZS por defecto (Liang et al., 2008;
     Wetzels & Wagenmakers, 2012) sobre la correlación parcial muestral:
         BF10 = Int_0^inf (1+g)^((m-2)/2) * [1+(1-R^2) g]^(-(m-1)/2) pi(g) dg
         pi(g) = (m*s^2/2)^(1/2)/Gamma(1/2) * g^(-3/2) * exp(-m*s^2/(2g)),
     con m = n - k y s = rscale (1.0 = prior JZS estándar; equivale a
     pingouin.bayesfactor_pearson).
  3. Potencia bayesiana = P(BF10 > umbral | rho). También:
     - evidencia engañosa bajo H1: P(BF01 > 3 | rho>0)
     - bajo H0 (rho=0): P(BF10 > umbral) [falsa evidencia] y P(BF01 > 3)
       [evidencia correcta a favor del nulo].
  4. BF direccional (sensibilidad): la hipótesis del proyecto es direccional
     (mayor desafío hormonal -> menor espesor). BF-0 = BF10 * 2 * P(beta<0|datos),
     con P(beta<0|datos) exacta mezclando sobre el posterior de g.
  5. Cohorte 2 (MCI vs HC): BF JZS para t de dos muestras (Rouder et al., 2009),
     potencia analítica exacta vía t no central.

Aceleración clave: para (n, rscale) fijos el BF es monótono en R^2, de modo que
P(BF10>k) = P(r_hat^2 > r_crit^2(n,k)); r_crit se busca una sola vez por (n,k)
y la potencia se estima sobre los r_parciales simulados.

Uso:
    python bfda_n_optimo.py                       # configuración por defecto
    python bfda_n_optimo.py --nsims 20000 --pool transition --noise empirico
    python bfda_n_optimo.py --extra-covariables 2 # + covariables tipo nse/IV

Requiere: numpy, pandas, scipy, matplotlib. Archivos de calibración (mismo dir):
    calibracion_swan_pc1.csv          (PC1 + edad por etapa STRAW, SWAN V10)
    calibracion_dkt_residuales.csv    (residuales z de espesor DKT, 40-55 años)
Si no están, puede recalcularlos desde el .dta de SWAN y el CSV DKT (--recalibrar).

Autor: preparado para Amaru Agüero (asesoría SWAN) — agosto 2026.
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
from scipy import optimize, special, stats

# ------------------------------------------------------------------ constantes
COL = {  # paleta consistente con las figuras previas del proyecto
    "azul": "#2563EB", "verde": "#059669", "morado": "#8B5CF6",
    "rojo": "#E11D48", "gris": "#64748B", "gris_claro": "#CBD5E1",
    "tinta": "#0F172A",
}
EFFECT_COLORS = ["#94A3B8", "#059669", "#2563EB", "#8B5CF6", "#E11D48"]

# nodos Gauss-Legendre para todas las integrales 1-D sobre u = log(g)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(400)
_U_LO, _U_HI = -30.0, 30.0
_U = 0.5 * (_U_HI - _U_LO) * _GL_X + 0.5 * (_U_HI + _U_LO)
_WU = 0.5 * (_U_HI - _U_LO) * _GL_W
_G = np.exp(_U)


# ============================================================ Bayes factor JZS
def log_bf10_jzs_corr(r2, m, rscale=1.0):
    """log BF10 JZS para correlación (Wetzels & Wagenmakers, 2012).

    Para correlaciones PARCIALES con k covariables, llamar con m = n - k
    y r2 = r_parcial^2 (extensión semiparcial de W&W 2012, sec. 'partial
    correlations'). Vectorizado sobre r2; integral en u=log(g) con
    Gauss-Legendre, en escala log para estabilidad numérica.
    """
    r2 = np.atleast_1d(np.asarray(r2, float))[:, None]        # (R,1)
    ms2 = m * rscale ** 2
    log_prior_jac = (0.5 * np.log(ms2 / 2.0) - special.gammaln(0.5)
                     - 0.5 * _U - ms2 / (2.0 * _G))            # incluye jacobiano g du
    logf = (log_prior_jac
            + ((m - 2.0) / 2.0) * np.log1p(_G)
            - ((m - 1.0) / 2.0) * np.log1p((1.0 - r2) * _G))   # (R,400)
    return special.logsumexp(logf + np.log(_WU), axis=1)


def r2_critico(m, bf_umbral, rscale=1.0):
    """Menor R^2 tal que BF10 >= umbral. NaN si inalcanzable (R^2<0.999)."""
    logk = np.log(bf_umbral)
    f = lambda r2: log_bf10_jzs_corr(r2, m, rscale)[0] - logk
    lo, hi = 1e-12, 0.999
    if f(hi) < 0:      # ni un R^2 casi perfecto alcanza el umbral
        return np.nan
    if f(lo) >= 0:     # el umbral se cumple ya en R^2=0 (solo pasa con k<1)
        return 0.0
    return optimize.brentq(f, lo, hi, xtol=1e-12)


def prob_beta_negativa(r_signed, m):
    """P(beta<0 | datos) exacta bajo el modelo JZS, mezclando sobre p(g|datos).

    Posterior condicional: beta | g, datos ~ t_{m-1} centrada en delta*beta_hat,
    delta = g/(1+g); estandarizando:
        P(beta<0 | g, datos) = T_{m-1}( -r_hat * sqrt(delta*(m-1)/(1-delta*r_hat^2)) ).
    Pesos sobre g: proporcionales al integrando del BF (posterior de g).
    """
    r_signed = np.atleast_1d(np.asarray(r_signed, float))[:, None]
    r2 = r_signed ** 2
    ms2 = m * 1.0
    log_prior_jac = (0.5 * np.log(ms2 / 2.0) - special.gammaln(0.5)
                     - 0.5 * _U - ms2 / (2.0 * _G))
    logf = (log_prior_jac
            + ((m - 2.0) / 2.0) * np.log1p(_G)
            - ((m - 1.0) / 2.0) * np.log1p((1.0 - r2) * _G))
    w = np.exp(logf - special.logsumexp(logf, axis=1, keepdims=True))
    delta = _G / (1.0 + _G)
    z0 = -r_signed * np.sqrt(delta * (m - 1.0) / (1.0 - delta * r2))
    return np.sum(w * stats.t.cdf(z0, df=m - 1), axis=1)


def log_bf_direccional(r_signed, m, rscale=1.0):
    """log BF-0 (H-: beta<0 vs H0) = log BF10 + log(2*P(beta<0|datos))."""
    r_signed = np.atleast_1d(np.asarray(r_signed, float))
    lb = log_bf10_jzs_corr(r_signed ** 2, m, rscale)
    p_neg = np.clip(prob_beta_negativa(r_signed, m), 1e-300, 1.0)
    return lb + np.log(2.0 * p_neg)


def log_bf10_jzs_ttest(t, n1, n2, rscale=np.sqrt(2) / 2):
    """log BF10 JZS para t de dos muestras independientes (Rouder et al., 2009)."""
    t = np.atleast_1d(np.asarray(t, float))[:, None]
    nu = n1 + n2 - 2.0
    neff = n1 * n2 / (n1 + n2)
    r2 = rscale ** 2
    log_prior_jac = (0.5 * np.log(r2 / 2.0) - special.gammaln(0.5)
                     - 0.5 * _U - r2 / (2.0 * _G))
    log_m1 = (log_prior_jac
              - 0.5 * np.log1p(neff * _G)
              - ((nu + 1.0) / 2.0) * np.log1p(t ** 2 / ((1.0 + neff * _G) * nu)))
    log_m1 = special.logsumexp(log_m1 + np.log(_WU), axis=1)
    log_m0 = -((nu + 1.0) / 2.0) * np.log1p(t.ravel() ** 2 / nu)
    return log_m1 - log_m0


def t_critico(n1, n2, bf_umbral, rscale=np.sqrt(2) / 2):
    logk = np.log(bf_umbral)
    f = lambda t: log_bf10_jzs_ttest(t, n1, n2, rscale)[0] - logk
    if f(50.0) < 0:
        return np.nan
    if f(1e-9) >= 0:
        return 0.0
    return optimize.brentq(f, 1e-9, 50.0, xtol=1e-10)


# ============================================================== calibración
def cargar_pools(base: Path, recalibrar: bool = False):
    """Devuelve (dict de DataFrames PC1+edad por pool, residuales z DKT)."""
    f_pc1 = base / "calibracion_swan_pc1.csv"
    f_res = base / "calibracion_dkt_residuales.csv"
    if recalibrar or not (f_pc1.exists() and f_res.exists()):
        _recalibrar(base, f_pc1, f_res)
    pc1 = pd.read_csv(f_pc1)
    pools = {
        "early_peri": pc1.loc[pc1.stage == "early_peri", ["PC1", "age"]],
        "transition": pc1.loc[pc1.stage.isin(["pre", "early_peri",
                                              "late_peri"]), ["PC1", "age"]],
        "all": pc1[["PC1", "age"]],
    }
    pools = {k: v.dropna().reset_index(drop=True) for k, v in pools.items()}
    resid = pd.read_csv(f_res)["z_residual"].to_numpy()
    return pools, resid


def _recalibrar(base, f_pc1, f_res):
    """Reconstruye la calibración desde SWAN (.dta) y los CSV DKT si existen."""
    dta = next(base.glob("**/32961-0001-Data.dta"), None)
    dkt = base / "DKTatlas_thickness.csv"
    if dta is None or not dkt.exists():
        raise FileNotFoundError(
            "Faltan los archivos de calibración y no se encuentran las fuentes "
            "(32961-0001-Data.dta y DKTatlas_thickness.csv) para recalcularlos.")
    sw = pd.read_stata(dta, convert_categoricals=False)
    d = sw[["AGE10", "FSH10", "E2AVE10", "STATUS10"]].dropna()
    d = d[(d.FSH10 > 0) & (d.E2AVE10 > 0)].reset_index(drop=True)
    lh = np.log10(d[["FSH10", "E2AVE10"]].to_numpy(float))
    z = (lh - lh.mean(0)) / lh.std(0, ddof=1)
    ev, C = np.linalg.eigh(np.cov(z, rowvar=False))
    o = np.argsort(ev)[::-1]
    C = C[:, o]
    if C[0, 0] < 0:
        C[:, 0] *= -1
    d["PC1"] = (z @ C)[:, 0]
    stage = d.STATUS10.map({2: "post", 3: "late_peri", 4: "early_peri",
                            5: "pre"}).fillna("other")
    pd.DataFrame({"PC1": d.PC1.round(6), "stage": stage,
                  "age": d.AGE10}).to_csv(f_pc1, index=False)

    th = pd.read_csv(dkt)
    age = th.EDAD_MESES / 12
    sub = th[(age >= 40) & (age <= 55)].copy()
    a = (sub.EDAD_MESES / 12).to_numpy()
    res = []
    for c in th.columns[2:]:
        y = sub[c].to_numpy(float)
        X = np.column_stack([np.ones_like(a), a])
        r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        res.append(r / r.std(ddof=1))
    pd.DataFrame({"z_residual": np.round(np.concatenate(res), 6)}
                 ).to_csv(f_res, index=False)


# ============================================================== simulación
def _corr_parcial_lote(y, x, Z):
    """Correlación parcial fila a fila entre y y x controlando por Z.

    y, x: (S, n); Z: (S, n, k) covariables (sin intercepto; se agrega aquí).
    Residualiza x e y sobre [1, Z] con ecuaciones normales por lote.
    """
    S, n = y.shape
    ones = np.ones((S, n, 1))
    Zc = np.concatenate([ones, Z], axis=2) if Z is not None else ones
    ZtZ = np.einsum("spk,spl->skl", Zc, Zc)
    Zty = np.einsum("spk,sp->sk", Zc, y)
    Ztx = np.einsum("spk,sp->sk", Zc, x)
    coef_y = np.linalg.solve(ZtZ, Zty[:, :, None])[:, :, 0]
    coef_x = np.linalg.solve(ZtZ, Ztx[:, :, None])[:, :, 0]
    ry = y - np.einsum("spk,sk->sp", Zc, coef_y)
    rx = x - np.einsum("spk,sk->sp", Zc, coef_x)
    num = np.einsum("ij,ij->i", rx, ry)
    den = np.sqrt(np.einsum("ij,ij->i", rx, rx)
                  * np.einsum("ij,ij->i", ry, ry))
    return num / den


def simular_r_parcial(n, rho_parcial, pool, resid_pool, nsims, rng,
                      noise="gaussiano", c_edad=-0.15, k_extra=0):
    """r parcial muestral (PC1 -> y | edad [+extras]) con rho poblacional dado.

    (PC1, edad) se re-muestrean POR PARES del pool empírico SWAN (preserva su
    dependencia real); se estandarizan con momentos del POOL. El componente de
    PC1 ortogonal a edad recibe coeficiente b = rho_parcial*sqrt(1-c_edad^2),
    lo que induce una correlación parcial poblacional EXACTA de rho_parcial.
    """
    pc1 = pool["PC1"].to_numpy()
    edad = pool["age"].to_numpy()
    r12 = np.corrcoef(pc1, edad)[0, 1]

    idx = rng.integers(0, len(pool), size=(nsims, n))
    x = pc1[idx]
    a = edad[idx]
    xz = (x - pc1.mean()) / pc1.std(ddof=1)
    az = (a - edad.mean()) / edad.std(ddof=1)
    x_perp = (xz - r12 * az) / np.sqrt(1.0 - r12 ** 2)   # ⊥ edad, var 1 (pobl.)

    b = rho_parcial * np.sqrt(1.0 - c_edad ** 2)
    var_resid = 1.0 - b ** 2 - c_edad ** 2
    if var_resid <= 0:
        raise ValueError("c_edad y rho_parcial incompatibles (var residual <=0)")
    if noise == "empirico":
        e = rng.choice(resid_pool, size=(nsims, n), replace=True)
        e = (e - resid_pool.mean()) / resid_pool.std(ddof=1)
    else:
        e = rng.standard_normal((nsims, n))
    y = c_edad * az + b * x_perp + np.sqrt(var_resid) * e

    Z = az[:, :, None]
    if k_extra > 0:  # covariables adicionales estilo lab (nse, IV, ...), sin efecto
        extras = rng.standard_normal((nsims, n, k_extra))
        Z = np.concatenate([Z, extras], axis=2)
    return _corr_parcial_lote(y, x, Z)


def correr_cohorte1(cfg, pools, resid, rng):
    """Grilla completa n x rho para la Cohorte 1 (PC1 -> cerebro | edad)."""
    pool = pools[cfg.pool]
    k = 1 + cfg.extra_covariables            # edad + extras
    filas = []
    bf_n_foco = {}
    grid_r = np.linspace(-0.995, 0.995, 1501)
    grid_r2 = np.linspace(0.0, 0.995, 800) ** 2

    for n in cfg.grid_n:
        m = n - k                             # n efectivo (W&W 2012, parciales)
        if m < 5:
            continue
        rc = {u: r2_critico(m, u, cfg.rscale) for u in cfg.umbrales}
        rc_null = r2_critico(m, 1.0 / 3.0, cfg.rscale)
        lbd = log_bf_direccional(grid_r, m, cfg.rscale)
        r_dir_crit = {}
        for u in cfg.umbrales:
            ok = np.where(lbd >= np.log(u))[0]
            r_dir_crit[u] = grid_r[ok.max()] if len(ok) else np.nan
        log_bf_grid = log_bf10_jzs_corr(grid_r2, m, cfg.rscale)

        for rho in [0.0] + list(cfg.efectos):
            # signo negativo: hipótesis direccional (más desafío -> menos espesor)
            r_hat = simular_r_parcial(n, -abs(rho), pool, resid, cfg.nsims,
                                      rng, cfg.noise, cfg.c_edad,
                                      cfg.extra_covariables)
            r2_hat = r_hat ** 2
            fila = {"n": n, "rho": rho}
            for u in cfg.umbrales:
                fila[f"p_bf{u}"] = np.nan if np.isnan(rc[u]) else float(
                    np.mean(r2_hat > rc[u]))
                fila[f"p_bf{u}_dir"] = np.nan if np.isnan(r_dir_crit[u]) \
                    else float(np.mean(r_hat < r_dir_crit[u]))
            fila["p_nulo"] = 0.0 if np.isnan(rc_null) else float(
                np.mean(r2_hat < rc_null))
            fila["bf_mediano"] = float(np.exp(log_bf10_jzs_corr(
                np.median(r2_hat), m, cfg.rscale)[0]))
            filas.append(fila)

            if n == cfg.n_foco:
                bf_n_foco[rho] = np.interp(r2_hat, grid_r2, log_bf_grid)
    return pd.DataFrame(filas), bf_n_foco


def correr_cohorte2(cfg):
    """Cohorte 2: MCI vs HC. Potencia analítica exacta (t no central)."""
    filas = []
    for npg in cfg.grid_n_grupo:
        neff = npg / 2.0
        tc = {u: t_critico(npg, npg, u, cfg.rscale_t) for u in cfg.umbrales}
        tc_null = t_critico(npg, npg, 1.0 / 3.0, cfg.rscale_t)
        for d in [0.0] + list(cfg.efectos_d):
            ncp = d * np.sqrt(neff)
            df = 2 * npg - 2
            fila = {"n_por_grupo": npg, "d": d}
            for u in cfg.umbrales:
                if np.isnan(tc[u]):
                    fila[f"p_bf{u}"] = np.nan
                else:
                    fila[f"p_bf{u}"] = float(stats.nct.sf(tc[u], df, ncp)
                                             + stats.nct.cdf(-tc[u], df, ncp))
            if np.isnan(tc_null) or tc_null == 0.0:
                fila["p_nulo"] = 0.0
            else:
                fila["p_nulo"] = float(stats.nct.cdf(tc_null, df, ncp)
                                       - stats.nct.cdf(-tc_null, df, ncp))
            filas.append(fila)
    return pd.DataFrame(filas)


def n_optimo(df, col, target, por="rho"):
    """Menor n (interpolado, redondeado hacia arriba) con potencia >= target."""
    out = {}
    xcol = "n" if por == "rho" else "n_por_grupo"
    for eff, g in df[df[por] > 0].groupby(por):
        g = g.sort_values(xcol)
        p = g[col].to_numpy()
        nn = g[xcol].to_numpy()
        idx = np.where(p >= target)[0]
        if len(idx) == 0:
            out[eff] = np.nan
            continue
        i = idx[0]
        if i == 0:
            out[eff] = float(nn[0])
        else:
            x0, x1, y0, y1 = nn[i - 1], nn[i], p[i - 1], p[i]
            out[eff] = float(np.ceil(x0 + (target - y0) * (x1 - x0) / (y1 - y0)))
    return out


# ================================================================= figuras
def _estilo(ax):
    ax.grid(color="#E2E8F0", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COL["gris_claro"])


def figura_cohorte1(df, bf_foco, nopt, cfg, out):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.2), dpi=220,
                             facecolor="white")

    # A) curvas de potencia BF>3 por rho -----------------------------------
    ax = axes[0, 0]
    for rho, c in zip(cfg.efectos, EFFECT_COLORS):
        g = df[df.rho == rho].sort_values("n")
        ax.plot(g.n, g[f"p_bf{cfg.umbral_1}"], color=c, lw=2.4,
                label=f"|ρ| = {rho:.1f}")
        if not np.isnan(nopt.get(rho, np.nan)):
            ax.plot(nopt[rho], cfg.target, "o", color=c, ms=7, mec="white",
                    mew=1.2, zorder=6)
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    ax.text(df.n.max(), cfg.target + 0.012, f"target = {cfg.target:.2f}",
            ha="right", fontsize=8.6, color=COL["tinta"])
    ax.axvline(cfg.n_foco, color=COL["rojo"], lw=1.2, ls=":", alpha=0.8)
    ax.text(cfg.n_foco + 1.5, 0.03, f"n = {cfg.n_foco}\n(proposal)",
            fontsize=8.6, color=COL["rojo"])
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel(f"P(BF$_{{10}}$ > {cfg.umbral_1})")
    ax.set_ylim(0, 1.02)
    ax.set_title("A. Bayesian power, two-sided JZS Bayes factor", loc="left",
                 pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right",
              title="Partial effect |ρ| (PC1 | age)")

    # B) tres umbrales para rho = 0.3 --------------------------------------
    ax = axes[0, 1]
    g = df[df.rho == 0.3].sort_values("n")
    for u, c in zip(cfg.umbrales, [COL["verde"], COL["azul"], COL["morado"]]):
        ax.plot(g.n, g[f"p_bf{u}"], color=c, lw=2.4, label=f"BF$_{{10}}$ > {u}")
        ax.plot(g.n, g[f"p_bf{u}_dir"], color=c, lw=1.6, ls="--", alpha=0.8)
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    ax.axvline(cfg.n_foco, color=COL["rojo"], lw=1.2, ls=":", alpha=0.8)
    ax.plot([], [], color=COL["gris"], lw=1.6, ls="--",
            label="directional prior (dashed)")
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("P(BF > threshold)")
    ax.set_ylim(0, 1.02)
    ax.set_title("B. Evidence thresholds at |ρ| = 0.3", loc="left", pad=10,
                 fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # C) comportamiento bajo H0 --------------------------------------------
    ax = axes[1, 0]
    g0 = df[df.rho == 0.0].sort_values("n")
    ax.plot(g0.n, g0.p_nulo, color=COL["verde"], lw=2.4,
            label="P(BF$_{01}$ > 3 | H0)  correct null evidence")
    ax.plot(g0.n, g0[f"p_bf{cfg.umbral_1}"], color=COL["rojo"], lw=2.4,
            label=f"P(BF$_{{10}}$ > {cfg.umbral_1} | H0)  misleading evidence")
    ax.axvline(cfg.n_foco, color=COL["rojo"], lw=1.2, ls=":", alpha=0.8)
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("C. Behaviour when the true effect is null", loc="left",
                 pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="center right")

    # D) distribución de BF en n_foco --------------------------------------
    ax = axes[1, 1]
    pos = np.arange(len(cfg.efectos))
    for i, (rho, c) in enumerate(zip(cfg.efectos, EFFECT_COLORS)):
        lb = bf_foco.get(rho)
        if lb is None:
            continue
        q = np.percentile(lb, [5, 25, 50, 75, 95]) / np.log(10)
        ax.plot([i, i], [q[0], q[4]], color=c, lw=1.6, alpha=0.85)
        ax.add_patch(plt.Rectangle((i - 0.18, q[1]), 0.36, q[3] - q[1],
                                   facecolor=c, alpha=0.55, edgecolor=c))
        ax.plot([i - 0.18, i + 0.18], [q[2], q[2]], color="white", lw=2.2,
                zorder=5)
        ax.plot([i - 0.18, i + 0.18], [q[2], q[2]], color=c, lw=1.1, zorder=6)
    for u, ls in zip(cfg.umbrales, [":", "--", "-."]):
        ax.axhline(np.log10(u), color=COL["tinta"], lw=1.0, ls=ls, alpha=0.55)
        ax.text(len(cfg.efectos) - 0.45, np.log10(u) + 0.03, f"BF = {u}",
                fontsize=8.2, color=COL["tinta"])
    ax.axhline(0, color=COL["gris_claro"], lw=0.9)
    ax.set_xticks(pos, [f"{r:.1f}" for r in cfg.efectos])
    ax.set_xlabel("True partial effect |ρ|")
    ax.set_ylabel("log$_{10}$ BF$_{10}$")
    ax.set_title(f"D. BF distribution at n = {cfg.n_foco} "
                 "(median, IQR, 5–95%)", loc="left", pad=10, fontweight="bold")
    _estilo(ax)

    fig.suptitle("Bayes Factor Design Analysis — Cohort 1: hormonal challenge "
                 "(PC1 FSH/estradiol) → brain feature, controlling for age",
                 y=0.995, fontsize=14.5, fontweight="bold", color=COL["tinta"])
    fig.text(0.5, 0.012,
             f"(PC1, age) pairs resampled from empirical SWAN Visit-10 "
             f"distribution ({cfg.pool}, n={cfg.pool_n}) · age effect on "
             f"outcome = {cfg.c_edad:+.2f} · covariates k = "
             f"{1 + cfg.extra_covariables} · noise: "
             f"{'Gaussian' if cfg.noise == 'gaussiano' else 'empirical DKT residuals'} · "
             f"{cfg.nsims:,} datasets/cell · JZS r-scale = {cfg.rscale:g} · "
             f"seed = {cfg.seed}",
             ha="center", fontsize=8.4, color=COL["gris"])
    plt.tight_layout(rect=(0, 0.025, 1, 0.975))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figura_cohorte2(df2, nopt2, cfg, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=220,
                             facecolor="white")
    ax = axes[0]
    for d, c in zip(cfg.efectos_d, [COL["verde"], COL["azul"], COL["morado"],
                                    COL["rojo"]]):
        g = df2[df2.d == d].sort_values("n_por_grupo")
        ax.plot(g.n_por_grupo, g[f"p_bf{cfg.umbral_1}"], color=c, lw=2.4,
                label=f"d = {d:.2f}")
        if not np.isnan(nopt2.get(d, np.nan)):
            ax.plot(nopt2[d], cfg.target, "o", color=c, ms=7, mec="white",
                    mew=1.2, zorder=6)
    ax.axhline(cfg.target, color=COL["tinta"], lw=1.1, ls="--", alpha=0.65)
    ax.axvline(30, color=COL["rojo"], lw=1.2, ls=":", alpha=0.8)
    ax.text(30.6, 0.04, "n = 30/group\n(proposal)", fontsize=8.6,
            color=COL["rojo"])
    ax.set_xlabel("n per group")
    ax.set_ylabel(f"P(BF$_{{10}}$ > {cfg.umbral_1})")
    ax.set_ylim(0, 1.02)
    ax.set_title("A. Bayesian power, MCI vs HC (JZS t-test BF)", loc="left",
                 pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right",
              title="Group difference")

    ax = axes[1]
    g0 = df2[df2.d == 0.0].sort_values("n_por_grupo")
    ax.plot(g0.n_por_grupo, g0.p_nulo, color=COL["verde"], lw=2.4,
            label="P(BF$_{01}$ > 3 | H0)")
    ax.plot(g0.n_por_grupo, g0[f"p_bf{cfg.umbral_1}"], color=COL["rojo"],
            lw=2.4, label=f"P(BF$_{{10}}$ > {cfg.umbral_1} | H0)")
    ax.axvline(30, color=COL["rojo"], lw=1.2, ls=":", alpha=0.8)
    ax.set_xlabel("n per group")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("B. Behaviour under the null", loc="left", pad=10,
                 fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9, loc="center right")

    fig.suptitle("BFDA — Cohort 2: MCI vs cognitively healthy controls",
                 y=1.0, fontsize=14, fontweight="bold", color=COL["tinta"])
    fig.text(0.5, -0.02, f"Exact power via non-central t · JZS r-scale = "
             f"{cfg.rscale_t:.3f} (default 'medium')",
             ha="center", fontsize=8.6, color=COL["gris"])
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figura_calibracion(pools, resid, cfg, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=220,
                             facecolor="white")
    ax = axes[0]
    bins = np.linspace(-6.5, 2.5, 46)
    ax.hist(pools["all"]["PC1"], bins=bins, color=COL["gris_claro"], alpha=0.8,
            density=True, label=f"All stages (n={len(pools['all']):,})")
    ax.hist(pools["early_peri"]["PC1"], bins=bins, histtype="step", lw=2.4,
            color=COL["azul"], density=True,
            label=f"Early perimenopause (n={len(pools['early_peri'])})")
    ax.set_xlabel("PC1 score (higher FSH / lower estradiol)")
    ax.set_ylabel("Density")
    ax.set_title("A. Empirical predictor distribution (SWAN Visit 10)",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    zz = np.linspace(-4, 4, 200)
    ax.hist(resid, bins=60, color=COL["morado"], alpha=0.55, density=True,
            label="DKT thickness residuals\n(age-adjusted, 40–55 y)")
    ax.plot(zz, stats.norm.pdf(zz), color=COL["tinta"], lw=1.8,
            label="Standard normal")
    ax.set_xlabel("Standardized residual")
    ax.set_ylabel("Density")
    ax.set_title("B. Outcome noise calibration (DKT cortical thickness)",
                 loc="left", pad=10, fontweight="bold")
    _estilo(ax)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Calibration of the simulation with real data", y=1.0,
                 fontsize=13.5, fontweight="bold", color=COL["tinta"])
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(description="BFDA / BSSD para n óptimo")
    ap.add_argument("--nsims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--pool", choices=["early_peri", "transition", "all"],
                    default="early_peri")
    ap.add_argument("--noise", choices=["gaussiano", "empirico"],
                    default="gaussiano")
    ap.add_argument("--rscale", type=float, default=1.0)
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--c-edad", type=float, default=-0.15,
                    help="efecto (corr) de edad sobre el biomarcador; "
                         "-0.15 según DKT 40-55 años")
    ap.add_argument("--extra-covariables", type=int, default=0,
                    help="covariables adicionales estilo lab (nse, IV...). "
                         "Sexo NO entra en Cohorte 1: solo mujeres.")
    ap.add_argument("--n-foco", type=int, default=60,
                    help="n de la propuesta, destacado en figuras")
    ap.add_argument("--n-min", type=int, default=20)
    ap.add_argument("--n-max", type=int, default=200)
    ap.add_argument("--n-step", type=int, default=5)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--recalibrar", action="store_true")
    cfg = ap.parse_args()

    cfg.umbrales = (3, 6, 10)
    cfg.umbral_1 = 3
    cfg.efectos = (0.1, 0.2, 0.3, 0.4, 0.5)
    cfg.efectos_d = (0.5, 0.65, 0.8, 1.0)
    cfg.grid_n = list(range(cfg.n_min, cfg.n_max + 1, cfg.n_step))
    if cfg.n_foco not in cfg.grid_n:
        cfg.grid_n = sorted(set(cfg.grid_n) | {cfg.n_foco})
    cfg.grid_n_grupo = list(range(10, 151, 2))
    cfg.rscale_t = float(np.sqrt(2) / 2)

    raiz = Path(__file__).resolve().parent.parent
    base = raiz / "data"
    outdir = raiz / cfg.outdir
    outdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    pools, resid = cargar_pools(base, cfg.recalibrar)
    cfg.pool_n = len(pools[cfg.pool])
    print(f"[calibración] pool '{cfg.pool}': n={cfg.pool_n} "
          f"(sd PC1={pools[cfg.pool]['PC1'].std(ddof=1):.2f}, "
          f"corr PC1-edad={np.corrcoef(pools[cfg.pool]['PC1'], pools[cfg.pool]['age'])[0, 1]:+.2f}) "
          f"· residuales DKT: {len(resid):,}")

    print(f"[cohorte 1] grilla n={cfg.grid_n[0]}–{cfg.grid_n[-1]}, "
          f"{cfg.nsims:,} sims/celda, covariables k={1 + cfg.extra_covariables} "
          "(edad" + (" + extras" if cfg.extra_covariables else "") + ") ...")
    df1, bf_foco = correr_cohorte1(cfg, pools, resid, rng)
    df1.to_csv(outdir / "resultados_bfda_cohorte1.csv", index=False)

    print("[cohorte 2] potencia exacta MCI vs HC ...")
    df2 = correr_cohorte2(cfg)
    df2.to_csv(outdir / "resultados_bfda_cohorte2.csv", index=False)

    # ------- tablas de n óptimo
    tablas = {}
    for u in cfg.umbrales:
        tablas[f"BF>{u} (dos colas)"] = n_optimo(df1, f"p_bf{u}", cfg.target)
        tablas[f"BF>{u} (direccional)"] = n_optimo(df1, f"p_bf{u}_dir",
                                                   cfg.target)
    nopt1 = tablas[f"BF>{cfg.umbral_1} (dos colas)"]
    nopt1_dir = tablas[f"BF>{cfg.umbral_1} (direccional)"]
    nopt2 = n_optimo(df2, f"p_bf{cfg.umbral_1}", cfg.target, por="d")

    tab = pd.DataFrame(tablas).rename_axis("|rho_parcial|")
    tab.to_csv(outdir / "tabla_n_optimo_cohorte1.csv")
    pd.Series(nopt2, name=f"n/grupo BF>{cfg.umbral_1}").rename_axis("d").to_csv(
        outdir / "tabla_n_optimo_cohorte2.csv")

    # ------- resumen en n_foco
    foco = df1[df1.n == cfg.n_foco].set_index("rho")
    resumen = {
        "config": {kk: vv for kk, vv in vars(cfg).items()
                   if isinstance(vv, (int, float, str, list, tuple))},
        "nota_covariables": "Cohorte 1: solo mujeres (sexo constante, no "
                            "entra); edad como covariable (BF sobre "
                            "correlacion parcial, W&W 2012, n_eff=n-k).",
        "n_optimo_cohorte1_dos_colas": nopt1,
        "n_optimo_cohorte1_direccional": nopt1_dir,
        "n_optimo_cohorte2_por_grupo": nopt2,
        "potencia_en_n_foco": {
            str(r): {"p_bf3": round(float(foco.loc[r, "p_bf3"]), 3),
                     "p_bf3_dir": round(float(foco.loc[r, "p_bf3_dir"]), 3),
                     "bf_mediano": round(float(foco.loc[r, "bf_mediano"]), 2)}
            for r in cfg.efectos},
        "H0_en_n_foco": {
            "p_evidencia_correcta_nulo": round(float(
                foco.loc[0.0, "p_nulo"]), 3),
            "p_evidencia_enganosa": round(float(foco.loc[0.0, "p_bf3"]), 3)},
    }
    (outdir / "resumen_n_optimo.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False))

    # ------- figuras
    figura_cohorte1(df1, bf_foco, nopt1, cfg, outdir / "Fig_BFDA_cohorte1.png")
    figura_cohorte2(df2, nopt2, cfg, outdir / "Fig_BFDA_cohorte2.png")
    figura_calibracion(pools, resid, cfg, outdir / "Fig_calibracion.png")

    # ------- consola
    print("\n========== n ÓPTIMO — Cohorte 1 (PC1 → biomarcador | edad) =====")
    print(tab.to_string(float_format=lambda v: f"{v:,.0f}"))
    print(f"\nEn n = {cfg.n_foco} (propuesta): potencia P(BF10>3) por |rho| parcial:")
    for r in cfg.efectos:
        print(f"  |rho|={r:.1f}: dos colas {foco.loc[r, 'p_bf3']:.3f} | "
              f"direccional {foco.loc[r, 'p_bf3_dir']:.3f} | "
              f"BF mediano {foco.loc[r, 'bf_mediano']:.2f}")
    print(f"H0 en n={cfg.n_foco}: P(BF01>3)={foco.loc[0.0, 'p_nulo']:.3f}, "
          f"P(BF10>3)={foco.loc[0.0, 'p_bf3']:.3f}")
    print("\n========== n ÓPTIMO — Cohorte 2 (MCI vs HC, por grupo) =========")
    for d, v in nopt2.items():
        print(f"  d={d:.2f}: n/grupo = {v:,.0f}")
    print(f"\nResultados en: {outdir}/")


if __name__ == "__main__":
    main()
