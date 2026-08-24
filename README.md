# Bayesian-sample-size-neuro

Bayesian sample size determination (Bayes Factor Design Analysis, BFDA) for two
neuroimaging cohort studies:

- **Cohort 1**: women aged 40-55 across the menopausal transition (hormonal
  challenge, PC1 of log10 FSH + estradiol) and, in a parallel exposure arm,
  long-term residential PM2.5 exposure.
- **Cohort 2**: adults aged 60-75 with Mild Cognitive Impairment (MCI) vs.
  cognitively healthy controls.

All simulations are calibrated with real data: SWAN Visit-10 hormone assays,
DKT-atlas cortical thickness (54 participants x 62 ROIs), and ACAG satellite
PM2.5 surfaces (2013-2022) mapped to the cohort's residential comunas in Chile.

## Reports

Four reports: versions 3 and 4, each for the hormonal arm and for the PM2.5
exposure arm. Version 3 dimensions the study under the laboratory's own
Bayesian-Lasso pipeline (success = at least one brain principal component with
a 95% credible interval excluding zero, plus posterior standardized beta
recovery). Version 4 uses a permutation-marginalized conditional multivariate
Gaussian generator with global-model Bayes factors (the exact equivalent of
`BayesFactor::regressionBF`).

| Report | Method | Outcome variable |
|---|---|---|
| `sample_size_report_v3` | Bayesian-Lasso pipeline | Hormonal PC1 |
| `sample_size_report_v4` | Global-model BF, conditional MVG | Hormonal PC1 |
| `sample_size_report_v3_pm25` | Bayesian-Lasso pipeline | Residential PM2.5 |
| `sample_size_report_v4_pm25` | Global-model BF, conditional MVG | Residential PM2.5 |

Each report exists in four formats: web page and Quarto source in `docs/`
(`.html` + `.qmd`), and PDF + LaTeX in `reports/`. `docs/index.html` (from
`docs/index.qmd`) is a landing page linking the four reports; the HTML is
standalone (figures embedded) and only loads the MathJax equation renderer from
a CDN. To publish the reports as a website, enable GitHub Pages with source
"Deploy from a branch", branch `main`, folder `/docs`.

## Folder structure

```
code/                        Simulation engines and figure scripts
  bfda_lasso_multivariante.py  Bayesian-Lasso BFDA engine (v3; --predictor pm25)
  bfda_v4_condicional_mvg.py   Conditional-MVG global-BF engine (v4; --predictor pm25)
  bfda_v4_condicional_mvg.R    R reference implementation (BayesFactor::regressionBF)
  bfda_n_optimo.py             Univariate JZS BFDA engine (cross-check; imported by v3)
  validacion_gibbs_lasso.py    Scalar Gibbs Lasso (JAGS port) + Savage-Dickey validation
  swan_figures.py              SWAN descriptive figures (needs the raw SWAN .dta file)
data/                        Frozen calibration inputs
  calibracion_swan_pc1.csv     Standardized hormonal PC1 + age, by STRAW stage
  calibracion_dkt_*.csv        DKT residuals, ROI SDs, eigenvalue spectrum
  calibracion_pm25.csv         Participant-level residential PM2.5 exposure
figures/                     SWAN descriptive figures (used by report v3)
results/                     Simulation outputs, hormonal arm (figures, CSV grids, JSON summaries)
results/pm25/                Simulation outputs, PM2.5 arm
docs/                        Website: index.qmd/.html + the four reports (.qmd + .html) + references.bib
reports/
  pdf/                       Compiled PDF reports
  tex/                       LaTeX sources
```

## Reproducing the simulations

Python 3.11+, `numpy`, `scipy`, `pandas`, `matplotlib`. The engines read
`data/` and write `results/` (both resolved relative to the repository root, so
they can be run from anywhere):

```bash
python code/bfda_lasso_multivariante.py                   # v3 main grid (~15 min)
python code/bfda_lasso_multivariante.py --extra
python code/bfda_lasso_multivariante.py --cohorte2
python code/bfda_lasso_multivariante.py --beta-posterior --nsims 1200
python code/bfda_v4_condicional_mvg.py                    # v4 grid (~2 min)
python code/bfda_v4_condicional_mvg.py --cohorte2
# PM2.5 arm: add --predictor pm25 to either engine (outputs in results/pm25/)
python code/bfda_n_optimo.py                              # univariate cross-check (~40 s)
```

All engines use fixed seeds. The R reference implementation is run from
`code/` with `Rscript bfda_v4_condicional_mvg.R`.

## Re-rendering the reports

[Quarto](https://quarto.org) 1.6+:

```bash
cd docs
quarto render sample_size_report_v4_pm25.qmd   # or index.qmd, or any report
```

The PDFs are compiled from `reports/tex/` with two `pdflatex` passes, e.g.
`pdflatex -jobname=sample_size_report_v4_pm25 report_v4_pm25.tex`, then moved
to `reports/pdf/`.

## Publishing this repository (first push)

```bash
cd bayesian-sample-size-neuro
git init -b main
git add -A
git commit -m "BFDA sample-size reports, engines and calibration data"
gh repo create bayesian-sample-size-neuro --private --source=. --push
# or without GitHub CLI: create an empty repo on github.com, then
# git remote add origin https://github.com/<your-user>/bayesian-sample-size-neuro.git
# git push -u origin main
```
