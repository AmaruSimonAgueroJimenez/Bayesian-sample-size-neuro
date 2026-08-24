# ==============================================================================
# Bayesian Sample Size Estimation via Permutation-Marginalized Conditional MVG
# VERSION 4 — reference R implementation (Luis Luarte, principal;
#             Amaru Aguero, coauthor). August 2026.
#
# Empirical inputs (same calibration files as the Python engine):
#   calibracion_swan_pc1.csv          -> Y pool: standardized hormonal PC1,
#                                        SWAN Visit 10, early perimenopause
#   calibracion_dkt_eigenvalues.csv   -> eigenvalue spectrum of the DKT
#                                        cortical-thickness PCA (40-55 y)
#
# The Python engine (bfda_v4_condicional_mvg.py) replicates regressionBF
# analytically (JZS, rscale = sqrt(2)/4) and runs the full K x tau grid in
# ~1 minute; this script is the faithful BayesFactor-based reference.
# ==============================================================================
library(BayesFactor)
library(ggplot2)
library(dplyr)

# ------------------------------------------------------------------------------
# 1. Inputs & Empirical Baseline
# ------------------------------------------------------------------------------
set.seed(42)
p <- 5   # Number of retained PCA components (explore 5, 10, 14, 20)

eig_df <- read.csv(file.path("..", "data", "calibracion_dkt_eigenvalues.csv"))
eigenvals_emp <- eig_df$eigenvalue[1:p]

pc1_df <- read.csv(file.path("..", "data", "calibracion_swan_pc1.csv"))
y_pool_raw <- pc1_df$PC1[pc1_df$stage == "early_peri"]

# Ensure Y is strictly standardized
y_pool <- as.vector(scale(y_pool_raw))
w_base <- eigenvals_emp / sum(eigenvals_emp)

# ------------------------------------------------------------------------------
# 2. Simulation Grid Parameters
# ------------------------------------------------------------------------------
sample_sizes <- c(50, 100, 150, 200, 250, 300)
r2_grid      <- c(0.05, 0.10, 0.15)
n_sims       <- 500   # Monte Carlo draws per grid point
bf_threshold <- 10    # Explore 4, 6, 10

# ------------------------------------------------------------------------------
# 3. Core Engine (Single Monte Carlo Iteration)
# ------------------------------------------------------------------------------
simulate_single_iteration <- function(N, R2, w_empirical, y_pool_std, p) {
  # Marginalize alignment by shuffling empirical eigenvalue weights
  w_shuffled <- sample(w_empirical)

  # Correlation vector and conditional covariance
  rho <- sqrt(w_shuffled * R2)
  Sigma_cond <- diag(p) - tcrossprod(rho)

  # Cholesky factor (lower triangular)
  L <- t(chol(Sigma_cond))

  # Bootstrap N empirical Y observations
  y_sampled <- matrix(sample(y_pool_std, size = N, replace = TRUE), ncol = 1)

  # Conditional X predictors in a single matrix operation
  Z <- matrix(rnorm(N * p), nrow = N, ncol = p)
  X_sim <- y_sampled %*% t(rho) + Z %*% t(L)

  df_sim <- data.frame(Y = as.vector(y_sampled), X_sim)
  colnames(df_sim) <- c("Y", paste0("PCA", 1:p))

  # Bayesian linear regression BF: full model vs intercept-only null
  bf_fit <- regressionBF(Y ~ ., data = df_sim, progress = FALSE)
  full_name <- paste(paste0("PCA", 1:p), collapse = " + ")
  bf_val <- as.numeric(extractBF(bf_fit)$bf[rownames(extractBF(bf_fit)) ==
                                              full_name])
  return(bf_val >= bf_threshold)
}

# ------------------------------------------------------------------------------
# 4. Monte Carlo Execution
# ------------------------------------------------------------------------------
sim_grid <- expand.grid(N = sample_sizes, R2 = r2_grid)
results  <- vector("list", nrow(sim_grid))

cat("Starting Monte Carlo simulation across parameter grid...\n")
for (i in seq_len(nrow(sim_grid))) {
  curr_N  <- sim_grid$N[i]
  curr_R2 <- sim_grid$R2[i]

  success_vector <- replicate(
    n_sims,
    simulate_single_iteration(curr_N, curr_R2, w_base, y_pool, p)
  )
  empirical_power <- mean(success_vector)

  results[[i]] <- data.frame(N = curr_N,
                             R2 = paste0("R² = ", curr_R2),
                             Power = empirical_power)
  cat(sprintf("Evaluated: N = %3d | R2 = %.2f | Power = %.3f\n",
              curr_N, curr_R2, empirical_power))
}
power_results <- bind_rows(results)

# ------------------------------------------------------------------------------
# 5. Power Curve Plotting
# ------------------------------------------------------------------------------
ggplot(power_results, aes(x = N, y = Power, color = R2, group = R2)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0.80, linetype = "dashed", color = "grey30") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1)) +
  scale_color_brewer(palette = "Set1") +
  theme_minimal(base_size = 13) +
  labs(
    title = "Bayes Factor Design Analysis (BFDA) — v4",
    subtitle = expression("Power to achieve " * BF[10] >= 10 *
                            " across sample sizes"),
    x = "Sample Size (N)",
    y = "Statistical Power (Pr[BF >= threshold])",
    color = "Effect Size"
  ) +
  theme(legend.position = "bottom", panel.grid.minor = element_blank())

ggsave("Fig_v4_power_R.png", width = 8.5, height = 5.5, dpi = 300)
