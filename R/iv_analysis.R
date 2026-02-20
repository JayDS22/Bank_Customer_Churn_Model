#!/usr/bin/env Rscript
# ==============================================================================
# Instrumental Variables (2SLS) Analysis in R
# ==============================================================================
# Uses the AER/ivreg package for IV estimation with robust standard errors.
#
# Usage:
#   Rscript R/iv_analysis.R [input_csv]
#
# If no input_csv is provided, generates synthetic data internally.
# ==============================================================================

suppressPackageStartupMessages({
  library(AER)       # ivreg
  library(sandwich)  # robust SEs
  library(lmtest)    # coeftest
})

cat("==============================================================\n")
cat("IV/2SLS Analysis (R) — Consumer Lending Fairness Audit\n")
cat("==============================================================\n\n")

# --- Load or generate data ---
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1 && file.exists(args[1])) {
  cat("Loading data from:", args[1], "\n")
  df <- read.csv(args[1])
} else {
  cat("No input file. Generating synthetic data...\n")
  set.seed(42)
  n <- 10000

  state_unemp <- runif(n, 3.5, 8.0)
  annual_inc <- exp(rnorm(n, mean = 10.8, sd = 0.7)) * (1 - 0.03 * (state_unemp - 5.5))
  annual_inc <- pmax(annual_inc, 10000)

  dti <- pmax(rnorm(n, 17, 8), 0)
  open_acc <- rpois(n, 11)
  revol_util <- rbeta(n, 2, 3) * 100
  total_acc <- open_acc + rpois(n, 14)

  logit <- -2.5 - 0.00001 * annual_inc + 0.015 * dti + 0.005 * revol_util
  logit <- logit + 0.25 * as.numeric(annual_inc < 40000)  # inject bias
  prob_default <- 1 / (1 + exp(-logit))
  default <- rbinom(n, 1, prob_default)

  df <- data.frame(
    default = default,
    annual_inc = annual_inc,
    log_income = log(annual_inc + 1),
    state_unemployment_rate = state_unemp,
    dti = dti,
    open_acc = open_acc,
    revol_util = revol_util,
    total_acc = total_acc
  )
}

if (!"log_income" %in% names(df)) {
  df$log_income <- log(df$annual_inc + 1)
}

cat("Data dimensions:", nrow(df), "x", ncol(df), "\n")
cat("Default rate:", mean(df$default, na.rm = TRUE), "\n\n")

# --- OLS baseline (biased) ---
cat("--- OLS Baseline (potentially biased) ---\n")
ols_model <- lm(default ~ log_income + dti + open_acc + revol_util + total_acc, data = df)
cat("OLS Results:\n")
print(coeftest(ols_model, vcov = vcovHC(ols_model, type = "HC1")))

# --- First Stage ---
cat("\n--- First Stage: log_income ~ state_unemployment_rate + controls ---\n")
first_stage <- lm(log_income ~ state_unemployment_rate + dti + open_acc + revol_util + total_acc, data = df)
cat("First Stage Summary:\n")
fs_summary <- summary(first_stage)
cat("F-statistic:", fs_summary$fstatistic[1], "\n")
cat("R-squared:", fs_summary$r.squared, "\n")
cat("Instrument coefficient:", coef(first_stage)["state_unemployment_rate"], "\n")
cat("Instrument p-value:", summary(first_stage)$coefficients["state_unemployment_rate", 4], "\n")

if (fs_summary$fstatistic[1] >= 10) {
  cat("✓ Strong instrument (F >= 10)\n")
} else {
  cat("⚠️  Weak instrument (F < 10)\n")
}

# --- IV/2SLS ---
cat("\n--- IV/2SLS Estimation ---\n")
iv_model <- ivreg(
  default ~ log_income + dti + open_acc + revol_util + total_acc |
    state_unemployment_rate + dti + open_acc + revol_util + total_acc,
  data = df
)

cat("IV/2SLS Results (robust SEs):\n")
iv_robust <- coeftest(iv_model, vcov = vcovHC(iv_model, type = "HC1"))
print(iv_robust)

# --- Key comparison ---
cat("\n--- Key Comparison: OLS vs IV ---\n")
ols_coef <- coef(ols_model)["log_income"]
iv_coef <- coef(iv_model)["log_income"]
cat(sprintf("OLS log(income) coefficient: %.6f\n", ols_coef))
cat(sprintf("IV  log(income) coefficient: %.6f\n", iv_coef))
cat(sprintf("Difference (IV - OLS):       %.6f\n", iv_coef - ols_coef))

if (abs(iv_coef) > abs(ols_coef)) {
  cat("→ IV estimate is larger in magnitude: OLS was attenuated (measurement error or omitted variable bias)\n")
} else {
  cat("→ IV estimate is smaller: some of the OLS association may be non-causal\n")
}

# --- Hausman test ---
cat("\n--- Hausman Test (OLS vs IV endogeneity) ---\n")
tryCatch({
  ht <- summary(iv_model, diagnostics = TRUE)
  cat("Wu-Hausman test p-value:", ht$diagnostics["Wu-Hausman", "p-value"], "\n")
  if (ht$diagnostics["Wu-Hausman", "p-value"] < 0.05) {
    cat("⚠️  Evidence of endogeneity — IV preferred over OLS\n")
  } else {
    cat("✓ No strong evidence of endogeneity\n")
  }
}, error = function(e) {
  cat("Hausman test not available:", e$message, "\n")
})

cat("\n==============================================================\n")
cat("R IV Analysis Complete\n")
cat("==============================================================\n")
