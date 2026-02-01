
# dml_functions.R

# Required libraries
library(xgboost)
library(ranger)
library(tidyr)
library(ggplot2)
library(purrr)
library(dplyr)
library(caret) #only necessary for createFolds
library(DoubleML)
library(mvtnorm)
library(foreach)
library(doParallel)
library(mlr3)
library(mlr3learners)

# Helper to handle missing defaults
`%||%` <- function(a, b) if (!is.null(a)) a else b

# Function to generate folds
generate_folds <- function(n, n_folds) {
  if (n_folds == 1) return(list(seq_len(n)))  # full data
  createFolds(1:n, k = n_folds, list = TRUE, returnTrain = FALSE)
  #split(sample(1:n), rep(1:n_folds, length.out = n)) #no caret necessary
}

# Main single-run simulation with or without cross-fitting
dml_single_run <- function(n = 500, p = 10, theta = 0.5, n_folds = 2, seed = NULL,
                           dgp = c("linear_uncorrelated", "linear_correlated", "nonlinear"),
                           learner_type = c("xgboost", "ranger"), learner_params = list()) {
  if (!is.null(seed)) set.seed(seed)
  dgp <- match.arg(dgp)
  learner_type <- match.arg(learner_type)

  if (dgp == "nonlinear") {
    dat <- make_plr_CCDDHNR2018(n_obs = n, dim_x = p, return_type = "data.frame")
    X <- as.matrix(dat[, paste0("X", 1:p)])
    D <- dat$d
    Y <- dat$y
  } else {
    if (dgp == "linear_uncorrelated") {
      X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    } else {
      Sigma <- 0.5 ^ abs(outer(1:p, 1:p, "-"))
      X <- rmvnorm(n, sigma = Sigma)
    }
    beta_shared <- rnorm(p)
    D <- X %*% beta_shared + rnorm(n)
    Y <- theta * D + X %*% beta_shared + rnorm(n)
  }

  if (n_folds == 1) {
    if (learner_type == "xgboost") {
      dtrain_D <- xgb.DMatrix(data = X, label = D)
      model_D <- xgboost(data = dtrain_D, nrounds = learner_params$nrounds %||% 20,
                         eta = learner_params$eta %||% 0.3,
                         objective = "reg:squarederror", verbose = 0)
      D_hat_full <- predict(model_D, newdata = X)

      dtrain_Y <- xgb.DMatrix(data = X, label = Y)
      model_Y <- xgboost(data = dtrain_Y, nrounds = learner_params$nrounds %||% 20,
                         eta = learner_params$eta %||% 0.3,
                         objective = "reg:squarederror", verbose = 0)
      Y_hat_full <- predict(model_Y, newdata = X)
    } else {
      model_D <- ranger(D ~ ., data = data.frame(D = D, X), num.trees = learner_params$num.trees %||% 100,
                        mtry = learner_params$mtry %||% floor(sqrt(p)))
      D_hat_full <- predict(model_D, data = data.frame(X))$predictions

      model_Y <- ranger(Y ~ ., data = data.frame(Y = Y, X), num.trees = learner_params$num.trees %||% 100,
                        mtry = learner_params$mtry %||% floor(sqrt(p)))
      Y_hat_full <- predict(model_Y, data = data.frame(X))$predictions
    }

    D_tilde_ncf <- D - D_hat_full
    Y_tilde_ncf <- Y - Y_hat_full
    theta_hat_ncf <- unname(coef(lm(Y_tilde_ncf ~ D_tilde_ncf))[2])

    data_df <- data.frame(Y = Y, D = D, X)
    dml_data <- DoubleMLData$new(data_df, y_col = "Y", d_cols = "D")
    learner <- lrn(paste0("regr.", learner_type), predict_type = "response")
    learner$param_set$values <- if (learner_type == "xgboost") {
      list(nrounds = learner_params$nrounds %||% 20, eta = learner_params$eta %||% 0.3,
           objective = "reg:squarederror", verbose = 0)
    } else {
      list(num.trees = learner_params$num.trees %||% 100, mtry = learner_params$mtry %||% floor(sqrt(p)))
    }
    dml_plr <- DoubleMLPLR$new(dml_data, ml_g = learner, ml_m = learner,
                               n_folds = 1, apply_cross_fitting = FALSE)
    dml_plr$fit()
    theta_hat_builtin <- dml_plr$coef[1]

    return(c(theta_hat_ncf = theta_hat_ncf, theta_hat_builtin = theta_hat_builtin))
  }

  folds <- generate_folds(n, n_folds)
  
  D_tilde <- rep(NA, n)
  Y_tilde <- rep(NA, n)

  for (k in 1:n_folds) {
    test_idx <- folds[[k]]
    train_idx <- setdiff(1:n, test_idx)

    if (learner_type == "xgboost") {
      dtrain_D <- xgb.DMatrix(data = X[train_idx, ], label = D[train_idx])
      model_D <- xgboost(data = dtrain_D, nrounds = learner_params$nrounds %||% 20,
                         eta = learner_params$eta %||% 0.3,
                         objective = "reg:squarederror", verbose = 0)
      D_hat <- predict(model_D, newdata = X[test_idx, ])

      dtrain_Y <- xgb.DMatrix(data = X[train_idx, ], label = Y[train_idx])
      model_Y <- xgboost(data = dtrain_Y, nrounds = learner_params$nrounds %||% 20,
                         eta = learner_params$eta %||% 0.3,
                         objective = "reg:squarederror", verbose = 0)
      Y_hat <- predict(model_Y, newdata = X[test_idx, ])
    } else {
      model_D <- ranger(D ~ ., data = data.frame(D = D[train_idx], X[train_idx, ]),
                        num.trees = learner_params$num.trees %||% 100,
                        mtry = learner_params$mtry %||% floor(sqrt(p)))
      D_hat <- predict(model_D, data = data.frame(X[test_idx, ]))$predictions

      model_Y <- ranger(Y ~ ., data = data.frame(Y = Y[train_idx], X[train_idx, ]),
                        num.trees = learner_params$num.trees %||% 100,
                        mtry = learner_params$mtry %||% floor(sqrt(p)))
      Y_hat <- predict(model_Y, data = data.frame(X[test_idx, ]))$predictions
    }

    D_tilde[test_idx] <- D[test_idx] - D_hat
    Y_tilde[test_idx] <- Y[test_idx] - Y_hat
  }

  theta_hat_cf <- unname(coef(lm(Y_tilde ~ D_tilde))[2])
  naive_theta <- unname(coef(lm(Y ~ D))[2])

  data_df <- data.frame(Y = Y, D = D, X)
  dml_data <- DoubleMLData$new(data_df, y_col = "Y", d_cols = "D")
  learner <- lrn(paste0("regr.", learner_type), predict_type = "response")
  learner$param_set$values <- if (learner_type == "xgboost") {
    list(nrounds = learner_params$nrounds %||% 20, eta = learner_params$eta %||% 0.3,
         objective = "reg:squarederror", verbose = 0)
  } else {
    list(num.trees = learner_params$num.trees %||% 100, mtry = learner_params$mtry %||% floor(sqrt(p)))
  }
  dml_plr <- DoubleMLPLR$new(dml_data, ml_g = learner, ml_m = learner, n_folds = n_folds)
  dml_plr$fit()
  theta_hat_builtin <- dml_plr$coef[1]

  return(c(theta_hat_cf = theta_hat_cf, naive_theta = naive_theta, theta_hat_builtin = theta_hat_builtin))
}

# Parallel simulation runner
runSim <- function(B = 500, num_cores = 4,
                   n = 500, p = 10, theta = 0.5, n_folds = 2, seed = NULL,
                   dgp = c("linear_uncorrelated", "linear_correlated", "nonlinear"),
                   learner_type = c("xgboost", "ranger"),
                   learner_params = list()) {
  
  dgp <- match.arg(dgp)
  learner_type <- match.arg(learner_type)
  
  if (num_cores == 1) {
    results <- replicate(B, dml_single_run(n = n, p = p, theta = theta, n_folds = n_folds,
                                           seed = seed, dgp = dgp, learner_type = learner_type,
                                           learner_params = learner_params),
                         simplify = "array")
    results_df <- as_tibble(t(results))
  } else {
    cl <- makeCluster(num_cores)
    registerDoParallel(cl)
    clusterExport(cl, varlist = c("dml_single_run", "generate_folds"))
    
    results <- foreach(i = 1:B, .combine = rbind, 
                       .packages = c("xgboost", "ranger", "DoubleML", "mvtnorm", "mlr3", "mlr3learners", "caret")) %dopar% {
                         dml_single_run(n = n, p = p, theta = theta, n_folds = n_folds,
                                        seed = seed, dgp = dgp, learner_type = learner_type,
                                        learner_params = learner_params)
                       }
    
    stopCluster(cl)
    results_df <- as_tibble(results)
  }
  
  return(results_df)
}


get_sim_config_old <- function(test = TRUE) {
  if (test) {
    list(
      B = 5,
      n = 100,
      num_cores = 1,
      grid = expand.grid(
        learner_type = c("xgboost", "ranger"),
        dgp = c("linear_uncorrelated"),  # just one for testing
        setting = c("test"), 
        stringsAsFactors = FALSE 
      ),
      param_override = list(
        xgboost = list(nrounds = 50, eta = 0.3),
        ranger = list(num.trees = 50, mtry = 2)
      )
    )
  } else {
    list(
      B = 500,
      n = 500,
      num_cores = 4,
      grid = expand.grid(
        learner_type = c("xgboost", "ranger"),
        dgp = c("linear_uncorrelated", "linear_correlated", "nonlinear"),
        setting = c(
          "nrounds=400,eta=0.3",
          "nrounds=600,eta=0.2",
          "nrounds=800,eta=0.1",
          "mtry=3,trees=500",
          "mtry=6,trees=250"
        ), stringsAsFactors = FALSE 
      ),
      param_override = list(
        xgboost = list(
          "nrounds=400,eta=0.3" = list(nrounds = 400, eta = 0.3),
          "nrounds=600,eta=0.2" = list(nrounds = 600, eta = 0.2),
          "nrounds=800,eta=0.1" = list(nrounds = 800, eta = 0.1)
        ),
        ranger = list(
          "mtry=3,trees=500" = list(num.trees = 500, mtry = 3),
          "mtry=6,trees=250" = list(num.trees = 250, mtry = 6)
        )
      )
    )
  }
}

run_all_simulations_old <- function(test = TRUE) {
  config <- get_sim_config(test)
  sim_grid <- config$grid
  
  all_results <- purrr::pmap_dfr(sim_grid, function(learner_type, dgp, setting) {
    params <- config$param_override[[learner_type]][[setting]]
    message(glue::glue("Running {learner_type} | {dgp} | {setting}"))
    
    sim_data <- runSim(
      B = config$B,
      n = config$n,
      num_cores = config$num_cores,
      learner_type = learner_type,
      dgp = dgp,
      learner_params = params
    )
    
    sim_data$learner_type <- learner_type
    sim_data$dgp <- dgp
    sim_data$setting <- setting
    sim_data
  })
  
  return(all_results)
}

#####################
get_sim_config <- function(test = FALSE) {
  if (test) {
    #num_cores = 1
    #if (!missing(n_cores)) num_cores=n_cores
    list(
      B = 5,
      n = 100,
      num_cores = 1,
      learner_grid = tribble(
        ~learner_type, ~learner_params,
        "xgboost", list(nrounds = 50, eta = 0.3),
        "ranger",  list(mtry = 3, num.trees = 50)
      ),
      dgp_grid = tibble(dgp = c("linear_uncorrelated"))
    )
  } else {
    #num_cores = 1
    #if (!missing(n_cores)) num_cores=n_cores
    list(
      B = 500,
      n = 500,
      num_cores = 1,
      learner_grid = tribble(
        ~learner_type, ~learner_params,
        "xgboost", list(nrounds = 400, eta = 0.3),
        "xgboost", list(nrounds = 600, eta = 0.2),
        "xgboost", list(nrounds = 800, eta = 0.1),
        "xgboost", list(),
        "ranger",  list(mtry = 3, num.trees = 500),
        "ranger",  list(mtry = 6, num.trees = 250),
        "ranger",  list()
      ),
      dgp_grid = tibble(dgp = c("linear_uncorrelated", "linear_correlated", "nonlinear"))
    )
  }
}

run_all_simulations <- function(test = FALSE, outdir = "sim_results") {
  config <- get_sim_config(test)
  cat("nCores:", config$num_cores, "\n")
  dir.create(outdir, showWarnings = FALSE)
  
  param_grid <- crossing(config$learner_grid, config$dgp_grid)
  
  for (i in seq_len(nrow(param_grid))) {
    row <- param_grid[i, ]
    learner <- row$learner_type
    params <- row$learner_params[[1]]
    dgp <- row$dgp
    
    # construct filename
    param_str <- paste(names(params), params, sep = "_", collapse = "_")
    if (param_str == "") param_str <- "default"
    fname <- sprintf("sim_%s_%s_%s.rds", learner, param_str, dgp)
    fpath <- file.path(outdir, fname)
    
    if (file.exists(fpath)) {
      message(glue::glue("Skipping existing: {fname}"))
      next
    }
    
    message(glue::glue("Running: {fname}"))
    res <- runSim(B = config$B, n = config$n, num_cores = config$num_cores,
                  dgp = dgp, learner_type = learner, learner_params = params)
    
    saveRDS(res, fpath)
  }
}

aggregate_sim_results <- function(result_dir = "sim_results") {
  files <- list.files(result_dir, pattern = "\\.rds$", full.names = TRUE)
  all_results <- purrr::map_dfr(files, function(f) {
    df <- readRDS(f)
    fname <- basename(f)
    
    # Extract params from filename
    parts <- strsplit(fname, "_")[[1]]
    learner <- parts[2]
    dgp <- gsub("\\.rds$", "", tail(parts, 1))
    
    # Extract param info (e.g., nrounds_400_eta_0.3 or default)
    param_str <- paste(parts[3:(length(parts)-1)], collapse = "_")
    if (param_str == "") param_str <- "default"
    
    df %>%
      mutate(learner_type = learner,
             dgp = dgp,
             param_setting = param_str)
  })
  
  return(all_results)
}

plot_theta_distributions <- function(results_df) {
  results_long <- results_df %>%
    pivot_longer(cols = starts_with("theta_hat"), names_to = "method", values_to = "theta_hat") %>%
    mutate(method = recode(method,
                           theta_hat_cf = "Manual CF",
                           theta_hat_builtin = "DoubleML",
                           theta_hat_ncf = "No CF",
                           naive_theta = "Naive OLS"))
  
  ggplot(results_long, aes(x = theta_hat, fill = method)) +
    geom_histogram(bins = 40, alpha = 0.7, position = "identity") +
    facet_grid(dgp ~ paste(learner_type, param_setting), scales = "free") +
    labs(title = "Distribution of Estimated θ", x = "Estimated θ", y = "Count") +
    theme_minimal() +
    scale_fill_brewer(palette = "Set2") +
    theme(legend.position = "bottom", strip.text = element_text(size = 9))
}


