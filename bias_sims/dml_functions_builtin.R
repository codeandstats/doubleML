
# dml_functions.R

# Required libraries
library(xgboost)
library(ranger)
library(tidyr)
library(ggplot2)
library(ggtext)
library(purrr)
library(dplyr)
#library(caret) #only necessary for createFolds
library(DoubleML)
library(mvtnorm)
library(foreach)
library(doParallel)
library(mlr3)
library(mlr3learners)
library(mlr3misc)
lgr::get_logger("mlr3")$set_threshold("warn")

# Helper to handle missing defaults
`%||%` <- function(a, b) if (!is.null(a)) a else b

#to add:
# get SE
# compute naive estimate !
dml_single_run_builtin_only <- function(n = 2000, p = 10, theta = 0.5, n_folds = 2,
                                        dgp = c("linear_uncorrelated", "linear_correlated", "nonlinear"),
                                        learner_type = c("xgboost", "ranger", "lm"),
                                        learner_params = list(), seed = NULL) {
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
      X <- matrix(rnorm(n * p), nrow = n)
    } else {
      Sigma <- 0.5 ^ abs(outer(1:p, 1:p, "-"))
      X <- mvtnorm::rmvnorm(n, sigma = Sigma)
    }
    beta <- rnorm(p)
    D <- X %*% beta + rnorm(n)
    Y <- theta * D + X %*% beta + rnorm(n)
  }
  
  data_df <- data.frame(Y = Y, D = D, X)
  dml_data <- DoubleML::DoubleMLData$new(data_df, y_col = "Y", d_cols = "D")
  
  #learner <- mlr3::lrn(learner_type, predict_type = "response")
  learner <- mlr3::lrn(paste0("regr.", learner_type), predict_type = "response")
  #if (num_cores > 1) 
  #learner$param_set$values$nthread <- 1#to avoid double parallelization
  
  if (learner_type == "xgboost") {
    learner$param_set$values <- list(
      nrounds = learner_params$nrounds %||% 20,
      eta = learner_params$eta %||% 0.3,
      objective = "reg:squarederror",
      nthread = 1,#to avoid double parallelization
      verbose = 0
    )
  } else if (learner_type == "ranger") {
    learner$param_set$values <- list(
      num.trees = learner_params$num.trees %||% 100,
      mtry = learner_params$mtry %||% floor(sqrt(p)),
      num.threads = 1##to avoid double parallelization
    )
  }
  
  apply_cf <- n_folds > 1
  dml_model <- DoubleML::DoubleMLPLR$new(
    data = dml_data,
    ml_l = learner,
    ml_m = learner,
    n_folds = n_folds,
    apply_cross_fitting = apply_cf
  )
  
  dml_model$fit(store_predictions = TRUE)
  theta_hat <- unname(dml_model$coef)[1]
  se_hat = unname(dml_model$se)[1]
  
  naive_theta <- unname(coef(lm(Y ~ D))[2])
  #browser()
  #print(summary(dml_model$predictions$ml_l))
  #print(summary(dml_model$predictions$ml_m))
  
  # Retrieve cross-fitted predictions
  g_hat <- as.vector(dml_model$predictions$ml_l)
  m_hat <- as.vector(dml_model$predictions$ml_m)
  
  # Compute diagnostics
  if (any(is.na(g_hat)) || any(is.na(m_hat))) {
    r2_g <- r2_m <- mse_g <- mse_m <- NA_real_
  } else {
    r2_g <- 1 - mean((Y - g_hat)^2) / var(Y)
    r2_m <- 1 - mean((D - m_hat)^2) / var(D)
    mse_g <- mean((Y - g_hat)^2)
    mse_m <- mean((D - m_hat)^2)
  }
  
  
  return(c(theta_hat = theta_hat, se_hat = se_hat, naive_theta = naive_theta, r2_g = r2_g, r2_m = r2_m, mse_g = mse_g, mse_m = mse_m))
}


get_sim_config <- function(test = FALSE) {
  if (test) {
    learner_grid <- tibble::tibble(
      learner_type = "xgboost",
      learner_params = list(list(nrounds = 50, eta = 0.3))
    )
  } else {
    learner_grid <- bind_rows(
      tibble::tibble(
        learner_type = "xgboost",
        learner_params = list(
          list(nrounds = 300, eta = 0.1),#example from doubleML() tutorial
          #list(nrounds = 600, eta = 0.2),
          list(nrounds = 500, eta = 0.3)
        )
      ),
      tibble::tibble(
        learner_type = "ranger",
        learner_params = list(
          list(mtry = 3, num.trees = 500),
          list(mtry = 6, num.trees = 250)
        )
      ),
      tibble::tibble(
        learner_type = "lm",
        learner_params = list(list())  # no tuning needed
      )
    )
    
  }
  
  dgp_grid <- tibble::tibble(dgp = c("linear_uncorrelated", "linear_correlated", "nonlinear"))
  #dgp_grid <- tibble::tibble(dgp = c( "nonlinear"))
  fold_grid <- tibble::tibble(n_folds = c(1, 2))
  
  config <- tidyr::crossing(learner_grid, dgp_grid, fold_grid)
  return(config)
}



get_sim_config_old <- function(test = FALSE) {
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
      num_cores = 40,
      learner_grid = tribble(
        ~learner_type, ~learner_params,
        "xgboost", list(nrounds = 1000, eta = 0.3),
        "xgboost", list(nrounds = 600, eta = 0.2),
        #"xgboost", list(nrounds = 800, eta = 0.1),
        #"xgboost", list(),
        "ranger",  list(mtry = 3, num.trees = 500),
        "ranger",  list(mtry = 5, num.trees = 250),
        #"ranger",  list()
      ),
      dgp_grid = tibble(dgp = c("linear_uncorrelated", "linear_correlated", "nonlinear"))
    )
  }
}

# Parallel simulation runner
runSim <- function(B = 500, num_cores = 4,
                   n = 500, p = 10, theta = 0.5, n_folds = 2, seed = NULL,
                   dgp = c("linear_uncorrelated", "linear_correlated", "nonlinear"),
                   learner_type = c("xgboost", "ranger", "lm"),
                   learner_params = list()) {
  
  dgp <- match.arg(dgp)
  learner_type <- match.arg(learner_type)
  
  if (num_cores == 1) {
    results <- replicate(B, dml_single_run_builtin_only(n = n, p = p, theta = theta, n_folds = n_folds,
                                           seed = seed, dgp = dgp, learner_type = learner_type,
                                           learner_params = learner_params),
                         simplify = "array")
    #browser()
    results_df <- as_tibble(t(results))
  } else {
    cl <- makeCluster(num_cores)
    registerDoParallel(cl)
    clusterExport(cl, varlist = c("dml_single_run_builtin_only"))
    
    results <- foreach(i = 1:B, .combine = rbind, 
                       .packages = c("xgboost", "ranger", "DoubleML", "mvtnorm", "mlr3", "mlr3learners", "caret")) %dopar% {
                         dml_single_run_builtin_only(n = n, p = p, theta = theta, n_folds = n_folds,
                                        seed = seed, dgp = dgp, learner_type = learner_type,
                                        learner_params = learner_params)
                       }
    
    stopCluster(cl)
    results_df <- as_tibble(results)
  }
  
  #colnames(results_df)[1] ="theta_hat"
  return(results_df)
}

run_all_simulations <- function(B = 500, num_cores = 4, test = FALSE, 
                                n = 500, p = 10, save_dir = "sim_results") {
  config <- get_sim_config(test)
  cat("Exploring a total of ", nrow(config) , "param options \n")
  dir.create(save_dir, showWarnings = FALSE)
  
  for (i in seq_len(nrow(config))) {
    row_cfg <- config[i, ]
    
    # Extract relevant info
    learner_type   <- row_cfg$learner_type
    learner_params <- row_cfg$learner_params[[1]]
    dgp            <- row_cfg$dgp
    n_folds        <- row_cfg$n_folds
    
    # Build filename
    param_str <- paste(names(learner_params), unlist(learner_params), sep = "_", collapse = "_")
    if (param_str == "") param_str <- "default"
    
    fname <- sprintf("sim__%s__%s__%s__nfolds_%s.rds", learner_type, param_str, dgp, n_folds)
    fpath <- file.path(save_dir, fname)
    
    
    if (file.exists(fpath)) {
      message("Skipping ", fname)
      next
    }
    
    # Run simulation
    message("Running ", fname)
    
    result <- tryCatch({
      runSim(B = B, num_cores = num_cores, n = n, p = p, theta = 0.5,
             dgp = dgp, learner_type = learner_type,
             learner_params = learner_params, n_folds = n_folds)
    }, error = function(e) {
      message("Error in runSim for ", fname, ": ", e$message)
      return(NULL)
    })
    
    if (!is.null(result)) {
      tryCatch({
        saveRDS(result, fpath)
        message("Saved: ", fname)
      }, error = function(e) {
        message("Failed to save file: ", fpath)
      })
    }
  }
}


run_all_simulations_old <- function(test = FALSE, outdir = "sim_results", num_cores = 1) {
  config <- get_sim_config(test)
  cat("nCores:", num_cores, "\n")
  dir.create(outdir, showWarnings = FALSE)
  
  param_grid <- config # crossing(config$learner_grid, config$dgp_grid)
  
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
    browser()
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

plot_sim_results <- function(results_dir = "sim_results") {
  files <- list.files(results_dir, pattern = "\\.rds$", full.names = TRUE)
  all_data <- do.call(rbind, lapply(files, readRDS))
  
  all_data_long <- all_data |>
    pivot_longer(cols = starts_with("theta_"), names_to = "estimate_type", values_to = "theta_hat")
  
  ggplot(all_data_long, aes(x = theta_hat)) +
    geom_histogram(bins = 40, fill = "steelblue", color = "white") +
    facet_grid(dgp ~ learner_type + learner_params, scales = "free_y") +
    theme_minimal(base_size = 14) +
    labs(title = "Distribution of θ̂ from DoubleML (only)", x = "θ̂", y = "Count")
}

aggregate_sim_results <- function(results_dir = "sim_results") {
  files <- list.files(results_dir, pattern = "^sim__.*\\.rds$", full.names = TRUE)
  
  #browser()
  
  all_dfs <- lapply(files, function(file) {
    df <- readRDS(file)
    
    # Fix unnamed column if needed
    if (ncol(df) == 1 && names(df)[1] == "") {
      names(df)[1] <- "theta_hat"
    }
    
    # Extract metadata manually
    fname <- basename(file)
    if (grepl("mtry_6", fname)) return(NULL)
    fname <- gsub("ranger__default", "ranger__mtry_3_num.trees_500", fname, fixed = TRUE)
    fname <- gsub("xgboost__default", "xgboost__nrounds_1000_eta_0.3", fname, fixed = TRUE)
    
    fname_core <- sub("^sim__", "", sub("\\.rds$", "", fname))
    parts <- strsplit(fname_core, "__")[[1]]
    
    if (length(parts) != 3) {
      warning("Filename could not be parsed: ", fname)
      #print("Hell")
      return(NULL)
    }
    
    learner_type   <- parts[1]
    param_setting  <- parts[2]
    dgp            <- parts[3]
    if (grepl("default", param_setting)) browser()
      
    df$learner_type   <- learner_type
    df$param_setting  <- param_setting
    df$dgp            <- dgp
    
    return(df)
  })
  
  results_df <- do.call(rbind, all_dfs)
  return(results_df)
}


aggregate_sim_results_old3 <- function(results_dir = "sim_results") {
  files <- list.files(results_dir, pattern = "\\.rds$", full.names = TRUE)
  
  all_dfs <- lapply(files, function(file) {
    df <- readRDS(file)
    
    # Fix column name if needed
    if (ncol(df) == 1 && names(df)[1] == "") {
      names(df)[1] <- "theta_hat"
    }
    
    # Parse filename for metadata
    fname <- basename(file)
    fname <- gsub("ranger_default", "ranger_mtry_3_num.trees_500", fname, fixed = TRUE)
    fname <- gsub("xgboost_default", "xgboost_nrounds_1000_eta_0.3", fname, fixed = TRUE)
    pattern <- "^sim_([^_]+)_([^_]+(?:_[^_]+)*)_(.*)\\.rds$"
    #for now we need this more complex matching, but in the future we will change this
    #pattern <- "^sim_([^_]+)_((?:[^_]+=[^_]+(?:_[^_]+=[^_]+)*)|default)_(.*)\\.rds$"
    
    matches <- regmatches(fname, regexec(pattern, fname))[[1]]
    
    learner_type   <- matches[2]
    param_setting  <- matches[3]
    dgp            <- matches[4]
    
    df$learner_type   <- learner_type
    df$param_setting  <- param_setting
    df$dgp            <- dgp
    
    return(df)
  })
  
  # Combine all
  results_df <- do.call(rbind, all_dfs)
  #browser()
  return(results_df)
}

aggregate_sim_results_old2 <- function(results_dir = "sim_results") {
  files <- list.files(results_dir, pattern = "\\.rds$", full.names = TRUE)
  
  # Helper to extract metadata from filename
  extract_metadata <- function(filename) {
    fname <- basename(filename)
    pattern <- "^sim_(.*?)_(.*?)_(.*?)\\.rds$"
    matches <- regmatches(fname, regexec(pattern, fname))[[1]]
    list(
      learner_type = matches[2],
      param_setting = matches[3],
      dgp = matches[4]
    )
  }
  
  # Read and tag each result
  all_dfs <- lapply(files, function(file) {
    df <- readRDS(file)
    
    # if column name is missing, name it
    if (ncol(df) == 1 && colnames(df)[1] == "") {
      colnames(df) <- "theta_hat"
    }
    
    meta <- extract_metadata(file)
    df$dgp <- meta$dgp
    df$learner_type <- meta$learner_type
    df$param_setting <- meta$param_setting
    df
  })
  
  # Combine into one long data frame
  results_df <- bind_rows(all_dfs)
  
  return(results_df)
}

aggregate_sim_results_old <- function(results_dir = "sim_results") {
  files <- list.files(results_dir, pattern = "\\.rds$", full.names = TRUE)
  
  # Read and bind all results
  all_data <- do.call(cbind, lapply(files, function(file) {
    df <- readRDS(file)
    
    # Ensure only the relevant column is kept (assumed to be the first one)
    if (ncol(df) > 1) {
      df <- df[, 1, drop = FALSE]  # Keep only the first column, theta_hat_builtin
    }
    return(df)
  }))
  
  #browser()
  # Rename the column to "theta_hat_builtin" if necessary
  if (names(all_data)[1] == "") {
    names(all_data)[1] <- "theta_hat_builtin"
  }
  
  # Add metadata columns (e.g., dgp, learner_type) from file name or other sources if needed
  all_data$dgp <- sub(".*/(.*)_[^_]+\\.rds$", "\\1", files)  # Example, adjust for naming pattern
  all_data$learner_type <- "xgboost"  # Adjust based on your naming scheme
  
  # Gather into long format
  results_long <- tidyr::pivot_longer(
    all_data,
    cols = "theta_hat_builtin",  # Only pivot the relevant column
    names_to = "estimate_type",
    values_to = "theta_hat"
  )
  
  return(results_long)
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

library(dplyr)

plot_overlayed_histograms <- function(results_df) {
  # 1. Compute means by facet
  means_df <- results_df %>%
    group_by(learner_type, dgp, param_setting) %>%
    summarise(mean_theta = mean(theta_hat), .groups = "drop")
  
  # 2. Main plot
  mp = ggplot(results_df, aes(x = theta_hat, fill = param_setting, color = param_setting)) +
    geom_histogram(position = "identity", bins = 40, alpha = 0.4) +
    geom_vline(data = means_df, aes(xintercept = mean_theta, color = param_setting),
               linetype = "dashed", linewidth = 0.8, show.legend = FALSE) +
    #geom_vline(xintercept = 0.5, color = "darkred", linewidth = 1.2) +
    #annotate("text", x = 0.5, y = Inf, label = "True θ = 0.5", vjust = -0.5,
    #         color = "darkred", fontface = "bold", size = 4) +
    facet_grid(rows = vars(learner_type), cols = vars(dgp), scales = "free_y") +
    theme_minimal(base_size = 14) +
    labs(
      title = "Overlayed Distributions of θ̂ with Empirical Means",
      x = "Estimated θ",
      y = "Count",
      fill = "Parameter Setting",
      color = "Parameter Setting"
    ) +
    theme(
      legend.position = "bottom",
      strip.text = element_text(size = 12)
    )
  
  mp = mp + scale_x_continuous(
    breaks = function(x) {
      b <- pretty(x)
      if (!any(abs(b - 0.5) < 1e-6)) b <- sort(unique(c(b, 0.5)))
      b
    },
    labels = function(b) {
      sapply(b, function(val) {
        if (!is.na(val) && abs(val - 0.5) < 1e-6) {
          "<span style='color:darkred; font-weight:bold;'>0.5</span>"
        } else {
          as.character(val)
        }
      })
    }
  ) +
    theme(axis.text.x = ggtext::element_markdown())
  
  
  mp
}


plot_overlayed_histograms_old <- function(results_df) {
  ggplot(results_df, aes(x = theta_hat, fill = param_setting, color = param_setting)) +
    geom_histogram(position = "identity", bins = 40, alpha = 0.4) +
    geom_vline(xintercept = 0.5, color = "red", linetype = "dashed") +
    facet_grid(rows = vars(learner_type), cols = vars(dgp), scales = "free_y") +
    stat_summary(fun = mean, geom = "vline", aes(xintercept = ..y..),
                 color = "black", linetype = "dashed", linewidth = 0.8, show.legend = FALSE) +
    theme_minimal(base_size = 14) +
    labs(
      title = "θ̂ by Learner and DGP",
      x = "Estimated θ",
      y = "Count",
      fill = "Param Setting",
      color = "Param Setting"
    ) +
    theme(
      legend.position = "bottom",
      strip.text = element_text(size = 12)
    )
}

plot_by_learner <- function(results_df, learner = "xgboost") {
  filtered <- subset(results_df, learner_type == learner)
  
  ggplot(filtered, aes(x = theta_hat, fill = param_setting, color = param_setting)) +
    geom_histogram(position = "identity", bins = 40, alpha = 0.4) +
    facet_wrap(~ dgp, nrow = 1, scales = "free_y") +
    theme_minimal(base_size = 14) +
    labs(title = paste("θ̂ for", learner),
         x = "Estimated θ",
         y = "Count",
         fill = "Parameter Setting",
         color = "Parameter Setting") +
    theme(legend.position = "bottom")
}

rename_sim_files <- function(dir = "sim_results") {
  files <- list.files(dir, pattern = "^sim_.*\\.rds$", full.names = TRUE)
  
  for (file in files) {
    fname <- basename(file)
    
    # Parse the existing pattern
    # Remove "sim_" prefix and ".rds" suffix
    core <- sub("^sim_", "", sub("\\.rds$", "", fname))
    parts <- strsplit(core, "_")[[1]]
    
    learner_type <- parts[1]
    
    # Heuristic: split into param_setting and dgp
    # We'll assume DGP always starts with "linear" or "nonlinear"
    dgp_start <- which(parts %in% c("linear", "nonlinear"))[1]
    
    param_setting <- paste(parts[2:(dgp_start - 1)], collapse = "_")
    dgp <- paste(parts[dgp_start:length(parts)], collapse = "_")
    
    # Clean fallback if something goes wrong
    if (is.na(dgp_start) || dgp_start < 2) {
      warning("Could not parse: ", fname)
      next
    }
    
    # New filename with double underscores
    new_fname <- sprintf("sim__%s__%s__%s.rds", learner_type, param_setting, dgp)
    old_path <- file.path(dir, fname)
    new_path <- file.path(dir, new_fname)
    
    if (!file.exists(new_path)) {
      file.rename(old_path, new_path)
      message("Renamed: ", fname, " → ", new_fname)
    } else {
      message("SKIPPED (already exists): ", new_fname)
    }
  }
}
