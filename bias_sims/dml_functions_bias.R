
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
library(caret)

# Helper to handle missing defaults
`%||%` <- function(a, b) if (!is.null(a)) a else b

make_betas <- function(p, rho = 0.3) {
  # rho in [0,1]: alignment between prognostic and predictive parts
  
  Sigma <- matrix(c(1, rho,
                    rho, 1), nrow = 2)
  
  Z <- MASS::mvrnorm(p, mu = c(0, 0), Sigma = Sigma)
  
  beta_g <- Z[, 1]
  beta_m <- Z[, 2]
  
  list(beta_g = beta_g, beta_m = beta_m)
}


dml_single_run_builtin_only <- function(n = 500, p = 10, theta = 0.5, n_folds = 2,
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
    #beta <- rnorm(p)
    #better:
    betas <- make_betas(p, rho=0.3)
    beta_g <- betas$beta_g
    beta_m <- betas$beta_m
    
    D <- X %*% beta_m + rnorm(n)
    Y <- theta * D + X %*% beta_g + rnorm(n)
  }
  
  data_df <- data.frame(Y = Y, D = D, X)
  dml_data <- DoubleML::DoubleMLData$new(data_df, y_col = "Y", d_cols = "D")
  
  #learner <- mlr3::lrn(learner_type, predict_type = "response")
  learner <- mlr3::lrn(paste0("regr.", learner_type), predict_type = "response")
  
  if (learner_type == "xgboost") {
    learner$param_set$values <- list(
      nrounds = learner_params$nrounds %||% 20,
      eta = learner_params$eta %||% 0.3,
      objective = "reg:squarederror",
      verbose = 0
    )
  } else if (learner_type == "ranger") {
    learner$param_set$values <- list(
      num.trees = learner_params$num.trees %||% 100,
      mtry = learner_params$mtry %||% floor(sqrt(p))
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
  
  
  return(c(theta_hat = theta_hat, r2_g = r2_g, r2_m = r2_m, mse_g = mse_g, mse_m = mse_m))
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
          list(nrounds = 1000, eta = 0.3),
          #list(nrounds = 600, eta = 0.2),
          list(nrounds = 800, eta = 0.1)
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
  fold_grid <- tibble::tibble(n_folds = c(1, 2))
  
  config <- tidyr::crossing(learner_grid, dgp_grid, fold_grid)
  return(config)
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
    clusterExport(cl, varlist = c("dml_single_run_builtin_only","make_betas"))
    
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
                                save_dir = "sim_results") {
  config <- get_sim_config(test)
  dir.create(save_dir, showWarnings = FALSE)
  print(config)
  
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
    message("Running: ", fname)
    if (1){
      results_df <- runSim(
        B = B,
        num_cores = num_cores,
        n = 500,
        p = 10,
        theta = 0.5,
        dgp = dgp,
        learner_type = learner_type,
        learner_params = learner_params,
        n_folds = n_folds
      )
      
      saveRDS(results_df, fpath)
    }
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
  
  all_dfs <- lapply(files, function(file) {
    fname <- basename(file)
    
    # Match: sim__{learner_type}__{param_setting}__{dgp}__nfolds_{n_folds}.rds
    pattern <- "^sim__([^_]+)__(.+)__(.+)__nfolds_([0-9]+)\\.rds$"
    matches <- regmatches(fname, regexec(pattern, fname))[[1]]
    
    if (length(matches) != 5) {
      warning("Filename does not match expected pattern: ", fname)
      return(NULL)
    }
    
    learner_type  <- matches[2]
    param_setting <- matches[3]
    dgp           <- matches[4]
    n_folds       <- as.integer(matches[5])
    #browser()
    df <- tryCatch(readRDS(file), error = function(e) {
      warning("Failed to read RDS file: ", file)
      return(NULL)
    })
    
    if (is.null(df)) return(NULL)
    
    # Ensure it's a data.frame or convert
    if (!inherits(df, "data.frame")) {
      df <- as.data.frame(df)
    }
    
    # Handle unnamed column (older formats)
    if (ncol(df) == 1 && names(df)[1] == "") {
      names(df)[1] <- "theta_hat_builtin"
    }
    
    df$learner_type  <- learner_type
    df$param_setting <- param_setting
    df$dgp           <- dgp
    df$n_folds       <- n_folds
    
    return(df)
  })
  
  results_df <- do.call(rbind, all_dfs)
  return(results_df)
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

plot_overlayed_histograms_old2 <- function(results_df) {
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

plot_overlayed_histograms_old <- function(results_df, y = "theta_hat_builtin") {
  # Clean param_setting labels
  results_df <- results_df %>%
    mutate(param_setting = ifelse(param_setting == "default", learner_type, param_setting))
  
  # Split data by n_folds
  plots <- results_df %>%
    split(.$n_folds) %>%
    lapply(function(df_subset) {
      y_label <- y
      if (!y %in% names(df_subset)) {
        warning("Variable not found: ", y)
        return(NULL)
      }
      
      # Compute group-wise means
      means_df <- df_subset %>%
        group_by(learner_type, dgp, param_setting) %>%
        summarise(mean_y = mean(.data[[y]], na.rm = TRUE), .groups = "drop")
      
      ggplot(df_subset, aes(x = .data[[y]], fill = param_setting, color = param_setting)) +
        geom_histogram(position = "identity", bins = 40, alpha = 0.4) +
        geom_vline(data = means_df, aes(xintercept = mean_y, color = param_setting),
                   linetype = "dashed", linewidth = 0.8, show.legend = FALSE) +
        scale_x_continuous(
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
        theme(axis.text.x = ggtext::element_markdown()) +
        facet_grid(rows = vars(learner_type), cols = vars(dgp), scales = "free_y") +
        theme_minimal(base_size = 14) +
        labs(
          title = paste("Distributions of", y, "for n_folds =", unique(df_subset$n_folds)),
          x = y,
          y = "Count",
          fill = "Param Setting",
          color = "Param Setting"
        ) +
        theme(
          legend.position = "bottom",
          strip.text = element_text(size = 12)
        )
    })
  
  return(plots)
}


plot_overlayed_histograms_old3 <- function(results_df, y = c("theta_hat", "r2_g", "r2_m", "mse_g", "mse_m"),
                                      highlight_05 = c("auto", "markdown", "vline")) {
  highlight_05 <- match.arg(highlight_05)
  
  # Fallback to vline if rendering as PDF (can override manually)
  if (highlight_05 == "auto") {
    highlight_05 <- if (capabilities("X11")) "markdown" else "vline"
  }
  
  results_df <- results_df %>%
    mutate(param_setting = ifelse(param_setting == "default", learner_type, param_setting)) %>%
  mutate(
    param_setting_clean = ifelse(learner_type == "lm", NA_character_, param_setting)
  )
  plots <- results_df %>%
    split(.$n_folds) %>%
    lapply(function(df_subset) {
      y_label <- y
      if (!y %in% names(df_subset)) {
        warning("Variable not found: ", y)
        return(NULL)
      }
      
      means_df <- df_subset %>%
        group_by(learner_type, dgp, param_setting) %>%
        summarise(mean_y = mean(.data[[y]], na.rm = TRUE), .groups = "drop")
      
      p <- ggplot(df_subset, aes(x = .data[[y]], fill = param_setting, color = param_setting)) +
        geom_histogram(position = "identity", bins = 40, alpha = 0.4) +
        geom_vline(data = means_df, aes(xintercept = mean_y, color = param_setting),
                   linetype = "dashed", linewidth = 0.8, show.legend = FALSE) +
        facet_grid(rows = vars(learner_type), cols = vars(dgp), scales = "free_y") +
        theme_minimal(base_size = 14) +
        labs(
          title = paste("Distributions of", y, "for n_folds =", unique(df_subset$n_folds)),
          x = y,
          y = "Count",
          fill = "Param Setting",
          color = "Param Setting"
        ) +
        theme(legend.position = "bottom", legend.box = "horizontal") +
        guides(
          fill = guide_legend(nrow = 2, byrow = TRUE),
          color = guide_legend(nrow = 2, byrow = TRUE)
        )
      
        # theme(
        #   legend.position = "bottom",
        #   strip.text = element_text(size = 12)
        # )
      
      if (highlight_05 == "markdown") {
        p <- p +
          scale_x_continuous(
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
      } else if (highlight_05 == "vline") {
        p <- p +
          geom_vline(xintercept = 0.5, color = "darkred", linewidth = 0.8) +
          annotate("text", x = 0.5, y = Inf, label = "true θ = 0.5",
                   color = "darkred", fontface = "bold", vjust = 2, hjust = 0.5, size = 4)
      }
      
      return(p)
    })
  
  return(plots)
}
plot_overlayed_histograms_April13 <- function(results_df, 
                                      y = c("theta_hat", "r2_g", "r2_m", "mse_g", "mse_m")[1],
                                      highlight_05 = c("auto", "markdown", "vline", "FALSE"),
                                      xlim = NULL
) {
  highlight_05 <- match.arg(highlight_05)
  if (highlight_05 == "auto") {
    highlight_05 <- if (capabilities("X11")) "markdown" else "vline"
  }
  
  results_df <- results_df %>%
    mutate(
      param_setting = ifelse(param_setting == "default", learner_type, param_setting),
      param_setting_clean = ifelse(learner_type == "lm", NA_character_, param_setting)
    )
  
  plots <- results_df %>%
    split(.$n_folds) %>%
    lapply(function(df_subset) {
      if (!y %in% names(df_subset)) {
        warning("Variable not found: ", y)
        return(NULL)
      }
      
      means_df <- df_subset %>%
        filter(!is.na(param_setting_clean)) %>%
        group_by(learner_type, dgp, param_setting_clean) %>%
        summarise(mean_y = mean(.data[[y]], na.rm = TRUE), .groups = "drop")
      
      p <- ggplot(df_subset, aes(x = .data[[y]])) +
        geom_histogram(
          data = df_subset %>% filter(!is.na(param_setting_clean)),
          aes(fill = param_setting_clean, color = param_setting_clean),
          position = "identity", bins = 40, alpha = 0.4
        ) +
        geom_histogram(
          data = df_subset %>% filter(learner_type == "lm"),
          #fill = "grey70", color = "grey50", 
          fill = "steelblue", color = "darkblue",
          bins = 40, alpha = 0.6,
          inherit.aes = FALSE, aes(x = .data[[y]])
        ) +
        geom_vline(data = means_df,
                   aes(xintercept = mean_y, color = param_setting_clean),
                   linetype = "dashed", linewidth = 0.8, show.legend = FALSE) +
        facet_grid(rows = vars(learner_type), cols = vars(dgp), scales = "free_y") +
        theme_minimal(base_size = 14) +
        labs(
          title = paste("Distributions of", y, "for n_folds =", unique(df_subset$n_folds)),
          x = y,
          y = "Count",
          fill = "Param Setting",
          color = "Param Setting"
        ) +
        guides(
          fill = guide_legend(nrow = 2, byrow = TRUE),
          color = guide_legend(nrow = 2, byrow = TRUE)
        ) +
        theme(
          legend.position = "bottom",
          legend.box = "horizontal",
          strip.text = element_text(size = 12)
        )
      
      if (highlight_05 == "markdown") {
        p <- p +
          scale_x_continuous(
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
      } else if (highlight_05 == "vline") {
        p <- p +
          geom_vline(xintercept = 0.5, color = "darkred", linewidth = 0.8) +
          annotate("text", x = 0.5, y = Inf, label = "true θ = 0.5",
                   color = "darkred", fontface = "bold", vjust = 2, hjust = 0.5, size = 4)
      }
      
      if (!is.null(xlim)) {
        p <- p + xlim(xlim[1], xlim[2])
      }
      
      return(p)
    })
  
  return(plots)
}

plot_overlayed_histograms <- function(results_df, 
                                              y = c("theta_hat", "r2_g", "r2_m", "mse_g", "mse_m")[1],
                                              highlight_05 = c("auto", "markdown", "vline", "FALSE")[3],
                                              xlim = NULL,
                                              free_x = TRUE,
                                              free_y = TRUE
) {
  highlight_05 <- match.arg(highlight_05)
  if (highlight_05 == "auto") {
    highlight_05 <- if (capabilities("X11")) "markdown" else "vline"
  }
  
  if (y != "theta_hat") highlight_05 = "FALSE"
  # Determine facet scaling mode
  facet_mode <- if (free_x && free_y) {
    "free"
  } else if (free_x) {
    "free_x"
  } else if (free_y) {
    "free_y"
  } else {
    "fixed"
  }
  if (!is.null(xlim) && free_x) {
    message("Note: `xlim` is ignored when `free_x = TRUE` (axes are independent).")
  }
  
  
  results_df <- results_df %>%
    mutate(
      param_setting = ifelse(param_setting == "default", learner_type, param_setting),
      param_setting_clean = ifelse(learner_type == "lm", NA_character_, param_setting)
    )
  
  #browser()
  plots <- results_df %>%
    split(.$n_folds) %>%
    lapply(function(df_subset) {
      if (!y %in% names(df_subset)) {
        warning("Variable not found: ", y)
        return(NULL)
      }
      
      means_df <- df_subset %>%
        filter(!is.na(param_setting_clean)) %>%
        group_by(learner_type, dgp, param_setting_clean) %>%
        summarise(mean_y = mean(.data[[y]], na.rm = TRUE), .groups = "drop")
      
      p <- ggplot(df_subset, aes(x = .data[[y]])) +
        geom_histogram(
          data = df_subset %>% filter(!is.na(param_setting_clean)),
          aes(fill = param_setting_clean, color = param_setting_clean),
          position = "identity", bins = 40, alpha = 0.4
        ) +
        geom_histogram(
          data = df_subset %>% filter(learner_type == "lm"),
          #fill = "grey70", color = "grey50", 
          fill = "steelblue", color = "darkblue",
          bins = 40, alpha = 0.6,
          inherit.aes = FALSE, aes(x = .data[[y]])
        ) +
        geom_vline(data = means_df,
                   aes(xintercept = mean_y, color = param_setting_clean),
                   linetype = "dashed", linewidth = 0.8, show.legend = FALSE) 
        #facet_grid(rows = vars(learner_type), cols = vars(dgp), scales = "free_y") +
        
        p <- p + facet_grid(rows = vars(learner_type), cols = vars(dgp), scales = facet_mode) +
        theme_minimal(base_size = 14) +
        labs(
          title = paste("n_folds =", unique(df_subset$n_folds)),
          x = y,
          y = "Count",
          fill = "Hyperparameters",
          color = "Hyperparameters"
        ) +
        guides(
          fill = guide_legend(nrow = 2, byrow = TRUE),
          color = guide_legend(nrow = 2, byrow = TRUE)
        ) 
        #browser()
        
        p = p + theme(
          legend.position = "bottom",
          legend.box = "horizontal",
          strip.text = element_text(size = 12)
        )
        
      if (highlight_05 == "markdown") {
        p <- p +
          scale_x_continuous(
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
      } else if (highlight_05 == "vline") {
        p <- p +
          geom_vline(xintercept = 0.5, color = "darkred", linewidth = 0.8) #+
          #annotate("text", x = 0.5, y = Inf, label = "true θ = 0.5",
          #         color = "darkred", fontface = "bold", vjust = 2, hjust = 0.5, size = 4)
      }
      
      if (!is.null(xlim)) {
        p <- p + xlim(xlim[1], xlim[2])
      }
      
      return(p)
    })
  
  return(plots)
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

aggregate_extraCols = function(yCol = "naive_theta",
    data_path = "/Users/loecherm/learning/Causality/sim_results/",
    indPaths = c("sim_results_n500_p10_April11_25/",
                 "sim_results_n1000_p05_April11_25",
                 "sim_results_n2000_p10_April11_25",
                 "sim_results_n50000_p10_April12_25")
){
  naive_theta = list()
  
  for (dirname in indPaths){
    matches <- regexec("sim_results_n(\\d+)_p(\\d+)", dirname)
    values <- regmatches(dirname, matches)[[1]]
    n <- as.integer(values[2])
    p <- as.integer(values[3])
    cat("n =", n, " | p =", p)
    
    results_df <- aggregate_sim_results(file.path(data_path,dirname))
    naive_theta[[dirname]] = results_df[,c(yCol, "dgp")]
    naive_theta[[dirname]]$n = n
    naive_theta[[dirname]]$p = p  
  }
  naive_theta = do.call("rbind",naive_theta)
  
  return(naive_theta)
}

make_plr_strongly_confounding_CCDDHNR <- function(n_obs = 500, dim_x = 10, theta = 0.5,
                                                  rho = 0.7, snr = 3, return_type = c("data.frame", "list"), seed = NULL) {
  return_type <- match.arg(return_type)
  if (!is.null(seed)) set.seed(seed)
  
  # Correlated covariates X ~ N(0, Σ) with AR(1) structure
  Sigma <- rho ^ abs(outer(1:dim_x, 1:dim_x, "-"))
  X <- mvtnorm::rmvnorm(n_obs, sigma = Sigma)
  colnames(X) <- paste0("X", 1:dim_x)
  
  # Sigmoid function
  sigmoid <- function(z) exp(z) / (1 + exp(z))
  
  # Shared structure: stronger overlap in m(x) and g(x)
  # Add more symmetric or shared components
  m_x <- X[, 1] + 0.4 * sigmoid(X[, 3]) + 0.3 * X[, 5]
  g_x <- sigmoid(X[, 1]) + 0.4 * X[, 3] + 0.3 * X[, 5]  # shared term: X[,5]
  
  # Add error terms with controlled noise (signal-to-noise ratio)
  v <- rnorm(n_obs, sd = 1 / sqrt(snr))
  zeta <- rnorm(n_obs, sd = 1 / sqrt(snr))
  
  # Generate treatment and outcome
  D <- m_x + v
  Y <- theta * D + g_x + zeta
  
  if (return_type == "data.frame") {
    df <- data.frame(y = Y, d = D, X)
    return(df)
  } else {
    return(list(Y = Y, D = D, X = X))
  }
}
