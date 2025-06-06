rm(list = ls())

# Load packages
library(keras)
library(rstan)
library(reshape2)
library(tidyverse)
library(dampack)
library(gridExtra)

set.seed(132342)

# Load data
df_hiv <- readRDS("data/df_hiv.rds")
source("R/00_helper_functions.R")
## Determinanstic
n_train <- 0.8 * nrow(df_hiv)
id_train <- sample(1:nrow(df_hiv), n_train)
params_train <- as.matrix(df_hiv[id_train,1:5])
params_test <- as.matrix(df_hiv[-id_train,1:5])
out_train <- as.matrix(df_hiv[id_train, 6:9])
out_test <- as.matrix(df_hiv[-id_train, 6:9])

# Helper Function================
prepare_data <- function(xtrain, ytrain, xtest, ytest){
  y_names <- colnames(ytrain)
  x_names <- colnames(xtrain)
  n_train <- nrow(xtrain)
  n_test <- nrow(xtest)
  x <- rbind(xtrain, xtest)
  y <- rbind(ytrain, ytest)
  n <- nrow(x)
  n_inputs <- length(x_names)
  n_outputs <- length(y_names)
  # scale the PSA inputs and outputs
  xresults <- scale_data(x)
  yresults <- scale_data(y)
  xscaled <- xresults$scaled_data
  yscaled <- yresults$scaled_data
  xmins <- xresults$vec.mins
  xmaxs <- xresults$vec.maxs
  ymins <- yresults$vec.mins
  ymaxs <- yresults$vec.maxs

  xtrain_scaled <- xscaled[1:n_train, ]
  ytrain_scaled <- yscaled[1:n_train, ]
  xtest_scaled  <- xscaled[(n_train+1):n, ]
  ytest_scaled  <- yscaled[(n_train+1):n, ]

  return(list(n_inputs = n_inputs,
              n_outputs = n_outputs,
              n_train = n_train,
              n_test = n_test,
              x_names = x_names,
              y_names = y_names,
              xscaled = xscaled,
              yscaled = yscaled,
              xtrain_scaled = xtrain_scaled,
              ytrain_scaled = ytrain_scaled,
              xtest_scaled  = xtest_scaled ,
              ytest_scaled  = ytest_scaled,
              xmins = xmins,
              xmaxs = xmaxs,
              ymins = ymins,
              ymaxs = ymaxs
  ))
}

scale_data <- function(unscaled_data){
  vec.maxs <- apply(unscaled_data, 2, max)
  vec.mins <- apply(unscaled_data, 2, min)
  vec.ones <- matrix(1, nrow = nrow(unscaled_data), 1)
  mat.maxs <- vec.ones %*% vec.maxs
  mat.mins <- vec.ones %*% vec.mins
  scaled_data <- 2 * (unscaled_data - mat.mins) / (mat.maxs - mat.mins) - 1
  results <- list(scaled_data = scaled_data, vec.mins = vec.mins, vec.maxs = vec.maxs)
  return(results)
}

unscale_data <- function(scaled_data, vec.mins, vec.maxs){
  vec.ones <- matrix(1, nrow = nrow(scaled_data), 1)
  mat.mins <- vec.ones %*% vec.mins
  mat.maxs <- vec.ones %*% vec.maxs
  unscaled_data <- (scaled_data + 1) * (mat.maxs - mat.mins) / 2 + mat.mins
}

# Root mean squared error
getRMSE <- function(predicted, observed) {

  out <- sqrt(mean((predicted - observed)^2))

  return(out)

}

# A function to calculate R squared
getR2 <- function(predicted, observed) {

  out <- 1 - sum((observed - predicted)^2) / sum((observed - mean(observed))^2)

  return(out)

}

# ==================
tensorflow::set_random_seed(41241)
# Input parameters
n_iter <- 10000
n_hidden_nodes <- 20
n_hidden_layers <- 2
n_epochs <- 100
verbose <- 0
n_batch_size <- 2000
n_chains <- 4

# Load the training and test data for ANN
prepared_data <- prepare_data(xtrain = params_train,
                              ytrain = out_train,
                              xtest  = params_test,
                              ytest  = out_test)

list2env(prepared_data, envir = .GlobalEnv)


# ============== TensorFlow Keras ANN Section ========================
model <- keras_model_sequential()
mdl_string <- paste("model %>% layer_dense(units = n_hidden_nodes, activation = 'tanh', input_shape = n_inputs) %>%",
                    paste(rep(x = "layer_dense(units = n_hidden_nodes, activation = 'tanh') %>%",
                              n_hidden_layers), collapse = " "),
                    "layer_dense(units = n_outputs)")
eval(parse(text = mdl_string))
summary(model)


model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam'
)
keras.time <- proc.time()
history <- model %>% fit(
  xtrain_scaled, ytrain_scaled,
  epochs = n_epochs,
  batch_size = n_batch_size,
  validation_data = list(xtest_scaled, ytest_scaled),
  verbose = verbose
)
proc.time() - keras.time #keras ann fitting time
plot(history)

pred <- model %>% predict(xscaled)
pred_unscaled <- unscale_data(pred, ymins, ymaxs)
colnames(pred_unscaled) <- c("SQ_Cost", "Adhr_Cost", "SQ_QALY", "Adhr_QALY")

pred_test <- model %>% predict(xtest_scaled)
pred_test_unscaled <- unscale_data(pred_test, ymins, ymaxs)

# Calculate the R2 of test dataset
r2_cost_sq <- getR2(pred_test_unscaled[,1], out_test[,1])
r2_cost_adhr <- getR2(pred_test_unscaled[,2], out_test[,2])
r2_qaly_sq <- getR2(pred_test_unscaled[,3], out_test[,3])
r2_qaly_adhr <- getR2(pred_test_unscaled[,4], out_test[,4])

# Calculate the RMSE of test dataset
rmse_cost_sq <- getRMSE(pred_test_unscaled[,1], out_test[,1])
rmse_cost_adhr <- getRMSE(pred_test_unscaled[,2], out_test[,2])
rmse_qaly_sq <- getRMSE(pred_test_unscaled[,3], out_test[,3])
rmse_qaly_adhr <- getRMSE(pred_test_unscaled[,4], out_test[,4])

# Plot the scatter plots for the validation set
g_cost_no_trt_ann <- plot_scatter(pred_test_unscaled[,1], out_test[,1]) + ggtitle("ANN, Cost Status Quo") +
  coord_cartesian(xlim = c(46110952611, 47650371315), ylim = c(46110952611, 47650371315)) + scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma)
g_cost_trtB_ann <- plot_scatter(pred_test_unscaled[,2], out_test[,2]) + ggtitle("ANN, Cost Intervention") +
  coord_cartesian(xlim = c(46110952611, 47650371315), ylim = c(46110952611, 47650371315)) + scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma)
g_qaly_no_trt_ann <- plot_scatter(pred_test_unscaled[,3], out_test[,3]) + ggtitle("ANN, QALY Status Quo") +
  coord_cartesian(xlim = c(3319736, 3330900), ylim = c(3319736, 3330900)) + scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma)
g_qaly_trtB_ann <- plot_scatter(pred_test_unscaled[,4], out_test[,4]) + ggtitle("ANN, QALY Intervention") +
  coord_cartesian(xlim = c(3319736, 3330900), ylim = c(3319736, 3330900)) + scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma)

g_ann_val <- grid.arrange(g_cost_no_trt_ann, g_cost_trtB_ann, g_qaly_no_trt_ann, g_qaly_trtB_ann, nrow = 2, ncol = 2)


# Create CEAC and PSA plots
psa_obj_ann <- make_psa_obj(cost = pred_unscaled[,1:2],
                            effectiveness = pred_unscaled[, 3:4],
                            parameters = df_hiv[,1:5],
                            strategies = c("Status Quo", "Adherence"),
                            currency = "$")
g_psa_ann <- plot(psa_obj_ann) + ggtitle("ANN Model")

ceac_ann <- ceac(wtp = seq(0, 250000, 1000),
                     psa = psa_obj_ann)
g_ceac_ann <- plot(ceac_ann ) + ggtitle("ANN Model") + ylim(c(0,1))

# Incremental PSA Scatter Plot
psa_obj_ann_inc <- make_psa_obj(cost = cbind(rep(0,1980), pred_unscaled[,2] - pred_unscaled[,1]),
                                effectiveness = cbind(rep(0,1980), pred_unscaled[,4] - pred_unscaled[,3]),
                                parameters = df_hiv[,1:5],
                                strategies = c("Status Quo", "Adherence"),
                                currency = "$")
g_psa_ann_inc <- plot(psa_obj_ann_inc, center = FALSE, alpha = 0.3) + ggtitle("ANN Model") + theme(legend.position = "none")

g_ann <- list(scatter_plot = g_psa_ann, ceac = g_ceac_ann, scatter_plot_inc = g_psa_ann_inc)
saveRDS(g_ann, file = "output/hiv/g_ann.rds")

