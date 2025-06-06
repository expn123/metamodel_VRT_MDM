rm(list = ls())

# Load packages
library(keras)
library(rstan)
library(reshape2)
library(tidyverse)
library(dampack)

set.seed(132342)

# Load data
df_psa_sto <- readRDS("data/df_psa_sto.rds")
## Determinanstic
df_psa_det <- readRDS("data/df_psa_det.rds")
n_train <- 0.8 * nrow(df_psa_sto)
id_train <- sample(1:nrow(df_psa_sto), n_train)
params_train <- as.matrix(df_psa_sto[id_train,c(1:12,14)])
params_test <- as.matrix(df_psa_sto[-id_train,c(1:12,14)])
out_train <- as.matrix(df_psa_sto[id_train,c(16,18,20,22)])
out_test <- as.matrix(df_psa_sto[-id_train,c(16,18,20,22)])

# Load Functions
source("R/00_helper_functions.R")

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

pred <- model %>% predict(xscaled[order(as.numeric(rownames(xscaled))),])
pred_unscaled <- unscale_data(pred, ymins, ymaxs)
colnames(pred_unscaled) <- c("No_Treatment_Cost", "Treatment_B_Cost", "No_Treatment_QALY", "Treatment_B_QALY")

pred_test <- model %>% predict(xtest_scaled)
pred_test_unscaled <- unscale_data(pred_test, ymins, ymaxs)

# Calculate the R2 of test dataset
r2_cost_no_trt <- getR2(pred_test_unscaled[,1], out_test[,1])
r2_cost_trtB <- getR2(pred_test_unscaled[,2], out_test[,2])
r2_qaly_no_trt <- getR2(pred_test_unscaled[,3], out_test[,3])
r2_qaly_trtB <- getR2(pred_test_unscaled[,4], out_test[,4])

# Calculate the RMSE of test dataset
rmse_cost_no_trt <- getRMSE(pred_test_unscaled[,1], out_test[,1])
rmse_cost_trtB <- getRMSE(pred_test_unscaled[,2], out_test[,2])
rmse_qaly_no_trt <- getRMSE(pred_test_unscaled[,3], out_test[,3])
rmse_qaly_trtB <- getRMSE(pred_test_unscaled[,4], out_test[,4])

# Plot the scatter plots for the validation set
g_cost_no_trt_ann <- plot_scatter(pred_test_unscaled[,1], out_test[,1]) + ggtitle("ANN, Cost Status Quo") +
  coord_cartesian(xlim = c(90000, 300000), ylim = c(90000, 300000))
g_cost_trtB_ann <- plot_scatter(pred_test_unscaled[,2], out_test[,2]) + ggtitle("ANN, Cost Intervention") +
  coord_cartesian(xlim = c(90000, 300000), ylim = c(90000, 300000))
g_qaly_no_trt_ann <- plot_scatter(pred_test_unscaled[,3], out_test[,3]) + ggtitle("ANN, QALY Status Quo") +
  coord_cartesian(xlim = c(15, 22), ylim = c(15, 22))
g_qaly_trtB_ann <- plot_scatter(pred_test_unscaled[,4], out_test[,4]) + ggtitle("ANN, QALY Intervention") +
  coord_cartesian(xlim = c(15, 22), ylim = c(15, 22))

g_ann_val <- grid.arrange(g_cost_no_trt_ann, g_cost_trtB_ann, g_qaly_no_trt_ann, g_qaly_trtB_ann, nrow = 2, ncol = 2)

# Calculate the variance
## Cost, Status Quo
var(pred_unscaled[,1]) # Variance of predicted values
var(df_psa_sto$No_Treatment_Cost - pred_unscaled[,1]) # Residual variance
var(pred_unscaled[,1]) / var(df_psa_sto$No_Treatment_Cost) # Variance percentage
var(df_psa_sto$No_Treatment_Cost - pred_unscaled[,1]) / var(df_psa_sto$No_Treatment_Cost) # Variance of deterministic model
## Cost, Intervention
var(pred_unscaled[,2]) # Variance of predicted values
var(df_psa_sto$Treatment_B_Cost - pred_unscaled[,2]) # Residual variance
var(pred_unscaled[,2]) / var(df_psa_sto$Treatment_B_Cost) # Variance percentage
var(df_psa_sto$Treatment_B_Cost - pred_unscaled[,2]) / var(df_psa_sto$Treatment_B_Cost) # Variance of deterministic model
## QALY, Status Quo
var(pred_unscaled[,3]) # Variance of predicted values
var(df_psa_sto$No_Treatment_QALY - pred_unscaled[,3]) # Residual variance
var(pred_unscaled[,3]) / var(df_psa_sto$No_Treatment_QALY) # Variance percentage
var(df_psa_sto$No_Treatment_QALY - pred_unscaled[,3]) / var(df_psa_sto$No_Treatment_QALY) # Variance of deterministic model
## QALY, Intervention
var(pred_unscaled[,4]) # Variance of predicted values
var(df_psa_sto$Treatment_B_QALY - pred_unscaled[,4]) # Residual variance
var(pred_unscaled[,4]) / var(df_psa_sto$Treatment_B_QALY) # Variance percentage
var(df_psa_sto$Treatment_B_QALY - pred_unscaled[,4]) / var(df_psa_sto$Treatment_B_QALY) # Variance of deterministic model


# Calculate the variance percentage
varp_cost_no_trt <- sum((pred_unscaled[,1] - mean(pred_unscaled[,1]))^2)/sum((df_psa_sto$No_Treatment_Cost - mean(df_psa_sto$No_Treatment_Cost))^2)
varp_cost_trtB <- sum((pred_unscaled[,2] - mean(pred_unscaled[,2]))^2)/sum((df_psa_sto$Treatment_B_Cost - mean(df_psa_sto$Treatment_B_Cost))^2)
varp_qaly_no_trt <- sum((pred_unscaled[,3] - mean(pred_unscaled[,3]))^2)/sum((df_psa_sto$No_Treatment_QALY - mean(df_psa_sto$No_Treatment_QALY))^2)
varp_qaly_trtB <- sum((pred_unscaled[,4] - mean(pred_unscaled[,4]))^2)/sum((df_psa_sto$Treatment_B_QALY - mean(df_psa_sto$Treatment_B_QALY))^2)

# Calculate the variance percentage of determininstic model
varp_cost_no_trt_det <- sum((pred_unscaled[,1] - mean(pred_unscaled[,1]))^2)/sum((df_psa_det$No_Treatment_Cost - mean(df_psa_det$No_Treatment_Cost))^2)
varp_cost_trtB_det <- sum((pred_unscaled[,2] - mean(pred_unscaled[,2]))^2)/sum((df_psa_det$Treatment_B_Cost - mean(df_psa_det$Treatment_B_Cost))^2)
varp_qaly_no_trt_det <- sum((pred_unscaled[,3] - mean(pred_unscaled[,3]))^2)/sum((df_psa_det$No_Treatment_QALY - mean(df_psa_det$No_Treatment_QALY))^2)
varp_qaly_trtB_det <- sum((pred_unscaled[,4] - mean(pred_unscaled[,4]))^2)/sum((df_psa_det$Treatment_B_QALY - mean(df_psa_det$Treatment_B_QALY))^2)

# Create CEAC and PSA plots
# PSA
psa_obj_ann <- make_psa_obj(cost = pred_unscaled[,1:2],
                            effectiveness = pred_unscaled[, 3:4],
                            parameters = df_psa_sto[,1:15],
                            strategies = c("Status Quo", "Intervention"),
                            currency = "$")
g_psa_ann <- plot(psa_obj_ann) + ggtitle("ANN Model") + ylim(80000, 300000) + xlim(15,24)
# CEAC
ceac_ann <- ceac(wtp = seq(0, 250000, 1000),
                     psa = psa_obj_ann)
g_ceac_ann <- plot(ceac_ann ) + ggtitle("ANN Model")
# Incremental PSA
psa_obj_ann_inc <- make_psa_obj(cost = cbind(rep(0, nrow(pred_unscaled)), pred_unscaled[,2] - pred_unscaled[,1]),
                                effectiveness = cbind(rep(0, nrow(pred_unscaled)), pred_unscaled[,4] - pred_unscaled[,3]),
                                parameters = df_psa_sto[,1:15],
                                strategies = c("Status Quo", "Intervention"),
                                currency = "$")
g_psa_ann_inc <- plot(psa_obj_ann_inc, center = FALSE) + ggtitle("ANN Model") + theme(legend.position = "none")

g_ann <- list(scatter_plot = g_psa_ann, ceac = g_ceac_ann, scatter_plot_inc = g_psa_ann_inc)
