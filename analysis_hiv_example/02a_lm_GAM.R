rm(list = ls())

# Load packages
library(gam)
library(dampack)

# Load data
## Stochastic
df_hiv <- readRDS("data/df_hiv.rds")
params <- df_hiv[,1:5]
source("R/00_helper_functions.R")

# Scaling Data
fun_scaling <- function(params) {
  vec.maxs <- apply(params, 2, max)
  vec.mins <- apply(params, 2, min)
  vec.ones <- matrix(1, nrow = nrow(params), 1)
  mat.maxs <- vec.ones %*% vec.maxs
  mat.mins <- vec.ones %*% vec.mins
  scaled_data <-  (params - mat.mins) / (mat.maxs - mat.mins)
  results <- list(scaled_data = scaled_data, vec.maxs = vec.maxs, vec.mins = vec.mins)
  return(results)
}

# Unscaling Data
fun_unscaling <- function(scaled_data, vec.maxs, vec.mins) {
  vec.ones <- matrix(1, nrow = nrow(scaled_data), 1)
  mat.maxs <- vec.ones %*% vec.maxs
  mat.mins <- vec.ones %*% vec.mins
  params <- (scaled_data + 1) * (mat.maxs - mat.mins) / 2 + mat.mins
  return(params)
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

# Prepared the data for metamodeling
features <- fun_scaling(params)$scaled_data
cost_sq <- as.data.frame(cbind(features, cost_no_trt = df_hiv$SQ_Cost))
qaly_sq <- as.data.frame(cbind(features, qaly_no_trt = df_hiv$SQ_QALY))
cost_adhr <- as.data.frame(cbind(features, cost_adhr = df_hiv$Adhr_Cost))
qaly_adhr <- as.data.frame(cbind(features, qaly_adhr = df_hiv$Adhr_QALY))

# Split the training and validation set
set.seed(1341232)
n_train <- 0.8 * nrow(cost_sq)
train_id <- sort(sample(1:nrow(cost_sq), n_train))

# Training datasets
cost_sq_train <- cost_sq[train_id,]
qaly_sq_train <- qaly_sq[train_id,]
cost_adhr_train  <- cost_adhr[train_id,]
qaly_adhr_train  <- qaly_adhr[train_id,]

# Validation sets
cost_sq_test <- cost_sq[-train_id,]
qaly_sq_test <- qaly_sq[-train_id,]
cost_adhr_test  <- cost_adhr[-train_id,]
qaly_adhr_test  <- qaly_adhr[-train_id,]


### Fitting Metamodels --------------------------------------------------------

## 1. Linear Regression Model

mm_lm_cost_sq <- lm(cost_no_trt ~ prep.start.prob * high.adhr * discont.rate.bl, data = cost_sq_train)
mm_lm_qaly_sq <- lm(qaly_no_trt ~ prep.start.prob * high.adhr * discont.rate.bl, data = qaly_sq_train)
mm_lm_cost_adhr <- lm(cost_adhr ~ prep.start.prob * high.adhr * discont.rate.bl +
                        adhr.shift.optim*high.adhr + prep.optim.adhr.init.pp, data = cost_adhr_train)
mm_lm_qaly_adhr <- lm(qaly_adhr  ~ prep.start.prob * high.adhr * discont.rate.bl +
                        adhr.shift.optim*high.adhr, data = qaly_adhr_train)

# Predict the validation set
predict_cost_sq_test <- predict(mm_lm_cost_sq, newdata = cost_sq_test)
predict_cost_adhr_test <- predict(mm_lm_cost_adhr, newdata = cost_adhr_test)
predict_qaly_sq_test <- predict(mm_lm_qaly_sq, newdata = qaly_sq_test)
predict_qaly_adhr_test <- predict(mm_lm_qaly_adhr, newdata = qaly_adhr_test)

# Calculate the RMSE of validation set
rmse_cost_no_trt <- getRMSE(predict_cost_sq_test, cost_sq_test$cost_no_trt)
rmse_cost_trtB <- getRMSE(predict_cost_adhr_test, cost_adhr_test$cost_adhr)
rmse_qaly_no_trt <- getRMSE(predict_qaly_sq_test, qaly_sq_test$qaly_no_trt)
rmse_qaly_trtB <- getRMSE(predict_qaly_adhr_test, qaly_adhr_test$qaly_adhr)

# Calculate the R squared of validation set
r2_cost_no_trt <- getR2(predict_cost_sq_test, cost_sq_test$cost_no_trt)
r2_cost_trtB <- getR2(predict_cost_adhr_test, cost_adhr_test$cost_adhr)
r2_qaly_no_trt <- getR2(predict_qaly_sq_test, qaly_sq_test$qaly_no_trt)
r2_qaly_trtB <- getR2(predict_qaly_adhr_test, qaly_adhr_test$qaly_adhr)

# Prediction with the whole set
pred_cost_sq_all <- predict(mm_lm_cost_sq, newdata = cost_sq)
pred_cost_adhr_all <- predict(mm_lm_cost_adhr, newdata = cost_adhr)
pred_qaly_sq_all <- predict(mm_lm_qaly_sq, newdata = qaly_sq)
pred_qaly_adhr_all <- predict(mm_lm_qaly_adhr, newdata = qaly_adhr)

# Plot the scatter plots for the validation set
g_cost_no_trt_lm <- plot_scatter(predict_cost_sq_test, cost_sq_test$cost_no_trt) + ggtitle("Linear Regression, Cost Status Quo") +
  coord_cartesian(xlim = c(46110952611, 47650371315), ylim = c(46110952611, 47650371315))
g_cost_adhr_lm <- plot_scatter(predict_cost_adhr_test, cost_adhr_test$cost_adhr) + ggtitle("Linear Regression, Cost Adherence") +
  coord_cartesian(xlim = c(46110952611, 47650371315), ylim = c(46110952611, 47650371315))
g_qaly_no_trt_lm <- plot_scatter(predict_qaly_sq_test, qaly_sq_test$qaly_no_trt) + ggtitle("Linear Regression, QALY Status Quo") +
  coord_cartesian(xlim = c(3319736, 3330900), ylim = c(3319736, 3330900))
g_qaly_trtB_lm <- plot_scatter(predict_qaly_adhr_test, qaly_adhr_test$qaly_adhr) + ggtitle("Linear Regression, QALY Adherence") +
  coord_cartesian(xlim = c(3319736, 3330900), ylim = c(3319736, 3330900))

g_lm_val_hiv <- grid.arrange(g_cost_no_trt_lm, g_cost_adhr_lm, g_qaly_no_trt_lm, g_qaly_trtB_lm, nrow = 2, ncol = 2)


# Use the predicted values for PSA and CEAC
# PSA
psa_obj_lm <- make_psa_obj(cost =  cbind(pred_cost_sq_all, pred_cost_adhr_all),
                            effectiveness = cbind(pred_qaly_sq_all, pred_qaly_adhr_all),
                            parameters = params,
                            strategies = c("Status Quo", "Adherence"),
                            currency = "$")
# CEAC
ceac_obj_lm <- ceac(wtp = seq(0, 250000, 1000),
                     psa = psa_obj_lm)
g_ceac_lm <- plot(ceac_obj_lm) + ggtitle("Linear Regression Model") + ylim(c(0, 1))
g_psa_lm  <- plot(psa_obj_lm) + ggtitle("Linear Regression Model")

# Incremental PSA Scatter Plot
psa_obj_lm_inc <- make_psa_obj(cost =  cbind(rep(0, 1980), pred_cost_adhr_all - pred_cost_sq_all),
                               effectiveness = cbind(rep(0, 1980), pred_qaly_adhr_all - pred_qaly_sq_all),
                               parameters = params,
                               strategies = c("Status Quo", "Adherence"),
                               currency = "$")
g_psa_lm_inc <- plot(psa_obj_lm_inc, center = FALSE, alpha = 0.3) + ggtitle("Linear Regression Model") + theme(legend.position = "none")

g_lm_hiv <- list(scatter_plot = g_psa_lm, ceac = g_ceac_lm, scatter_plot_inc = g_psa_lm_inc)


## 2. GAM
mm_gam_cost_sq <- gam(formula = cost_no_trt ~ s(prep.start.prob)*s(high.adhr)*s(discont.rate.bl),
                          data = cost_sq_train)
mm_gam_cost_adhr  <- gam(formula = cost_adhr ~ s(prep.start.prob)*s(high.adhr)*s(discont.rate.bl) + s(adhr.shift.optim)*s(high.adhr) + s(prep.optim.adhr.init.pp),
                          data = cost_adhr_train)
mm_gam_qaly_sq <- gam(formula = qaly_no_trt ~ s(prep.start.prob)*s(high.adhr)*s(discont.rate.bl),
                          data = qaly_sq_train)
mm_gam_qaly_adhr <- gam(formula = qaly_adhr ~ s(prep.start.prob)*s(high.adhr)*s(discont.rate.bl) + s(adhr.shift.optim)*s(high.adhr),
                          data = qaly_adhr_train)

# Calculate the RMSE for the validation set
rmse_cost_sq   <- getRMSE(predict(mm_gam_cost_sq, newdata = cost_sq_test), cost_sq_test$cost_no_trt)
rmse_cost_adhr <-   getRMSE(predict(mm_gam_cost_adhr, newdata = cost_adhr_test), cost_adhr_test$cost_adhr)
rmse_qaly_sq   <- getRMSE(predict(mm_gam_qaly_sq, newdata = qaly_sq_test), qaly_sq_test$qaly_no_trt)
rmse_qaly_adhr <-   getRMSE(predict(mm_gam_qaly_adhr, newdata = qaly_adhr_test), qaly_adhr_test$qaly_adhr)

# Calculate the R-Square for the validation set
r2_cost_sq   <- getR2(predict(mm_gam_cost_sq, newdata = cost_sq_test), cost_sq_test$cost_no_trt)
r2_cost_adhr <- getR2(predict(mm_gam_cost_adhr, newdata = cost_adhr_test), cost_adhr_test$cost_adhr)
r2_qaly_sq   <- getR2(predict(mm_gam_qaly_sq, newdata = qaly_sq_test), qaly_sq_test$qaly_no_trt)
r2_qaly_adhr <- getR2(predict(mm_gam_qaly_adhr, newdata = qaly_adhr_test), qaly_adhr_test$qaly_adhr)

# Plot the scatter plots for the validation set
g_cost_no_trt_gam <- plot_scatter(predict(mm_gam_cost_sq, newdata = cost_sq_test), cost_sq_test$cost_no_trt) + ggtitle("GAM, Cost Status Quo") +
  coord_cartesian(xlim = c(46110952611, 47650371315), ylim = c(46110952611, 47650371315))
g_cost_adhr_gam <- plot_scatter(predict(mm_gam_cost_adhr, newdata = cost_adhr_test), cost_adhr_test$cost_adhr) + ggtitle("GAM, Cost Adherence") +
  coord_cartesian(xlim = c(46110952611, 47650371315), ylim = c(46110952611, 47650371315))
g_qaly_no_trt_gam <- plot_scatter(predict(mm_gam_qaly_sq, newdata = qaly_sq_test), qaly_sq_test$qaly_no_trt) + ggtitle("GAM, QALY Status Quo") +
  coord_cartesian(xlim = c(3319736, 3330900), ylim = c(3319736, 3330900))
g_qaly_trtB_gam <- plot_scatter(predict(mm_gam_qaly_adhr, newdata = qaly_adhr_test), qaly_adhr_test$qaly_adhr) + ggtitle("GAM, QALY Adherence") +
  coord_cartesian(xlim = c(3319736, 3330900), ylim = c(3319736, 3330900))

g_gam_val_hiv <- grid.arrange(g_cost_no_trt_gam, g_cost_adhr_gam, g_qaly_no_trt_gam, g_qaly_trtB_gam, nrow = 2, ncol = 2)


# Prediction with the whole set
pred_cost_sq_all_gam <- predict(mm_gam_cost_sq, newdata = cost_sq)
pred_cost_adhr_all_gam <- predict(mm_gam_cost_adhr, newdata = cost_adhr)
pred_qaly_sq_all_gam <- predict(mm_gam_qaly_sq, newdata = qaly_sq)
pred_qaly_adhr_all_gam <- predict(mm_gam_qaly_adhr, newdata = qaly_adhr)

psa_obj_gam <- make_psa_obj(cost =  cbind(pred_cost_sq_all_gam, pred_cost_adhr_all_gam),
                            effectiveness = cbind(pred_qaly_sq_all_gam, pred_qaly_adhr_all_gam),
                            parameters = params,
                            strategies = c("Status Quo", "Adherence"),
                            currency = "$")
ceac_obj_gam <- ceac(wtp = seq(0, 250000, 1000),
                    psa = psa_obj_gam)
g_ceac_gam <- plot(ceac_obj_gam) + ggtitle("GAM") + ylim(c(0, 1))
g_psa_gam  <- plot(psa_obj_gam) + ggtitle("GAM")
# Incremental plot
psa_obj_gam_inc <- make_psa_obj(cost =  cbind(rep(0, 1980), pred_cost_adhr_all_gam - pred_cost_sq_all_gam),
                               effectiveness = cbind(rep(0, 1980), pred_qaly_adhr_all_gam - pred_qaly_sq_all_gam),
                               parameters = params,
                               strategies = c("Status Quo", "Adherence"),
                               currency = "$")
g_psa_gam_inc <- plot(psa_obj_gam_inc, center = FALSE, alpha = 0.3) + ggtitle("GAM") + theme(legend.position = "none")

g_gam <- list(scatter_plot = g_psa_gam, ceac = g_ceac_gam, scatter_plot_inc = g_psa_gam_inc)
