rm(list = ls())

# Load packages
library(gam)
library(dampack)
library(gridExtra)

# Load data
## Stochastic
df_psa_sto <- readRDS("data/df_psa_sto.rds")
params <- df_psa_sto[,c(1:12, 14)]
## Determinanstic
df_psa_det <- readRDS("data/df_psa_det.rds")

# Load the functions
source("R/00_helper_functions.R")


# Prepared the data for metamodeling
features <- fun_scaling(params)$scaled_data
cost_no_treatment <- as.data.frame(cbind(features, cost_no_trt = df_psa_sto$No_Treatment_Cost))
qaly_no_treatment <- as.data.frame(cbind(features, qaly_no_trt = df_psa_sto$No_Treatment_QALY))
cost_treatment_b <- as.data.frame(cbind(features, cost_trt_b = df_psa_sto$Treatment_B_Cost))
qaly_treatment_b <- as.data.frame(cbind(features, qaly_trt_b = df_psa_sto$Treatment_B_QALY))

# Split the training and validation set
set.seed(1342)
n_train <- 0.8 * nrow(cost_no_treatment)
train_id <- sort(sample(1:nrow(cost_no_treatment), n_train))

# Training datasets
cost_no_treatment_train <- cost_no_treatment[train_id,]
qaly_no_treatment_train <- qaly_no_treatment[train_id,]
cost_treatment_b_train  <- cost_treatment_b[train_id,]
qaly_treatment_b_train  <- qaly_treatment_b[train_id,]

# Validation sets
cost_no_treatment_test <- cost_no_treatment[-train_id,]
qaly_no_treatment_test <- qaly_no_treatment[-train_id,]
cost_treatment_b_test  <- cost_treatment_b[-train_id,]
qaly_treatment_b_test  <- qaly_treatment_b[-train_id,]


### Fitting Metamodels --------------------------------------------------------

## 1. Linear Regression Model

mm_lm_cost_no_trt <- lm(cost_no_trt ~ ., data = cost_no_treatment_train)
mm_lm_qaly_no_trt <- lm(qaly_no_trt ~ ., data = qaly_no_treatment_train)
mm_lm_cost_trtB <- lm(cost_trt_b ~ ., data = cost_treatment_b_train)
mm_lm_qaly_trtB <- lm(qaly_trt_b  ~ ., data = qaly_treatment_b_train)

# Predict the validation set
predict_cost_no_trt <- predict(mm_lm_cost_no_trt, newdata = cost_no_treatment_test)
predict_cost_trtB <- predict(mm_lm_cost_trtB, newdata = cost_treatment_b_test)
predict_qaly_no_trt <- predict(mm_lm_qaly_no_trt, newdata = qaly_no_treatment_test)
predict_qaly_trtB <- predict(mm_lm_qaly_trtB, newdata = qaly_treatment_b_test)

# Calculate the RMSE of validation set
rmse_cost_no_trt <- getRMSE(predict_cost_no_trt, cost_no_treatment_test$cost_no_trt)
rmse_cost_trtB <- getRMSE(predict_cost_trtB, cost_treatment_b_test$cost_trt_b)
rmse_qaly_no_trt <- getRMSE(predict_qaly_no_trt, qaly_no_treatment_test$qaly_no_trt)
rmse_qaly_trtB <- getRMSE(predict_qaly_trtB, qaly_treatment_b_test$qaly_trt_b)

# Calculate the R squared of validation set
r2_cost_no_trt <- getR2(predict_cost_no_trt, cost_no_treatment_test$cost_no_trt)
r2_cost_trtB <- getR2(predict_cost_trtB, cost_treatment_b_test$cost_trt_b)
r2_qaly_no_trt <- getR2(predict_qaly_no_trt, qaly_no_treatment_test$qaly_no_trt)
r2_qaly_trtB <- getR2(predict_qaly_trtB, qaly_treatment_b_test$qaly_trt_b)

# Plot the scatter plots for the validation set
g_cost_no_trt_lm <- plot_scatter(predict_cost_no_trt, cost_no_treatment_test$cost_no_trt) + ggtitle("Linear Regression, Cost Status Quo") +
  coord_cartesian(xlim = c(90000, 300000), ylim = c(90000, 300000))
g_cost_trtB_lm <- plot_scatter(predict_cost_trtB, cost_treatment_b_test$cost_trt_b) + ggtitle("Linear Regression, Cost Intervention") +
  coord_cartesian(xlim = c(90000, 300000), ylim = c(90000, 300000))
g_qaly_no_trt_lm <- plot_scatter(predict_qaly_no_trt, qaly_no_treatment_test$qaly_no_trt) + ggtitle("Linear Regression, QALY Status Quo") +
  coord_cartesian(xlim = c(15, 22), ylim = c(15, 22))
g_qaly_trtB_lm <- plot_scatter(predict_qaly_trtB, qaly_treatment_b_test$qaly_trt_b) + ggtitle("Linear Regression, QALY Intervention") +
  coord_cartesian(xlim = c(15, 22), ylim = c(15, 22))

g_lm_val <- grid.arrange(g_cost_no_trt_lm, g_cost_trtB_lm, g_qaly_no_trt_lm, g_qaly_trtB_lm, nrow = 2, ncol = 2)

# Prediction with the whole set
pred_cost_no_trt_all <- predict(mm_lm_cost_no_trt, newdata = cost_no_treatment)
pred_cost_trtB_all <- predict(mm_lm_cost_trtB, newdata = cost_treatment_b)
pred_qaly_no_trt_all <- predict(mm_lm_qaly_no_trt, newdata = qaly_no_treatment)
pred_qaly_trtB_all <- predict(mm_lm_qaly_trtB, newdata = qaly_treatment_b)

# Calculate the variance values
## Cost, Status Quo
var(pred_cost_no_trt_all) # Variance of predicted values
var(df_psa_sto$No_Treatment_Cost - pred_cost_no_trt_all) # Variance of residuals
var(pred_cost_no_trt_all)/var(df_psa_sto$No_Treatment_Cost) # Variance percentage
var(df_psa_sto$No_Treatment_Cost - pred_cost_no_trt_all)/var(df_psa_sto$No_Treatment_Cost) # Variance percentage of residuals
## QALY, Status Quo
var(pred_qaly_no_trt_all) # Variance of predicted values
var(df_psa_sto$No_Treatment_QALY - pred_qaly_no_trt_all) # Variance of residuals
var(pred_qaly_no_trt_all)/var(df_psa_sto$No_Treatment_QALY) # Variance percentage
var(df_psa_sto$No_Treatment_QALY - pred_qaly_no_trt_all)/var(df_psa_sto$No_Treatment_QALY) # Variance percentage of residuals
## Cost, Intervention
var(pred_cost_trtB_all) # Variance of predicted values
var(df_psa_sto$Treatment_B_Cost - pred_cost_trtB_all) # Variance of residuals
var(pred_cost_trtB_all)/var(df_psa_sto$Treatment_B_Cost) # Variance percentage
var(df_psa_sto$Treatment_B_Cost - pred_cost_trtB_all)/var(df_psa_sto$Treatment_B_Cost) # Variance percentage of residuals
## QALY, Intervention
var(pred_qaly_trtB_all) # Variance of predicted values
var(df_psa_sto$Treatment_B_QALY - pred_qaly_trtB_all) # Variance of residuals
var(pred_qaly_trtB_all)/var(df_psa_sto$Treatment_B_QALY) # Variance percentage
var(df_psa_sto$Treatment_B_QALY - pred_qaly_trtB_all)/var(df_psa_sto$Treatment_B_QALY) # Variance percentage of residuals


# Use the predicted values for PSA and CEAC
# PSA
psa_obj_lm <- make_psa_obj(cost =  cbind(pred_cost_no_trt_all, pred_cost_trtB_all),
                            effectiveness = cbind(pred_qaly_no_trt_all, pred_qaly_trtB_all),
                            parameters = params,
                            strategies = c("Status Quo", "Intervention"),
                            currency = "$")
g_psa_lm  <- plot(psa_obj_lm) + ggtitle("Linear Regression Model") + ylim(80000, 300000) + xlim(15,24)
# CEAC
ceac_obj_lm <- ceac(wtp = seq(0, 250000, 1000),
                     psa = psa_obj_lm)
g_ceac_lm <- plot(ceac_obj_lm) + ggtitle("Linear Regression Model")
# Incremental PSA
psa_obj_lm_inc <- make_psa_obj(cost =  cbind(rep(0, length(pred_cost_trtB_all)), pred_cost_trtB_all - pred_cost_no_trt_all),
                               effectiveness = cbind(rep(0, length(pred_qaly_trtB_all)), pred_qaly_trtB_all - pred_qaly_no_trt_all),
                               parameters = params,
                               strategies = c("Status Quo", "Intervention"),
                               currency = "$")
g_psa_lm_inc  <- plot(psa_obj_lm_inc, center = FALSE) + ggtitle("Linear Regression Model") + theme(legend.position = "none")
g_lm <- list(scatter_plot = g_psa_lm, ceac = g_ceac_lm, scatter_plot_inc = g_psa_lm_inc)


## 2. GAM
mm_gam_cost_no_trt <- gam(formula = cost_no_trt ~ s(r_HS1) + s(r_S1H) + s(r_S1S2) + s(hr_S1) + s(hr_S2) +
                            s(hr_S1S2_trtB) + s(c_H) + s(c_S1) + s(c_S2) + s(u_H) + s(u_S1) + s(u_S2) + s(c_trtB),
                          data = cost_no_treatment_train)
mm_gam_cost_trtB  <- gam(formula = cost_trt_b ~ s(r_HS1) + s(r_S1H) + s(r_S1S2) + s(hr_S1) + s(hr_S2) +
                            s(hr_S1S2_trtB) + s(c_H) + s(c_S1) + s(c_S2) + s(u_H) + s(u_S1) + s(u_S2) + s(c_trtB),
                          data = cost_treatment_b_train)
mm_gam_qaly_no_trt <- gam(formula = qaly_no_trt ~ s(r_HS1) + s(r_S1H) + s(r_S1S2) + s(hr_S1) + s(hr_S2) +
                            s(hr_S1S2_trtB) + s(c_H) + s(c_S1) + s(c_S2) + s(u_H) + s(u_S1) + s(u_S2) + s(c_trtB),
                          data = qaly_no_treatment_train)
mm_gam_qaly_trtB <- gam(formula = qaly_trt_b ~ s(r_HS1) + s(r_S1H) + s(r_S1S2) + s(hr_S1) + s(hr_S2) +
                            s(hr_S1S2_trtB) + s(c_H) + s(c_S1) + s(c_S2) + s(u_H) + s(u_S1) + s(u_S2) + s(c_trtB),
                          data = qaly_treatment_b_train)

# Calculate the RMSE for the validation set
rmse_cost_no_trt <- getRMSE(predict(mm_gam_cost_no_trt, newdata = cost_no_treatment_test), cost_no_treatment_test$cost_no_trt)
rmse_cost_trtB <- getRMSE(predict(mm_gam_cost_trtB, newdata = cost_treatment_b_test), cost_treatment_b_test$cost_trt_b)
rmse_qaly_no_trt <- getRMSE(predict(mm_gam_qaly_no_trt, newdata = qaly_no_treatment_test), qaly_no_treatment_test$qaly_no_trt)
rmse_qaly_trtB <- getRMSE(predict(mm_gam_qaly_trtB, newdata = qaly_treatment_b_test), qaly_treatment_b_test$qaly_trt_b)

# Calculate the R-Square for the validation set
r2_cost_no_trt <- getR2(predict(mm_gam_cost_no_trt, newdata = cost_no_treatment_test), cost_no_treatment_test$cost_no_trt)
r2_cost_trtB <- getR2(predict(mm_gam_cost_trtB, newdata = cost_treatment_b_test), cost_treatment_b_test$cost_trt_b)
r2_qaly_no_trt <- getR2(predict(mm_gam_qaly_no_trt, newdata = qaly_no_treatment_test), qaly_no_treatment_test$qaly_no_trt)
r2_qaly_trtB <- getR2(predict(mm_gam_qaly_trtB, newdata = qaly_treatment_b_test), qaly_treatment_b_test$qaly_trt_b)

# Plot the scatter plots for the validation set
g_cost_no_trt_gam <- plot_scatter(predict(mm_gam_cost_no_trt, newdata = cost_no_treatment_test), cost_no_treatment_test$cost_no_trt) + ggtitle("GAM, Cost Status Quo") +
  coord_cartesian(xlim = c(90000, 300000), ylim = c(90000, 300000))
g_cost_trtB_gam <- plot_scatter(predict(mm_gam_cost_trtB, newdata = cost_treatment_b_test), cost_treatment_b_test$cost_trt_b) + ggtitle("GAM, Cost Intervention") +
  coord_cartesian(xlim = c(90000, 300000), ylim = c(90000, 300000))
g_qaly_no_trt_gam <- plot_scatter(predict(mm_gam_qaly_no_trt, newdata = qaly_no_treatment_test), qaly_no_treatment_test$qaly_no_trt) + ggtitle("GAM, QALY Status Quo") +
  coord_cartesian(xlim = c(15, 22), ylim = c(15, 22))
g_qaly_trtB_gam <- plot_scatter(predict(mm_gam_qaly_trtB, newdata = qaly_treatment_b_test), qaly_treatment_b_test$qaly_trt_b) + ggtitle("GAM, QALY Intervention") +
  coord_cartesian(xlim = c(15, 22), ylim = c(15, 22))

g_gam_val <- grid.arrange(g_cost_no_trt_gam, g_cost_trtB_gam, g_qaly_no_trt_gam, g_qaly_trtB_gam, nrow = 2, ncol = 2)

# Prediction with the whole set
pred_cost_no_trt_all_gam <- predict(mm_gam_cost_no_trt, newdata = cost_no_treatment)
pred_cost_trtB_all_gam <- predict(mm_gam_cost_trtB, newdata = cost_treatment_b)
pred_qaly_no_trt_all_gam <- predict(mm_gam_qaly_no_trt, newdata = qaly_no_treatment)
pred_qaly_trtB_all_gam <- predict(mm_gam_qaly_trtB, newdata = qaly_treatment_b)

# Calculate the variance values
## Cost, Status Quo
var(pred_cost_no_trt_all_gam) # Variance of predicted values
var(df_psa_sto$No_Treatment_Cost - pred_cost_no_trt_all_gam) # Variance of residuals
var(pred_cost_no_trt_all_gam)/var(df_psa_sto$No_Treatment_Cost) # Variance percentage
var(df_psa_sto$No_Treatment_Cost - pred_cost_no_trt_all_gam)/var(df_psa_sto$No_Treatment_Cost) # Variance percentage of residuals
## QALY, Status Quo
var(pred_qaly_no_trt_all_gam) # Variance of predicted values
var(df_psa_sto$No_Treatment_QALY - pred_qaly_no_trt_all_gam) # Variance of residuals
var(pred_qaly_no_trt_all_gam)/var(df_psa_sto$No_Treatment_QALY) # Variance percentage
var(df_psa_sto$No_Treatment_QALY - pred_qaly_no_trt_all_gam)/var(df_psa_sto$No_Treatment_QALY) # Variance percentage of residuals
## Cost, Intervention
var(pred_cost_trtB_all_gam) # Variance of predicted values
var(df_psa_sto$Treatment_B_Cost - pred_cost_trtB_all_gam) # Variance of residuals
var(pred_cost_trtB_all_gam)/var(df_psa_sto$Treatment_B_Cost) # Variance percentage
var(df_psa_sto$Treatment_B_Cost - pred_cost_trtB_all_gam)/var(df_psa_sto$Treatment_B_Cost) # Variance percentage of residuals
## QALY, Intervention
var(pred_qaly_trtB_all_gam) # Variance of predicted values
var(df_psa_sto$Treatment_B_QALY - pred_qaly_trtB_all_gam) # Variance of residuals
var(pred_qaly_trtB_all_gam)/var(df_psa_sto$Treatment_B_QALY) # Variance percentage
var(df_psa_sto$Treatment_B_QALY - pred_qaly_trtB_all_gam)/var(df_psa_sto$Treatment_B_QALY) # Variance percentage of residuals


# Calculate the variance percentage
varp_cost_no_trt_gam <- sum((pred_cost_no_trt_all_gam - mean(pred_cost_no_trt_all_gam))^2) / sum((df_psa_sto$No_Treatment_Cost - mean(df_psa_sto$No_Treatment_Cost))^2)
varp_cost_trtB_gam <- sum((pred_cost_trtB_all_gam - mean(pred_cost_trtB_all_gam))^2) / sum((df_psa_sto$Treatment_B_Cost - mean(df_psa_sto$Treatment_B_Cost))^2)
varp_qaly_no_trt_gam <- sum((pred_qaly_no_trt_all_gam - mean(pred_qaly_no_trt_all_gam))^2) / sum((df_psa_sto$No_Treatment_QALY - mean(df_psa_sto$No_Treatment_QALY))^2)
varp_qaly_trtB_gam <- sum((pred_qaly_trtB_all_gam - mean(pred_qaly_trtB_all_gam))^2) / sum((df_psa_sto$Treatment_B_QALY - mean(df_psa_sto$Treatment_B_QALY))^2)

varp_cost_no_trt_gam_det <- sum((pred_cost_no_trt_all_gam - mean(pred_cost_no_trt_all_gam))^2) / sum((df_psa_det$No_Treatment_Cost - mean(df_psa_det$No_Treatment_Cost))^2)
varp_cost_trtB_gam_det <- sum((pred_cost_trtB_all_gam - mean(pred_cost_trtB_all_gam))^2) / sum((df_psa_det$Treatment_B_Cost - mean(df_psa_det$Treatment_B_Cost))^2)
varp_qaly_no_trt_gam_det <- sum((pred_qaly_no_trt_all_gam - mean(pred_qaly_no_trt_all_gam))^2) / sum((df_psa_det$No_Treatment_QALY - mean(df_psa_det$No_Treatment_QALY))^2)
varp_qaly_trtB_gam_det <- sum((pred_qaly_trtB_all_gam - mean(pred_qaly_trtB_all_gam))^2) / sum((df_psa_det$Treatment_B_QALY - mean(df_psa_det$Treatment_B_QALY))^2)

# Use the predicted values for PSA and CEAC
# PSA
psa_obj_gam <- make_psa_obj(cost =  cbind(pred_cost_no_trt_all_gam, pred_cost_trtB_all_gam),
                            effectiveness = cbind(pred_qaly_no_trt_all_gam, pred_qaly_trtB_all_gam),
                            parameters = params,
                            strategies = c("Status Quo", "Intervention"),
                            currency = "$")
g_psa_gam  <- plot(psa_obj_gam) + ggtitle("GAM") + ylim(80000, 300000) + xlim(15,24)
# CEAC
ceac_obj_gam <- ceac(wtp = seq(0, 250000, 1000),
                    psa = psa_obj_gam)
g_ceac_gam <- plot(ceac_obj_gam) + ggtitle("GAM")
# Incremental PSA
psa_obj_gam_inc <- make_psa_obj(cost =  cbind(rep(0, length(pred_cost_trtB_all_gam)), pred_cost_trtB_all_gam - pred_cost_no_trt_all_gam),
                               effectiveness = cbind(rep(0, length(pred_qaly_trtB_all_gam)), pred_qaly_trtB_all_gam - pred_qaly_no_trt_all_gam),
                               parameters = params,
                               strategies = c("Status Quo", "Intervention"),
                               currency = "$")
g_psa_gam_inc  <- plot(psa_obj_gam_inc, center = FALSE) + ggtitle("GAM") + theme(legend.position = "none")
g_gam <- list(scatter_plot = g_psa_gam, ceac = g_ceac_gam, scatter_plot_inc = g_psa_gam_inc)

