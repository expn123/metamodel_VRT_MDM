rm(list = ls())

# Load external packages
library(devtools)
library(dplyr)
library(dampack)
library(gridExtra)

# Load the data (PSA Results)
df_psa_det <- readRDS("data/df_psa_det.rds")
df_psa_sto <- readRDS("data/df_psa_sto.rds")


# Take a look at treatment B and no treatment in stochastic model
sum((df_psa_sto$Treatment_B_QALY - df_psa_sto$No_Treatment_QALY) < 0)

# Calculate the variance
## Cost, Status Quo
var(df_psa_sto$No_Treatment_Cost) # Total
var(df_psa_det$No_Treatment_Cost) # Deterministic
var(df_psa_sto$No_Treatment_Cost - df_psa_det$No_Treatment_Cost) # Residual
## Cost, Treatment B
var(df_psa_sto$Treatment_B_Cost) # Total
var(df_psa_det$Treatment_B_Cost) # Deterministic
var(df_psa_sto$Treatment_B_Cost - df_psa_det$Treatment_B_Cost) # Residual
## QALY, Status Quo
var(df_psa_sto$No_Treatment_QALY) # Total
var(df_psa_det$No_Treatment_QALY) # Deterministic
var(df_psa_sto$No_Treatment_QALY - df_psa_det$No_Treatment_QALY) # Residual
var(df_psa_det$No_Treatment_QALY)/var(df_psa_sto$No_Treatment_QALY) # Proportion of variance
var(df_psa_sto$No_Treatment_QALY - df_psa_det$No_Treatment_QALY)/var(df_psa_sto$No_Treatment_QALY) # Residual proportion of variance
## QALY, Treatment B
var(df_psa_sto$Treatment_B_QALY) # Total
var(df_psa_det$Treatment_B_QALY) # Deterministic
var(df_psa_sto$Treatment_B_QALY - df_psa_det$Treatment_B_QALY) # Residual
var(df_psa_det$Treatment_B_QALY)/var(df_psa_sto$Treatment_B_QALY) # Proportion of variance
var(df_psa_sto$Treatment_B_QALY - df_psa_det$Treatment_B_QALY)/var(df_psa_sto$Treatment_B_QALY) # Residual proportion of variance

# 1. Deterministic model----------------------------------------------
# Create the PSA object
psa_obj_det <- make_psa_obj(cost = df_psa_det[, c("No_Treatment_Cost", "Treatment_B_Cost")],
                            effectiveness = df_psa_det[, c("No_Treatment_QALY", "Treatment_B_QALY")],
                            parameters = df_psa_det[,1:15],
                            strategies = c("Status Quo", "Intervention"),
                            currency = "$")
# Plot the PSA Scatter Plot
g_psa_det <- plot(psa_obj_det) + ggtitle("Deterministic Model") + ylim(80000, 300000) + xlim(15,24)
# Plot CEAC
ceac_obj_det <- ceac(wtp = seq(0, 250000, 1000),
                 psa = psa_obj_det)
g_ceac_det <- plot(ceac_obj_det) + ggtitle("Deterministic Model")
# Incremental PSA Scatter Plot
psa_obj_det_inc <- make_psa_obj(cost = cbind(rep(0, nrow(df_psa_det)), df_psa_det$Treatment_B_Cost - df_psa_det$No_Treatment_Cost),
                                effectiveness = cbind(rep(0, nrow(df_psa_det)), df_psa_det$Treatment_B_QALY - df_psa_det$No_Treatment_QALY),
                                parameters = df_psa_det[,1:15],
                                strategies = c("Status Quo", "Intervention"),
                                currency = "$")
g_psa_det_inc <- plot(psa_obj_det_inc, center = FALSE) + ggtitle("Deterministic Model")

# 2. Stochastic model----------------------------------------------
# Create the PSA object
psa_obj_sto <- make_psa_obj(cost = df_psa_sto[, c("No_Treatment_Cost", "Treatment_B_Cost")],
                            effectiveness = df_psa_sto[, c("No_Treatment_QALY", "Treatment_B_QALY")],
                            parameters = df_psa_sto[,1:15],
                            strategies = c("Status Quo", "Intervention"),
                            currency = "$")
# Plot the PSA Scatter Plot
g_psa_sto <- plot(psa_obj_sto) + ggtitle("Stochastic Model") + ylim(80000, 300000) + xlim(15, 24)
# Plot CEAC
ceac_obj_sto <- ceac(wtp = seq(0, 250000, 1000),
                     psa = psa_obj_sto)
g_ceac_sto <- plot(ceac_obj_sto) + ggtitle("Stochastic Model")
# Incremental PSA Scatter Plot
psa_obj_sto_inc <- make_psa_obj(cost = cbind(rep(0, nrow(df_psa_sto)), df_psa_sto$Treatment_B_Cost - df_psa_sto$No_Treatment_Cost),
                                effectiveness = cbind(rep(0, nrow(df_psa_sto)), df_psa_sto$Treatment_B_QALY - df_psa_sto$No_Treatment_QALY),
                                parameters = df_psa_sto[,1:15],
                                strategies = c("Status Quo", "Intervention"),
                                currency = "$")
g_psa_sto_inc <- plot(psa_obj_sto_inc, center = FALSE) + ggtitle("Stochastic Model")


