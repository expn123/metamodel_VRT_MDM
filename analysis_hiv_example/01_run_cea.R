rm(list = ls())

# Load external packages
library(devtools)
library(dplyr)
library(dampack)
library(gridExtra)

# Load the data (PSA Results)
df_hiv <- readRDS("data/df_hiv.rds")


# Take a look at Adherence and Status Quo in stochastic model
sum((df_hiv$Adhr_QALY - df_hiv$SQ_QALY) < 0)/nrow(df_hiv)

psa_obj_hiv <- make_psa_obj(cost = df_hiv[, c("SQ_Cost", "Adhr_Cost")],
                            effectiveness = df_hiv[, c("SQ_QALY", "Adhr_QALY")],
                            parameters = df_hiv[,1:5],
                            strategies = c("Status Quo", "Adherence"),
                            currency = "$")
ceac_hiv <- ceac(wtp = seq(0, 250000, 1000),
                 psa = psa_obj_hiv)

# Incremental PSA Scatter Plot
psa_obj_hiv_inc <- make_psa_obj(cost = cbind(rep(0, 1980), df_hiv$Adhr_Cost-df_hiv$SQ_Cost),
                                effectiveness = cbind(rep(0, 1980), df_hiv$Adhr_QALY-df_hiv$SQ_QALY),
                                parameters = df_hiv[,1:5],
                                strategies = c("Status Quo", "Adherence"),
                                currency = "$")

# Save the results
g_psa_hiv <- plot(psa_obj_hiv) + ggtitle("Original HIV Model")
g_psa_hiv_inc <- plot(psa_obj_hiv_inc, center = FALSE, alpha = 0.3) + ggtitle("Original HIV Model") + theme(legend.position = "none")
g_ceac_hiv <- plot(ceac_hiv) + ggtitle("Original HIV Model") + ylim(c(0, 1))

g_hiv <- list(scatter_plot = g_psa_hiv, ceac = g_ceac_hiv, scatter_plot_inc = g_psa_hiv_inc)

