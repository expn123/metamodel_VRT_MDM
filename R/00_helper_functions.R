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

# A function to have a scatter plot of predicted values and observed values with ggplot2
plot_scatter <- function(predicted, observed) {

  df <- data.frame(predicted = predicted, observed = observed)

  p <- ggplot(df, aes(x = predicted, y = observed)) +
    geom_point(color = "tomato") +
    geom_abline(intercept = 0, slope = 1, linetype = 'dashed', color = 'black') +
    labs(x = "Predicted", y = "Observed") +
    theme_minimal()

  return(p)

}
