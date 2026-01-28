ar1_simulate <- function(init, J, phi, tau) {
  # evolves the AR(1) process for J steps
  
  # init = initial value
  # J = number of steps
  # phi = slope, |phi|<1
  # tau = scale, tau>0
  y <- init
  for (j in 1:J) {
    y <- phi*y + tau*rnorm(1)
  }
  return(y)
}

besag_clifford_ar1 <- function(dat, J, M, phi, tau) {
  # runs the Besag and Clifford parallel method with an AR(1) process
  
  # dat = observed data
  # J = number of steps
  # M = number of samples
  # phi = slope, |phi|<1
  # tau = scale
  
  ## first, generate y_init
  y_init <- ar1_simulate(init = dat, J=J, phi=phi, tau=tau)
  
  # then, generate the M samples
  y <- rep(0,M)
  for (m in 1:M) {
    y[m] <- ar1_simulate(init = y_init, J=J, phi=phi, tau=tau)
  }
  
  return(list(y_init=y_init, y=y))
}