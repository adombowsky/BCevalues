## preliminaries: functions
# log-likelihood for the product of t-distributions
log_t_product <- function(x,nu,ncp){
  # x = data
  # nu = vector of degrees of freedom
  # ncp = vector of non-centrality parameters
  loglik <- 0
  n <- length(x)
  K <- length(nu)
  for (i in 1:n) {
    loglik <- loglik + sum(dt(x[i],df=nu, ncp=ncp,log=T))
  }
  return(loglik)
}

# unnormalized likelihood ratio
ulr <- function(x, mu, sd, nu0, ncp0) {
  return( dnorm(x,mean=mu,sd=sd)* (1+(x-ncp0[1])^2/nu0[1])^(0.5*(nu0[1]+1))*(1+(x-ncp0[2])^2/nu0[2])^(0.5*(nu0[2]+1)) )
}

# log unnormalized likelihood ratio
log_ulr <- function(x, mu, sd, nu0, ncp0) {
  return( dnorm(x,mean=mu,sd=sd, log=T) -
            ( -0.5*(nu0[1]+1)*log(1+(x-ncp0[1])^2/nu0[1]) +
                0.5*(nu0[2]+1)* log(1+(x-ncp0[2])^2/nu0[2])   )  )
}

# Besag-Clifford Samples
library(MCMCpack)
besag_clifford_t_product <- function(dat, J, M, nu, ncp, init_seed=1) {
  # runs the Besag and Clifford parallel method with an AR(1) process
  
  # dat = observed data
  # J = number of steps
  # M = number of samples
  # init_seed = seed of initial draw
  
  ## first, generate y_init
  y_init <- MCMCmetrop1R(log_t_product, theta.init = dat, nu=nu, ncp=ncp, burnin=J, mcmc=1, seed=init_seed, verbose = 0)
  
  # then, generate the M samples
  y <- rep(0,M)
  for (m in 1:M) {
    y[m] <- MCMCmetrop1R(log_t_product, theta.init = y_init, nu=nu, ncp=ncp, burnin=J, mcmc=1, seed=m, verbose = 0)
  }
  return(list(y_init=y_init, y=y))
}

# Besag-Clifford E-value
besag_clifford_evals_t_product <- function(x, J, M, nu, ncp, mu, sd, init_seed=1) {
  # x = data point
  # J = number of steps
  # M = number of samples
  # nu, ncp = df and noncentral parameter of t-distributions
  # mu, sd = parameters of normal distribution
  # init_seed = seed of initial draw
  bc <- besag_clifford_t_product(dat = x, J = J, M = M, nu = nu, ncp = ncp, init_seed=init_seed)
  samps <- bc$y
  t_x <- c(ulr(x, mu, sd, nu, ncp),ulr(samps, mu, sd, nu, ncp))
  return( ulr(x, mu, sd, nu, ncp)/mean(t_x))
}

# Besag-Clifford E-value Multi-Chain
besag_clifford_evals_t_product_chains <- function(x, J, M, nu, ncp, mu, sd, S) {
  # x = data point
  # J = number of steps
  # M = number of samples
  # nu, ncp = df and noncentral parameter of t-distributions
  # mu, sd = parameters of normal distribution
  # S = number of chains
  evals <- rep(0,S)
  for (s in 1:S) {
    evals[s] <- besag_clifford_evals_t_product(x= x, J = J, M = M, nu = nu, ncp = ncp, mu=mu, sd=sd, init_seed=s)
    # samps <- bc$y
    # t_x <- c(ulr(x, mu, sd, nu, ncp),ulr(samps, mu, sd, nu, ncp))
  }
  return(mean(evals))
}

# testing several trajectories
R <- 500 # number of replications
n <- 50 # number of time points
mu <- 0 # Gaussian mean
sd <- 1 # Gaussian sd
J <- 4 # number of steps
M <- 25 # number of samples
S_med <- 4 # number of chains (for comparison)
S_large <- 10 # number of chains (for comparison)
nu <- c(1,10) # product of t df
ncp <- c(-3,0) # product of t ncp
bc_evals <- bc_evals_med <- bc_evals_large <- matrix(0, nrow = R, ncol = n)
for (r in 1:R) {
  set.seed(r)
  X <- rnorm(n, mean=mu, sd=sd)
  for (i in 1:n) {
    bc_evals[r,i] <- besag_clifford_evals_t_product(X[i], J, M, nu, ncp, mu, sd)
    bc_evals_med[r,i] <- besag_clifford_evals_t_product_chains(X[i], J, M, nu, ncp, mu, sd, S_med)
    bc_evals_large[r,i] <- besag_clifford_evals_t_product_chains(X[i], J, M, nu, ncp, mu, sd, S_large)
  }
}

lbc_evals <- t(apply(log(bc_evals),1,cumsum))
lbc_evals_med <- t(apply(log(bc_evals_med),1,cumsum))
lbc_evals_large <- t(apply(log(bc_evals_large),1,cumsum))


# plot(lbc_evals[1,],type="l")
# lines(lbc_evals_mc[1,],type="l",col="red")

matplot(
  t(lbc_evals),
  type = "l",
  lty  = 1,
  col  = rgb(0, 0, 0, 0.2),
  xlab = "time",
  ylab = "value"
)

matlines(
  t(lbc_evals_med),
  type = "l",
  lty  = 1,
  col  = rgb(0, 0, 0.5, 0.2),
  xlab = "time",
  ylab = "value"
)

matlines(
  t(lbc_evals_large),
  type = "l",
  lty  = 1,
  col  = rgb(0, 0.5, 0, 0.2),
  xlab = "time",
  ylab = "value"
)
abline(h=log(1/0.05),col="red")

# plot(colMeans(lbc_evals), type = "l", col="blue")
# lines(colMeans(lbc_evals_med), col="red")
# lines(colMeans(lbc_evals_large), col="purple")

cms <- data.frame(small = colMeans(lbc_evals),
                  small_l = apply(lbc_evals,2,quantile,probs=.025),
                  small_u = apply(lbc_evals,2,quantile,probs=.975),
                  med = colMeans(lbc_evals_med),
                  med_l = apply(lbc_evals_med,2,quantile,probs=.025),
                  med_u = apply(lbc_evals_med,2,quantile,probs=.975),
                  large=colMeans(lbc_evals_large),
                  large_l = apply(lbc_evals_large,2,quantile,probs=.025),
                  large_u = apply(lbc_evals_large,2,quantile,probs=.975))

# plotting
library(ggplot2)
library(cowplot)
library(latex2exp)

# first find common axes
lwb <- min(cms)
upb <- max(cms)

plot_small <- ggplot(cms,aes(x=1:n,y=small)) + 
  ylim(lwb,upb) +
  ylab(TeX("$\\log \\hat{E}_t$")) + xlab("t") +
  geom_hline(yintercept=log(1/0.05), linetype="dashed") +
  labs(title="1 Chain") +
  geom_ribbon(aes(ymin=small_l,ymax=small_u), fill="blue", alpha=0.1) + 
  geom_line(col="blue") + theme_bw() + theme(text = element_text(size=12))
plot_med <-  ggplot(cms,aes(x=1:n,y=med)) +
  ylim(lwb,upb) +
  ylab(TeX("$\\log \\hat{E}_t$")) + xlab("t") +
  geom_hline(yintercept=log(1/0.05), linetype="dashed") +
  labs(title = paste(toString(S_med), "Chains")) +
  geom_ribbon(aes(ymin=med_l,ymax=med_u), fill="purple", alpha=0.1) + 
  geom_line(col="purple") + theme_bw() + theme(text = element_text(size=12))
plot_large <-  ggplot(cms,aes(x=1:n,y=large)) + 
  ylim(lwb,upb) +
  ylab(TeX("$\\log \\hat{E}_t$")) + xlab("t") +
  geom_hline(yintercept=log(1/0.05), linetype="dashed") +
  labs(title = paste(toString(S_large), "Chains")) +
  geom_ribbon(aes(ymin=large_l,ymax=large_u), fill="red", alpha=0.1) +
  geom_line(col="red") + theme_bw() + theme(text = element_text(size=12))

plot_grid(plot_small, plot_med, plot_large, nrow=1, labels = c("(a)","(b)","(c)"))


## if you want to compare betting strategies

# # empirically adaptive e-process
# grapa_obj <- function(lambda, e_vals) {
#   # objective function, note it is negative for minimization purposes
#   return(-mean(log(1-lambda + lambda*e_vals)))
# }
# 
# # implementing GRAPA
# lambda <- rep(0,n)
# for (i in 2:n) {
#   lambda[i] <- optimize(grapa_obj, lower=0, upper=1, e_vals = bc_evals[1,1:(i-1)])$minimum
# }


