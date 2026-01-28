# implements the AR(1) example for testing mean-shift
source("r/ar_one.R")
library(ggplot2)
library(latex2exp)
library(cowplot)

# the ULR for this example
ulr <- function(x,mu) {
  return(exp(x*mu))
}

# the LR for this example
lr <- function(x,mu) {
  return(exp(x*mu - (mu^2/2)))
}

# simulation parameters
R <- 1000 # number of replications
Phi <- c(0.3,0.5,0.8) # slope
mu <- 1 # alternative 
J <- 1
M <- 1000

# run iterations and create plots
ehat <- delta_ehat <- likerat <- matrix(0,nrow=R,ncol=length(Phi))
for (s in 1:length(Phi)) {
  phi <- Phi[s]
  tau <- sqrt(1-phi^2) # scale
  ehat_samps <- delta_samps <- lr_samps <- rep(0,R)
  t_x <- rep(0,M)
  for (r in 1:R) {
    set.seed(r)
    #x <- rnorm(1) # simulate from null
    x <- rnorm(1,mean=mu) # simulate from alternative
    bc <- besag_clifford_ar1(dat = x, J = J, M = M, phi = phi, tau = tau)
    samps <- bc$y
    y_init <- bc$y_init
    # computing delta
    delta_samps[r] <- lr(y_init,phi*mu)
    # computing the other e-variables
    t_x <- c(ulr(x,mu),ulr(samps,mu))
    ehat_samps[r] <- ulr(x,mu)/mean(t_x)
    lr_samps[r] <- lr(x,mu)
  }
  ehat[,s] <- ehat_samps
  delta_ehat[,s] <- ehat_samps*delta_samps
  likerat[,s] <- lr_samps
}

# plotting
## ehat vs likerat
ehat_plots <- list()
for (s in 1:length(Phi)) {
  df <- data.frame(ehat = ehat[,s], likerat = likerat[,s])
  ehat_plots[[s]] <- ggplot(df, aes(x=likerat,y=ehat)) + geom_point() +
    geom_abline(slope = 1, intercept = 0, color="red") + 
    xlab("LR") + ylab(TeX("$\\hat{E}_M$")) +
    labs(title = TeX(paste0("$\\phi = ", toString(Phi[s]), "$"))) +
    theme_bw()
}
## ehat*delta vs likerat
delta_ehat_plots <- list()
for (s in 1:length(Phi)) {
  df <- data.frame(delta_ehat = delta_ehat[,s], likerat = likerat[,s])
  delta_ehat_plots[[s]] <- ggplot(df, aes(x=likerat,y=delta_ehat)) + geom_point() +
    geom_abline(slope = 1, intercept = 0, color="red") + 
    xlab("LR") + ylab(TeX("$\\Delta(Y^{(0)}) \\hat{E}_M$")) +
    labs(title = TeX(paste0("$\\phi = ", toString(Phi[s]), "$"))) +
    theme_bw()
}

# displaying
plot_grid(plotlist=c(ehat_plots,delta_ehat_plots),nrow=2)


### next, plot the power as M increases
phi <- 0.5
tau <- sqrt(1-phi^2)
J <- c(1,3,5,10,20)
M <- c(25,50,100,500,1000,2500,5000)
bc_evals <- rep(0,R)
alpha <- 0.05
mu <- 2 # alternative
power_curves <- matrix(0,nrow=length(J),ncol=length(M))
for (l in 1:length(J)) {
  for (i in 1:length(M)) {
    for (r in 1:R) {
      set.seed(r)
      x <- rnorm(1,mean=mu)
      bc <- besag_clifford_ar1(dat = x, J = J[l], M = M[i], phi = phi, tau = tau)
      samps <- bc$y
      # computing the other e-variables
      t_x <- c(ulr(x,mu),ulr(samps,mu))
      bc_evals[r] <- ulr(x,mu)/mean(t_x)
    }
    power_curves[l,i] <- mean(bc_evals>=(1/alpha))
  }
}

## sanity check, power of the LR
lr_samps <- rep(0,R)
for (r in 1:R) {
  set.seed(r)
  x <- rnorm(1,mean=mu)
  lr_samps[r] <- lr(x,mu)
}
power_lr <- mean(lr_samps>=(1/alpha))

power_curves_df <- data.frame(power=as.vector(t(log(power_curves))),
                              J = rep(factor(J),each=length(M)),
                              M = rep(factor(M),length(J)))

ggplot(power_curves_df, aes(x=M,y=power,group=J,color=J)) + geom_point(size=2) + geom_line() +
  theme_bw() +
  ylab("log(power)") +
  labs(title="Log Power for Varying J and M") +
  geom_hline(yintercept = log(power_lr), linetype="dashed",linewidth=0.5) +
  theme(text = element_text(size=13))
