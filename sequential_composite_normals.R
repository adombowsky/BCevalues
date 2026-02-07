# functions
source("r/composite_e_value.R")

# empirically adaptive e-process
grapa_obj <- function(lambda, e_vals) {
  # objective function, note it is negative for minimization purposes
  return(-mean(log(1-lambda + lambda*e_vals)))
}

# parameters
r <- 1000 # number of iterations
theta <- 1 # values of the alternative mean
sigma <- 2
#theta <- seq(-.5,0,by=0.025) # values of the null mean
alpha <- 0.05 # significance level
n <- 50 # sample size
M <- 1000 # samples from the null
U_samps <- LR_samps <- matrix(0,nrow=r,ncol=n)
lambda_U <- lambda_LR <- 0 # betting strategy
# simulate
for (s in 1:r) {
    set.seed(s) # saving output
    x <- rnorm(n,mean=theta,sd=sigma) # sample from alternative (or null)
    U <- LR <- U_evals <- LR_evals <- rep(1,n)
    for (i in 2:n) {
      # first, handle Besag-Clifford e-values # storage for e-values and e-processes
      lambda_U <- optimize(grapa_obj, lower=0, upper=1, e_vals = U_evals[1:(i-1)])$minimum # betting strategy
      x_tilde <- rnorm(M,mean=0,sd=1)
      dat_tilde <- do.call(rbind, lapply(c(x[i],x_tilde), function(vi) c(x[1:(i-1)], vi)))
      mu_tilde <- rowMeans(dat_tilde)
      sigmasq_tilde <- apply(dat_tilde, 1, function(x) mean((x-mean(x))^2))
      e_tilde <- (dnorm(c(x[i], x_tilde), mu_tilde, sqrt(sigmasq_tilde)) )/dnorm(c(x[i],x_tilde),mean=0,sd=1)
      U_evals[i] <- dnorm(x[i], mu_tilde[1], sd=sqrt(sigmasq_tilde[1]))/dnorm(x[i],mean=0,sd=1)/mean(e_tilde)
      U[i] = U[i-1]*(1-lambda_U + lambda_U*U_evals[i])
      # next, handle UI e-values
      lambda_LR <- optimize(grapa_obj, lower=0, upper=1, e_vals = LR_evals[1:(i-1)])$minimum # betting strategy
      LR_evals[i] = (dnorm(x[i],mean=mean(x[1:(i-1)]),sd=ifelse(i>2,((i-2)/(i-1))*sd(x[1:(i-1)]),1))/dnorm(x[i],mean=0,sd=1))
      LR[i] = LR[i-1]*(1-lambda_LR + lambda_LR*LR_evals[i])
    }
    # save output
    U_samps[s,] <- U
    LR_samps[s,] <- LR
}

matplot(
  t(log(LR_samps)),
  type = "l",
  lty  = 1,
  col  = rgb(0, 0, 0, 0.2),
  xlab = "time",
  ylab = "value"
)

matlines(
  t(log(U_samps)),
  type = "l",
  lty  = 1,
  col  = rgb(0, 0, 0.5, 0.2),
  xlab = "time",
  ylab = "value"
)
abline(h=log(1/0.05),col="red")

# plotting w/gg
library(ggplot2)
library(latex2exp)
library(cowplot)

U_mean <- colMeans(log(U_samps))
U_CI <- apply(log(U_samps),2,quantile,probs=c(0.025,0.975))
LR_mean <- colMeans(log(LR_samps))
LR_CI <- apply(log(LR_samps),2,quantile,probs=c(0.025,0.975))

plot_df <- data.frame(U_mean = U_mean,
                      U_l = U_CI[1,],
                      U_u = U_CI[2,],
                      LR_mean = LR_mean,
                      LR_l = LR_CI[1,],
                      LR_u = LR_CI[2,])
# initial plot
logwealth <- ggplot(plot_df, aes(x=1:n,y=U_mean)) + geom_line(col="blue") + 
  ylab(TeX("$\\log \\hat{E}_t$")) + xlab("t") +
  labs(title="Log Wealth of BC and LR E-Processes") +
  geom_ribbon(aes(ymin=U_l,ymax=U_u),fill="blue",alpha=0.1) +
  geom_line(aes(x=1:n,y=LR_mean),col="red") + 
  geom_ribbon(aes(ymin=LR_l,ymax=LR_u),fill="red",alpha=0.1) + 
  geom_hline(yintercept = log(1/0.05),linetype="dashed") + theme_bw()
# create legend
legend_plot <- ggplot(
  data.frame(x = 1, y = 1, lab = c("LR", "BC")),
  aes(x, y, color = lab)
) +
  labs(color="Process") +
  geom_point() +
  scale_color_manual(values = c(LR = "red", BC = "blue")) +
  theme_void() +
  theme(legend.position = "right", text = element_text(size=13))
leg <- get_legend(legend_plot)
plot_grid(logwealth, leg, rel_widths = c(1, 0.15))
