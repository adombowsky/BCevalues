# libraries
library(dplyr)
library(ggplot2)
library(latex2exp)
# hello, this is code to create the plots for the Shapley application
## cumsum_lbcevs.csv is the output of train_product_of_t.py
## null_samps.csv and alt_samps.csv are output from simulate_product_of_t.py


# loading and plotting data
shapley <- read.csv("cumsum_lbcevs.csv", header = F) # cumulative sums
colnames(shapley) <- "lbcev"
shapley$index <- 1:nrow(shapley)
ggplot(shapley, aes(x=index,y=lbcev)) + geom_point(color="blue") +
  xlab("Index") +
  ylab(TeX("$\\log \\hat{E}_t$")) +
  labs(title="Log Wealth for the Shapley Velocities") +
  theme_bw() +
  theme(text = element_text(size=12)) # log-wealth plot

# comparing samples to data
samps_null <- read.csv("null_samps.csv", header = F)
samps_null <- c(samps_null$V1, samps_null$V2, samps_null$V3, samps_null$V4)
samps_alt <- read.csv("alt_samps.csv", header=F)
samps_alt <- c(samps_alt$V1,samps_alt$V2, samps_alt$V3, samps_alt$V4)
X <- read.table("Shapley_galaxy.dat") # available online
v <- as.numeric(X$V4[-1])
v <- scale(v)
comp_df <- data.frame(x = c(v[,1], samps_null, samps_alt),
                      id = c(rep("Data",nrow(v)), 
                             rep("Null Samples", length(samps_null)), 
                             rep("Alternative Samples", length(samps_alt))))
ggplot(comp_df, aes(x=x, group=id, color = id, fill=id)) + 
  geom_density(alpha=0.1) + 
  theme_bw() +
  xlab("Velocity") +
  ylab("Samples") +
  labs(title="Comparison of HMC Samples and Shapley Velocities",
       color="ID",
       fill="ID") # comparing null and alternative samples to data

