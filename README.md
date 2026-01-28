# Introduction
For any test statistic $T(X)$, a Besag-Clifford e-value is
$$\hat E_M(X) = \frac{(M+1)T(X)}{T(X) + \sum_{m=1}^M T(Y^{(m)}) }$$
where $(X,Y^{(1)}, \dots, Y^{(M)})$ are sampled using the parallel algorithm of Besag and Clifford.
This repository provides code to implement the experiments and applications in ``Besag-Clifford e-values for unnormalized testing."

## AR(1) Process
We test $H_0: \mathcal N(0,1)$ against $H_1: \mathcal N(\mu,1)$. The BC e-value is $$\hat E_M(X) = \frac{(M+1)\exp(\mu x)}{\exp(\mu x) + \sum_{m=1}^M \exp(\mu Y^{(m)}) }.$$

* ```besag_clifford_ar1(dat,J,M,phi,tau)```
  * computes a single BC e-value using the AR(1) process as an MCMC algorithm
  * ```dat```: observed data, real number
  * ```J``` number of steps in the parallel algorithm
  * ```M``` number of Besag-Clifford samples
* ```r/ar_one.R```
  * doc of functions for creating AR(1) BC e-values
* ```ar_one_meahshift.R```
  * script to implement the simulation study in Section 6.1, creates figures 2 and 3


## Sequential PoE
The null hypothesis is a product of experts (PoE) with $2$ experts and the alternative is the $\mathcal N(0,1)$ distribution.
The BC e-value is constructed from $$T(x_i) = \frac{e^{-\frac{(x_i-\mu)^2}{2}}}{\sqrt{2\pi \sigma^2}} \prod_{w=1}^2 \left( 1 + \frac{(x_i-\psi_w)^2}{\theta_w} \right)^{\frac{\theta_w+1}{2}}.$$
* ```besag_clifford_evals_t_product_chains(x,J,M,nu,ncp,mu,sd,S)```
  * calculates a BC e-value for a single object
  * ``x``: observed data, real number
  * ```J``` number of steps in the parallel algorithm
  * ```M``` number of Besag-Clifford samples
  * ```nu```, ```ncp```: df and noncentral parameter of t-distributions, both vectors
  * ```mu```, ```sd```: mean and standard deviation of normal distribution
  * ```S```: number of chains
* ```sequential_product_of_t.R```
  * script to implement the simulation study in Section 6.2, creates figure 4

## Sequential Composite Alternatives
The null is $\mathcal N(0,1)$ and the alternative is $\mathcal N(\mu, \sigma^2)$ for unknown $\mu$ and $\sigma^2$.
The BC e-value is constructed from $$T(x_t \mid x_{1:(t-1)}) = \frac{\N(x_t; \bar x_t, \hat \sigma_t^2)}{\mathcal N(x_t; 0, 1)}.$$
* ```grapa_obj(lambda, e_vals)```
  * objective function for the GRAPA betting strategy, passed on to ```optim```
  * ```lambda```: weight, in (0,1)
  * ```e_vals```: vector of observed e-values
* ```sequential_composite_normals.R```
  * script to implement the simulation study in Section 6.3, creates figure 5