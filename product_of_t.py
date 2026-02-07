# hello, these are necessary functions for training the PoE
import torch
import math
import numpy as np
import jax
import jax.numpy as jnp

def log_mvt_dens(x, mu, Sigma, nu):
    # log unnormalized density of a single t distribution (for one observation)
    # x = 1D tensor, point that will be evaluated
    # mu = 1D tensor, location parameter
    # Sigma = 2D tensor, scale
    # nu = float, degrees of freedom
    nu = float(nu) # often an integer
    p = x.shape[0]
    Omega = torch.linalg.cholesky(Sigma)
    S = torch.cholesky_inverse(Omega)  # inverse scale
    l = x - mu
    return -0.5*(nu+p)*torch.log(1 + l@S@l/nu)

def log_mvt_prod_dens(x, params):
    # log unnormalized density of a product of t distributions (for one observation)
    # x = 1D tensor, point that will be evaluated
    # params = list consisting of tuples for parameters of the multivariate T
    dens = 0.0 # density
    for mu, Sigma, nu in params:
        dens += log_mvt_dens(x,mu,Sigma,nu)
    return dens

def build_params(mu, L, rho):
    # translate the unconstrained parameters mu, L, and rho into constrained versions
    # mu = location parameter
    # L = general p x p matrix, will make postive definite
    # rho = float, will make positive 
    # for use after each step in stochastic gradient descent
    params=[]
    for w in range(mu.shape[0]):
        # step 1: making L into a positive definite Sigma
        Lw = torch.tril(L[w])
        diag = torch.diagonal(Lw)
        Lw = Lw - torch.diag(Lw) + torch.diag(torch.exp(Lw))
        Sigma = Lw @ Lw.T
        # step 2: making rho into a positive number
        nu = torch.nn.functional.softplus(rho[w]) + 1e-3 # constrain to be positive
        params.append((mu[w],Sigma,nu))
    return params


## mext two functions are for use in the MCMC sampler, hence the jnp (jax.numpy) suffix
## all objects are jnp arrays
def log_mvt_dens_jnp(x, mu, Sigma, nu):
    # log density of a single t distribution (for one observation)
    # x = 1D array, point that will be evaluated
    # mu = 1D array, location parameter
    # Sigma = 2D array, scale
    # nu = float, degrees of freedom
    nu = float(nu) # often an integer
    p = x.shape[0]
    S = jnp.linalg.inv(Sigma)
    l = x - mu
    return -0.5*(nu+p)*jnp.log(1 + l@S@l/nu)

def log_mvt_prod_dens_jnp(x, params):
    # log unnormalized density of a product of t distributions (for one observation)
    # x = 1D array, point that will be evaluated
    # params = list consisting of tuples for parameters of the multivariate T
    # params constructed from the build_params function
    dens = 0.0 # density
    for mu, Sigma, nu in params:
        dens += log_mvt_dens_jnp(x,mu,Sigma,nu)
    return dens

def log_mvt_prod_dens_sample_jnp(x, params):
    # log unnormalized density of a product of t distributions (for multiple observations)
    # x should be a jnp array of observations
    # params = list consisting of tuples for parameters of the multivariate T
    # params constructed from the build_params function
    dens = jnp.zeros(x.shape[0])
    for i in range(x.shape[0]):
        dens = dens.at[i].set(log_mvt_prod_dens_jnp(x[i],params))
    return(dens)


def log_mvt_prod_likelihood(X, params):
    # log unnormalized likelihood of the product of t distributions
    # x = n x p tensor, data that will be evaluated
    # params = list consisting of tuples for parameters of the multivariate T
    # params constructed from the build_params function
    dens = 0.0 # density
    for i in range(X.shape[0]):
        dens +=log_mvt_prod_dens(X[i,:],params)
    return dens

## next are functions for score matching
def score_matching_loss(x,params):
    # computes the score-matching loss in a product of t's
    # x = (p,) tensor, point that will be evaluated
    # params = list of W tuples, each is a triple (mu, Sigma, nu)
    # params built by build_params
    
    # preliminaries
    x = x.clone().detach().requires_grad_(True) # allows for the automatic differentiation
    logf = log_mvt_prod_dens(x,params) # log density for a single point

    # compute score function
    gradf = torch.autograd.grad(logf, x, create_graph=True)[0]
    gradfnorm = torch.sum(gradf**2)

    # compute second derivatives
    hess = 0.0
    for j in range(x.shape[0]):
        hess += torch.autograd.grad(gradf[j], x, create_graph=True)[0][j]

    # full objective function
    return 0.5*gradfnorm + hess

def score_matching_loss_sample(X,params):
    # computes the score-matching loss for an entire dataset
    # X = (n,p) tensor, data that will be evaluate
    loss = 0.0
    for i in range(X.shape[0]):
        loss+=score_matching_loss(X[i],params)
    return loss


    