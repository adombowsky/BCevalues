#  hello, this is the code to train the PoE and compute BC e-values
import torch
import math
import numpy as np
import product_of_t as pt
import pandas as pd
import random as rd
import jax
import jax.numpy as jnp
import blackjax as bjax

# simulate some data
rd.seed(1996) # ranodm number seed
#X = torch.normal(0,1,(500,1))
# load in Galaxy dataset
#Xdf = pd.read_csv("galaxy.csv") # initial data frame
Xdf = pd.read_csv("Shapley_galaxy.dat", sep='\s+')
X = Xdf["Vel"]
X = torch.from_numpy(X.to_numpy()).float()
#X = torch.log(torch.unsqueeze(X,dim=1)) # getting correct dimensions
X =torch.unsqueeze(X,dim=1) # getting correct dimensions
mX = torch.mean(X,0) # for centering
sX = torch.std(X,0) # for scaling
X = (X-mX)/sX
# # split the data
n = X.shape[0] // 2 # splitting the data in half
p = X.shape[1] # dimension
inds = rd.sample(range(0,X.shape[0]),n)
X_a = X[inds]
not_inds = torch.ones(X.shape[0],dtype=torch.bool)
not_inds[inds] = False
X_b = X[not_inds]

# # initialize the parameters
# W =5   # number of experts
# mu = torch.normal(0,1,(W,p),requires_grad=True) # centers
# L = torch.normal(0,1,(W,p,p),requires_grad=True) # scales
# rho = torch.normal(0,1,(W,),requires_grad=True) # dfs



# # load in optimizer (adam)
# optimizer = torch.optim.Adam(
#     [mu, L, rho],
#     lr=1e-3
# )

# # run the loop, get estimates
# steps = 3000 # number of steps
# stops = 500 # when to print progress
# for r in range(steps):
#     optimizer.zero_grad()

#     params = pt.build_params(mu, L, rho) # adjust current parameter values
#     loss = pt.score_matching_loss_sample(X_a, params) # compute loss function on the data

#     loss.backward()
#     optimizer.step()

#     if r % stops==0:
#         print(f"Step {r}, loss = {loss.item():.4f}")

# trained_params = pt.build_params(mu, L, rho)

# # compute alternative parameters
# # initialize the parameters
# W = 25 # number of experts
# mu = torch.normal(0,1,(W,p),requires_grad=True) # centers
# L = torch.normal(0,1,(W,p,p),requires_grad=True) # scales
# rho = torch.normal(0,1,(W,),requires_grad=True) # dfs

# # load in optimizer (adam)
# optimizer = torch.optim.Adam(
#     [mu, L, rho],
#     lr=1e-3
# )
# for r in range(steps):
#     optimizer.zero_grad()

#     params = pt.build_params(mu, L, rho) # adjust current parameter values
#     loss = pt.score_matching_loss_sample(X_a, params) # compute loss function on the data

#     loss.backward()
#     optimizer.step()

#     if r % stops==0:
#         print(f"Step {r}, loss = {loss.item():.4f}")
    

# alternative_params = pt.build_params(mu, L, rho)

# # saving parameters
# torch.save(trained_params, "null_params.pt")
# torch.save(alternative_params, "alt_params.pt")

# load in parameters
trained_params = torch.load("null_params.pt")
alternative_params = torch.load("alt_params.pt")


# converting the trained params into jnp arrays
trained_params_jnp =[]
for mu, Sigma, nu in trained_params:
    mu_jnp = jnp.array(mu.detach().numpy())
    Sigma_jnp = jnp.array(Sigma.detach().numpy())
    nu_jnp = jnp.array(nu.detach().numpy())
    trained_params_jnp.append((mu_jnp, Sigma_jnp, nu_jnp))

# converting the alt params into jnp arrays
alternative_params_jnp =[]
for mu, Sigma, nu in alternative_params:
    mu_jnp = jnp.array(mu.detach().numpy())
    Sigma_jnp = jnp.array(Sigma.detach().numpy())
    nu_jnp = jnp.array(nu.detach().numpy())
    alternative_params_jnp.append((mu_jnp, Sigma_jnp, nu_jnp))

#print(trained_params)
# return the optimized unnormalized likelihood
# loglike = pt.log_mvt_prod_likelihood(X_b,trained_params) 
# print(loglike)

# setting up MCMC
J =  500 # number of steps
M = 1000 # number of iterations
target_poe = lambda x: pt.log_mvt_prod_dens_jnp(x,trained_params_jnp)
hmc_setup = bjax.hmc(target_poe, step_size = 0.1, inverse_mass_matrix = jnp.ones(p), num_integration_steps = 1000)

# writing a function for just running a single chain
def run_single_chain(initial_state, keys):
    def one_step(state, key):
        new_state, info = hmc_setup.step(key, state)
        # None as the second value so don't store history
        return new_state, None
        
    # final_state is the state after J
    final_state, _ = jax.lax.scan(one_step, initial_state, keys)
    return final_state.position

# initialization
bcevs = np.zeros(X_b.shape[0])
# looping across dataset to get BC e-values
for i in range(X_b.shape[0]):
    # generate Besag-Clifford samples
    ## initial state
    initial_state = hmc_setup.init(jnp.array(X_b[i].numpy()))
    rng_keys = jax.random.split(jax.random.PRNGKey(1),J)
    y_zero = run_single_chain(initial_state, rng_keys)

    ## generate samples
    new_initial_state = hmc_setup.init(y_zero) # setting initial state to be the new generated one
    initial_states = jax.vmap(lambda _: new_initial_state)(jnp.arange(M)) # setting the same initial point for each run
    rng_key = jax.random.PRNGKey(1)
    chain_keys = jax.random.split(rng_key, M)
    step_keys = jax.vmap(lambda k: jax.random.split(k, J))(chain_keys)
    # parallelize across M chains
    parallel_sampler = jax.jit(jax.vmap(run_single_chain))
    all_samples = parallel_sampler(initial_states, step_keys)
    null_density= pt.log_mvt_prod_dens_sample_jnp(all_samples, trained_params_jnp)
    alternative_density= pt.log_mvt_prod_dens_sample_jnp(all_samples, alternative_params_jnp)
    t_x = pt.log_mvt_prod_dens(X_b[i],alternative_params)-pt.log_mvt_prod_dens(X_b[i],trained_params)
    d_tilde = alternative_density-null_density # e-values = alt denisty/null density
    log_e_tilde = np.append(np.array(d_tilde),t_x.detach().numpy()) # for log-sum-exp
    log_e_star = np.max(log_e_tilde)
    bcevs[i] = np.log(M+1) + t_x - log_e_star - np.log(np.sum( np.exp(log_e_tilde - log_e_star)))
    #bcevs[i] = t_x.detach().numpy()/(np.mean( np.concatenate(t_x.detach().numpy(), np.array(e_tilde))))

    # progress
    if i % 50==0:
        print(f"Observation {i}")

#lbcevs = np.log(bcevs)

# # saving everything
np.savetxt('bcevs.csv', bcevs, delimiter=',')
np.savetxt('cumsum_lbcevs.csv', np.cumsum(bcevs), delimiter=',')