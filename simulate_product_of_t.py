# hello, code to simulate from the pre-trained PoE model in "train_product_of_t.py"
import numpy as np
import blackjax as bjax
import jax.numpy as jnp
import torch
import jax
import product_of_t as pt

# load in trained parameters
trained_params = torch.load("null_params.pt") # for null samples
#trained_params = torch.load("alt_params.pt") # for alternative samples

# converting the trained params into jnp arrays, for HMC sampling
trained_params_jnp =[]
for mu, Sigma, nu in trained_params:
    mu_jnp = jnp.array(mu.detach().numpy())
    Sigma_jnp = jnp.array(Sigma.detach().numpy())
    nu_jnp = jnp.array(nu.detach().numpy())
    trained_params_jnp.append((mu_jnp, Sigma_jnp, nu_jnp))

# setting up HMC
target_poe = lambda x: pt.log_mvt_prod_dens_jnp(x,trained_params_jnp)
hmc_setup = bjax.hmc(target_poe, step_size = 0.05, inverse_mass_matrix = jnp.ones(1), num_integration_steps = 1000)

# function to generate samples using HMC in BlackJAX
def run_single_chain(initial_state, keys):
    def one_step(state, key):
        new_state,info= hmc_setup.step(key, state)
        return new_state, new_state
        
    # run multiple iterations of HMC
    current, samps = jax.lax.scan(one_step, initial_state, keys)
    return current, samps


# running the HMC sampler
R = 10000 # number of simulations
initial_state = hmc_setup.init(jnp.array(jax.random.normal(jax.random.PRNGKey(1),1))) # initialization
rng_keys = jax.random.split(jax.random.PRNGKey(1),R) # random number seeds
_, samps = run_single_chain(initial_state, rng_keys) # generating HMC samples
np.savetxt('null_samps.csv', np.array(samps[0]), delimiter=',') # save null samples
#np.savetxt('alt_samps.csv', np.array(samps[0]), delimiter=',') # save alternative samples

