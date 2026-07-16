import numpy as np
from scipy.stats import lognorm
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pickle, corner, time
from scipy.special import logsumexp
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from scipy.integrate import simpson
import jax
from log_normal_h2 import get_lnorm_mu_sig2, get_h2_mean_var
from pdf_compound import pdf_saddlepoint, lnpdf_saddlepoint
from ng_refit_astro import LikelihoodFS
from mcmc_ds import get_chains
from sim_astro_data import get_psrs

# constants
m_sun = 2e30
c = 3e8
G = 6.67e-11
Mpc = 3.086e22
yr = 365.25 * 24 * 3600

dirname = 'Nbright_50_astro_model_KAKA'
get_psrs(dirname, gaussian=True, npsrs=60, Tobs=20, ntoas=200, log10_toaerr=-7., nfreqs=50, n0_dot=1.26e-3, alpha=0.5, m0=10**9.3 * m_sun, beta=0.5, z0=1., seed=1321)
chains = get_chains(dirname)

# dirname = 'astro_psrs'
chains = pickle.load(open('chains/'+dirname+'_chains.pkl', 'rb'))
Tspan = 20 * 365.25 * 24 * 3600
like_fs = LikelihoodFS(chains, Tspan, prior='lognormal')

# injected values
log10_n0_dot = np.log10(1.26e-3)
alpha = 0.5
log10_m0 = 9.3
beta = 0.5
z0 = 1.
x_inj = np.array([log10_n0_dot, alpha, log10_m0, beta, z0])

# init sampler
x0 = np.copy(x_inj)
ndim = len(x0)
cov = np.diag(np.ones(ndim) * 0.01**2)
sampler = ptmcmc(ndim, like_fs.get_lnlike, like_fs.get_lnprior, cov,
outDir='./ptmcmc_'+dirname+'_refit/', resume=False)  # Default chain filename will be used

# sample for N steps
N = int(1e5)
sampler.sample(x0, N, SCAMweight=50, AMweight=0, DEweight=50)

# # plot corner
# labels = ['log10 n0_dot', 'alpha', 'log10 M0', 'beta', 'z0']
# # chains = np.genfromtxt('./ptmcmc_astro_psrs_test_dirac/chain_1.txt')
# # fig = corner.corner(chains[2000:, :-4], color='C0', plot_datapoints=False, labels=labels, bins=20, smooth=1., smooth1d=1.) #, hist_kwargs={'density':True})
# # chains = np.genfromtxt('./ptmcmc_astro_psrs_full/chain_1.txt')
# # corner.corner(chains[2000:, :-4], color='C1', plot_datapoints=False, labels=labels, bins=20, truths=x_inj, truth_color="k", smooth=1., smooth1d=1., fig=fig) #, hist_kwargs={'density':True})
# chains = np.genfromtxt('./ptmcmc_'+dirname+'_refit/chain_1.txt')
# corner.corner(chains[2000:, :-4], color='C2', plot_datapoints=False, labels=labels, bins=20, truths=x_inj, truth_color="k", smooth=1., smooth1d=1.)#, fig=fig) #, hist_kwargs={'density':True})
# plt.savefig('corner_astro_Nbright50.pdf', bbox_inches='tight')
# plt.savefig('corner_astro_Nbright50.png', bbox_inches='tight')
# plt.show()