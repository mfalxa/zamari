import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
import healpy as hp
from scipy.stats import gamma as gdist
from numpy.fft import rfft, irfft
import scipy.special as ss
from scipy.stats import gamma as gdist
from scipy.stats import invgauss, lognorm


m_sun = 1.989e30
c = 2.998e8
G = 6.674e-11
Mpc = 3.086e22
yr = 365.25 * 24 * 3600

def h2_binary(z, mc, f):

    dl = cosmo.luminosity_distance(z).to(u.meter).value
    return (32 * np.pi**(4/3) / 5 / c**8) * ((1 + z) * G * mc)**(10/3) * f**(4/3) / dl**2

def mc_from_hz(z, h, f):

    dl = cosmo.luminosity_distance(z).to(u.meter).value
    return (h**2 / ((32 * np.pi**(4/3) / 5 / c**8) * ((1 + z) * G)**(10/3) * f**(4/3) / dl**2))**(3/10)

def dn_dzdlog10M(z, mc, n0_dot, alpha, m0, beta, z0):

    # cosmology terms
    Hz = cosmo.H(z)
    dt_dz = 1 / ((1 + z) * Hz)
    dt_dz = dt_dz.to(u.Gyr).value

    # astrophysics terms
    mass_term = (mc / (1e7 * m_sun))**(-alpha) * np.exp(-mc/m0)
    redshift_term = (1 + z)**beta * np.exp(-z/z0) * dt_dz

    return n0_dot * mass_term * redshift_term 

def dN_dzdlog10Mdf(z, mc, f, n0_dot, alpha, m0, beta, z0):

    # cosmology terms
    dm = cosmo.comoving_distance(z).value

    # astrophysics terms
    mass_redshift_term = dn_dzdlog10M(z, mc, n0_dot, alpha, m0, beta, z0)
    frequency_term = df_dt(mc, f)

    return mass_redshift_term * (1 + z) * 4*np.pi * (c / Mpc) * dm**2  / frequency_term

def df_dt(mc, f):

    return (96/5) * np.pi**(8/3) * (G * mc / c**3)**(5/3) * f**(11/3)

def lognormal(x, mu, sig):

    return np.exp(-0.5 * (np.log(x) - mu)**2 / sig**2) / (x * sig * (2*np.pi)**0.5)

def get_h2_mean_var(freqs, n0_dot, alpha, m0, beta, z0, only_mean=False):

    df = freqs[0]
    N_grid = dN_dzdlog10Mdf(z[:, None], mc[None, :], (1 + z[:, None]), n0_dot, alpha, m0, beta, z0) * dlog10_mc * dz[:, None] * (1 + z[:, None])
    h2_grid = np.array(h2_binary(z[:, None], 10**log10_mc[None, :], 1.)) # * freq / df

    Nh2 = np.sum(N_grid * h2_grid)
    h2c = Nh2 * freqs**(-11/3) * df * (freqs**(4/3) * freqs / df)
    if only_mean:
        return h2c, 0.
    else:
        Nh4 = np.sum(N_grid * h2_grid**2)
        var_h2c = Nh4 * freqs**(-11/3) * df * (freqs**(4/3) * freqs / df)**2
        return h2c, var_h2c
    
def get_h2_n_cumul(freqs, ncumul, n0_dot, alpha, m0, beta, z0):

    df = freqs[0]
    N_grid = dN_dzdlog10Mdf(z[:, None], mc[None, :], 1., n0_dot, alpha, m0, beta, z0) * dlog10_mc * dz[:, None]
    h2_grid = np.array(h2_binary(z[:, None], 10**log10_mc[None, :], 1.)) # * freq / df

    Nhn = np.sum(N_grid * h2_grid**ncumul)
    n_h2c = Nhn * freqs**(-11/3) * df * (freqs**(4/3) * freqs / df)**ncumul
    return n_h2c

def get_lnorm_mu_sig2(mean, var):

    # lognorm params
    mu = np.log(mean) - 0.5 * np.log(1 + (var/mean**2))
    sig2 = np.log(1 + (var / mean**2))

    return mu, sig2

def get_lnorms(h2, freqs, n0_dot, alpha, m0, beta, z0):

    mean, var = get_h2_mean_var(freqs, n0_dot, alpha, m0, beta, z0)
    mu, sig2 = get_lnorm_mu_sig2(mean, var)
    lnorms = [lognorm.pdf(h2, s=sig2[n]**0.5, scale=np.exp(mu[n])) for n in range(len(freqs))]

    return lnorms

# setup frequencies
df = 1 / (20 * 365.25 * 24 * 3600)
# freq = 1. / (7. * 365.25 * 24 * 3600)
freqs = np.arange(1, 20) * df

# model
alpha = 0.5
m0 = 10**9.3 * m_sun
beta = 0.5
z0 = 1.
n0_dot = 1.26e-3

# coefficients
L_f = (96/5) * np.pi**(8/3) * (G/c**3)**(5/3)
K_f = (32/5) * (np.pi)**(4/3) * G**(10/3) / c**8

# bin parameter space
log10_mc = np.linspace(7, 11, 50) + np.log10(m_sun)
mc = 10**log10_mc
dlog10_mc = (log10_mc[1] - log10_mc[0])
z = np.logspace(-2, np.log10(4), 51)
dz = np.diff(z)
z = z[:-1]