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


m_sun = 2e30
c = 3e8
G = 6.67e-11
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

# X, Y = np.meshgrid(log10_mc - np.log10(m_sun), z)
# d2n_dzdM = dn_dzdlog10M(z[:, None], 10**log10_mc[None, :], n0_dot, alpha, m0, beta, z0)
# plt.contourf(X, Y, dn_dzdlog10M(z[:, None], 10**log10_mc[None, :], n0_dot, alpha, m0, beta, z0) * dz[:, None] * dlog10_mc)
# plt.ylim((1e-3, 3))
# plt.xlim((7, 9))
# plt.ylabel('z')
# plt.xlabel('log10 M')
# plt.savefig('d2ndzdM.png', bbox_inches='tight')
# plt.show()
# print(kaixo)
# plt.loglog(10**log10_mc / m_sun, np.sum(d2n_dzdM * dz[:, None], axis=0))
# plt.axvline(m0 / m_sun, ls='--')
# plt.xlabel(r'$\log_{10} M / M_{\odot} $')
# plt.ylim((1e-7, 1))
# plt.xlim((1e7, 1e11))
# plt.savefig('dndz.png', bbox_inches='tight')
# plt.show()
# plt.loglog(z, np.sum(d2n_dzdM * dlog10_mc, axis=1))
# plt.axvline(z0, ls='--')
# plt.xlabel('z')
# plt.ylim((1e-7, 1))
# plt.xlim((1e7, 1e11))
# plt.savefig('dndz.png', bbox_inches='tight')
# plt.show()
# print(kaixo)


# dm = cosmo.comoving_distance(z).to(u.meter).value
# # df = 1e-9
# # dz = z[1] - z[0]

# # set population grid
# h2c = []
# var_h2c = []
# for freq in freqs:
#     N_grid = dN_dzdlog10Mdf(z[:, None], mc[None, :], freq, n0_dot, alpha, m0, beta, z0) * dlog10_mc * dz[:, None] * df
#     h2_grid = np.array(h2_binary(z[:, None], 10**log10_mc[None, :], freq)) * freq / df
#     h2c.append(np.sum(N_grid * h2_grid))
#     var_h2c.append(np.sum(N_grid * h2_grid**2))

# h2c = np.array(h2c)
# var_h2c = np.array(var_h2c)
# plt.loglog(freqs, h2c)
# plt.loglog(freqs, (h2c + var_h2c**0.5))

# h2c, var_h2c = get_h2_mean_var(freqs, n0_dot, alpha, m0, beta, z0)
# plt.loglog(freqs, h2c, ls='--', lw=3.)
# plt.loglog(freqs, (h2c + var_h2c**0.5), ls='--', lw=3.)

# plt.show()
# dataset = []
# for i in range(len(freqs)):
#     mean = h2c[i]
#     var = var_h2c[i]

#     # lognorm params
#     mu = np.log(mean) - 0.5 * np.log(1 + (var/mean**2))
#     sig2 = np.log(1 + (var / mean**2))

#     print(mu, sig2)
#     lnorm2 = lognorm.rvs(s=sig2**0.5, scale=np.exp(mu), size=10000)
#     dataset.append(lnorm2)

# print(get_lnorm_mu_sig2(h2c, var_h2c))

# dataset = np.log10(np.array(dataset)**0.5)
# plt.violinplot(dataset.T, positions=freqs, widths=0.1 * freqs, showextrema=False)
# plt.plot(freqs, np.log10(h2c**0.5), lw=3., ls='--', c='C0', label='Ensemble average')
# plt.plot(freqs, dataset.T[3006], alpha=0.2, color='k', label='Single realisation')
# # plt.plot(freqs, dataset.T[1612], alpha=0.2, color='k')
# # plt.plot(freqs, dataset.T[2408], alpha=0.2, color='k')
# # plt.plot(freqs, dataset.T[2109], alpha=0.2, color='k')
# plt.axvline(1/yr, ls='--', c='k')
# plt.xscale('log')
# # plt.yscale('log')
# plt.ylim((-15, -13.))
# plt.ylabel('log10 hc')
# plt.xlabel('Frequency [Hz]')
# plt.legend(loc='lower left')
# plt.savefig('log10_M0_8.png', bbox_inches='tight')
# plt.show()
