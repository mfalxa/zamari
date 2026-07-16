import numpy as np
import matplotlib.pyplot as plt
from fakepta.fake_pta import make_fake_array
import fakepta.correlated_noises as cn
from fakepta.correlated_noises import hd, dipole, monopole, curn, anisotropic
import healpy as hp
from scipy.stats import norminvgauss, invgauss
from numpy.fft import rfft
import pickle
from enterprise_extensions import model_orfs
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
import constants as const


m_sun = 2e30
c = 3e8
G = 6.67e-11
Mpc = 3.086e22
yr = 365.25 * 24 * 3600

def h2_binary(z, mc, f):

    dl = cosmo.luminosity_distance(z).to(u.meter).value
    return (32 * np.pi**(4/3) / 5 / c**8) * ((1 + z) * G * mc)**(10/3) * f**(4/3) / dl**2 / (1 + z)**(4/3)

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
    redshift_term = z**beta * np.exp(-z/z0) * dt_dz

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

def get_h2_iso_and_cgw(freqs, n0_dot=1.26e-3, alpha=0.5, m0=10**9.3 * m_sun, beta=0.5, z0=1., N_brightest=50, seed=1321):

    # set seed
    np.random.seed(seed)

    # pixel map
    nside = 16
    npixels = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npixels))

    # bin parameter space
    df = freqs[0]
    log10_mc = np.linspace(7, 10, 500) + np.log10(m_sun)
    mc = 10**log10_mc
    dlog10_mc = (log10_mc[1] - log10_mc[0])
    z = np.logspace(-2, np.log10(4), 501)
    dz = np.diff(z)
    z = z[:-1]

    # z-M meshgrid
    MM, ZZ = np.meshgrid(log10_mc, z)
    MM = np.ravel(MM)
    ZZ = np.ravel(ZZ)

    h2_f = []
    h2_f_tot = []
    # 0: frequency, 1: redshift, 2: chirp_mass, 3: theta, 4: phi, 5: cosinc, 6: psi, 7: phase0, 8: h
    cgw_params = []
    for freq in freqs:

        # set population grid
        h2_grid = np.array(h2_binary(z[:, None], 10**log10_mc[None, :], freq)) * freq / df
        N_grid = dN_dzdlog10Mdf(z[:, None], mc[None, :], freq, n0_dot, alpha, m0, beta, z0) * dlog10_mc * dz[:, None] * df

        # get poisson grid and sort source contributions from brightest
        N_poisson = np.random.poisson(N_grid)
        h2_grid[N_poisson == 0] = 0.
        isort = np.argsort(np.ravel(h2_grid))[::-1]

        # keep first N_brightest sources
        N_cumul = np.cumsum(np.ravel(N_poisson)[isort])
        ibright = np.arange(len(N_cumul))[N_cumul <= N_brightest]

        # iterate over source redshift and mass and draw other random properties
        zb, mcb = (ZZ[isort])[ibright], (MM[isort])[ibright]
        ns = 0
        for zbk, mcbk in zip(zb, mcb):
            theta = np.arccos(np.random.uniform(-1, 1., size=N_cumul[ns]))
            phi = np.random.uniform(0., 2*np.pi, size=N_cumul[ns])
            for i in range(len(theta)):
                cgw_params.append(np.array([freq, # / (1 + z[idx_zk]),
                                        zbk,
                                        mcbk - np.log10(m_sun),
                                        theta[i],
                                        phi[i],
                                        np.random.uniform(-1, 1),
                                        np.random.uniform(0., np.pi),
                                        np.random.uniform(0., 2*np.pi),
                                        h2_binary(zbk, 10**mcbk, freq)**0.5]))
            ns += 1

        Nh2 = np.ravel(N_poisson * h2_grid)[isort]
        h2_f_tot.append(np.sum(Nh2))
        Nh2[ibright] = 0.
        # print('==============================')
        h2_f.append(np.sum(Nh2))

    return np.array(h2_f), np.array(h2_f_tot), np.array(cgw_params)

def lognormal(x, mu, sig):

    return np.exp(-0.5 * (np.log(x) - mu)**2 / sig**2) / (x * sig * (2*np.pi)**0.5)

# # setup frequencies
# Tobs = 20

# # model
# alpha = 0.5
# m0 = 10**9.3 * m_sun
# beta = 0.5
# z0 = 1.
# n0_dot = 1.26e-3

# # coefficients
# L_f = (96/5) * np.pi**(8/3) * (G/c**3)**(5/3)
# K_f = (32/5) * (np.pi)**(4/3) * G**(10/3) / c**8

def get_psrs(filename, gaussian=True, npsrs=60, Tobs=20, ntoas=200, log10_toaerr=-7., nfreqs=50, n0_dot=1.26e-3, alpha=0.5, m0=10**9.3 * m_sun, beta=0.5, z0=1., seed=1321):

    # get properties
    df = 1 / (Tobs * yr)
    freqs = np.arange(1, nfreqs+1) * df

    # bin parameter space
    log10_mc = np.linspace(7, 11, 500) + np.log10(m_sun)
    mc = 10**log10_mc
    dlog10_mc = (log10_mc[1] - log10_mc[0])
    z = np.logspace(-1, np.log10(4), 501)
    dz = np.diff(z)
    z = z[:-1]

    dm = cosmo.comoving_distance(z).to(u.meter).value
    # df = 1e-9
    # dz = z[1] - z[0]

    # set population grid
    Nsources = 0
    h2c = []
    for freq in freqs:
        N_grid = dN_dzdlog10Mdf(z[:, None], mc[None, :], freq, n0_dot, alpha, m0, beta, z0) * dlog10_mc * dz[:, None] * df
        N_poisson = np.random.poisson(N_grid)
        h2_grid = np.array(h2_binary(z[:, None], 10**log10_mc[None, :], freq)) * freq / df

    h2c, h2ctot, cgw_params = get_h2_iso_and_cgw(freqs, n0_dot, alpha, m0, beta, z0, seed=seed, N_brightest=50)

    # print(len(cgw_params))

    h2c = np.array(h2c)

    # plt.loglog(freqs, h2c**0.5)
    # plt.loglog(freqs, h2ctot**0.5)
    # plt.loglog(freqs, 3e-15 * (freqs*yr)**(-2/3))
    # plt.show()

    ################ inject gwb ##############
    np.random.seed(seed)

    noisedict = {}
    custom_models = {}
    psrs = make_fake_array(npsrs=npsrs, Tobs=Tobs, ntoas=ntoas, gaps=False, backends=['backend.1400'], custom_model={'RN':None, 'DM':None, 'Sv':None}, toaerr=10**log10_toaerr, noisedict={'efac':1, 'log10_tnequad':-12.})
    for psr in psrs:
        psr.make_ideal()
        noisedict.update(psr.noisedict)
        psr.add_white_noise()
        custom_models[psr.name] = psr.custom_model

    # Inject Gaussian GWB 
    if gaussian:

        custom_psd = h2ctot / (12 * np.pi**2 * freqs**3) # / (Tobs * yr)
        cn.add_common_correlated_noise(psrs, orf='hd', spectrum='custom', name='gw', custom_psd=custom_psd, f_psd=freqs)

    else:

        custom_psd = h2c / (12 * np.pi**2 * freqs**3) # / (Tobs * yr)
        cn.add_common_correlated_noise(psrs, orf='hd', spectrum='custom', name='gw', custom_psd=custom_psd, f_psd=freqs)

        # for n, psr in enumerate(psrs[:2]):
        #     plt.plot(psr.toas, psr.residuals, color='C'+str(n))

        # Inject single source foreground
        for i, source in enumerate(cgw_params):

            print(str(round(i * 100 / len(cgw_params), 1)) + '%       ', end='\r')

            log10_mc = source[2]
            mc = 10**log10_mc * const.Tsun
            z = source[1]
            dist = cosmo.luminosity_distance(z).to(u.parsec).value / (1e6)
            dist *= const.Mpc / const.c
            log10_fgw = np.log10(source[0])
            phi = source[4]
            costheta = np.cos(source[3])
            cosinc = source[5]
            psi = source[6]
            phase0 = source[7]
            # log10_h = np.log10(2 * mc ** (5 / 3) * (np.pi * 10**log10_fgw) ** (2 / 3) / dist)
            log10_h = np.log10(source[8])

            psrterm = True
            for psr in psrs:
                psr.add_cgw(costheta, phi, cosinc, log10_mc, log10_fgw, log10_h, phase0, psi, psrterm=psrterm)


    pickle.dump(psrs, open('./pkl/'+filename+'.pkl', 'wb'))
    pkl_to_feathers('./pkl/'+filename, psrs=psrs)

# for n, psr in enumerate(psrs[:2]):
#     plt.plot(psr.toas, psr.residuals, ls='--', color='C'+str(n))

# plt.show()