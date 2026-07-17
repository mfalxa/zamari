import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
import matplotlib.pyplot as plt


jax.config.update('jax_enable_x64', True)


m_sun = 2e30
c = 3e8
G = 6.67e-11
Mpc = 3.086e22
yr = 365.25 * 24 * 3600


class AstroPopulation:

    def __init__(self, freqs, params={'n0_dot':1.26e-3, 'alpha':0.5, 'log10_m0':9.3, 'beta':0.5, 'z0':1.}, d2n_dzdlog10M=None, nz=50, nlogM=50):

        self.freqs = freqs
        self.df = freqs[0]
        self.params = params
        if d2n_dzdlog10M == None:
            self.d2n_dzdlog10M = dn_dzdlog10M
        else:
            self.d2n_dzdlog10M = d2n_dzdlog10M

        # bin parameter space
        self.log10_mc = jnp.asarray(np.linspace(7, 11, nlogM) + np.log10(m_sun))
        self.dlog10_mc = float(self.log10_mc[1] - self.log10_mc[0])

        _z = np.logspace(-2, np.log10(4), nz + 1)
        self.dz = jnp.asarray(np.diff(_z))
        self.z = jnp.asarray(_z[:-1])
        self.dm = jnp.asarray(cosmo.comoving_distance(_z[:-1]).value)

        # cosmology terms
        self.dt_dz = jnp.asarray((1 / ((1 + _z[:-1]) * cosmo.H(_z[:-1]))).to(u.Gyr).value)

    def get_lnpdf(self, h2s, parameters):

        mass_redshift_term = self.d2n_dzdlog10M(self.z[:, None], self.dt_dz[:, None], 10**self.log10_mc[None, :], **parameters)
        lnpdf_saddle = lnpdf_saddlepoint_vmap(h2s, self.freqs, self.df, self.log10_mc, self.dlog10_mc, self.z, self.dz, self.dm, mass_redshift_term, tol=1e-3, N_floor=1e-3)

        return lnpdf_saddle
    
    def get_pdf(self, h2s, parameters):

        return np.exp(self.get_lnpdf(h2s, parameters))


# vanilla functions with phenomenological model
def h2_binary(z, dm, mc, f):

    dl = dm * (1 + z) * Mpc
    return (32 * jnp.pi**(4/3) / 5 / c**8) * ((1 + z) * G * mc)**(10/3) * f**(4/3) / dl**2 / (1 + z)**(4/3)

def df_dt(mc, f):

    return (96/5) * jnp.pi**(8/3) * (G * mc / c**3)**(5/3) * f**(11/3)

def dn_dzdlog10M(z, dt_dz, mc, n0_dot, alpha, log10_m0, beta, z0):

    m0 = m_sun * 10**log10_m0

    # astrophysics terms
    mass_term = (mc / (1e7 * m_sun))**(-alpha) * jnp.exp(-mc/m0)
    redshift_term = z**beta * jnp.exp(-z/z0) * dt_dz

    return n0_dot * mass_term * redshift_term

def dN_dzdlog10Mdf(z, dm, mc, f, mass_redshift_term):

    # astrophysics terms
    frequency_term = df_dt(mc, f)

    return mass_redshift_term * (1 + z) * 4*jnp.pi * (c / Mpc) * dm**2 / frequency_term


def build_population(freq, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term):
    """Per-bin squared strains h2_i and expected counts N_i (grid of get_cgf)."""

    h2_grid = jnp.ravel(h2_binary(z[:, None], dm[:, None], 10**log10_mc[None, :], freq)) * freq / df
    N_grid = jnp.ravel(dN_dzdlog10Mdf(z[:, None], dm[:, None], 10**log10_mc[None, :], freq, mass_redshift_term) * dz[:, None] * dlog10_mc * df)
    
    # plt.imshow(N_grid)
    # plt.show()
    return h2_grid, N_grid

def get_h2_mean_var(freqs, n0_dot, alpha, m0, beta, z0):

    df = freqs[0]
    h2_grid, N_grid  = build_population(1., 1., n0_dot, alpha, m0, beta, z0)

    Nh2 = jnp.sum(N_grid * h2_grid)
    h2c = Nh2 * freqs**(-11/3) * df * (freqs**(4/3) * freqs / df)

    Nh4 = jnp.sum(N_grid * h2_grid**2)
    var_h2c = Nh4 * freqs**(-11/3) * df * (freqs**(4/3) * freqs / df)**2
    return h2c, var_h2c


# functions for saddlepoint approximation
def _outer(t, h2):
    """np.multiply.outer replacement (jnp.multiply has no .outer)."""

    return jnp.asarray(t)[..., None] * h2

def cgf(t, h2, N):

    return jnp.expm1(_outer(t, h2)) @ N

def cgf_prime(t, h2, N):

    return jnp.exp(logsumexp(_outer(t, h2), b=N * h2, axis=-1))

def cgf_prime2(t, h2, N):

    return jnp.exp(logsumexp(_outer(t, h2), b=N * h2**2, axis=-1))

def _solve_saddlepoint(y, a, N, s0=0., tol=1e-10, max_iter=100):

    Na, Na2 = N * a, N * a**2
    keep = N > 0
    log_y = jnp.log(y)

    def body(state):
        s, _, i = state
        t = s * a
        t_max = jnp.max(jnp.where(keep, t, -jnp.inf))
        w = jnp.where(keep, jnp.exp(t - t_max), 0.)
        Kp, K2 = Na @ w, Na2 @ w        # K'(s), K''(s), both times exp(-t_max)
        g = t_max + jnp.log(Kp) - log_y
        return s - g * Kp / K2, g, i + 1    # g' = K''/K'

    def cond(state):
        _, g, i = state
        return (jnp.abs(g) > tol) & (i < max_iter)

    s, _, _ = lax.while_loop(cond, body, (jnp.asarray(s0, dtype=float), jnp.inf, 0))
    return s

def logpdf_saddlepoint(x, h2, N, N_floor=1e-4, tol=1e-10):

    keep = N > N_floor
    h2 = jnp.where(keep, h2, 0.)
    N = jnp.where(keep, N, 0.)

    x = jnp.atleast_1d(jnp.asarray(x, dtype=float))
    mu = N @ h2
    a = h2 / mu
    y = x / mu

    # solve in ascending order of y so each root warm-starts the next one
    # (s(y) is increasing)
    order = jnp.argsort(y)

    def step(s, y_j):
        s = _solve_saddlepoint(y_j, a, N, s0=s, tol=tol)
        K = jnp.expm1(s * a) @ N
        log_K2 = logsumexp(s * a, b=N * a**2)
        logp_j = K - s * y_j - 0.5 * (jnp.log(2 * jnp.pi) + log_K2)
        return s, logp_j

    _, logp_sorted = lax.scan(step, jnp.asarray(-1000000., dtype=float), y[order])
    logp = jnp.zeros(len(y)).at[order].set(logp_sorted) - jnp.log(mu)

    if x.shape[0] < 2:
        return logp

    # if peak is unresolved, return a Dirac delta with width 1/dx and mean given by the CGF
    sigma = jnp.sqrt(N @ h2**2)
    dx = jnp.abs(jnp.gradient(x))
    i_star = jnp.argmin(jnp.abs(x - mu))
    delta = jnp.where(jnp.arange(x.shape[0]) == i_star,
                      -jnp.log(dx[i_star]), -jnp.inf)

    return jnp.where(sigma < dx[i_star], delta, logp)

def pdf_saddlepoint(x, freq, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol=1e-3, N_floor=1e-3):

    h2_grid, N_grid = build_population(freq, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term)
    return jnp.exp(logpdf_saddlepoint(x, h2_grid, N_grid, N_floor=N_floor, tol=tol))

def lnpdf_saddlepoint(x, freq, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol=1e-3, N_floor=1e-3):

    # build_population(freq, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term)
    h2_grid, N_grid = build_population(freq, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term)
    return logpdf_saddlepoint(x, h2_grid, N_grid, tol=tol, N_floor=N_floor)


lnpdf_saddlepoint_jit = jax.jit(lnpdf_saddlepoint)
pdf_saddlepoint_jit = jax.jit(pdf_saddlepoint)

def _lnpdf_over_freqs(x, freqs, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol=1e-3, N_floor=1e-3):
    """vmap of lnpdf_saddlepoint over a 1d array of frequencies.

    x is either (n_x,), the same grid for every frequency, or
    (n_freq, n_x), one grid per frequency; output is (n_freq, n_x).
    """

    x = jnp.asarray(x, dtype=float)
    fun = lambda x_, f_: lnpdf_saddlepoint(x_, f_, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol, N_floor)
    return jax.vmap(fun, in_axes=(0 if x.ndim == 2 else None, 0))(x, freqs)


@jax.jit
def lnpdf_saddlepoint_vmap(x, freqs, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol, N_floor):
    """(n_freq, n_x) array of log p(x | freq); jitted vmap over freqs."""

    return _lnpdf_over_freqs(x, freqs, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol, N_floor)

@jax.jit
def pdf_saddlepoint_vmap(x, freqs, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol=1e-3, N_floor=1e-3):
    """(n_freq, n_x) array of p(x | freq); jitted vmap over freqs."""

    return jnp.exp(_lnpdf_over_freqs(x, freqs, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol, N_floor))

def lnpdf_saddlepoint_pmap(x, freqs, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol=1e-3, N_floor=1e-3):

    n_dev = jax.local_device_count()
    freqs = jnp.asarray(freqs, dtype=float)
    x = jnp.asarray(x, dtype=float)
    n = freqs.shape[0]
    n_pad = (-n) % n_dev
    freqs_p = jnp.pad(freqs, (0, n_pad), mode='edge').reshape(n_dev, -1)

    if x.ndim == 2:
        x_p = jnp.pad(x, ((0, n_pad), (0, 0)), mode='edge').reshape(n_dev, -1, x.shape[-1])
        fun = lambda x_, f_: _lnpdf_over_freqs(x_, f_, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol, N_floor)
        out = jax.pmap(fun)(x_p, freqs_p)
    else:
        fun = lambda f_: _lnpdf_over_freqs(x, f_, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol, N_floor)
        out = jax.pmap(fun)(freqs_p)

    return out.reshape(-1, out.shape[-1])[:n]

def pdf_saddlepoint_pmap(x, freqs, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol=1e-3, N_floor=1e-4):

    return jnp.exp(lnpdf_saddlepoint_pmap(x, freqs, df, log10_mc, dlog10_mc, z, dz, dm, mass_redshift_term, tol, N_floor))


# draw h2 values from Poisson draw of the population and sum to get h2c
def sample_compound(h2, N, n_real, rng=0, chunk=5000):

    key = jax.random.PRNGKey(rng if rng is not None else 0)
    out = []
    for i in range(0, n_real, chunk):
        key, sub = jax.random.split(key)
        n = min(chunk, n_real - i)
        out.append(jax.random.poisson(sub, N, shape=(n, len(N))) @ h2)
    return jnp.concatenate(out)