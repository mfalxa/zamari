import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from inspect import signature
from functools import partial
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u


jax.config.update('jax_enable_x64', True)


m_sun = 1.989e30
c = 2.998e8
G = 6.674e-11
Mpc = 3.086e22
yr = 365.25 * 24 * 3600


class AstroPopulation:

    def __init__(self, freqs, params=['n0_dot', 'alpha', 'log10_m0', 'beta', 'z0'], d2n_dzdlog10M=None, dfdt=None, nz=50, nlogM=50):

        self.freqs = jnp.array(freqs)
        self.df = freqs[0]
        self.params = params
        if d2n_dzdlog10M == None:
            self.d2n_dzdlog10M = dn_dzdlog10M
        else:
            self.d2n_dzdlog10M = d2n_dzdlog10M
        if dfdt == None:
            self.dfdt = df_dt
        else:
            self.dfdt = dfdt

        # bin parameter space
        self.log10_mc = jnp.asarray(np.linspace(7, 11, nlogM) + np.log10(m_sun))
        self.dlog10_mc = float(self.log10_mc[1] - self.log10_mc[0])

        _z = np.logspace(-2, np.log10(4), nz + 1)
        self.dz = jnp.asarray(np.diff(_z))
        self.z = jnp.asarray(_z[:-1])
        self.dm = jnp.asarray(cosmo.comoving_distance(_z[:-1]).value)

        # cosmology terms
        self.dt_dz = jnp.asarray((1 / ((1 + _z[:-1]) * cosmo.H(_z[:-1]))).to(u.Gyr).value)

        self.mr_keys = [p for p in self.params if p in signature(self.d2n_dzdlog10M).parameters]
        self.f_keys = [p for p in self.params if p in signature(self.dfdt).parameters]

    @partial(jax.jit, static_argnums=0)
    def get_massredshift_term(self, z, dt_dz, mc, mr_params):

        return self.d2n_dzdlog10M(z, dt_dz, mc, **mr_params)

    @partial(jax.jit, static_argnums=0)
    def get_frequency_term(self, mc, f, df_params):

        return self.dfdt(mc, f, **df_params)

    @partial(jax.jit, static_argnums=0)
    def dN_dzdlog10Mdf(self, z, dt_dz, dm, mc, f, parameters):

        # map parameters
        mr_params = {k: parameters[k] for k in self.mr_keys}
        df_params = {k: parameters[k] for k in self.f_keys}
    
        # astrophysics terms
        mass_redshift_term = self.get_massredshift_term(z, dt_dz, mc, mr_params)
        frequency_term = self.get_frequency_term(mc, f, df_params)

        return mass_redshift_term * (1 + z) * 4*jnp.pi * (c / Mpc) * dm**2 / frequency_term

    @partial(jax.jit, static_argnums=0)
    def h2_binary(self, z, dm, mc, f):

        dl = dm * (1 + z) * Mpc
        return (32 * jnp.pi**(4/3) / 5 / c**8) * ((1 + z) * G * mc)**(10/3) * f**(4/3) / dl**2

    @partial(jax.jit, static_argnums=0)
    def build_population(self, freq, df, log10_mc, dlog10_mc, z, dz, dt_dz, dm, parameters):

        h2_grid = jnp.ravel(self.h2_binary(z[:, None], dm[:, None], 10**log10_mc[None, :], freq)) * freq / df
        N_grid = jnp.ravel(self.dN_dzdlog10Mdf(z[:, None], dt_dz[:, None], dm[:, None], 10**log10_mc[None, :], freq * (1 + z[:, None]), parameters)
                           * dz[:, None] * dlog10_mc * df * (1 + z[:, None]))
        
        return h2_grid, N_grid

    @partial(jax.jit, static_argnums=0)
    def lnpdf(self, x, freq, parameters, tol=1e-5, N_floor=1e-3):

        h2_grid, N_grid = self.build_population(freq, self.df, self.log10_mc, self.dlog10_mc, self.z, self.dz, self.dt_dz, self.dm, parameters)
        return logpdf_saddlepoint(x, h2_grid, N_grid, tol=tol, N_floor=N_floor)

    # @partial(jax.jit, static_argnums=0)
    def _lnpdf_over_freqs(self, x, freqs, parameters, tol=1e-5, N_floor=1e-3):

        x = jnp.asarray(x, dtype=float)
        fun = lambda x_, f_: self.lnpdf(x_, f_, parameters, tol, N_floor)
        return jax.vmap(fun, in_axes=(0 if x.ndim == 2 else None, 0))(x, freqs)

    def lnpdf_vmap(self, x, parameters, tol=1e-5, N_floor=1e-3):

        return self._lnpdf_over_freqs(x, self.freqs, parameters, tol, N_floor)

    def lnpdf_pmap(self, x, parameters, tol=1e-5, N_floor=1e-3):

        n_dev = jax.local_device_count()
        freqs = jnp.asarray(self.freqs, dtype=float)
        x = jnp.asarray(x, dtype=float)
        n = freqs.shape[0]
        n_pad = (-n) % n_dev
        freqs_p = jnp.pad(freqs, (0, n_pad), mode='edge').reshape(n_dev, -1)

        if x.ndim == 2:
            x_p = jnp.pad(x, ((0, n_pad), (0, 0)), mode='edge').reshape(n_dev, -1, x.shape[-1])
            fun = lambda x_, f_: self._lnpdf_over_freqs(x_, f_, parameters, tol, N_floor)
            out = jax.pmap(fun)(x_p, freqs_p)
        else:
            fun = lambda f_: self._lnpdf_over_freqs(x, f_, parameters, tol, N_floor)
            out = jax.pmap(fun)(freqs_p)

        return out.reshape(-1, out.shape[-1])[:n]




# GW driven binaries
def df_dt(mc, f):

    return (96/5) * jnp.pi**(8/3) * (G * mc / c**3)**(5/3) * f**(11/3)

# Phenomenological mass redshift function
def dn_dzdlog10M(z, dt_dz, mc, log10_n0_dot, alpha, log10_m0, beta, z0):

    m0 = m_sun * 10**log10_m0

    # astrophysics terms
    mass_term = (mc / (1e7 * m_sun))**(-alpha) * jnp.exp(-mc/m0)
    redshift_term = (1 + z)**beta * jnp.exp(-z/z0) * dt_dz

    return 10**log10_n0_dot * mass_term * redshift_term




# functions for saddlepoint approximation
@jax.jit
def _outer(t, h2):
    """np.multiply.outer replacement (jnp.multiply has no .outer)."""

    return jnp.asarray(t)[..., None] * h2

@jax.jit
def cgf(t, h2, N):

    return jnp.expm1(_outer(t, h2)) @ N

@jax.jit
def cgf_prime(t, h2, N):

    return jnp.exp(logsumexp(_outer(t, h2), b=N * h2, axis=-1))

@jax.jit
def cgf_prime2(t, h2, N):

    return jnp.exp(logsumexp(_outer(t, h2), b=N * h2**2, axis=-1))

@jax.jit
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

@jax.jit
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




# draw h2 values from Poisson draw of the population and sum to get h2c
@jax.jit
def sample_compound(h2, N, n_real, rng=0, chunk=5000):

    key = jax.random.PRNGKey(rng if rng is not None else 0)
    out = []
    for i in range(0, n_real, chunk):
        key, sub = jax.random.split(key)
        n = min(chunk, n_real - i)
        out.append(jax.random.poisson(sub, N, shape=(n, len(N))) @ h2)
    return jnp.concatenate(out)