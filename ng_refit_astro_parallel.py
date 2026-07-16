from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from scipy.stats import gaussian_kde
import pickle

# from your module:
# from saddlepoint import _lnpdf_over_freqs, m_sun
from pdf_compound_jax import _lnpdf_over_freqs, get_h2_mean_var
from ng_refit_astro import LikelihoodFS as LinFS



class LikelihoodFS:

    # prior box: log10_n0_dot, alpha, log10_m0 [Msun], beta, z0
    _lo = jnp.array([-5.0, 0.0, 7.0, 0.1, 0.1])
    _hi = jnp.array([-1.0, 3.0, 10.0, 3.0, 3.0])

    def __init__(self, chains, Tspan, ngrid=5000, tol=1e-5, N_floor=1e-3):
        self.Tspan = Tspan
        self.tol = float(tol)
        self.N_floor = float(N_floor)

        # ---------- host-side, one-off precomputation (numpy/scipy is fine here)
        nfreqs = chains.shape[1]
        self.nfreqs = nfreqs

        kdes = [gaussian_kde(chains[:, n]) for n in range(nfreqs)]
        x = np.linspace(chains.min(), chains.max(), ngrid)          # log10_rho grid
        freqs = (1.0 + np.arange(nfreqs)) / Tspan
        self.df = float(freqs[0])

        conversion = freqs[0] / (12.0 * np.pi**2 * freqs**3)
        p_rhos = np.array([np.log(k(x) + 1e-200) for k in kdes])    # (nfreqs, ngrid)

        # change of variable log10_rho -> v = rho^2
        self.v = 10.0 ** (2.0 * x)                                       # (ngrid,)
        h2 = self.v[None, :] / conversion[:, None]                       # (nfreqs, ngrid)
        self.dv_row = (x[1] - x[0]) * 2.0 * np.log(10.0) * v             # measure dv
        dv = np.tile(self.dv_row, (nfreqs, 1))                           # (nfreqs, ngrid)
        # NOTE: this fixes the original bug where `v` was tiled instead of `dv`.

        # ---------- shard everything across local devices, once
        n_dev = jax.local_device_count()
        n_pad = (-nfreqs) % n_dev

        def shard(a):
            pad = ((0, n_pad),) + ((0, 0),) * (a.ndim - 1)
            a = np.pad(a, pad, mode="edge")
            return jnp.asarray(a.reshape(n_dev, -1, *a.shape[1:]))

        self.freqs_s  = shard(freqs)     # (n_dev, chunk)
        self.h2_s     = shard(h2)        # (n_dev, chunk, ngrid)
        self.p_rhos_s = shard(p_rhos)    # (n_dev, chunk, ngrid)
        self.dv_s     = shard(dv)        # (n_dev, chunk, ngrid)

        mask = np.pad(np.ones(nfreqs), (0, n_pad))                  # 0 on padding
        self.mask_s = jnp.asarray(mask.reshape(n_dev, -1))          # (n_dev, chunk)

        # ---------- the single compiled entry point
        # params is broadcast (in_axes=None); everything else is sharded on axis 0.
        self._lnlike_pmapped = jax.pmap(
            self._chunk_lnlike,
            axis_name="dev",
            in_axes=(None, 0, 0, 0, 0, 0),
        )

    # ------------------------------------------------------------------ core
    def _chunk_lnlike(self, params, freqs, h2, p_rhos, dv, mask):
        """Runs on ONE device, on its chunk of frequencies. Fully traced."""
        n0_dot   = 10.0 ** params[0]
        alpha    = params[1]
        log10_m0 = params[2]        # already in solar masses: log10(10**x * m_sun / m_sun) = x
        beta     = params[3]
        z0       = params[4]

        # (chunk, ngrid): log saddlepoint pdf on the h2 grid, per frequency
        ig = _lnpdf_over_freqs(h2, freqs, self.df, n0_dot, alpha,
                               log10_m0, beta, z0, self.tol, self.N_floor)

        conversion = self.df / (12.0 * np.pi**2 * freqs**3)
        log_norm = logsumexp(ig, b=self.dv_row[None, :] / conversion[:, None], axis=1)
        ig -= log_norm[:, None]

        # h2m, h2v = get_h2_mean_var(freqs, n0_dot, alpha, log10_m0, beta, z0)

        # per-frequency marginal:  log ∫ dv  p(rho) * pdf(v)
        per_freq = logsumexp(p_rhos + ig, b=dv, axis=-1)            # (chunk,)

        # sum over this device's real (unpadded) frequencies, then over devices
        local = jnp.sum(per_freq * mask)
        return jax.lax.psum(local, axis_name="dev")                  # scalar, replicated

    # ------------------------------------------------------------------ API
    def get_lnlike(self, x):
        x = jnp.asarray(x, dtype=float)
        # result is replicated on every device; take the first copy
        return self._lnlike_pmapped(x, self.freqs_s, self.h2_s,
                                    self.p_rhos_s, self.dv_s, self.mask_s)[0]

    def get_lnprior(self, x):
        """Traceable box prior (works under jit/grad, no Python branching)."""
        x = jnp.asarray(x, dtype=float)
        inside = jnp.all((x > self._lo) & (x < self._hi))
        return jnp.where(inside, -jnp.sum(jnp.log(self._hi - self._lo)), -jnp.inf)

    def get_lnprob(self, x):
        """Log-posterior for host-driven samplers (emcee, PTMCMC...).

        Short-circuits in Python so the expensive pmap call is skipped
        when the point is outside the prior box.
        """
        lp = self.get_lnprior(x)
        if not np.isfinite(lp):
            return -np.inf
        return float(lp + self.get_lnlike(x))

    def get_lnprob_traced(self, x):
        """Fully traceable log-posterior for JAX-native samplers / grad.

        Both branches are evaluated under tracing, so the likelihood must
        return finite values for any in-box params (it will; out-of-box
        points are masked to a safe value before the pmap call).
        """
        lp = self.get_lnprior(x)
        # clamp params into the box so the likelihood never sees garbage,
        # then mask the result with -inf via the prior
        x_safe = jnp.clip(jnp.asarray(x, dtype=float), self._lo + 1e-9, self._hi - 1e-9)
        return jnp.where(jnp.isfinite(lp), lp + self.get_lnlike(x_safe), -jnp.inf)
    

dirname = 'Nbright_50_astro_model_data'

# dirname = 'astro_psrs'
chains = pickle.load(open('chains/'+dirname+'_chains.pkl', 'rb'))
Tspan = 20 * 365.25 * 24 * 3600
like_fs = LikelihoodFS(chains, Tspan)

print(f"JAX devices: {jax.local_device_count()}", flush=True)  # should print 10

# injected values
log10_n0_dot = np.log10(1.26e-3)
alpha = 0.5
log10_m0 = 9.3
beta = 0.5
z0 = 1.
x_inj = np.array([log10_n0_dot, alpha, log10_m0, beta, z0])

print(like_fs.get_lnlike(x_inj))
