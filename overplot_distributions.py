import numpy as np
import matplotlib.pyplot as plt
from sim_astro_data import get_h2_iso_and_cgw
from log_normal_h2 import get_lnorms, get_h2_mean_var
from pdf_compound_jax import pdf_saddlepoint_vmap, build_population


m_sun = 2e30
c = 3e8
G = 6.67e-11
Mpc = 3.086e22
yr = 365.25 * 24 * 3600

import numpy as np

def sample_from_pdf(x, pdf, size=1, rng=None):

    x = np.asarray(x)
    pdf = np.asarray(pdf)

    if rng is None:
        rng = np.random.default_rng()

    # Normalize the PDF (works for uneven spacing)
    pdf = pdf / np.trapz(pdf, x)

    # Compute the CDF
    dx = np.diff(x)
    cdf = np.concatenate([[0],
                          np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dx)])
    cdf /= cdf[-1]

    # Draw uniform random numbers
    u = rng.random(size)

    # Inverse transform sampling
    samples = np.interp(u, cdf, x)

    return samples

def plot_violin(x, pdf_x, position, width=1., color='C0', pmin=0.01, label=None):

    mask = pdf_x > pmin * np.amax(pdf_x)
    p = 0.5 * width * pdf_x / np.amax(pdf_x)
    plt.fill_betweenx(x[mask], position + p[mask], position - p[mask], alpha=0.2, color=color, label=label)


# model
alpha = 0.5
log10_m0 = 8.
m0 = 10**log10_m0 * m_sun
beta = 0.5
z0 = 1.
n0_dot = 4.15e-2


Tspan = 20 * yr
freqs = np.arange(1, 20) / Tspan

h2_f, h2_f_tot, params = get_h2_iso_and_cgw(freqs, n0_dot, alpha, m0, beta, z0, N_brightest=0, seed=1321)

# plt.loglog(freqs, h2_f_tot**0.5)
# plt.loglog(freqs, h2_f**0.5)
# plt.loglog(freqs, 3e-15 * (freqs * yr)**(-2/3), ls='--', c='k')
# plt.show()

h2 = np.logspace(-36, -26, 1001)
dh2 = np.diff(h2)
h2 = h2[:-1]

# get lognorm
lnorms = get_lnorms(h2, freqs, n0_dot, alpha, m0, beta, z0)

# get saddlepoint
pdfs = pdf_saddlepoint_vmap(h2, freqs, 1/Tspan, n0_dot, alpha, log10_m0, beta, z0, tol=1e-10, N_floor=1e-3)
# pdfs6 = pdf_saddlepoint_vmap(h2, freqs, 1/Tspan, n0_dot, alpha, log10_m0, beta, z0, tol=1e-10, N_floor=1e-10)

# plt.loglog(freqs, h2_f_tot)
# plt.loglog(freqs, (3e-15 * (freqs * yr)**(-2/3))**2, ls='--', c='k')
# plt.show()

# plt.violinplot(X, freqs, widths=freqs[0], showextrema=False)
h2c, var_h2c = get_h2_mean_var(freqs, n0_dot, alpha, m0, beta, z0)
for n, pdf in enumerate(pdfs):
    if n == 0:
        plot_violin(h2**0.5, pdf, freqs[n] * 1e9, width=freqs[0] * 1e9, label='Saddlepoint approximation')
    else:
        plot_violin(h2**0.5, pdf, freqs[n] * 1e9, width=freqs[0] * 1e9)
plt.plot(freqs * 1e9, h2_f_tot**0.5, c='k', alpha=0.3, label='Single realisation')
plt.plot(freqs * 1e9, h2c**0.5, ls='--', c='k', label='Ensemble average')
plt.plot(freqs * 1e9, 3e-15 * (freqs * yr)**(-2/3))
# plt.ylim((1e-31, 1e-26))
plt.yscale('log')
plt.ylabel('$h_c$')
plt.xlabel('Frequency [nHz]')
plt.legend(loc='upper right')
# plt.savefig('plots/overplot_distrib.pdf', bbox_inches='tight')
plt.show()

# for n in range(len(freqs)):
#     print('FREQ', n)
#     print('theo mean', h2c[n])
#     m4 = np.sum(pdfs[n] * h2 * dh2)
#     m6 = np.sum(pdfs6[n] * h2 * dh2)
#     print('mean floor 4', m4)
#     print('mean floor 6', m6)
#     print('-----------')
#     print('theo var', var_h2c[n])
#     print('var floor 4', np.sum(pdfs[n] * (h2 - m4)**2 * dh2))
#     print('var floor 6', np.sum(pdfs6[n] * (h2 - m6)**2 * dh2))
#     print('================')

# for n in range(len(freqs)):
#     # plt.semilogx(h2, lnorms[n])
#     plt.loglog(h2, pdfs[n], ls='--')
#     plt.loglog(h2, pdfs6[n], ls='-')
#     plt.axvline(h2_f[n], c='k')
#     plt.show()

# for ln in lnorms:
#     plt.semilogx(h2, ln)

# for pdf in pdfs:
#     plt.semilogx(h2, pdf, ls='--')

# for h in h2_f:
#     plt.axvline(h, c='k')


# plt.show()