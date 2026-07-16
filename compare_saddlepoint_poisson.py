import numpy as np
import matplotlib.pyplot as plt
from sim_astro_data import get_h2_iso_and_cgw
from log_normal_h2 import get_lnorms, get_h2_mean_var
from pdf_compound_jax import pdf_saddlepoint_vmap, build_population, sample_compound, lnpdf_saddlepoint_vmap
from scipy.special import logsumexp


m_sun = 2e30
c = 3e8
G = 6.67e-11
Mpc = 3.086e22
yr = 365.25 * 24 * 3600

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

def plot_violin(x, pdf_x, position, width=1., color='C0'):

    p = 0.5 * width * pdf_x / np.amax(pdf_x)
    plt.fill_betweenx(x, position + p, position - p, alpha=0.2, color=color)


# model
alpha = 0.5
log10_m0 = 9.3
m0 = 10**log10_m0 * m_sun
beta = 0.5
z0 = 1.
n0_dot = 1.26e-3

Tspan = 20 * yr
freqs = np.arange(1, 22) / Tspan
conversion = freqs[0] / (12.0 * np.pi**2 * freqs**3)

h2c, var_h2c = get_h2_mean_var(freqs, n0_dot, alpha, m0, beta, z0)
ratio = var_h2c**0.5 / h2c

h2_f, h2_f_tot, params = get_h2_iso_and_cgw(freqs, n0_dot, alpha, m0, beta, z0, N_brightest=0, seed=1321)

h2 = np.logspace(-36, -26, 5001)
dh2 = np.diff(h2)
h2 = h2[:-1]

# get lognorm
lnorms = get_lnorms(h2, freqs, n0_dot, alpha, m0, beta, z0)

# get saddlepoint
pdfs = lnpdf_saddlepoint_vmap(h2, freqs, 1/Tspan, n0_dot, alpha, log10_m0, beta, z0, tol=1e-3, N_floor=10**(-3.))
# pdfs6 = pdf_saddlepoint_vmap(h2, freqs, 1/Tspan, n0_dot, alpha, log10_m0, beta, z0, tol=1e-10, N_floor=1e-10)

print('norm', np.exp(logsumexp(pdfs, b=dh2, axis=1)))

bins = np.logspace(np.log10(h2.min()), np.log10(h2.max()), 100)

print(freqs[19] * yr)
print(freqs[2] * yr)

for n in range(len(freqs)):

    print(ratio[n])

    h2_grid, N_grid = build_population(freqs[n], df=1/(20*yr), n0_dot=n0_dot, alpha=alpha, log10_m0=log10_m0, beta=beta, z0=z0)
    xc = sample_compound(h2_grid, N_grid,  n_real=5000, rng=0)

    print('norm', np.exp(logsumexp(pdfs[n], b=dh2)))

    # plt.hist(xc, bins=bins, density=True, label='Compound poisson draws')

    # one jitted, vmapped call replaces the loop over frequencies
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.title.set_text('Lin scale')
    ax1.hist(xc, bins=bins, density=True, label='Monte carlo')
    ax1.plot(h2, np.exp(pdfs[n]), label='Saddelpoint')
    ax1.set_xscale('log')
    # ax1.set_xlim(5e-31, 6e-29)
    ax1.set_xlabel(r'$h^2_c$')
    ax1.set_ylabel('pdf')
    ax1.legend(loc='upper right')

    ax2.title.set_text('Log scale')
    ax2.hist(xc, bins=bins, density=True, label='Monte carlo')
    ax2.plot(h2, np.exp(pdfs[n]), lw=3., label='Saddelpoint')
    ax2.set_xscale('log')
    # ax2.set_xlim(5e-31, 6e-29)
    # ax2.set_ylim((1e25, 1e30))
    ax2.set_xlabel(r'$h^2_c$')
    ax2.legend(loc='upper right')
    ax2.set_yscale('log')

    fig.suptitle('$h^2_c$ distribution at f=1/(1yr) Hz')
    # plt.savefig('plots/poisson_mc_vs_saddlepoint.png', bbox_inches='tight')
    # plt.savefig('plots/poisson_mc_vs_saddlepoint.pdf', bbox_inches='tight')

    plt.show()
