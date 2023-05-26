# code mostley based on "Complex Momentum for Optimization in Games"
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib as mpl
from tqdm import tqdm


def get_spectral_radius(g, eta, beta, is_eg=False, is_eg_and_cm=False,
                        is_og=False):
    if is_eg:
        eig_vals, eig_vecs = np.linalg.eig(g)
        eg_eig_vals = 1 - eta * (eig_vals * (1 - beta.real * eig_vals))
        eig_norms = np.real(np.sqrt(eg_eig_vals * np.conjugate(eg_eig_vals)))
        max_norm = np.max(eig_norms)
        return max_norm
    elif is_eg_and_cm:
        eig_vals, eig_vecs = np.linalg.eig(g)
        eg_eig_vals = eig_vals * (1 - eta * eig_vals)
        g = np.diag(eg_eig_vals)
    elif is_og:
        n = len(g)
        R_t_min_1 = np.concatenate((np.eye(n) * 0.0, np.eye(n)))
        R_t = np.concatenate((beta.real * g,
                              np.eye(n) - 2.0 * eta.real * g))
        R = np.concatenate([R_t_min_1, R_t], axis=1)

        eig_vals, eig_vecs = np.linalg.eig(R)

        eig_norms = np.real(np.sqrt(eig_vals * np.conjugate(eig_vals)))

        max_norm = np.max(eig_norms)
        return max_norm
    n = len(g)
    R_re = np.concatenate((beta.real * np.eye(n),
                           -beta.imag * np.eye(n), -g))
    R_im = np.concatenate([(beta.imag) * np.eye(n),
                           beta.real * np.eye(n),
                           0 * np.eye(n)])

    R_param = np.concatenate([
        (eta * beta).real * np.eye(n),
        -(eta * beta).imag * np.eye(n),
        np.eye(n) - eta.real * g])
    R = np.concatenate([R_re, R_im, R_param], axis=1)
    eig_vals, eig_vecs = np.linalg.eig(R)

    eig_norms = np.real(np.sqrt(eig_vals * np.conjugate(eig_vals)))

    max_norm = np.max(eig_norms)
    return max_norm



def get_spectral_radius_LEAD(nabla_v, M, D, P, eta, beta, alpha):
    n = nabla_v.shape[0] // 2

    j11 = (1 + beta) * np.eye(2 * n) - eta * nabla_v - alpha * np.dot(D, P)
    j12 = - beta * np.eye(2 * n) + alpha * nabla_v + alpha * np.dot(D, P)
    j21 = np.eye(2 * n)
    j22 = np.zeros((2 * n, 2 * n))

    row_1 = np.concatenate([j11, j12], 1)
    row_2 = np.concatenate([j21, j22], 1)
    j = np.concatenate([row_1, row_2])

    eig_vals, eig_vecs = np.linalg.eig(j)
    eig_norms = np.real(np.sqrt(eig_vals * np.conjugate(eig_vals)))
    max_norm = np.max(eig_norms)
    return max_norm


np.random.seed(0)
n_game = 14
sample_constant_adv = 4.0
rand_vals_1 = np.linspace(
    1.0 / sample_constant_adv, sample_constant_adv, num=n_game // 2)
matrix_A = np.diag(rand_vals_1)
adv_component = np.zeros((n_game, n_game))
adv_component[:n_game // 2, n_game // 2:] = -matrix_A
adv_component[n_game // 2:, :n_game // 2] = matrix_A

sample_constant_coop = sample_constant_adv
matrix_B = np.linspace(
    1.0 / sample_constant_coop, sample_constant_coop, num=n_game)
matrix_B[:n_game // 2] = np.linspace(
    1.0 / sample_constant_coop, sample_constant_coop, num=n_game // 2)
matrix_B[-n_game // 2:] = matrix_B[:n_game // 2]

coop_component = np.diag(matrix_B)

num_hist_bins = 50
dist_flag = False

g = None
plt.figure(figsize=(8, 5))
ax2 = plt.subplot(1, 1, 1)
cmap = mpl.cm.get_cmap('viridis')
# beta is the momentum
# eta is the learning rate that we do the for-loop over
for beta_phase, beta_phase_string in (
    [(None, 'LEAD'),
     (0.0, 'PM'),
     (np.pi / 2.0, r'CM (pi/2)'),
     (np.pi, r'NM'),
     (None, r'EG'),
     (None, r'OG'),
     (None, r'GDA')]):

    mixture_scores = []
    mixture_space = np.linspace(0, 1, num=20)
    alpha_space = np.linspace(-0.5, 0.5, num=num_hist_bins)
    for mixture in tqdm(mixture_space):
        np.random.seed(0)
        gamma = np.zeros(n_game)
        gamma[:n_game // 2] = np.random.uniform(
            0, mixture, size=n_game // 2)
        gamma[0] = 0
        gamma[n_game // 2 - 1] = mixture

        gamma[-n_game // 2:] = gamma[:n_game // 2]
        g = gamma * adv_component + (1 - gamma) * coop_component
        beta_mags, eta_mags, rates = [], [], []
        if beta_phase is not None:
            beta_mag_space = np.linspace(0.0, 1.0, num=num_hist_bins)
            for beta_mag in beta_mag_space:
                beta = beta_mag * np.exp(1j * beta_phase)
                eta_space = np.linspace(.0, 1.0, num=num_hist_bins)
                for eta in eta_space:
                    is_eg_and_cm = False
                    spectral_radius = get_spectral_radius(g=g, eta=eta, beta=beta, is_eg_and_cm=is_eg_and_cm)
                    beta_mags += [beta_mag]
                    eta_mags += [eta]
                    if is_eg_and_cm:
                        rates += [np.sqrt(spectral_radius)]
                    else:
                        rates += [spectral_radius]
        else:

            if beta_phase_string == 'LEAD':
                n = adv_component.shape[0]
                gamma = np.diag(gamma)
                M = (np.dot(np.eye(n) - gamma, coop_component) +
                     np.dot(coop_component.T, (np.eye(n) - gamma).T)) / 2

                row_1 = np.concatenate([M, np.dot(gamma, adv_component)], 1)
                row_2 = np.concatenate([-np.dot(adv_component.T, gamma.T), M], 1)
                nabla_v = np.concatenate([row_1, row_2])

                row_1 = np.concatenate([np.dot(gamma, adv_component), np.zeros((n, n))], 1)
                row_2 = np.concatenate([np.zeros((n, n)), -np.dot(adv_component.T, gamma.T)], 1)
                D = np.concatenate([row_1, row_2])

                row_1 = np.concatenate([np.zeros((n, n)), np.eye(n)], 1)
                row_2 = np.concatenate([np.eye(n), np.zeros((n, n))], 1)
                P = np.concatenate([row_1, row_2])

            eta_space = np.linspace(0.0, 1.75, num=num_hist_bins)
            alphas = []
            for eta in eta_space:
                beta_mag_space = eta_space
                for beta_mag in beta_mag_space:
                    beta = beta_mag
                    if beta_phase_string == r'EG':
                        spectral_radius = get_spectral_radius(g=g, eta=eta, beta=beta, is_eg=True)
                        beta_mags += [beta_mag]
                        eta_mags += [eta]
                        rates += [np.sqrt(spectral_radius)]
                    elif beta_phase_string == r'OG':
                        spectral_radius = get_spectral_radius(g=g, eta=eta, beta=beta, is_og=True)
                        beta_mags += [beta_mag]
                        eta_mags += [eta]
                        rates += [spectral_radius]
                    elif beta_phase_string == 'LEAD':
                        for alpha in alpha_space:
                            spectral_radius = get_spectral_radius_LEAD(nabla_v, M, D, P, eta, beta, alpha)
                            beta_mags += [beta_mag]
                            eta_mags += [eta]
                            alphas += [alpha]
                            rates += [spectral_radius]
                    else:
                        spectral_radius = get_spectral_radius(g=g, eta=eta, beta=0.0, is_eg=False)
                        beta_mags += [0.0]
                        eta_mags += [eta]
                        rates += [spectral_radius]

        best_rate = np.min(rates)
        mixture_scores += [best_rate]
        best_index = np.argmin(rates)
        print('Method:', beta_phase_string)
        if len(alphas) > best_index:
            print('best alpha:', alphas[best_index])
        if len(eta_mags) > best_index:
            print('best eta:', eta_mags[best_index])
        if len(beta_mags) > best_index:
            print('best beta:', beta_mags[best_index])

    label = None
    if beta_phase_string is not None:
        label = beta_phase_string

    c = None
    marker = None
    markersize = None
    if beta_phase is not None:
        c = cmap(beta_phase / np.pi)

    if beta_phase_string == 'PM':
        c = 'tab:purple'
        marker = 'x'
        markersize = 12
    elif beta_phase_string == 'LEAD':
        c = 'tab:orange'
        marker = 'x'
        markersize = 12
    elif beta_phase_string == r'NM':
        c = 'tab:green'
        marker = 'x'
        markersize = 12
    elif beta_phase_string == r'EG':
        c = 'tab:red'
        marker = 'o'
        label = beta_phase_string
    elif beta_phase_string == r'OG':
        c = 'tab:gray'
        marker = 'o'
        label = beta_phase_string
    elif beta_phase_string == r'GDA':
        c = 'tab:blue'
        marker = '+'
        markersize = 12
        label = beta_phase_string
    ax2.set_xlim([0.0, 1.0])
    ax2.semilogy(mixture_space, mixture_scores,
                 label=label, c=c, alpha=0.75,
                 marker=marker, markersize=markersize,
                 linewidth=3, zorder=10)
    ax2.legend(framealpha=1.0)
    ax2.set_xlabel('Max adversarialness γmax')
    ax2.set_ylabel('# grad. eval. to converge')
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    ax2.grid(color="k", linestyle="--", linewidth=0.5, alpha=0.3)

    if dist_flag:
        ax3 = ax2.twinx()
        cur_xs = np.linspace(0.0, 1.0, 10)
        cur_ys = norm.pdf(cur_xs, 0.25, 0.27)
        ax3.fill_between(cur_xs, cur_ys, alpha=0.25, zorder=0)
        ax3.set_ylim([0.0, 1.5])
        dist_flag = False

    plt.savefig('phase_compare.pdf', transparent=True, bbox_inches='tight')
