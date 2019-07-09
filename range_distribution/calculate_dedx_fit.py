import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit
from argparse import ArgumentParser

def fit_iterative(func, x_val, y_val, eps=1e-14, max_iterations=100, bounds=(-np.inf, np.inf)):
    '''
    Least Squares Fit using sqrt(lambda) as error.
    First iteration is unweighted, then we iterate using the previous
    solution as error estimation until we converge
    '''
    opt, cov = curve_fit(func, x_val, y_val, bounds=bounds)

    # refit with error estimation from previous fit until converged
    delta = np.inf
    idx = 0
    while delta > eps:
        assert idx < max_iterations, 'Fit does not converge'

        # calculate error from last solution
        sigma = func(x_val, *opt)

        old_opt = opt
        opt, cov = curve_fit(
            func, x_val, y_val,
            p0=opt,
            sigma=sigma, absolute_sigma=True,
            bounds=bounds,
        )
        delta = np.sum(np.abs((opt - old_opt)/opt))
        idx += 1
    print("iterations: {}".format(idx))
    return opt#, cov


def test_fitter(energies, dedx_arr, plot_name):
    dedx_sum = np.sum(dedx_arr, axis=1)

    fig = plt.figure(figsize=(8,10))
    gs = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)
    ax1.set_title(r'try to fit $\langle \mathrm{d}E/\mathrm{d}x \rangle = a + b \cdot E$')
    # for jdx in range(len(cs_calc_list)):
    #     ax1.plot(energies, dedx_arr[:,jdx], label=cs_calc_list[jdx].parametrization.name)
    ax1.plot(energies, dedx_sum, label='Sum dEdx')

    def func(x, a, b):
        return a + b*x

    def func2(x, a, b, c, d):
        return a + b*x + c*np.log(x) + d*x*np.log(x)

    fit_ioniz = curve_fit(func, np.log10(energies), np.log10(dedx_arr[:,0]))[0]
    fit_stoch = curve_fit(func, np.log10(energies), np.log10(np.sum(dedx_arr[:,1:], axis=1)))[0]
    fit_eye = [2, 3.5e-6]
    # fit_easy = np.polyfit(energies, dedx_sum, deg=1)[::-1]
    fit_easy = curve_fit(func, energies, dedx_sum, p0=fit_eye, sigma=dedx_sum)[0]
    fit_iter = fit_iterative(func, energies, dedx_sum, bounds=(0,np.inf))
    fit_three = curve_fit(func2, energies, dedx_sum, sigma=dedx_sum)[0]

    fit_lines = [
        #10**(fit_ioniz[0] + fit_ioniz[1]*np.log10(energies)) + 10**(fit_stoch[0] + fit_stoch[1]*np.log10(energies)),
        # fit_eye[0] + fit_eye[1]*energies,
        fit_easy[0] + fit_easy[1]*energies,
        fit_iter[0] + fit_iter[1]*energies,
        fit_three[0] + fit_three[1]*energies + fit_three[2]*np.log(energies) + fit_three[3]*energies*np.log(energies),
    ]
    fit_labels = [
        #r'separate log fit ($10^{a_{i} + b_i\cdot\mathrm{log}_{10}E} + 10^{a_r + b_r\cdot\mathrm{log}_{10}E}$)',
        # 'handfit a={}, b={}'.format(fit_eye[0], fit_eye[1]),
        'simple fit a={:.4g}, b={:.4g}'.format(fit_easy[0], fit_easy[1]),
        'iter. fit a={:.4g}, b={:.4g}'.format(fit_iter[0], fit_iter[1]),
        # 'fit a={:.4g}, b={:.4g}, c={:.4g}, d={:.4g}'.format(fit_three[0], fit_three[1], fit_three[2], fit_three[3])
        r'fit $a + b \cdot E + c\cdot \mathrm{log} E + d \cdot E \cdot \mathrm{log}E$'
    ]
    for idx in range(len(fit_lines)):
        ax1.plot(energies, fit_lines[idx], label=fit_labels[idx])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_ylabel(r'$\langle \mathrm{d}E/\mathrm{d}x \rangle$')
    # ax1.set_xlabel('Energy / MeV')

    ax2.plot(energies, np.ones(len(energies)), label='True')
    for idx in range(len(fit_lines)):
        ax2.plot(energies, fit_lines[idx]/dedx_sum,
                 label=r'$\sum|\Delta|/N=${:.4f}'.format(np.sum(np.abs(fit_lines[idx]/dedx_sum-1))/len(energies)))
    ax2.set_xscale('log')
    ax2.set_xlabel('Energy / MeV')
    ax2.set_ylabel('Fit/Truth')
    ax2.legend()

    plt.subplots_adjust(hspace=.0)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.savefig(plot_name)
    plt.show()

def plot_dedx(energies, dedx_arr, fit_params, plot_name):
    dedx_sum = np.sum(dedx_arr, axis=1)
    xs_labels = ['Ionization', 'Pair Production', 'Bremsstrahlung', 'Inelastic Nuclear Interaction']
    fit_line = fit_params[0] + fit_params[1]*energies

    fig = plt.figure(figsize=(7,7))
    gs = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1], sharex=ax1)
    ax1.set_title(r'fit $\langle \mathrm{d}E/\mathrm{d}x \rangle = a + b \cdot E$')
    for jdx in range(len(xs_labels)):
        ax1.plot(energies, dedx_arr[:,jdx], label=xs_labels[jdx])
    ax1.plot(energies, dedx_sum, label='Sum dEdx')
    ax1.plot(energies, fit_line, label='Fit: a= {:.4g}, b={:.4g}'.format(fit_params[0], fit_params[1]))
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_ylabel(r'$\langle \mathrm{d}E/\mathrm{d}x \rangle$')

    ax2.plot(energies, dedx_sum/dedx_sum, label=r'True')
    ax2.plot(energies, fit_line/dedx_sum, label=r'fit')
    ax2.set_xscale('log')
    ax2.set_xlabel('Energy / MeV')
    ax2.set_ylabel('Fit/Truth')
    ax2.legend()
    plt.subplots_adjust(hspace=.0)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.savefig(plot_name)
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument('-f','--file',
                        type=str,
                        dest='settings_file',
                        default="build/settings.json",
                        help='json file containing the settings')
    args = parser.parse_args()

    with open(args.settings_file) as file:
        settings_dict = json.load(file)

    dedx_arr = np.genfromtxt(settings_dict['dedx_data_filename_all'])
    dedx_sum = np.genfromtxt(settings_dict['dedx_data_filename_sum'])
    tmp_arr = np.array(settings_dict['dedx_energies'])

    def func(x, a, b):
        return a + b*x
    fit_params = curve_fit(func, tmp_arr, dedx_sum, sigma=dedx_sum)[0]
    np.savetxt(settings_dict['dedx_data_fitparams'], fit_params)

    test_fitter(tmp_arr, dedx_arr, settings_dict['dedx_plot_test_fitter'])
    plot_dedx(tmp_arr, dedx_arr, fit_params, settings_dict['dedx_plot'])

if __name__ == '__main__':
    main()
