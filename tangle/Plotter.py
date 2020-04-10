import PyGnuplot as gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle

rc('font',**{'family':'Times New Roman','sans-serif':['Times New Roman'], 'size': 12})
rc('text', usetex=True)

def plot_results(collision_number, numerical_tangle_list, fidelity_S_list, fidelity_T_list,
                 numerical_tangle_list_sim, fidelity_S_list_sim, fidelity_T_list_sim,
                 theta, initial_statevector):

    collision_number = np.arange(collision_number + 1)

    c1 = '#5f15ff'
    c2 = '#ba0c2f'
    c3 = '#0cba97'

    s1 = 8
    s2 = 8.68
    mew1 = 0.8
    mew2 = 0.8
    lw = 0.8

    def general(ax):
        ax.set_xlim(0, 2)
        ax.xaxis.set_ticks(np.linspace(0, 2, 3))
        ax.set_xlabel(r'Collision Number ($n$)')
        ax.grid(linestyle='--', linewidth=0.5, color='lightgray', zorder=4)
        ax.spines['top'].set_linestyle('-')
        ax.spines['right'].set_linestyle('-')
        ax.spines['bottom'].set_linestyle('-')
        ax.spines['left'].set_linestyle('-')
        ax.spines['top'].set_linewidth(lw)
        ax.spines['right'].set_linewidth(lw)
        ax.spines['bottom'].set_linewidth(lw)
        ax.spines['left'].set_linewidth(lw)
        ax.spines['top'].set_color('k')
        ax.spines['right'].set_color('k')
        ax.spines['bottom'].set_color('k')
        ax.spines['left'].set_color('k')

    fig = plt.figure()
    DPI = fig.get_dpi()
    fig.set_size_inches(1920 / float(DPI), 1080 / float(DPI))

    x = np.linspace(0, 10, 100000)
    gamma = np.log(np.cos(theta))
    tau = np.exp(2 * x * gamma)
    F_GHZ = (1 + np.exp(gamma * x)) / 2
    F_T = 1 - 2 * np.abs(initial_statevector[0]) ** 2 * np.abs(initial_statevector[1]) ** 2 * (1 - np.exp(gamma * x))
    f = (1 + np.sqrt(x)) / 2
    g = (1 - np.sqrt(x)) / 2

    ax = plt.figure().add_subplot()
    general(ax)
    ax.set_ylim(0, 1)
    ax.yaxis.set_ticks(np.linspace(0, 1, 11))
    ax.set_ylabel(r'3-Tangle ($\tau_3$)')
    ax.plot(collision_number, numerical_tangle_list, ls='', ms=s1, marker='x', mew=mew1, label=r'ibmq\_5\_yorktown', clip_on=False,
               c=c1, zorder=5)
    ax.plot(collision_number, numerical_tangle_list_sim, ls='', ms=s2, marker='+', mew=mew2, label=r'qasm\_simulator', clip_on=False,
               c=c2, zorder=4)
    ax.plot(x, tau, label=r'Theoretical', c=c3, ls='-', lw=lw, zorder=3)
    ax.legend()

    plt.savefig('tangle.pdf')

    ax = plt.figure().add_subplot()
    general(ax)
    a = np.floor(np.min(fidelity_S_list) * 10)/10
    ax.set_ylim(a, 1)
    ax.yaxis.set_ticks(np.linspace(a, 1, int((1 - a) * 10 + 1)))
    ax.set_ylabel(r'Fidelity ($F$)')
    ax.plot(collision_number, fidelity_S_list, ls='', ms=s1, marker='+', mew=mew1, label=r'ibmq\_5\_yorktown', clip_on=False,
               color=c1, zorder=5)
    ax.plot(collision_number, fidelity_S_list_sim, ls='', ms=s2, marker='x', mew=mew2, label=r'qasm\_simulator', clip_on=False,
               color=c2, zorder=4)
    ax.plot(x, F_GHZ, label=r'Theoretical', color=c3, ls='-', lw=lw, zorder=3)
    ax.legend()

    plt.savefig('GHZ fidelity.pdf')

    ax = plt.figure().add_subplot()
    general(ax)
    a = np.floor(np.min(fidelity_T_list) * 10) / 10
    ax.set_ylim(a, 1)
    ax.yaxis.set_ticks(np.linspace(a, 1, int((1 - a) * 10 + 1)))
    ax.set_ylabel(r'Fidelity ($F$)')
    ax.plot(collision_number, fidelity_T_list, ls='', ms=s1, marker='+', mew=mew1, label=r'ibmq\_5\_yorktown', clip_on=False,
               color=c1, zorder=5)
    ax.plot(collision_number, fidelity_T_list_sim, ls='', ms=s2, marker='x', mew=mew2, label=r'qasm\_simulator', clip_on=False,
               color=c2, zorder=4)
    ax.plot(x, F_T, label=r'Theoretical', color=c3, ls='-', lw=lw, zorder=3)
    ax.legend()

    plt.savefig('teleported fidelity.pdf')

    ax = plt.figure().add_subplot()
    general(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_ticks(np.linspace(0, 1, 11))
    ax.yaxis.set_ticks(np.linspace(0, 1, 11))
    ax.set_xlabel(r'3-Tangle ($\tau_3$)')
    ax.set_ylabel(r'Fidelity ($F$)')
    ax.plot(numerical_tangle_list, fidelity_S_list, ls='', ms=s1, marker='+', mew=mew1, label=r'ibmq\_5\_yorktown', clip_on=False,
               color=c1, zorder=5)
    ax.plot(numerical_tangle_list_sim, fidelity_S_list_sim, ls='', ms=s2, marker='x', mew=mew2, label=r'qasm\_simulator', clip_on=False,
               color=c2, zorder=4)
    ax.plot(x, f, label=r'Theoretical', color=c3, ls='-', lw=lw, zorder=3)
    ax.plot(x, g, color=c3, ls='-', lw=lw, zorder=3)
    ax.legend()

    plt.savefig('fidelity-tangle.pdf', bbox_inches='tight')

data = pickle.load(open('data.json', 'rb'))

plot_results(data['collision_number'], data['numerical_tangle_list'], data['fidelity_S_list'],
             data['fidelity_T_list'], data['numerical_tangle_list_sim'], data['fidelity_S_list_sim'],
             data['fidelity_T_list_sim'], data['theta'], data['initial_statevector'])