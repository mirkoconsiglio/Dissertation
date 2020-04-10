import PyGnuplot as gp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import rc
import pickle

rc('font',**{'family':'Times New Roman','sans-serif':['Times New Roman'], 'size': 12})
rc('text', usetex=True)

def plot_results(collision_number, concurrence_list, fidelity_S_list, fidelity_T_list,
                 concurrence_list_sim, fidelity_S_list_sim, fidelity_T_list_sim,
                 theta, initial_statevector):

    c1 = '#5f15ff'
    c2 = '#ba0c2f'
    c3 = '#0cba97'

    s1 = 8
    s2 = 8.68
    mew1 = 0.8
    mew2 = 0.8
    lw = 0.8

    def general(ax):
        ax.set_xlim(0, 10)
        ax.xaxis.set_ticks(np.linspace(0, 10, 11))
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
    C = np.exp(x * gamma)
    F_P = (1 + np.exp(gamma * x)) / 2
    F_T = 1 - 2 * np.abs(initial_statevector[0]) ** 2 * np.abs(initial_statevector[1]) ** 2 * (1 - np.exp(gamma * x))
    f = (1 + x) / 2
    g = (1 - x) / 2

    def h(x, A, B):
        return ((A - 1) * np.exp(-B * x) + 1) / A

    ax = plt.figure().add_subplot()
    general(ax)
    ax.set_ylim(0, 1)
    ax.yaxis.set_ticks(np.linspace(0, 1, 11))
    ax.set_ylabel(r'Concurrence ($C$)')
    ax.plot(collision_number, concurrence_list, ls='', ms=s1, marker='x', mew=mew1, label=r'ibmq\_16\_melbourne', clip_on=False,
               c=c1, zorder=5)
    ax.plot(collision_number, concurrence_list_sim, ls='', ms=s2, marker='+', mew=mew2, label=r'qasm\_simulator', clip_on=False,
               c=c2, zorder=4)
    ax.plot(x, C, label=r'Theoretical', c=c3, ls='-', lw=lw, zorder=3)
    ax.legend()

    plt.savefig('concurrence.pdf')

    ax = plt.figure().add_subplot()
    general(ax)
    a = np.floor(np.min(fidelity_S_list) * 10)/10
    ax.set_ylim(a, 1)
    ax.yaxis.set_ticks(np.linspace(a, 1, int((1 - a) * 10 + 1)))
    ax.set_ylabel(r'Fidelity ($F$)')
    ax.plot(collision_number, fidelity_S_list, ls='', ms=s1, marker='x', mew=mew1, label=r'ibmq\_16\_melbourne', clip_on=False,
               color=c1, zorder=5)
    ax.plot(collision_number, fidelity_S_list_sim, ls='', ms=s2, marker='+', mew=mew2, label=r'qasm\_simulator', clip_on=False,
               color=c2, zorder=4)
    ax.plot(x, F_P, label=r'Theoretical', color=c3, ls='-', lw=lw, zorder=3)
    p = curve_fit(h, collision_number, fidelity_S_list, bounds=[[2, 0], [np.inf, np.inf]])[0]
    print(p)
    y = h(x, p[0], p[1])
    ax.plot(x, y, color=c3, ls='-', lw=lw, zorder=3)
    ax.legend()

    plt.savefig('bell_fidelity.pdf')

    ax = plt.figure().add_subplot()
    general(ax)
    a = np.floor(np.min(fidelity_T_list) * 10) / 10
    ax.set_ylim(a, 1)
    ax.yaxis.set_ticks(np.linspace(a, 1, int((1 - a) * 10 + 1)))
    ax.set_ylabel(r'Fidelity ($F$)')
    ax.plot(collision_number, fidelity_T_list, ls='', ms=s1, marker='x', mew=mew1, label=r'ibmq\_16\_melbourne', clip_on=False,
               color=c1, zorder=5)
    ax.plot(collision_number, fidelity_T_list_sim, ls='', ms=s2, marker='+', mew=mew2, label=r'qasm\_simulator', clip_on=False,
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
    ax.set_xlabel(r'Concurrence ($C$)')
    ax.set_ylabel(r'Fidelity ($F$)')
    ax.plot(concurrence_list, fidelity_S_list, ls='', ms=s1, marker='x', mew=mew1, label=r'ibmq\_16\_melbourne', clip_on=False,
               color=c1, zorder=5)
    ax.plot(concurrence_list_sim, fidelity_S_list_sim, ls='', ms=s2, marker='+', mew=mew2, label=r'qasm\_simulator', clip_on=False,
               color=c2, zorder=4)
    ax.plot(x, f, label=r'Theoretical', color=c3, ls='-', lw=lw, zorder=3)
    ax.plot(x, g, color=c3, ls='-', lw=lw, zorder=3)
    ax.legend()

    plt.savefig('fidelity-concurrence.pdf', bbox_inches='tight')

data = pickle.load(open('data.json', 'rb'))

plot_results(data['collision_number'], data['concurrence_list'], data['fidelity_S_list'],
             data['fidelity_T_list'], data['concurrence_list_sim'], data['fidelity_S_list_sim'],
             data['fidelity_T_list_sim'], data['theta'], data['initial_statevector'])