import PyGnuplot as gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle

rc('font',**{'family':'Times New Roman','sans-serif':['Times New Roman'], 'size': 22})
rc('text', usetex=True)

def plot_concurrence(collision_number, concurrence_list, fidelity_S_list, fidelity_T_list,
                 concurrence_list_sim, fidelity_S_list_sim, fidelity_T_list_sim,
                 theta, initial_statevector, fig, i, labels, j, min_S_fidelity, min_T_fidelity):
    x = np.linspace(0, 10, 100000)
    gamma = np.log(np.cos(theta))
    tau = np.exp(x * gamma)
    F_S = (1 + np.exp(gamma * x)) / 2
    F_T = 1 - 2 * np.abs(initial_statevector[0]) ** 2 * np.abs(initial_statevector[1]) ** 2 * (1 - np.exp(gamma * x))
    f = (1 + x) / 2
    g = (1 - x) / 2

    c1 = '#5912ff'
    c2 = '#ba0c2f'
    c3 = '#0cba97'

    s1 = 16
    s2 = 17.36
    mew1 = 1.2
    mew2 = 1.2
    lw = 1.2

    ax = fig.add_subplot(2, 2, i)
    ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, color='lightgray', zorder=4)
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
    ax.text(-0.07, 1.1, labels[i - 1], transform=ax.transAxes, fontsize=22, fontweight='bold', va='top', ha='right')

    if j == 'concurrence':
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.xaxis.set_ticks(np.linspace(0, 10, 6))
        ax.yaxis.set_ticks(np.linspace(0, 1, 6))
        ax.set_xlabel(r'Collision Number ($n$)')
        ax.set_ylabel(r'Concurrence ($C$)')

        ax.plot(collision_number, concurrence_list, ls='', ms=s1, mew=mew1, marker='x', label=r'ibmq\_16\_melbourne', clip_on=False, color=c1, zorder=5)
        ax.plot(collision_number, concurrence_list_sim, ls='', ms=s2, mew=mew2, marker='+', label=r'qasm\_simulator', clip_on=False, color=c2, zorder=4)
        ax.plot(x, tau, label=r'Theoretical ', color=c3, ls='-', lw=lw, zorder=3)

    if j == 'bell_fidelity':
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.xaxis.set_ticks(np.linspace(0, 10, 6))
        ax.yaxis.set_ticks(np.linspace(0, 1, 6))
        ax.set_xlabel(r'Collision Number ($n$)')
        ax.set_ylabel(r'Fidelity ($F$)')
        ax.plot(collision_number, fidelity_S_list, ls='', ms=s1, mew=mew1, marker='x', label=r'ibmq\_16\_melbourne',
                   clip_on=False, color=c1, zorder=5)
        ax.plot(collision_number, fidelity_S_list_sim, ls='', ms=s2, mew=mew2, marker='+', label=r'qasm\_simulator',
                   clip_on=False, color=c2, zorder=4)
        ax.plot(x, F_S, label=r'Theoretical ', color=c3, ls='-', lw=lw, zorder=3)

    if j == 'teleported_fidelity':
        ax.set_xlim(0, 10)
        a = np.floor(min_T_fidelity * 10) / 10
        ax.set_ylim(a, 1)
        ax.xaxis.set_ticks(np.linspace(0, 10, 6))
        ax.yaxis.set_ticks(np.linspace(a, 1, int((1 - a) * 5 + 1)))
        ax.set_xlabel(r'Collision Number ($n$)')
        ax.set_ylabel(r'Fidelity ($F$)')
        ax.plot(collision_number, fidelity_T_list, ls='', ms=s1, mew=mew1, marker='x', label=r'ibmq\_16\_melbourne',
                   clip_on=False, color=c1, zorder=5)
        ax.plot(collision_number, fidelity_T_list_sim, ls='', ms=s2, mew=mew2, marker='+', label=r'qasm\_simulator',
                   clip_on=False, color=c2, zorder=4)
        ax.plot(x, F_T, label=r'Theoretical ', color=c3, ls='-', lw=lw, zorder=3)

    if j == 'fidelity-concurrence':
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_ticks(np.linspace(0, 1, 6))
        ax.yaxis.set_ticks(np.linspace(0, 1, 6))
        ax.set_xlabel(r'Concurrence ($C$)')
        ax.set_ylabel(r'Fidelity ($F$)')
        ax.plot(concurrence_list, fidelity_S_list, ls='', ms=s1, mew=mew1, marker='x', label=r'ibmq\_16\_melbourne',
                   clip_on=False, color=c1, zorder=5)
        ax.plot(concurrence_list_sim, fidelity_S_list_sim, ls='', ms=s2, mew=mew2, marker='+', label=r'qasm\_simulator',
                   clip_on=False, color=c2, zorder=4)
        ax.plot(x, f, label=r'Theoretical', color=c3, ls='-', lw=lw, zorder=3)
        ax.plot(x, g, color=c3, ls='-', lw=lw, zorder=3)

    if i == 2:
        if j == 'fidelity-concurrence':
            ax.legend(loc='center right')
        else:
            ax.legend(loc='upper right')

f_g = []
f_t = []
for i in range(1, 5):
    f_g.append(pickle.load(open("plots/data{}.json".format(i), "rb"))['fidelity_S_list'])
    f_t.append(pickle.load(open("plots/data{}.json".format(i), "rb"))['fidelity_T_list'])
min_S_fidelity = np.min(f_g)
min_T_fidelity = np.min(f_t)

plots = ('concurrence', 'bell_fidelity', 'teleported_fidelity', 'fidelity-concurrence')
labels = (r'\textbf{(a)}', r'\textbf{(b)}', r'\textbf{(c)}', r'\textbf{(d)}')
for j in plots:
    fig = plt.figure()
    DPI = fig.get_dpi()
    fig.set_size_inches(1920 / float(DPI), 1080 / float(DPI))
    fig.subplots_adjust(wspace=0.18)
    fig.subplots_adjust(hspace=0.28)
    for i in range(1, 5):
        data = pickle.load(open("plots/data{}.json".format(i), "rb"))

        plot_concurrence(data['collision_number'], data['concurrence_list'], data['fidelity_S_list'],
                    data['fidelity_T_list'], data['concurrence_list_sim'], data['fidelity_S_list_sim'],
                    data['fidelity_T_list_sim'], data['theta'], data['initial_statevector'], fig, i, labels, j,
                    min_S_fidelity, min_T_fidelity)

    plt.savefig('plots/{}.pdf'.format(j), bbox_inches='tight')