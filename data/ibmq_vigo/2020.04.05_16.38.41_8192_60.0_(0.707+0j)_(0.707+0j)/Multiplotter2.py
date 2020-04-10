import PyGnuplot as gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle

rc('font',**{'family':'Times New Roman','sans-serif':['Times New Roman'], 'size': 22})
rc('text', usetex=True)

def plot(collision_number, concurrence_list, fidelity_S_list, fidelity_T_list,
                 concurrence_list_sim, fidelity_S_list_sim, fidelity_T_list_sim,
                 theta, initial_statevector, fig, labels):

    collision_number = np.arange(collision_number + 1)

    x = np.linspace(0, 10, 100000)
    gamma = np.log(np.cos(theta))
    tau = np.exp(x * gamma)
    F_GHZ = (1 + np.exp(gamma * x)) / 2
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

    min_S_fidelity = np.min(fidelity_S_list)
    min_T_fidelity = np.min(fidelity_T_list)

    for i in range(1, 5):
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

        if i == 1:
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 1)
            ax.xaxis.set_ticks(np.linspace(0, 2, 3))
            ax.yaxis.set_ticks(np.linspace(0, 1, 6))
            ax.set_xlabel(r'Collision Number ($n$)')
            ax.set_ylabel(r'Concurrence ($C$)')

            ax.plot(collision_number, concurrence_list, ls='', ms=s1, mew=mew1, marker='x', label=r'ibmq\_vigo', clip_on=False, color=c1, zorder=5)
            ax.plot(collision_number, concurrence_list_sim, ls='', ms=s2, mew=mew2, marker='+', label=r'qasm\_simulator', clip_on=False, color=c2, zorder=4)
            ax.plot(x, tau, label=r'Theoretical', color=c3, ls='-', lw=lw, zorder=3)

        if i == 2:
            ax.set_xlim(0, 2)
            ax.set_ylim(0.4, 1)
            ax.xaxis.set_ticks(np.linspace(0, 2, 3))
            ax.yaxis.set_ticks(np.linspace(0.4, 1, 4))
            ax.set_xlabel(r'Collision Number ($n$)')
            ax.set_ylabel(r'Fidelity ($F$)')
            ax.plot(collision_number, fidelity_S_list, ls='', ms=s1, mew=mew1, marker='x', label=r'ibmq\_vigo',
                       clip_on=False, color=c1, zorder=5)
            ax.plot(collision_number, fidelity_S_list_sim, ls='', ms=s2, mew=mew2, marker='+', label=r'qasm\_simulator',
                       clip_on=False, color=c2, zorder=4)
            ax.plot(x, F_GHZ, label=r'Theoretical', color=c3, ls='-', lw=lw, zorder=3)

        if i == 4:
            ax.set_xlim(0, 2)
            ax.set_ylim(0.4, 1)
            ax.xaxis.set_ticks(np.linspace(0, 2, 3))
            ax.yaxis.set_ticks(np.linspace(0.4, 1, 4))
            ax.set_xlabel(r'Collision Number ($n$)')
            ax.set_ylabel(r'Fidelity ($F$)')
            ax.plot(collision_number, fidelity_T_list, ls='', ms=s1, mew=mew1, marker='x', label=r'ibmq\_vigo',
                       clip_on=False, color=c1, zorder=5)
            ax.plot(collision_number, fidelity_T_list_sim, ls='', ms=s2, mew=mew2, marker='+', label=r'qasm\_simulator',
                       clip_on=False, color=c2, zorder=4)
            ax.plot(x, F_T, label=r'Theoretical', color=c3, ls='-', lw=lw, zorder=3)

        if i == 3:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.xaxis.set_ticks(np.linspace(0, 1, 6))
            ax.yaxis.set_ticks(np.linspace(0, 1, 6))
            ax.set_xlabel(r'Concurrence ($C$)')
            ax.set_ylabel(r'Fidelity ($F$)')
            ax.plot(concurrence_list, fidelity_S_list, ls='', ms=s1, mew=mew1, marker='x', label=r'ibmq\_vigo',
                       clip_on=False, color=c1, zorder=5)
            ax.plot(concurrence_list_sim, fidelity_S_list_sim, ls='', ms=s2, mew=mew2, marker='+', label=r'qasm\_simulator',
                       clip_on=False, color=c2, zorder=4)
            ax.plot(x, f, label=r'Theoretical', color=c3, ls='-', lw=lw, zorder=3)
            ax.plot(x, g, color=c3, ls='-', lw=lw, zorder=3)

        if i == 2:
            ax.legend(loc='upper right')

plots = ('concurrence', 'Bell_fidelity', 'teleported_fidelity', 'fidelity-concurrence')
labels = (r'\textbf{(a)}', r'\textbf{(b)}', r'\textbf{(c)}', r'\textbf{(d)}')
fig = plt.figure()
DPI = fig.get_dpi()
fig.set_size_inches(1920 / float(DPI), 1080 / float(DPI))
fig.subplots_adjust(wspace=0.18)
fig.subplots_adjust(hspace=0.28)

data = pickle.load(open("data.json", "rb"))

plot(data['collision_number'], data['concurrence_list'], data['fidelity_S_list'],
            data['fidelity_T_list'], data['concurrence_list_sim'], data['fidelity_S_list_sim'],
            data['fidelity_T_list_sim'], data['theta'], data['initial_statevector'], fig, labels)

plt.savefig('multiplot.pdf', bbox_inches='tight')