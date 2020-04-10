import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt

rc('font',**{'family':'Times New Roman','sans-serif':['Times New Roman'], 'size': 12})
rc('text', usetex=True)

lw = 0.8

fig = plt.figure()
DPI = fig.get_dpi()
fig.set_size_inches(1920 / float(DPI), 1080 / float(DPI))

ax = plt.figure().add_subplot()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.xaxis.set_ticks(np.linspace(0, 1, 11))
ax.yaxis.set_ticks(np.linspace(0, 1, 11))
ax.set_xlabel(r'$p$')
ax.set_ylabel(r'3-Tangle ($\tau_3$)')
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

p = np.linspace(0, 1, 100000)
tau = (2 * p - 1) ** 2

ax.plot(p, tau, ls='-', c='#ba0c2f', zorder=3)
plt.savefig('p_tangle.pdf')