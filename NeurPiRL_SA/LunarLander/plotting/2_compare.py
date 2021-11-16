import numpy as np
import matplotlib.pyplot as plt

configs = ['D000', 'D010', 'D100', 'D110', 'E010_D010']

for count, config in enumerate(configs):

    time = np.load('./plot_data/' + config + '_time.npy')
    r_avg = np.load('./plot_data/' + config + '_avg.npy')
    r_std = np.load('./plot_data/' + config + '_std.npy')

    plt.plot(time, r_avg)
    plt.fill_between(time, r_avg - r_std, r_avg + r_std, alpha=0.2)

plt.xlabel('Sequential Runtime (s)')
plt.ylabel('Reward')
plt.title('LunarLander-v2')
plt.ylim([-200, 250])
plt.legend(configs, loc='lower right')
plt.savefig('./plots/COMPARE.png', dpi=1080, bbox_inches="tight")

