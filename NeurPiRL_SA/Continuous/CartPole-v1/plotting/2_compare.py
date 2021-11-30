import numpy as np
import matplotlib.pyplot as plt


configs = ['2x4', '64x64', 'BaseDSL']

for count, config in enumerate(configs):

    time = np.load('./plot_data/' + config + '_time.npy')
    r_avg = np.load('./plot_data/' + config + '_avg.npy')
    r_std = np.load('./plot_data/' + config + '_std.npy')

    plt.plot(time, r_avg)
    plt.fill_between(time, r_avg - r_std, r_avg + r_std, alpha=0.2)

plt.xlabel('Sequential Runtime (s)')
plt.ylabel('Reward')
plt.title('CartPole-v1 with BO')
plt.ylim([0, 500])
plt.legend(configs, loc='lower right')
#
plt.savefig('./plots/COMPARE.png', dpi=1080, bbox_inches="tight")

