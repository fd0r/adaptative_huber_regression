from ada_hub.huber_regressor import HuberLoss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


if __name__ == "__main__":
    taus = range(1, 10)
    colors = cm.rainbow(np.linspace(0, 1, len(taus)))
    x = np.linspace(-10,10,10000)
    for tau, color in zip(taus, colors):
        loss = HuberLoss(tau=tau)
        y = loss.loss(x)
        plt.plot(x, y, color=color,label='tau = {}'.format(tau))
        plt.vlines(loss.tau, 0, np.max(y), linestyles="dashed", color=color)
        plt.vlines(-loss.tau, 0, np.max(y), linestyles="dashed", color=color)
    plt.savefig('huber_loss.png')