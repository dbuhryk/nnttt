from gameplay.tournaments.ova import OVATournament
from gameplay.tournaments.fr import FRTournament
from gameplay.players.mlplayer import MLPlayer01
from gameplay.players.rndplayer import RNDPlayer
import logging
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


def main():

    try:
        with h5py.File('statistics.h5', 'r') as h5f:
            statistics = h5f['statistics'][:]  # [?, 25, 2, 4]
            #statistics = np.concatenate((statistics, statistics), 0)
            ys = np.array(range(0, 25, 2))
            statistics_wins = np.subtract(statistics[:, :, 1, 2], statistics[:, :, 0, 2])
            zs = np.array([statistics_wins[:, idx] for idx in ys])
            xs = np.array(range(statistics.shape[0]))
            #delta = [statistics[rec_idx, idx, 1, 2]-statistics[rec_idx, idx, 0, 2]
            #         for rec_idx in statistics.shape[0]
            #         for idx in rg]
            #plt.plot(rg, [, 'o-', label='curPerform')
            xs, ys = np.meshgrid(xs, ys)
            fig = p.figure()
            ax = p3.Axes3D(fig)
            ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.show()

    except IOError:
        logging.warning("Statistic file \"statistics.h5\" is missing or structure is corrupted")



if __name__ == "__main__":
    main()
