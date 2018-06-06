from gameplay.tournaments.ova import OVATournament
from gameplay.players.mlplayer import *
from gameplay.players.rndplayer import *
from gameplay.game import Game
import logging
import h5py
import numpy as np
import time
import argparse
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def main():
    np.random.seed(seed=int(time.time()))

    parser = argparse.ArgumentParser(description='Tournament generator.')
    parser.add_argument('--class', type=str, action='append', dest='clazz', help='ML Class', required=True)

    args = parser.parse_args()

    clazzs = [getattr(sys.modules[__name__], clazz) for clazz in args.clazz]
    players = [clazz() for clazz in clazzs]
    for player in players:
        if isinstance(player, MLPlayer):
            player.load_model()

    game = Game(5, 4)
    status = None

    while status is None:
        next_player = game.next_player()

        position = game.position.serialize_out(asplayer=next_player)
        positions = [position]  # [?, x, x, 3] shape is required

        moves = players[next_player - 1].make_game_moves(positions=positions)
        xy = moves[0][0] + 1, moves[0][1] + 1

        game.makemove(player=next_player, position=xy)
        game.printboard()
        status, _ = game.game_status()

if __name__ == "__main__":
    main()
