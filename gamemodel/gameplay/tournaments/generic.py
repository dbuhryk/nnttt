import logging
from abc import ABCMeta
from joblib import Parallel, delayed
import numpy as np


class Tournament(metaclass=ABCMeta):
    size = 2
    players = []
    games = []
    statuses = []  # game statuses, for fast computing
    # status: None - game continues, otherwise wining player (1 or 2), 0 is a draw
    active_games = []  # [(game index, player 1 index, player 2 index)]
    history = []  # [(game index, player 1 index, player 2 index, status)]

    def run_tournament(self):
        """
        :return: returns tournament dynamics per step
        """
        return

    def collect_statistics(self, players=None):
        """
        :param players: if None collect statistic for all games, if a list then cumulative statistic for players
        :return: [?, 4], (points '-1' or '0' or '1', player 1 idx, player 2 idx, game length)
        """
        return
