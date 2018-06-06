import random
import logging
from gameplay.game import Game
from gameplay.tournaments.generic import Tournament
import numpy as np


class RNDTournament(Tournament):
    """
    Random tournament
    Every turn a random player is picked to make all moves on all boards
    """
    active_size = None
    count = None
    size = None
    players = []
    games = []
    statuses = []  # game statuses, for fast computing
    active_games = []  # [game index]
    weights = []
    debug_print_boards = False

    def __init__(self, players, count, active_size=None, weights=None, debug_print_boards=False):
        """
        :param players: array of Player objects
        :param count: Total number of games to play
        :param active_size: Total number of games to play simultaneously
        """
        self.players = []
        self.games = []
        self.active_games = []
        self.count = count
        self.active_size = count if active_size is None else active_size
        self.size = len(players)
        self.players = players
        self.debug_print_boards = debug_print_boards

        if weights is None:
            self.weights = np.ones(shape=(len(self.players)))
        else:
            self.weights = weights

        norm = np.linalg.norm(self.weights, ord=1)
        if norm == 0:
            norm = np.finfo(self.weights.dtype).eps
        self.weights = self.weights/norm

    def is_complete(self):
        """
        :return:
        """
        if len(self.games) >= self.count and len(self.active_games) == 0:
            return True
        return False

    def update_active(self):
        """
        :return:
        """

        while len(self.games) < self.count and len(self.active_games) < self.active_size:
            game = Game(5, 4)
            self.games.append(game)
            self.statuses.append(None)
            self.active_games.append(len(self.games) - 1)

    def step_active(self):

        # batch_count = 16
        # #active_games_batches = [self.active_games[x:x+batchsize] for x in range(0, len(self.active_games), batchsize)]
        # active_games_batches = np.array_split(self.active_games, batch_count)
        # with Parallel(n_jobs=batch_count) as parallel:
        #     results = parallel(delayed(play_parallel_game)(active_game_batch, self.statuses, self.games, self.players)
        #                        for active_game_batch in active_games_batches)

        active_games = [self.games[rec] for rec in self.active_games]
        active_positions = [game.position.serialize_out(asplayer=game.next_player()) for game in active_games]

        player_idx = np.random.choice(len(self.players), size=1, p=self.weights)[0]
        #player_idx = np.random.choice(len(self.players), size=1)[0]
        player = self.players[player_idx]
        logging.debug("Player %d picked (class: %s)" % (player_idx, player.__class__.__name__))

        if len(active_positions) > 0:
            moves = player.make_game_moves(positions=active_positions)
            moves = moves + 1  # xys = moves[:, 0] + 1, moves[:, 1] + 1

            [game.makemove(player=game.next_player(), position=xy) for game, xy in zip(active_games, moves)]

    def cleanup_active(self):
        self.active_games = [rec for rec in self.active_games if self.games[rec].game_status()[0] is None]

    def run_tournament(self):
        """
        :return: returns tournament dynamics per step
        """
        self.update_active()
        counter = 0
        statistics = []
        while not self.is_complete():
            statuses = [self.games[rec].game_status()[0] for rec in self.active_games]
            statistic = (statuses.count(None), statuses.count(0), statuses.count(1), statuses.count(2))
            self.step_active()

            statuses = [self.games[rec].game_status()[0] for rec in self.active_games]
            statistic_after = (statuses.count(None), statuses.count(0), statuses.count(1), statuses.count(2))

            logging.info("Step: %d, Games total: %d, active: %d (None=%d[-%d], DRAW=%d[+%d], P1=%d[+%d], P2=%d[+%d])" %
                         (counter, len(self.games), statuses.count(None),
                          statistic_after[0], statistic[0] - statistic_after[0],
                          statistic_after[1], statistic_after[1] - statistic[1],
                          statistic_after[2], statistic_after[2] - statistic[2],
                          statistic_after[3], statistic_after[3] - statistic[3]
                          ))
            statistics.append((statistic, statistic_after))
            counter += 1
            if self.debug_print_boards:
                [self.games[rec].printboard() for rec in self.active_games]

            self.cleanup_active()
            self.update_active()
        logging.info("Tournament ended")
        return statistics
