import random
import logging
from gameplay.game import Game
from gameplay.tournaments.generic import Tournament


def play_parallel_game(active_game_batch, statuses, games, players):
    for rec in active_game_batch:
        gameindex = rec[0]
        if statuses[gameindex] is None:
            game = games[gameindex]
            players = [players[rec[1]], players[rec[2]]]

            next_player = game.next_player()
            position = game.position.serialize_out(asplayer=next_player)
            positions = [position]  # [?, x, x, 3] shape is required
            moves = players[next_player - 1].make_game_moves(positions=positions)
            xy = moves[0][0] + 1, moves[0][1] + 1
            game.makemove(player=next_player, position=xy)
            status, _ = game.game_status()
            #logging.debug("Game #%d (%d vs %d), turn: %d, outcome: %s" % (gameindex, rec[1], rec[2], game.move, str(status)))
            if status is not None:
                statuses[gameindex] = status


class OVATournament:
    """
    One Versus All Tournament
    Matches player index 0 vs all players
    """
    active_size = None
    count = None
    size = 2
    players = []
    games = []
    statuses = []  # game statuses, for fast computing
    # status: None - game continues, otherwise wining player (1 or 2), 0 is a draw
    active_games = []  # [(game index, player 1 index, player 2 index)]
    history = []  # [(game index, player 1 index, player 2 index, status)]

    def __init__(self, players, count, active_size=None):
        self.players = []
        self.games = []
        self.statuses = []
        self.active_games = []
        self.history = []
        self.count = count
        self.active_size = count if active_size is None else active_size
        self.size = len(players)
        self.players = players

    def is_complete(self):
        """
        :return:
        """
        if None in self.statuses:
            return False
        return True

    def update_active(self):
        """
        :return:
        """

        while len(self.games) < self.count and len(self.active_games) < self.active_size:
            game = Game(5, 4)
            self.games.append(game)
            self.statuses.append(None)
            players = random.sample(range(1, self.size), 1)
            self.active_games.append((len(self.games) - 1, 0, players[0]))

    def step_active(self):

        # batch_count = 16
        # #active_games_batches = [self.active_games[x:x+batchsize] for x in range(0, len(self.active_games), batchsize)]
        # active_games_batches = np.array_split(self.active_games, batch_count)
        # with Parallel(n_jobs=batch_count) as parallel:
        #     results = parallel(delayed(play_parallel_game)(active_game_batch, self.statuses, self.games, self.players)
        #                        for active_game_batch in active_games_batches)

        for rec in self.active_games:
            gameindex = rec[0]
            if self.statuses[gameindex] is None:
                game = self.games[gameindex]
                players = [self.players[rec[1]], self.players[rec[2]]]

                next_player = game.next_player()
                position = game.position.serialize_out(asplayer=next_player)
                positions = [position]  # [?, x, x, 3] shape is required
                moves = players[next_player - 1].make_game_moves(positions=positions)
                xy = moves[0][0] + 1, moves[0][1] + 1
                game.makemove(player=next_player, position=xy)
                status, _ = game.game_status()
                logging.debug("Game #%d (%d vs %d), turn: %d, outcome: %s" % (gameindex, rec[1], rec[2], game.move, str(status)))

                if status is not None:
                    self.statuses[gameindex] = status

    def cleanup_active(self):

        self.history += [(active_game[0],
                          active_game[1],
                          active_game[2],
                          self.games[active_game[0]].game_status()
                          ) for active_game in self.active_games if self.statuses[active_game[0]] is not None]

        # pop_idx = [active_game_idx
        #            for active_game_idx in range(len(self.active_games))
        #            if self.statuses[self.active_games[active_game_idx][0]] is not None]
        #
        # for idx in pop_idx:
        #     active_game = self.active_games[idx]
        #     #self.active_games.pop(idx)
        #     self.history.append((active_game[0],
        #                          active_game[1],
        #                          active_game[2],
        #                          self.games[active_game[0]].game_status()
        #                          ))

        self.active_games = [active_game for active_game in self.active_games if self.statuses[active_game[0]] is None]

    def run_tournament(self):
        """
        :return: returns tournament dynamics per step
        """
        self.update_active()
        counter = 0
        statistics = []
        while not self.is_complete():
            statistic = (self.statuses.count(None), self.statuses.count(0), self.statuses.count(1), self.statuses.count(2))
            self.step_active()
            statistic_after = (self.statuses.count(None), self.statuses.count(0), self.statuses.count(1), self.statuses.count(2))
            logging.info("Step: %d, games: %d (None=%d[-%d], DRAW=%d[+%d], P1=%d[+%d], P2=%d[+%d])" %
                         (counter, self.statuses.count(None),
                          statistic_after[0], statistic[0] - statistic_after[0],
                          statistic_after[1], statistic_after[1] - statistic[1],
                          statistic_after[2], statistic_after[2] - statistic[2],
                          statistic_after[3], statistic_after[3] - statistic[3]
                          ))
            statistics.append((statistic, statistic_after))
            counter += 1
            self.cleanup_active()
            self.update_active()
        logging.info("Tournament ended")
        return statistics

    def collect_statistics(self, players=None):
        """
        :param players: if None collect statistic for all games, if a list then cumulative statistic for players
        :return: [?, 4], (points '-1' or '0' or '1', player 1 idx, player 2 idx, game length)
        """
        if players is None:
            statistics = [[1 if history_rec[3] == 1 else -1 if history_rec[3] == 2 else 0,
                           history_rec[1],
                           history_rec[2],
                           len(self.games[history_rec[0]].moves)]
                          for history_rec in self.history]
            return statistics

        for player_idx in players:
            statistics = [[1 if history_rec[3] == 1 else -1 if history_rec[3] == 2 else 0,
                           history_rec[1],
                           history_rec[2],
                           len(self.games[history_rec[0]].moves)]
                          for history_rec in self.history if history_rec[1] == player_idx]
            return statistics
