import numpy as np

from gameplay.players.player import Player


class RNDPlayer(Player):
    model = None

    def make_game_moves(self, positions, asplayer = None):
        """
        returns valid moves for every position as player 1
        asplayer is not used
        :param positions: [?, 5, 5, 3]
        :param asplayer: makes move as player with index, normally either 1 or 2
        :return: [?, 2], where x, y = [:, 1], [:, 2]
        """
        positions = np.array(positions)  # [?, 5, 5, 3]
        shape = positions.shape[:-1] + (1,)
        predictions = np.random.rand(*shape)  # [?, 5, 5, 1]
        moves = Player.get_best_valid_move(predictions, positions)
        return moves
