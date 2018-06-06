from abc import ABCMeta, abstractmethod
import numpy as np

class Player(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def make_game_moves(self, positions, asplayer):
        """
        :param positions:
        :param asplayer:
        :return: position (x,y) = [1...size], where size is the size of the board
        """
        return

    @staticmethod
    def get_best_valid_move(predictions, positions):
        moves, _ = Player.get_best_valid_moves(predictions, positions)  # [?, n, 2]
        return moves[:, 0, :]  # [?, 2]

    @staticmethod
    def get_best_valid_moves(predictions, positions):
        """
        :param predictions: shape [?, 5, 5, 1]
        :param positions: shape [?, 5, 5, 3]
        :return: shape [?, 2], position [x,y], x,y=[0...size)
        """

        shape_pos = positions.shape
        size = shape_pos[1]
        predictions = np.reshape(predictions, (-1, size, size, 1))
        shape_pr = predictions.shape


        np.testing.assert_array_equal(shape_pr[:-1], shape_pos[:-1])
        np.testing.assert_equal(shape_pr[1], shape_pr[2])
        np.testing.assert_equal(shape_pos[1], shape_pos[2])

        size = shape_pr[1]

        validmoves = positions[:, :, :, 0]  # [?, size, size]
        predictions = np.reshape(predictions, (-1, size, size))  # [?, size, size]

        moves = np.multiply(predictions, validmoves)  # [?, size, size]
        moves = np.reshape(moves, (-1, size * size))  # [?, size * size]

        #moves[moves < 0.7] = 0
        pos_best3 = np.argsort(moves, axis=-1)[:, -3:]  # [?, 3]
        pos_best3 = np.stack((pos_best3[:, 2], pos_best3[:, 1], pos_best3[:, 0]), axis=1)
        moves_best3 = moves[np.arange(np.shape(moves)[0])[:, np.newaxis], pos_best3]
        pos_best3 = np.reshape(pos_best3, (-1, 3))  # [?, 3]
        pos_best3 = np.expand_dims(pos_best3, 2)  # [?, 3, 1]
        pos_y, pos_x = np.divmod(pos_best3, size)  # [?, 3, 1], [?, 3, 1]
        xy_best3 = np.concatenate((pos_x, pos_y), axis=2)  # [?, 3, 2]
        return xy_best3, moves_best3

        #good_pos = np.where(moves > 0.7)
        #good_pos = np.column_stack(good_pos)
        # good_pos = np.split(good_pos, range(moves.shape[0]), axis=0)
        # good_moves = moves[good_pos]


        #moves = Player.softmax(moves, axis=-1)  # [?, size * size]
        # Original solution
        # moves = np.argmax(moves, axis=1)  # (1,)
        # moves = np.reshape(moves, (-1, 1))  # [?, 1]
        # moves_y, moves_x = np.divmod(moves, size)  # [?, 1], [?, 1]
        # moves = np.concatenate((moves_x, moves_y), axis=1)  # [?, 2]
        #
        # return moves

    @staticmethod
    def softmax(x, axis=0):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), keepdims=True, axis=axis)