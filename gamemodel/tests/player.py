from proboscis import test

from gameplay.players.rndplayer import RNDPlayer
from learning.model import *

K.set_image_data_format('channels_last')

@test(groups=["unit"])
class PlayerTests:

    @test
    def test_01(self):
        p = RNDPlayer()
        x = generate_initial_input()
        #model = build_model(x.shape[1:])
        moves = p.make_game_moves(x)
        pass

def run_tests():
    from proboscis import TestProgram
    TestProgram().run_and_exit()


def generate_initial_input():
    x = []
    board = 25 * [0, 0, 0]
    board[0] = 1
    x.extend(board)
    r = np.reshape(x, (-1, 5, 5, 3))
    return r


if __name__ == '__main__':
    run_tests()
