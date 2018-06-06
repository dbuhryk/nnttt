from proboscis import SkipTest
import keras.backend as K
from learning.model import *
from proboscis import SkipTest
from proboscis import test

K.set_image_data_format('channels_last')

@test(groups=["unit"])
class LearningTests:

    @test
    def test_01(self):
        raise SkipTest("Skipping palette tests")

        print()
        #x = np.arange(75)
        size = 2
        x = np.random.rand(2, size, size, 3)
        x = np.reshape(x, (-1, size, size, 3))
        print(x)
        x = x[:, :, :, 2]
        print(x)
        x = np.reshape(x, (-1, size * size))
        print(x)
        x = softmax(x, -1)
        print(x)
        y = np.argmax(x, axis=1)
        print(y)
        pass

    @test(depends_on=[test_01])
    def test_02(self):
        x = generate_initial_input()
        model = build_model(x.shape[1:])
        model1 = build_model(x.shape[1:])
        predict = predict_moves(model, x)
        predict1 = predict_moves(model1, x)
        moves = get_best_valid_move(predict, x)
        moves1 = get_best_valid_move(predict1, x)
        pass

def run_tests():
    from proboscis import TestProgram
    TestProgram().run_and_exit()


def generate_initial_input():
    x = []
    #board = 25 * [1, 0, 0]
    #x.extend(board)
    board = 25 * [0, 0, 0]
    board[0] = 1
    x.extend(board)
    #r = np.reshape(x, (5, 5, 3))
    #r = np.expand_dims(r, axis=0)
    #r = [r]
    r = np.reshape(x, (-1, 5, 5, 3))
    return r


if __name__ == '__main__':
    run_tests()
