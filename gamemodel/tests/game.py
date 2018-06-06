from proboscis.asserts import assert_equal, assert_not_equal
from proboscis import after_class
from proboscis import before_class
from proboscis import SkipTest
from proboscis import test
from gameplay.game import Game


@test(groups=["unit"])
class GameTests:

    @test
    def test_01(self):
        """
        testing:
        x x x x -
        - o - - -
        - - o - -
        - - - o -
        - - - - -
        """
        game = Game(5, 4)
        moves = [(1, 1), (2, 2), (2, 1), (3, 3), (3, 1), (4, 4), (4, 1)]
        for move in moves:
            game.makemove(game.next_player(), move)

        game.printboard()
        print(game.position.serialize_out())


    @test
    def test_game01(self):
        """
        testing:
        x x x x -
        - o - - -
        - - o - -
        - - - o -
        - - - - -
        """
        game = Game(5, 4)
        moves = [(1, 1), (2, 2), (2, 1), (3, 3), (3, 1), (4, 4)]
        for move in moves:
            game.makemove(game.next_player(), move)
            status, _ = game.game_status()
            assert_equal(status, None)

        game.makemove(game.next_player(), (4, 1))
        status, _ = game.game_status()
        game.printboard()
        assert_equal(status, 1)


    @test
    def test_game02(self):
        """
        testing:
        x o x o x
        o x o x o
        x o x o x
        x o x o x
        x o x o o
        """
        game = Game(5, 4)
        moves = [
            (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (1, 3), (2, 3), (3, 3),
            (4, 3), (5, 3), (5, 5), (1, 4), (2, 4), (3, 4), (4, 4), (1, 5), (2, 5), (3, 5), (4, 5)]
        for move in moves:
            game.makemove(game.next_player(), move)
            status, _ = game.game_status()
            #game.printboard()
            assert_equal(status, None)

        game.makemove(game.next_player(), (5, 4))
        status, _ = game.game_status()
        game.printboard()
        assert_equal(status, 0)


def run_tests():
    from proboscis import TestProgram
    TestProgram().run_and_exit()

if __name__ == '__main__':
    run_tests()