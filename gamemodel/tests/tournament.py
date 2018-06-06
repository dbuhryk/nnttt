from proboscis import test
from gameplay.tournaments.generic import Tournament
from gameplay.tournaments.fr import FRTournament
from gameplay.tournaments.ova import OVATournament
import logging


@test(groups=["unit"])
class TournamentTests:
    """"""

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    @test
    def test_01(self):
        t = OVATournament(10, 100)
        statistics = t.run_tournament()
        #t.games[0].serialize_moves_and_positions(1)
        #print(statistics)


def run_tests():
    from proboscis import TestProgram
    TestProgram().run_and_exit()


if __name__ == '__main__':
    run_tests()
