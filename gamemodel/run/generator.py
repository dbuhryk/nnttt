from gameplay.tournaments.ova import OVATournament
from gameplay.tournaments.rnd import RNDTournament
from gameplay.players.mlplayer import *
from gameplay.players.rndplayer import RNDPlayer
import logging
import h5py
import numpy as np
#import matplotlib.pyplot as plt
import argparse
import sys
from joblib import Parallel, delayed

#logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def play_single_tournament(t_idx, t, args):
    statistics = t.run_tournament()

    statuses = [game.game_status()[0] for game in t.games]
    logging.info("Tournament #%d Results GAMES:%d, WINS:%d, LOSSES:%d, DRAWS:%d" % (t_idx, len(statuses), statuses.count(1), statuses.count(2), statuses.count(0)))

    # pick only winning games
    win_pos_prediction = [[pos_prediction[0], pos_prediction[1]]
                          for game in t.games if (game.game_status()[0] == 1 or game.game_status()[0] == 2) and len(game.moves) < args.gamelen
                          for pos_prediction in game.serialize_moves_and_positions(asplayer=game.game_status()[0])]

    # win_pos_prediction [?, 2]
    # win_pos_prediction [?, 0] [5, 5, 3]
    # win_pos_prediction [?, 1] [5, 5, 1]

    win_data_positions = np.array([item[0] for item in win_pos_prediction])  # [?, 5, 5, 3]
    win_data_predictions = np.array([item[1] for item in win_pos_prediction])  # [?, 5, 5, 1]

    logging.info("Samples/Predictions generated %s, %s " % (str(win_data_positions.shape), str(win_data_predictions.shape)))

    data_positions = win_data_positions
    data_predictions = win_data_predictions

    if data_positions.size > 0:
        try:
            with h5py.File(args.file, 'r') as h5f:
                positions = h5f['set_positions'][:]
                predictions = h5f['set_predictions'][:]
                logging.info("Samples/Predictions loaded %s, %s " % (str(positions.shape), str(predictions.shape)))
                data_positions = np.concatenate((positions, data_positions), axis=0)
                data_predictions = np.concatenate((predictions, data_predictions), axis=0)
                logging.info("Appended data from file \"%s\"" % args.file)
                logging.info("Samples/Predictions concatenated %s, %s " % (str(data_positions.shape), str(data_predictions.shape)))
        except (IOError, ValueError, KeyError):
            logging.warning("Data file \"%s\" is missing or structure is corrupted" % args.file)

        ins_unique, ins_pos = np.unique(data_positions, return_index=True, axis=0)
        logging.info("Unique Samples %s (%d)" % (str(ins_unique.shape), ins_unique.shape[0] - data_positions.shape[0]))

        data = [(
            np.expand_dims(sample, 0),
            np.sum(data_predictions[np.count_nonzero(np.subtract(data_positions, sample), axis=(1, 2, 3)) == 0], axis=0, keepdims=True)
        ) for sample in ins_unique]
        dataset1, dataset2 = zip(*data)
        data_positions = np.concatenate(dataset1, axis=0)
        data_predictions = np.concatenate(dataset2, axis=0)

        logging.debug("Normalising predictions %s " % str(data_predictions.shape))
        row_sums = np.reshape(data_predictions.sum(axis=(1, 2, 3)), (-1, 1, 1, 1))
        row_sums = np.multiply(row_sums, np.ones((1, 5, 5, 1)))
        data_predictions = np.divide(data_predictions, row_sums)

        logging.info("Samples/Predictions reduced %s, %s " % (str(data_positions.shape), str(data_predictions.shape)))

        with h5py.File(args.file, 'w') as h5f:
            h5f.create_dataset('set_positions', data=data_positions)
            h5f.create_dataset('set_predictions', data=data_predictions)
            logging.info("Saved data file \"%s\"" % args.file)

def main():
    parser = argparse.ArgumentParser(description='Process generator.')
    parser.add_argument('--count', nargs='?', type=int, dest='count', default=100000, help='Games to generate')
    parser.add_argument('--size', nargs='?', type=int, dest='size', default=10000, help='Games to play at once')
    parser.add_argument('--game-length', nargs='?', type=int, dest='gamelen', default=14, help='length of games')
    parser.add_argument('--file', nargs='?', type=str, dest='file', default='data.h5', help='Data file')
    parser.add_argument('--class', type=str, action='append', dest='clazz', help='ML Class', required=True)
    parser.add_argument('--classsuffix', type=str, action='append', dest='classsuffix', help='ML Class suffix')
    parser.add_argument('--weights', type=float, action='append', dest='weights', help='Class weights')
    parser.add_argument('--parallel', nargs='?', type=int, dest='parallel', default=1, help='Tournaments to play')
    parser.add_argument('--debug_print_boards', dest='debug_print_boards', action='store_true')
    args = parser.parse_args()
    logging.info("generating TOURNAMENTS:%d, GAMES:%d, CONCURRENT:%d, GAMELEN:%d" % (args.parallel, args.count, args.size, args.gamelen))

    clazzs = [getattr(sys.modules[__name__], clazz) for clazz in args.clazz]
    players = [clazz() for clazz in clazzs]
    mlplayers = [player for player in players if isinstance(player, MLPlayer)]
    if args.classsuffix is not None:
        assert(len(args.classsuffix) == len(mlplayers))
        [player.load_model(suffix=suffix) for player, suffix in zip(mlplayers, args.classsuffix)]

    tournaments = [RNDTournament(players, args.count, args.size, weights=args.weights, debug_print_boards=args.debug_print_boards) for _ in range(args.parallel)]

    with Parallel(n_jobs=args.parallel) as parallel:
        results = parallel(delayed(play_single_tournament)(t_idx, tournaments[t_idx], args)
                           for t_idx in range(len(tournaments)))

    #t = RNDTournament(players, args.count, args.size, weights=args.weights, debug_print_boards=args.debug_print_boards)
    #play_single_tournament(t, args)


if __name__ == "__main__":
    main()
