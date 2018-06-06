from gameplay.tournaments.ova import OVATournament
from gameplay.players.mlplayer import *
from gameplay.players.rndplayer import RNDPlayer
import logging
import h5py
import numpy as np
import keras
import argparse
import sys
import time

#logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def main():
    np.random.seed(seed=int(time.time()))

    parser = argparse.ArgumentParser(description='Process generator.')
    parser.add_argument('--epochn', nargs='?', type=int, dest='epoch', default=10, help='Number of epochs')
    parser.add_argument('--samplen', nargs='?', type=int, dest='sample', default=0, help='Sample size')
    parser.add_argument('--validation', nargs='?', type=float, dest='validation', default=0.1, help='Validation')
    parser.add_argument('--class', type=str, dest='clazz', default="MLPlayer01", help='ML Class', required=True)
    parser.add_argument('--file', type=str, dest='file', default="data.h5", help='Learn file')
    parser.add_argument('--classsuffix', nargs='?', type=str, dest='classsuffix', help='ML Class suffix')
    parser.add_argument('--reset', dest='reset', action='store_true')
    args = parser.parse_args()
    logging.info("learning EPOCH:%d, SAMPLE:%d, VALIDATION:%d, CLASS:%s" %
                 (args.epoch, args.sample, args.validation, args.clazz))

    clazz = getattr(sys.modules[__name__], args.clazz)
    mlplayer = clazz()
    keras.utils.print_summary(mlplayer.model)
    if not args.reset:
        mlplayer.load_model(suffix=args.classsuffix)

    positions = None
    predictions = None
    try:
        with h5py.File(args.file, 'r') as h5f:
            positions = h5f['set_positions'][:]
            predictions = h5f['set_predictions'][:]
            logging.info("Samples/Predictions loaded %s, %s " % (str(positions.shape), str(predictions.shape)))
            if args.sample != 0:
                sample = args.sample
                if args.sample > positions.shape[0]:
                    sample = positions.shape[0]
                p = np.random.permutation(positions.shape[0])
                positions = positions[p]
                predictions = predictions[p]
                positions = positions[:sample]
                predictions = predictions[:sample]

    except IOError:
        logging.warning("Data file \"%s\" is missing or structure is corrupted" % args.file)

    if positions is not None:
        predictions = np.reshape(predictions, (-1, 25))
        mlplayer.model.fit(x=positions, y=predictions, validation_split=args.validation, verbose=2, epochs=args.epoch, shuffle=True)

    mlplayer.save_model(suffix=args.classsuffix)


def refine_data():
    with h5py.File("data.h5", 'r') as h5f:
        ins = h5f['set_positions'][:]
        predictions = h5f['set_predictions'][:]

        # Find all unique inputs
        ins_unique, ins_pos = np.unique(ins, return_index=True, axis=0)

        # # Iterate via all unique po
        # for sample in ins_unique:
        #     #sample = ins[pos]
        #     # difference between current sample and input samples (broadcast)
        #     deltas = np.subtract(ins, sample)
        #     # count non-zero elements for each delta
        #     deltas_abs = np.count_nonzero(deltas, axis=(1, 2, 3))
        #     same_samples = deltas_abs == 0
        #     prediction_sum = np.sum(predictions[same_samples], axis=0, keepdims=True)

        data = [(
            np.expand_dims(sample, 0),
            np.sum(predictions[np.count_nonzero(np.subtract(ins, sample), axis=(1, 2, 3)) == 0], axis=0, keepdims=True)
        ) for sample in ins_unique]
        dataset1, dataset2 = zip(*data)
        ins = np.concatenate(dataset1, axis=0)
        predictions = np.concatenate(dataset2, axis=0)

        row_sums = np.reshape(predictions.sum(axis=(1, 2, 3)), (-1, 1, 1, 1))
        ones = np.ones((1, 5, 5, 1))
        row_sums = np.multiply(row_sums, np.ones((1, 5, 5, 1)))

        predictions = np.divide(predictions, row_sums)
        pass


if __name__ == "__main__":
    #refine_data()
    main()
