####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Email	: mengchil@cs.cmu.edu      #
####################################


import numpy as np
import time
import argparse

from .gen2out import gen2Out
from .utils import sythetic_group_anomaly, plot_results


if __name__ == '__main__':	

    parser = argparse.ArgumentParser(description='Parameters for gen2Out')
    parser.add_argument('--lower_bound', default=9, type=int, help='Lower bound of sampling (2^i)')
    parser.add_argument('--upper_bound', default=12, type=int, help='Upper bound of sampling (2^i)')
    parser.add_argument('--max_depth', default=7, type=int, help='Maximum depth of each tree')
    parser.add_argument('--rotate', default=True, type=bool, help='Whether to use the rotated IF or not')
    parser.add_argument('--contamination', default='auto', type=str, help='Contamination rate of the dataset')
    parser.add_argument('--random_state', default=0, type=int, help='Control the randomness')
    args = parser.parse_args()

    model = gen2Out(lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    max_depth=args.max_depth,
                    rotate=args.rotate,
                    contamination=args.contamination,
                    random_state=args.random_state)

    X = sythetic_group_anomaly()

    print('Start point anomaly detection:')
    t1 = time.time()
    pscores = model.point_anomaly_scores(X)
    t2 = time.time()
    print('Finish in %.1f seconds!\n' % (t2 - t1))

    print('Start group anomaly detection:')
    t1 = time.time()
    gscores = model.group_anomaly_scores(X)
    t2 = time.time()
    print('Finish in %.1f seconds!\n' % (t2 - t1))

    print('Generating plots...')
    plot_results(X, model)
    print('Finish!')

    
