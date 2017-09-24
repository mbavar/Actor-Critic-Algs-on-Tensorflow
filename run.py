import tensorflow as tf
import numpy as np
import gym
import util as U
import argparse
from scipy import signal

LOG_FILE = args.outdir
ANIMATE = args.animate
ROLLS_PER_EPISODE = 10
MAX_PATH_LENGTH = 400
ITER = 100000
BATCH = 32
MULT = 5
LOG_ROUND = 10
EP_LENGTH_STOP = 2000
FRAMES = 2


def main():
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("job", choices=["ps", "worker"])
parser.add_argument("task", dtype=int, )
parser.add_argument("--outdir", default='log.txt')
parser.add_argument("--animate", default=False, action='store_true')
parser.add_argument("--env", default='Pendulum-v0')
parser.add_argument("--seed", default=12321)
parser.add_argument("--tboard", default=False)
args = parser.parse_args()





if __name__ == '__main__':
    main()


#all_vars = tf.trainable_variables()
#u = [v for v in all_vars if 'Critic' in v.name]
