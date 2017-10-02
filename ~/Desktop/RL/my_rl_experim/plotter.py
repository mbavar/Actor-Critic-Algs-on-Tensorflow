import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter as sav
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--outdir', default='log.txt')
args = parser.parse_args()

with open(args,outdir) as f:
    t = f.read()
u = t.replace('\n', ' ').replace('\t', ' ').replace(',', ' ').split()
z = np.array(u).reshape(-1,8)
data = z[1:].astype(float)
rews = data[:, 3]
smooth_rews= sav(polyorder=3, window_length=11,x=rews1)

pltplot(smooth_rews)

