import numpy as np
import os
def make_np(*t):
    return (np.array(x) for x in t)

class Logger(object):
    def __init__(self, logfile):
        self.logfile = logfile
        self.f = open(logfile, 'w')
        self.last_write = 0
        self.f.write('step, avg_rew, act_loss_1, act_loss_2, act_loss_full, crit_loss')
        self._reset()
        
    def __call__(self, t, act_loss1, act_loss2, act_loss_full, circ_loss, avg_rew, print_tog):
        if print_tog:
            print('Iteration %d' % t)
            print('EpRewMean %.4f ' % avg_rew )
            print('Act losses %.4f    %.4f    %.4f' %(act_loss1, act_loss2, act_loss_full))
            print('Critic loss  %.4f' % circ_loss)
        self.act_loss1.append(act_loss1)
        self.act_loss2.append(act_loss2)
        self.act_loss_full.append(act_loss_full)
        self.circ_loss.append(circ_loss)
        self.rews.append(avg_rew)

    def __del__(self):
        self.f.close()


    def _reset(self):
        self.act_loss1 = []
        self.act_loss2 = []
        self.act_loss_full = []
        self.circ_loss = []
        self.rews = []


    def write(self):
        n = len(self.rews)
        for i in range(n):
            avg_rew, act_loss1, act_loss2, act_loss_full, circ_loss = self.rews[i], self.act_loss1[i], self.act_loss2[i],  \
                                                                      self.act_loss_full[i], self.circ_loss[i]
            self.f.write('%d, %.4f, %.4f, %.4f, %.4f, %.4f\n' % (i + self.last_write, avg_rew,
                                                               act_loss1, act_loss2, act_loss_full,
                                                               circ_loss))
        self._reset()
        self.last_write = + n


