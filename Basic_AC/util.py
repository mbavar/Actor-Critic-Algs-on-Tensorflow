import numpy as np
import os
def make_np(*t):
    return (np.array(x) for x in t)

class Logger(object):
    def __init__(self, logfile):
        self.logfile = logfile
        self.f = open(logfile, 'w')
        self.last_write = 0
        self.f.write('step avg_rew ev_before ev_after act_loss crit_loss kl_dist avg_ent\n')
        self._reset()
        
    def __call__(self, t, act_loss, circ_loss, kl_dist, avg_rew, print_tog, act_lr, avg_ent, worker_id=0, ev_before= -1 , ev_after=-1):
        if print_tog:
            print('Iteration %d' % t)
            print('EpRewMean %.4f ' % avg_rew )
            print('EV Before %f' % ev_before)
            print('EV After %f' % ev_after )
            print('Act losses %.4f  ' %(act_loss))
            print('Critic loss  %.4f' % circ_loss)
            print('Actor lr %f' % act_lr)
            print('KL dist %.4f' % kl_dist )
            print('Avg Ent %.4f' % avg_ent)
            print('Performed by worker %d' % worker_id)
            

        self.act_loss.append(act_loss)
        self.circ_loss.append(circ_loss)
        self.rews.append(avg_rew)
        self.ev_before.append(ev_before)
        self.ev_after.append(ev_after)
        self.kl_dist.append(kl_dist)
        self.ents.append(avg_ent)

    def __del__(self):
        self.f.close()


    def _reset(self):
        self.act_loss = []
        self.circ_loss = []
        self.rews = []
        self.ev_before = []
        self.ev_after = []
        self.kl_dist = []
        self.ents = []


    def write(self):
        n = len(self.rews)
        for i in range(n):
            avg_rew, act_loss, circ_loss, kl_dist = self.rews[i], self.act_loss[i], self.circ_loss[i], self.kl_dist[i]
            ev_before, ev_after, ent = self.ev_before[i], self.ev_after[i], self.ents[i]
            self.f.write('%d %.4f %.4f  %.4f %.4f  %.4f %.4f %.4f\n' % (i + self.last_write, avg_rew, ev_before, ev_after,
                                                                        act_loss, circ_loss, kl_dist, ent))
        self._reset()
        self.last_write = + n


