import numpy as np
import os

def var_accounted_for(target, pred):
    """
    Computes the covariance between target and pred (after scaling).
    Return in [-1, 1].
    """
    pred, target = pred.reshape(-1),  target.reshape(-1)
    pred = (pred - np.mean(pred))/np.std(pred)
    target =  (target - np.mean(target))/np.std(target)
    return np.mean(target * pred)


class LinearSchedule(object):
    """
    For scheduling the annealing of training hyperparameters. 

    Args:
       init_t: step at which the annealing starts
       end_t : step at which the annealing end
       init_val: value of hyperparameter at init_t and before
       end_val: value of hyperparameter at end_t and after
       updated_every_t: updates happen only every update_every_t steps
    """
    def __init__(self, init_t, end_t, init_val, end_val, update_every_t):
        self.init_t = init_t
        self.end_t = end_t
        self.init_val = init_val
        self.end_val = end_val
        self.update_every_t = update_every_t

    def val(self, t):
        if t < self.init_t:
            return self.init_val
        if t > self.end_t:
            return self.end_val
        return ((t-self.init_t)*self.end_val + (self.end_t-t)*self.init_val)/ float(self.end_t-self.init_t)

    def update_time(self, t):
        return t %self.update_every_t == 0


def make_np(*t):
    """
    Transorfms as many lists to numpy version
    """
    return (np.array(x) for x in t)   

class Logger(object):
    """  
    Simple logger which stores training stats and writes to file occasionally.
    """
    def __init__(self, logfile):
        self.logfile = logfile
        dir_name= os.path.dirname(logfile)
        if not os.path.exists(dir_name) and dir_name != '':
            os.makedirs(dir_name, exist_ok=True)
        self.f = open(logfile, 'w')
        self.last_write = 0
        self.f.write('step avg_rew ev_before ev_after act_loss crit_loss kl_dist avg_ent\n')
        self._reset()
        
    def __call__(self, t, act_loss, circ_loss, kl_dist, avg_rew, print_tog, act_lr, avg_ent, worker_id=0, ev_before= -1 , ev_after=-1):
        if print_tog:
            print('\nIteration %d' % t)
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

    def flush(self):
        """
        Writes the gathered stats to file specified in init
        """
        n = len(self.rews)
        for i in range(n):
            avg_rew, act_loss, circ_loss, kl_dist = self.rews[i], self.act_loss[i], self.circ_loss[i], self.kl_dist[i]
            ev_before, ev_after, ent = self.ev_before[i], self.ev_after[i], self.ents[i]
            self.f.write('%d %.4f %.4f  %.4f %.4f  %.4f %.4f %.4f\n' % (i + self.last_write, avg_rew, ev_before, ev_after,                                                                     act_loss, circ_loss, kl_dist, ent))
        self._reset()
        self.last_write = + n

def ob_feature_augment(obs_path):
    """
    Method for feature augmentation for observations. Only used in old versions of the code. 
    """
    obs_path = np.array(obs_path)
    obs2 = obs_path ** 2
    l = len(obs_path)
    time = np.arange(l, dtype=np.float32).reshape(-1,1) / (l-1)
    time2 = time ** 2
    time = time * 2 - 1
    return list(np.concatenate([obs_path, obs2, time, time2], axis=1))
