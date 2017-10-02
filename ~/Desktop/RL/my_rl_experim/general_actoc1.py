import tensorflow as tf
import numpy as np
import gym
import util as U
import argparse
from scipy import signal

import Policies1 as pol

def discount(x, gamma):
    ret = np.array(signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1])
    return ret

def lrelu(x, alpha=0.2):
    return (1-alpha) * tf.nn.relu(x) + alpha * x

def var_accounted_for(target, pred):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    return 1- (np.var(target-pred)/ (np.var(target)+1e-8))
    
    #target = target /  np.sqrt(np.sum(np.square(target)))
    #pred = pred/  np.sqrt(np.sum(np.square(pred)))
    #return np.sum(target * pred)

class Framer(object):
    def __init__(self, frame_num):
        self.frame_num =  frame_num
    def _extend(self, obs):
        obs = list(obs)
        init = [obs[0]] * (self.frame_num-1)
        return init + obs

    def last(self, obs):
        obs = self._extend(obs)
        li = [obs[i] for i in range(-self.frame_num, 0)]
        return np.concatenate(li)
    def full(self, obs):
        obs = self._extend(obs)
        frames = []
        for i in range(len(obs)-self.frame_num+1):
            li = [obs[i+j] for j in range(self.frame_num)]
            frames.append(np.concatenate(li))
        return frames



class PathAdv(object):
    def __init__(self, gamma=0.98, look_ahead=30):
        self.reset(gamma, look_ahead)
    
    def __call__(self, rews, vals, terminal):
        
        action_val =np.convolve(rews[::-1], self.kern)[len(rews)-1::-1]
        assert len(rews) == len(action_val)   
        assert len(vals) == len(rews) + 1
        max_id = len(vals) -1 
        advs = np.zeros(len(rews))
        for i in range(len(action_val)):
            horizon_id = min(i+self.look_ahead, max_id)
            if not terminal or horizon_id != max_id:
                action_val[i] += np.power(self.gamma, horizon_id-i) * vals[horizon_id]    
            advs[i] = action_val[i]- vals[i]
        return list(action_val), list(advs)        
        
    def reset(self, gamma, look_ahead):
        self.kern = [np.power(gamma, k) for k in range(look_ahead)]
        self.look_ahead = look_ahead
        self.gamma = gamma

def rollout(env, sess, policy, framer, max_path_length=100, render=False):
    t = 0
    ob = env.reset()
    obs = [ob]
    logps = []
    rews = []
    acs = []
    done = False
    while t < max_path_length and not done:
        if render:
            env.render()
        t += 1
        ac, logp = policy(framer.last(obs), sess=sess)
        ob, rew, done, _ = env.step(ac)
        obs.append(ob)
        rews.append(rew)
        acs.append(ac)
        logps.append(logp)
    path = {'rews': rews, 'obs':obs, 'acs':acs, 'terminated': done, 'logps':logps}
    return path


def ob_feature_augment(obs_path):
    obs_path = np.array(obs_path)
    obs2 = obs_path ** 2
    l = len(obs_path)
    time = np.arange(l, dtype=np.float32).reshape(-1,1) / (l-1)
    time2 = time ** 2
    time = time * 2 - 1
    return list(np.concatenate([obs_path, obs2, time, time2], axis=1))


def train_ciritic(critic, sess, batch_size, repeat, obs, targets):
    assert len(obs) == len(targets)
    n = len(obs)
    #perm = np.random.permutation(n)
    #obs = obs[perm]
    #targets = targets[perm]
    #targets = targets.reshape(-1)
    pre_preds = critic.value(obs, sess=sess)
    ev_before = var_accounted_for(targets, pre_preds)
    #print(np.mean(targets), np.mean(pre_preds))
    #ev_before = var_accounted_for(targets, targets)
    tot_loss = 0.0
    l = int(repeat*len(obs)/batch_size+1)
    for i in range(l):
        low = (i* batch_size) % n
        high = min(low+batch_size, n)
        loss, _= critic.optimize(obs=obs[low:high], targets=targets[low:high],sess=sess)
        tot_loss += loss
    post_preds = critic.value(obs, sess=sess)
    ev_after = var_accounted_for(targets, post_preds)
    #print(ev_before, ev_after)
    return tot_loss/ l, ev_before, ev_after



def train_actor(actor, sess, batch_size, repeat, obs, advs, logps, acs):
    assert len(obs) == len(advs)
    assert len(advs) == len(acs)
    n = len(obs)
    perm = np.random.permutation(n)
    obs, advs, logps, acs = obs[perm], advs[perm], logps[perm], acs[perm]
    tot_rew_loss, tot_p_dist, tot_comb_loss = 0.0, 0.0, 0.0 
    l = int(repeat*len(obs)/batch_size+1)
    for i in range(l):
        low = (i* batch_size) % n
        high = min(low+batch_size, n)
        rew_loss, p_dist, comb_loss, _ = actor.optimize(sess=sess, obs=obs[low:high], acs=acs[low:high],  advs=advs[low:high], logps=logps[low:high])
        tot_comb_loss += comb_loss
        tot_p_dist += p_dist
        tot_rew_loss += rew_loss
    
    return (tot_rew_loss/l, tot_p_dist/l, tot_comb_loss/l)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--outdir", default='log.txt')
parser.add_argument("--animate", default=False, action='store_true')
parser.add_argument("--env", default='Pendulum-v0')
parser.add_argument("--seed", default=12321)
parser.add_argument("--tboard", default=False, action='store_true')
args = parser.parse_args()
LOG_FILE = args.outdir
ANIMATE = args.animate

MAX_PATH_LENGTH = 400
ITER = 100000
BATCH = 64
MULT = 2
LOG_ROUND = 10
EP_LENGTH_STOP = 1200
FRAMES = 2

desired_kl = 0.002
max_lr, min_lr = 1. , 1e-6

env = gym.make(args.env)
framer = Framer(frame_num=FRAMES)
ob_dim = env.observation_space.shape[0] * FRAMES
critic = pol.Critic(num_ob_feat=ob_dim*2+4)
rew_to_advs =  PathAdv(gamma=0.98, look_ahead=10)
logger = U.Logger(logfile=LOG_FILE)
np.random.seed(args.seed)
env.seed(args.seed)

try: 
    ac_dim = env.action_space.shape[0]
    actor = pol.Actor(num_ob_feat=ob_dim, num_ac=ac_dim, act_type='cont')
    print('Continuous Action Space')
except:
    print('Discrete Action Space')
    ac_n = env.action_space.n
    actor = pol.Actor(num_ob_feat=ob_dim, num_ac=ac_n, act_type='disc')

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./summaries/'+args.outdir.split('.')[0], tf.get_default_graph())

#all_vars = tf.trainable_variables()
#u = [v for v in all_vars if 'Critic' in v.name]


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(ITER):
        ep_obs, ep_advs, ep_logps, ep_target_vals, ep_acs = [], [], [], [], []
        ep_unproc_obs = []
        ep_rews = []
        tot_rews, j = 0, 0
        while len(ep_rews)<EP_LENGTH_STOP:
            path = rollout(env=env, sess= sess, policy=actor.act, 
                           max_path_length=MAX_PATH_LENGTH, framer=framer,
                           render= j==0 and  i % 20 == 0 and ANIMATE)
            obs_aug = framer.full(ob_feature_augment(path['obs']))
            ep_unproc_obs += framer.full(path['obs'][:-1])
            ep_obs += obs_aug[:-1]
            ep_logps += path['logps']
            ep_acs += path['acs']
            obs_vals = critic.value(obs=obs_aug, sess=sess).reshape(-1)
            target_val, advs = rew_to_advs(rews=path['rews'], terminal=path['terminated'], vals=obs_vals)
            #target_val = discount(path['rews'], gamma=0.97)
            #advs = target_val - obs_vals[:-1]
            ep_target_vals += list(target_val)
            ep_advs += list(advs)
            ep_rews += path['rews']
            tot_rews += sum(path['rews'])

            if j ==0 and i%10 ==0:
                actor.printoo(obs=ep_unproc_obs, sess=sess)
                critic.printoo(obs=ep_obs, sess=sess)
                print('Path length %d' % len(path['rews']))
                print('Terminated {}'.format(path['terminated']))
            j +=1

        avg_rew = float(tot_rews)/ j  
        ep_obs, ep_advs, ep_logps, ep_target_vals, ep_acs, ep_rews, ep_unproc_obs = U.make_np(ep_obs, ep_advs, ep_logps, 
                                                                                ep_target_vals, ep_acs, ep_rews, ep_unproc_obs)
        ep_advs.reshape(-1)
        ep_target_vals.reshape(-1)
        ep_advs = (ep_advs - np.mean(ep_advs))/ (1e-8+ np.std(ep_advs))
        """
        if i%10 ==0:
            print('Advantage mean & std {}, {}'.format(np.mean(ep_advs), np.std(ep_advs)))       
        if i % 50 == 13:
            perm = np.random.choice(len(ep_advs), size=20)
            print('Some obs', ep_obs[perm])
            print('Some acs', ep_obs[perm])
            print('Some advs', ep_obs[perm])
            print('Some rews', ep_rews[perm])
        """
        
        cir_loss, ev_before, ev_after = train_ciritic(critic=critic, sess=sess, batch_size=BATCH, repeat= MULT, obs=ep_obs, targets=ep_target_vals)
        act_loss1, act_loss2, act_loss_full = train_actor(actor=actor, sess=sess, 
                                                         batch_size=BATCH, repeat=MULT, obs=ep_unproc_obs, 
                                                          advs=ep_advs, acs=ep_acs, logps=ep_logps)
        if args.tboard:
            summ, _, _ = sess.run([merged, actor.ac, critic.v], feed_dict={actor.ob: ep_unproc_obs[:1000], critic.obs:ep_obs[:1000]})
            writer.add_summary(summ,i)
        #logz
        act_lr, _ = actor.get_opt_param(sess)
        logger(i, act_loss1=act_loss1, act_loss2=act_loss2,  act_loss_full=act_loss_full, circ_loss=np.sqrt(cir_loss), 
                avg_rew=avg_rew, ev_before=ev_before, ev_after=ev_after,act_lr=act_lr, print_tog= (i %20) == 0)
        if i % 100 == 50:
            logger.write()
        if act_loss2 < desired_kl/4:
            new_lr = min(max_lr,act_lr*1.5)

            actor.set_opt_param(sess=sess, new_lr=new_lr)
        elif act_loss2 > desired_kl * 4:
            new_lr = max(min_lr,act_lr/1.5)
            actor.set_opt_param(sess=sess, new_lr=new_lr)


del logger
