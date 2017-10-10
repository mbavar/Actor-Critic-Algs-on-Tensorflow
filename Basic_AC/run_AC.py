import tensorflow as tf
import numpy as np
import gym
import util as U
import argparse
from scipy import signal
import os 

import Policies as pol

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--animate", default=False, action='store_true') 
parser.add_argument("--env", default='Pendulum-v0') 
parser.add_argument("--seed", default=12321)  
parser.add_argument("--tboard", default=False, action='store_true')
parser.add_argument("--save_every", default=600)  
parser.add_argument("--outdir", default=os.path.join('tmp', 'log.txt'))  # file for the statistics of training
parser.add_argument("--checkpoint_dir", default=os.path.join('tmp', 'checkpoints'))   #where to save checkpoint
parser.add_argument("--frames", default=1)    #how many recent frames to send to model 
parser.add_argument("--mode", choices=["train", "debug"], default="train") #how verbose to print to stdout
parser.add_argument("--desired_kl", default=0.002)   #An important to tune. The learning rate is adjusted when KL dist falls 
                                                     #far above or below the desired_kl

class Framer(object):
    """
    Ceates the augmentd obs features, to be fed to Actor & Critics nets, by concatenating a few (frame_num)
    bare obs together. So ob_dim = env.observation_space.shape[0] * frame_num

    Members:
      last: given the current stack of obs creates the last feature for t=len(obs)
      full: same as last but gives features for the whole whole history t= 0, 1, ..., len(obs)

    """
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
    """
    Given a few rewards and values (given from critic valuation of obs) sampled in a rollout
    gives the advantage function and updated values for the obs.
    """
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
    """
    Gather an episode of experience by running the environment. Continues until env.done is True
    or length of episode exceed max_path_length
    """
    t = 0
    ob = env.reset()
    obs = [ob]
    logps = []
    rews = []
    acs = []
    ents = []
    done = False
    while t < max_path_length and not done:
        if render:
            env.render()
        t += 1
        ac, logp, ent = policy(framer.last(obs), sess=sess)
        ob, rew, done, _ = env.step(ac)
        obs.append(ob)
        rews.append(rew)
        acs.append(ac)
        ents.append(ent)
        logps.append(logp)
    path = {'rews': rews, 'obs':obs, 'acs':acs, 'terminated': done, 'logps':logps, 'entropy':ents}
    return path

def train_ciritic(critic, sess, obs, targets):
    assert len(obs) == len(targets)
    pre_preds = critic.value(obs, sess=sess)
    ev_before = U.var_accounted_for(targets, pre_preds)
    loss, _= critic.optimize(obs=obs, targets=targets, sess=sess)
    post_preds = critic.value(obs, sess=sess)
    ev_after = U.var_accounted_for(targets, post_preds)
    return loss, ev_before, ev_after

def train_actor(actor, sess, obs, advs, logps, acs):
    assert len(obs) == len(advs)
    assert len(advs) == len(acs)
    loss, _ = actor.optimize(sess=sess, obs=obs, acs=acs,  advs=advs, logps=logps) 
    return loss

def get_roll_params(env_id):
    """
    Creates environment and sets up the rollout params.
    """
    env = gym.make(env_id)
    max_path_length, ep_length_stop = 1200, 3000
    if env.spec.max_episode_steps is not None:
        max_path_length = env.spec.max_episode_steps
        ep_length_stop  = min(max_path_length * 6, 3000)
    return env, max_path_length, ep_length_stop

def main():        
    args = parser.parse_args()
    LOG_FILE = args.outdir
    ANIMATE = args.animate
    DEBUG = (args.mode == "debug")
    CHECKPOINT_PATH = args.checkpoint_dir + '-'+ args.env.split('-')[0]
    MAX_ROLLS = 7
    ITER = 5000000
    LOG_ROUND = 10
    env, MAX_PATH_LENGTH, EP_LENGTH_STOP = get_roll_params(args.env)

    desired_kl = args.desired_kl
    max_lr, min_lr = 1. , 1e-6

    framer = Framer(frame_num=args.frames) 
    log_gamma_schedule = U.LinearSchedule(init_t=100, end_t=3000, init_val=-2, end_val=-8, update_every_t=100) #This is base 10
    log_beta_schedule = U.LinearSchedule(init_t=100, end_t=3000, init_val=0, end_val=-4, update_every_t=100) #This is base 10
    rew_to_advs = PathAdv(gamma=0.98, look_ahead=40)
    logger = U.Logger(logfile=LOG_FILE)
    np.random.seed(args.seed)
    env.seed(args.seed)

    if type(env.action_space) == gym.spaces.discrete.Discrete:
        act_type = 'disc'
        ac_dim, ac_scale = env.action_space.n, None
        print('Discrete Action Space. Numer of actions is {}.'.format(env.action_space.n))
    else:
        act_type = 'cont'
        ac_dim, ac_scale = env.action_space.shape[0], np.maximum(env.action_space.high, np.abs(env.action_space.low))    
        print('Continuous Action Space. Action Scale is {}.'.format(ac_scale))
    ob_dim = env.observation_space.shape[0] * args.frames
    critic = pol.Critic(num_ob_feat=ob_dim)
    actor = pol.Actor(num_ob_feat=ob_dim, ac_dim=ac_dim, act_type=act_type, ac_scale=ac_scale)
    saver = tf.train.Saver(max_to_keep=3) 

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join('summaries', args.outdir.split('.')[0]+'.data'), tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        tot_rolls = 0

        for i in range(ITER):
            ep_obs, ep_advs, ep_logps, ep_target_vals, ep_acs = [], [], [], [], []
            ep_rews = []
            tot_rews, tot_ent, rolls = 0, 0, 0
            while len(ep_rews)<EP_LENGTH_STOP and rolls<MAX_ROLLS:
                path = rollout(env=env, sess= sess, policy=actor.act, 
                               max_path_length=MAX_PATH_LENGTH, framer=framer,
                               render= rolls==0 and  i % 20 == 0 and ANIMATE)
                obs_aug = framer.full(path['obs'])
                ep_obs += obs_aug[:-1]
                ep_logps += path['logps']
                ep_acs += path['acs']
                tot_ent += np.sum(path['entropy'])
                obs_vals = critic.value(obs=obs_aug, sess=sess).reshape(-1)
                target_val, advs = rew_to_advs(rews=path['rews'], terminal=path['terminated'], vals=obs_vals)
                ep_target_vals += list(target_val)
                ep_advs += list(advs)
                ep_rews += path['rews']
                tot_rews += sum(path['rews'])

                if rolls ==0 and i%10 ==0 and DEBUG:
                    actor.printoo(obs=ep_obs, sess=sess)
                    critic.printoo(obs=ep_obs, sess=sess)
                    print('Path length %d' % len(path['rews']))
                    print('Terminated {}'.format(path['terminated']))
                rolls +=1
                
            avg_rew = float(tot_rews)/ rolls
            avg_ent = tot_ent/ float(len(ep_logps))
            ep_obs, ep_advs, ep_logps, ep_target_vals, ep_acs, ep_rews = U.make_np(ep_obs, ep_advs, ep_logps, 
                                                                                    ep_target_vals, ep_acs, ep_rews)
            ep_advs.reshape(-1)
            ep_target_vals.reshape(-1)
            ep_advs = (ep_advs - np.mean(ep_advs))/ (1e-8+ np.std(ep_advs))
             
            if i % 50 == 13 and DEBUG:
                perm = np.random.choice(len(ep_advs), size=20)
                print('Some targets', ep_target_vals[perm])
                print('Some preds', critic.value(ep_obs[perm], sess=sess) )
                print('Some logps', ep_logps[perm])
          
               
            cir_loss, ev_before, ev_after = train_ciritic(critic=critic, sess=sess, obs=ep_obs, targets=ep_target_vals)
            act_loss = train_actor(actor=actor, sess=sess, obs=ep_obs, advs=ep_advs, acs=ep_acs, logps=ep_logps)

            if args.tboard:
                summ, _, _ = sess.run([merged, actor.ac, critic.v], feed_dict={actor.ob: ep_obs, critic.obs:ep_obs})
                writer.add_summary(summ,i)
            #logz
            act_lr, cur_beta, cur_gamma = actor.get_opt_param(sess)
            kl_dist = actor.get_kl(sess=sess, obs=ep_obs, logp_feeds=ep_logps, acs=ep_acs)

            #updates the learning rate based on the observed kl_distance and its multiplicative distance to desired_kl
            if kl_dist < desired_kl/4:
                new_lr = min(max_lr,act_lr*1.5)
                actor.set_opt_param(sess=sess, new_lr=new_lr)
            elif kl_dist > desired_kl * 4:
                new_lr = max(min_lr,act_lr/1.5)
                actor.set_opt_param(sess=sess, new_lr=new_lr)

            if log_gamma_schedule.update_time(i):
                new_gamma = np.power(10., log_gamma_schedule.val(i))
                actor.set_opt_param(sess=sess, new_gamma=new_gamma)
                print('\nUpdated gamma from %.4f to %.4f.' % (cur_gamma, new_gamma))
            if log_beta_schedule.update_time(i):
                new_beta = np.power(10., log_beta_schedule.val(i))
                actor.set_opt_param(sess=sess, new_beta=new_beta)
                print('Updated beta from %.4f to %.4f.' % (cur_beta, new_beta))
           
            logger(i, act_loss=act_loss, circ_loss=np.sqrt(cir_loss), avg_rew=avg_rew, ev_before=ev_before, 
                   ev_after=ev_after, act_lr=act_lr, print_tog= (i %20) == 0, kl_dist=kl_dist, avg_ent=avg_ent)  
            if i % 100 == 50:
                logger.flush() 

            if  i%args.save_every == 0:   
                saver.save(sess, CHECKPOINT_PATH, global_step=tot_rolls)
            tot_rolls += rolls

    del logger

if __name__ == '__main__':
    main()
