import tensorflow as tf
import numpy as np
import gym
import util as U

from scipy import signal

import Policies as pol

MAX_PATH_LENGTH = 400
LOG_ROUND = 10
EP_LENGTH_STOP = 1200
MAX_ITERS = 1e7
DESIRED_KL = 0.002
MAX_LR, MIN_LR = .1 , 1e-6


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
    sum_ents = 0
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
        logps.append(logp)
        sum_ents += ent
    path = {'rews': rews, 'obs':obs, 'acs':acs, 'terminated': done, 'logps':logps, 'entropy': sum_ents}
    return path


def train_ciritic(critic, sess, obs, targets):
    assert len(obs) == len(targets)
    ev_before = U.var_accounted_for(obs=obs, target=targets, sess=sess, critic=critic)
    loss, _ = critic.optimize(obs=obs, targets=targets, sess=sess)
    return loss, ev_before


def train_actor(actor, sess, obs, advs, logps, acs, rolls):
    assert len(obs) == len(advs)
    assert len(advs) == len(acs)
    loss, _ = actor.optimize(sess=sess, obs=obs, acs=acs, advs=advs, logps=logps)
    gstep = actor.update_global_step(sess=sess, batch_size=rolls)
    return loss, gstep


def process_fn(cluster, task_id, job, env_id, logger, save_path, random_seed=12321, gamma=0.98, look_ahead=40, 
               stack_frames=3, animate=False, TB_log=False, ):

    cluster = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster, job_name=job, task_index=task_id)

    if job == 'ps':
        server.join()
    else:
        env = gym.make(env_id)
        framer = Framer(frame_num=stack_frames)
        ob_dim = env.observation_space.shape[0] * stack_frames
        rew_to_advs =  PathAdv(gamma=gamma, look_ahead=look_ahead)
        is_chief = (task_id == 0)
        log_gamma_schedule = U.LinearSchedule(init_t=100, end_t=3000, init_val=-2, end_val=-8, update_every_t=100) #This is base 10
        log_beta_schedule = U.LinearSchedule(init_t=100, end_t=3000, init_val=0, end_val=-4, update_every_t=100) #This is base 10
        
        np.random.seed(random_seed)
        env.seed(random_seed)
        if type(env.action_space) == gym.spaces.discrete.Discrete:
            act_type = 'disc'
            ac_dim, ac_scale = env.action_space.n, None
        else:
            act_type = 'cont'
            ac_dim, ac_scale = env.action_space.shape[0], np.maximum(env.action_space.high, np.abs(env.action_space.low))
        if is_chief:
            print('Initilizing chief. Envirnoment action type {}.'.format(act_type))

        worker_device = '/job:worker/task:{}/cpu:0'.format(task_id)
        #ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy() 
        with tf.device(tf.train.replica_device_setter(
            worker_device=worker_device,
            cluster=cluster,)):
            global_critic = pol.Critic(num_ob_feat=ob_dim, name='global_critic')
            global_actor = pol.Actor(name='global_actor', num_ob_feat=ob_dim, num_ac=ac_dim, act_type=act_type, ac_scale=ac_scale)     
            
            #saver = tf.train.Saver(var_list=global_vars, max_to_keep=3)

        with tf.device(worker_device):
            local_critic = pol.Critic(num_ob_feat=ob_dim, name='local_critic_{}'.format(task_id), global_critic=global_critic)
            local_actor = pol.Actor(num_ob_feat=ob_dim, num_ac=ac_dim, act_type=act_type, name='local_actor_{}'.format(task_id), 
                                    global_actor=global_actor) 

        local_init_op = tf.global_variables_initializer()
        with tf.Session(server.target) as sess:
                sess.run(local_init_op)
        print('\n\nREACHING THE MAIN LOOP OF WORKER %d\n' % task_id)
        desired_kl, max_lr, min_lr = DESIRED_KL, MAX_LR, MIN_LR
        kl_dist= 0.

        with tf.train.MonitoredTrainingSession(master=server.target) as sess:
            i, gstep = 0, 0 
            while not sess.should_stop() and gstep < MAX_ITERS:
                ep_obs, ep_advs, ep_logps, ep_target_vals, ep_acs = [], [], [], [], []
                ep_rews = []
                tot_rews, tot_ent, rolls = 0., 0., 0

                while len(ep_rews)<EP_LENGTH_STOP:
                    path = rollout(env=env, sess= sess, policy=local_actor.act, 
                                   max_path_length=MAX_PATH_LENGTH, framer=framer,
                                   render= rolls==0 and  i % 20 == 0 and animate and is_chief)
                    obs_aug = framer.full(path['obs'])
                    ep_obs += obs_aug[:-1]
                    ep_logps += path['logps']
                    ep_acs += path['acs']
                    obs_vals = local_critic.value(obs=obs_aug, sess=sess).reshape(-1)  
                    target_val, advs = rew_to_advs(rews=path['rews'], terminal=path['terminated'], vals=obs_vals)
                    ep_target_vals += list(target_val)
                    ep_advs += list(advs)
                    ep_rews += path['rews']
                    tot_rews += sum(path['rews'])
                    tot_ent += path['entropy']
                    if rolls ==0 and i%50 ==0:
                        print('Total Rolls %d' % gstep)
                        print('Path length %d' % len(path['rews']))
                        print('Terminated {}'.format(path['terminated']))       
                    rolls +=1

                avg_rew = tot_rews/ rolls  
                ep_obs, ep_advs, ep_logps, ep_target_vals, ep_acs, ep_rews,  = U.make_np(ep_obs, ep_advs, ep_logps, 
                                                                                        ep_target_vals, ep_acs, ep_rews)
                ep_advs.reshape(-1)
                ep_target_vals.reshape(-1)
                ep_advs = (ep_advs - np.mean(ep_advs))/ (1e-8+ np.std(ep_advs))
                avg_ent = tot_ent/ float(len(ep_logps))
                
                if is_chief and i%50 == 13:
                    perm = np.random.choice(len(ep_advs), size=20)
                    print('Some preds', local_critic.value(sess=sess, obs=ep_obs[perm]))
                    print('Some target vals', ep_target_vals[perm])
                    print('Some logps', ep_logps[perm])
                    local_actor.printoo(obs=ep_obs, sess=sess)
                    local_critic.printoo(obs=ep_obs, sess=sess)
                
                cir_loss, ev_before = train_ciritic(critic=local_critic, sess=sess, obs=ep_obs, targets=ep_target_vals,)
                act_loss, gstep = train_actor(actor=local_actor, sess=sess,  obs=ep_obs, advs=ep_advs, acs=ep_acs, logps=ep_logps, rolls=rolls) 

                local_actor.sync_w_global(sess)
                local_critic.sync_w_global(sess)             
                ev_after =  U.var_accounted_for(obs=ep_obs, target=ep_target_vals, sess=sess, critic=local_critic)
                kl_dist =  local_actor.get_kl(sess=sess, logp_feeds=ep_logps, obs=ep_obs, acs=ep_acs)
                act_lr, cur_beta, cur_gamma = local_actor.get_opt_param(sess)
        
                if kl_dist < desired_kl/4:
                    new_lr = min(max_lr,act_lr*1.5)
                    local_actor.set_opt_param(sess=sess, new_lr=new_lr)
                elif kl_dist > desired_kl * 4:
                    new_lr = max(min_lr,act_lr/1.5)
                    local_actor.set_opt_param(sess=sess, new_lr=new_lr)
                if log_gamma_schedule.update_time(i):
                    new_gamma = np.power(10., log_gamma_schedule.val(i))
                    local_actor.set_opt_param(sess=sess, new_gamma=new_gamma)
                    print('Updated gamma from %.4f to %.4f.' % (cur_gamma, new_gamma))
                if log_beta_schedule.update_time(i):
                    new_beta = np.power(10., log_beta_schedule.val(i))
                    local_actor.set_opt_param(sess=sess, new_beta=new_beta)
                    print('Updated beta from %.4f to %.4f.' % (cur_beta, new_beta))

                
                logger(i, act_loss=act_loss, worker_id = task_id, act_lr=act_lr, kl_dist=kl_dist, circ_loss=np.sqrt(cir_loss), avg_rew=avg_rew, 
                ev_before=ev_before, ev_after=ev_after, print_tog= (i %20) == 0, avg_ent=avg_ent)
                if i % 100 == 50:
                    logger.write()
                i += 1
                

        del logger


#all_vars = tf.trainable_variables()
#u = [v for v in all_vars if 'Critic' in v.name]
