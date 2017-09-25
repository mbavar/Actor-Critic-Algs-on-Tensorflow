import tensorflow as tf
import numpy as np
import gym
import util as U

from scipy import signal

import Policies1 as pol

ROLLS_PER_EPISODE = 10
MAX_PATH_LENGTH = 400
ITER = 100000
BATCH = 32
MULT = 5
LOG_ROUND = 10
EP_LENGTH_STOP = 2000


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


def sync_local_to_global(local_scope, global_scope):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=local_scope)
    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=global_scope)
    for v_l, v_g in zip(local_vars, global_vars):
        print(v_l, v_g)



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
    l = int(repeat*len(obs)/n+1)
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
    obs, advs, acs, logps = obs[perm], advs[perm], acs[perm], logps[perm]
    tot_rew_loss, tot_p_dist, tot_comb_loss = 0.0, 0.0, 0.0 
    l = int(repeat*len(obs)/n+1)
    for i in range(l):
        low = (i* batch_size) % n
        high = min(low+batch_size, n)
        rew_loss, p_dist, comb_loss, _ = actor.optimize(sess=sess, obs=obs[low:high], acs=acs[low:high],  advs=advs[low:high], logps=logps[low:high])
        tot_comb_loss += comb_loss
        tot_p_dist += p_dist
        tot_rew_loss += rew_loss
    
    return (tot_rew_loss/l, tot_p_dist/l, tot_comb_loss/l)


def process_fn(cluster, task_id, job, env_id, logger, random_seed=12321, gamma=0.97, look_ahead=30, 
               stack_frames=2, animate=False, TB_log=False):

    cluster = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster, job, task_id,)
    if job == 'ps':
        server.join()
    else:
        env = gym.make(env_id)
        framer = Framer(frame_num=stack_frames)
        ob_dim = env.observation_space.shape[0] * stack_frames
        rew_to_advs =  PathAdv(gamma=gamma, look_ahead=look_ahead)
        is_chief = task_id == 0
        
        np.random.seed(random_seed)
        env.seed(random_seed)
        if type(env.action_space) == gym.spaces.discrete.Discrete:
            act_type = 'disc'
            ac_dim = env.action_space.n
        else:
            act_type = 'cont'
            ac_dim = env.action_space.shape[0]
        if is_chief:
            print('Initilizing chief. Envirnoment action type {}.'.format(act_type))

        local_critic = pol.Critic(num_ob_feat=ob_dim, name='local_critic_{}'.format(task_id))
        local_actor = pol.Actor( num_ob_feat=ob_dim, num_ac=ac_dim, act_type=act_type, name='local_actor_{}'.format(task_id)) 
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % task_id,
            cluster=cluster)):
            global_critic = pol.Critic(num_ob_feat=ob_dim, name='global_critic')
            global_actor = pol.Actor(name='global_actor', num_ob_feat=ob_dim, num_ac=ac_dim, act_type=act_type) 

        sync_local_to_global(local_scope=local_actor.name, global_scope=global_actor.name)

"""
        #merged = tf.summary.merge_all()
        #writer = tf.summary.FileWriter('./summaries/'+logger.logfile.split('_')[0], tf.get_default_graph())

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
                                   render= j==0 and  i % 20 == 0 and animate)
                    obs_aug = framer.full(path['obs'])
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
                        actor.printoo(obs=ep_obs, sess=sess)
                        critic.printoo(obs=ep_obs, sess=sess)
                        print('Path length %d' % len(path['rews']))
                        print('Terminated {}'.format(path['terminated']))
                    j +=1

                avg_rew = float(tot_rews)/ ROLLS_PER_EPISODE  
                ep_obs, ep_advs, ep_logps, ep_target_vals, ep_acs, ep_rews,  = U.make_np(ep_obs, ep_advs, ep_logps, 
                                                                                        ep_target_vals, ep_acs, ep_rews)
                ep_advs.reshape(-1)
                ep_target_vals.reshape(-1)
                ep_advs = (ep_advs - np.mean(ep_advs))/ (1e-8+ np.std(ep_advs))
                ""
                if i%10 ==0:
                    print('Advantage mean & std {}, {}'.format(np.mean(ep_advs), np.std(ep_advs)))       
                if i % 50 == 13:
                    perm = np.random.choice(len(ep_advs), size=20)
                    print('Some obs', ep_obs[perm])
                    print('Some acs', ep_obs[perm])
                    print('Some advs', ep_obs[perm])
                    print('Some rews', ep_rews[perm])
                ""
                
                cir_loss, ev_before, ev_after = train_ciritic(critic=critic, sess=sess, batch_size=BATCH, repeat= MULT, obs=ep_obs, targets=ep_target_vals)
                act_loss1, act_loss2, act_loss_full = train_actor(actor=actor, sess=sess, 
                                                                 batch_size=BATCH, repeat=MULT, obs=ep_obs, 
                                                                  advs=ep_advs, acs=ep_acs, logps=ep_logps)
                if TB_log:
                    summ, _, _ = sess.run([merged, actor.ac, critic.v], feed_dict={actor.ob: ep_obs[:1000], critic.obs:ep_obs[:1000]})
                    writer.add_summary(summ,i)
                #logz
                logger(i, act_loss1=act_loss1, act_loss2=act_loss2,  act_loss_full=act_loss_full, circ_loss=np.sqrt(cir_loss), 
                        avg_rew=avg_rew, ev_before=ev_before, ev_after=ev_after, print_tog= (i %20) == 0)
                if i % 100 == 50:
                    logger.write()


        del logger
"""

#all_vars = tf.trainable_variables()
#u = [v for v in all_vars if 'Critic' in v.name]