import tensorflow as tf
import numpy as np
import gym
import util as U

from scipy import signal
from time import sleep
import policies as pol

LOG_ROUND = 10
MAX_ITERS = 1e7
MAX_LR, MIN_LR = .1 , 1e-6

class Framer(object):
    """
    Ceates the augmentd obs features from the bare observations. Any obs fed to Actor & Critics nets must go through Framer. 
    Currently it simply concatenates a few (frame_num) recent bare obs together. 
    So ob_dim = env.observation_space.shape[0] * frame_num

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
    Gather an episode of experiences by running the environment. Continues until env.done is True
    or length of episode exceed max_path_length
    """
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
        sum_ents += ent
        logps.append(logp)
    path = {'rews': rews, 'obs':obs, 'acs':acs, 'terminated': done, 'logps':logps, 'entropy':sum_ents}
    return path

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

def train_ciritic(critic, sess, obs, targets):
    assert len(obs) == len(targets)
    pre_preds = critic.value(obs, sess=sess)
    ev_before = U.var_accounted_for(pred=pre_preds, target=targets)
    loss, _ = critic.optimize(obs=obs, targets=targets, sess=sess)
    return loss, ev_before


def train_actor(actor, sess, obs, advs, logps, acs, rolls):
    assert len(obs) == len(advs)
    assert len(advs) == len(acs)
    loss, _ = actor.optimize(sess=sess, obs=obs, acs=acs, advs=advs, logps=logps)
    return loss

def test_process(env_id, random_seed, stack_frames, model_path, num_episodes, animate=True):
	"""
	A minimalistic version of process_fn that is used instead when just testing the model.
	Omits any use of cluster management, learning rate scheduling, etc. 
	"""
    env, MAX_PATH_LENGTH, _ = get_roll_params(env_id)
    framer = Framer(frame_num=stack_frames)
    ob_dim = env.observation_space.shape[0] * stack_frames
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        act_type = 'disc'
        ac_dim, ac_scale = env.action_space.n, None
    else:
        act_type = 'cont'
        ac_dim, ac_scale = env.action_space.shape[0], np.maximum(env.action_space.high, np.abs(env.action_space.low))
    actor = pol.Actor(name='global_actor', num_ob_feat=ob_dim, num_ac=ac_dim, act_type=act_type, ac_scale=ac_scale) 
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess=sess, save_path=model_path)
        avg_rew = 0
        for i in range(num_episodes):
            path = rollout(env=env, sess= sess, policy=actor.act, max_path_length=MAX_PATH_LENGTH, framer=framer, render= animate)
            rew = np.sum(path['rews'])
            print("Iteration {}".format(i))
            print("Reward {}".format(rew))
            print("Episode Length {}\n".format(len(path['rews'])))
            avg_rew += rew/float(num_episodes)
        print('Average reward over {} was {}'.format(num_episodes, avg_rew))


def process_fn(cluster, task_id, job, env_id, logger, save_path, stdout_freq, random_seed=12321, gamma=0.98, 
               look_ahead=40, stack_frames=3, animate=False, save_every=600, desired_kl=0.002, TB_log=False,
               run_mode='train', checkpoint_basename='model',):
   
    num_ps, num_workers = len(cluster['ps']), len(cluster['worker'])

    cluster = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster, job_name=job, task_index=task_id)

    if job == 'ps':
        server.join()
    else:
        env, MAX_PATH_LENGTH, EP_LENGTH_STOP = get_roll_params(env_id)
        framer = Framer(frame_num=stack_frames)
        ob_dim = env.observation_space.shape[0] * stack_frames
        rew_to_advs =  PathAdv(gamma=gamma, look_ahead=look_ahead)
        is_chief = (task_id == 0)
        log_gamma_schedule = U.LinearSchedule(init_t=100, end_t=3000, init_val=-2, end_val=-8, update_every_t=100) #This is base 10
        log_beta_schedule = U.LinearSchedule(init_t=100, end_t=3000, init_val=0, end_val=-4, update_every_t=100) #This is base 10
        DEBUG = run_mode == "debug-full" or (run_mode == "debug-light" and is_chief)
        #if not is_chief:   #heuristic for output cleanness
        #    stdout_freq = min(6, num_workers) * stdout_freq
        
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
        with tf.device(tf.train.replica_device_setter(cluster=cluster,             #Makes sure global variables defined in  
                                                      worker_device=worker_device, #this contexts are synced across processes
                                                      ps_strategy=U.greedy_ps_strategy(ps_tasks=num_ps))):

            global_critic = pol.Critic(num_ob_feat=ob_dim, name='global_critic')
            global_actor = pol.Actor(name='global_actor', num_ob_feat=ob_dim, num_ac=ac_dim, act_type=act_type, ac_scale=ac_scale)            
            saver = tf.train.Saver(max_to_keep=3) #saver defined here so it only saves the global models vars
            global_step_tensor = tf.train.get_or_create_global_step()  

        with tf.device(worker_device):
            local_critic = pol.Critic(num_ob_feat=ob_dim, name='local_critic_{}'.format(task_id), global_critic=global_critic)
            local_actor = pol.Actor(num_ob_feat=ob_dim, num_ac=ac_dim, act_type=act_type, name='local_actor_{}'.format(task_id), 
                                    global_actor=global_actor, global_step=global_step_tensor ) 

        local_init_op = tf.global_variables_initializer()
        with tf.Session(server.target) as sess:
                sess.run(local_init_op)
        print('\n\nREACHING THE MAIN LOOP OF WORKER %d\n' % task_id)
        max_lr, min_lr = MAX_LR, MIN_LR
        kl_dist, i = 0., 0
        saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=save_path, save_steps=save_every, 
                                                  checkpoint_basename=checkpoint_basename, saver=saver)

        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief, chief_only_hooks=[saver_hook]) as sess:
            gstep = tf.train.global_step(sess, global_step_tensor)
            
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
                        print('Total Steps %d' % gstep)
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
                
                if DEBUG and i%50 == 13:
                    perm = np.random.choice(len(ep_advs), size=20)
                    print('Some preds', local_critic.value(sess=sess, obs=ep_obs[perm]))
                    print('Some target vals', ep_target_vals[perm])
                    print('Some logps', ep_logps[perm])
                    local_actor.printoo(obs=ep_obs, sess=sess)
                    local_critic.printoo(obs=ep_obs, sess=sess)     
                cir_loss, ev_before = train_ciritic(critic=local_critic, sess=sess, obs=ep_obs, targets=ep_target_vals,)
                act_loss = train_actor(actor=local_actor, sess=sess,  obs=ep_obs, advs=ep_advs, acs=ep_acs, logps=ep_logps, rolls=rolls) 
                local_actor.sync_w_global(sess)
                local_critic.sync_w_global(sess)             
                ev_after =  U.var_accounted_for(target=ep_target_vals, pred=local_critic.value(sess=sess, obs=ep_obs))
                kl_dist = local_actor.get_kl(sess=sess, logp_feeds=ep_logps, obs=ep_obs, acs=ep_acs)
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

                logger(i, act_loss=act_loss, worker_id=task_id, act_lr=act_lr, kl_dist=kl_dist, circ_loss=np.sqrt(cir_loss), avg_rew=avg_rew, 
                ev_before=ev_before, ev_after=ev_after, print_tog= (i%stdout_freq) == 0, avg_ent=avg_ent)
                if i % 100 == 50:
                    logger.flush()
                i += 1
                gstep = tf.train.global_step(sess, global_step_tensor)
                

        del logger


#all_vars = tf.trainable_variables()
#u = [v for v in all_vars if 'Critic' in v.name]
