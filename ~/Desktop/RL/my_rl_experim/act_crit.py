import tensorflow as tf
import numpy as np
import gym
import util as U

xavier = tf.contrib.layers.xavier_initializer()
SCALE = 0.1
def xav(*t):
    return SCALE * xavier(*t)


def dense(name, input, in_dim, out_dim, activation, initializer=xavier):
    W = tf.get_variable(shape=[in_dim, out_dim], initializer=xavier, name=name)
    b = tf.get_variable(shape=[out_dim], initializer= tf.constant(0.0), name=name)
    return tf.matmul(input, W) + b 

class Actor(object):
    def __init__(self, num_ob_feat, num_ac, init_lr = 0.005, init_beta = 1):
        
        ob = tf.placeholder(shape=[None, num_ob_feat], dtype=tf.float32)
        #code for sampling a new action
        x = tf.contrib.layers.fully_connected(inputs = ob, activation_fn=tf.nn.relu, num_outputs=64)
        x1 = tf.contrib.layers.fully_connected(inputs = x, activation_fn=tf.nn.relu, num_outputs=32)
        mu = tf.contrib.layers.fully_connected(inputs = x1, num_outputs=num_ac)
        log_std = tf.contrib.layers.fully_connected(inputs = ob, num_outputs=num_ac)
        
        # Think about action range issue and obs range.
        std = tf.exp(log_std)+ 1e-8
        ac = mu + tf.random_normal(shape=tf.shape(mu)) * std
        logp =  tf.reduce_sum(- tf.square((ac - mu)/std)/2.0 - log_std, axis=1)
        self.ac, self.logp, self.ob = ac, logp, ob
        #code for computing reward loss given the advantage of each action
        adv = tf.placeholder(shape=[None], dtype=tf.float32)
        ac_hist = tf.placeholder(shape=[None, num_ac], dtype=tf.float32)
        logp_newpolicy_oldac = tf.reduce_sum(- tf.square( (ac_hist - mu) / (2.0*std)) - log_std, axis=1)
        rew_loss = -tf.reduce_mean(adv * logp_newpolicy_oldac)
        self.rew_loss, self.adv, self.ac_hist = rew_loss, adv, ac_hist 
        #some measure of distance between new and old policy
        logp_feed = tf.placeholder(shape=[None], dtype=tf.float32)   
        p_dist = tf.reduce_mean(tf.square(logp_feed-logp_newpolicy_oldac))   
        self.p_dist, self.logp_feed = p_dist, logp_feed
        # Actual loss stuff...
        # Can try to add action entropy here too
        self.beta = tf.Variable(initial_value=init_beta, dtype=tf.float32, trainable=False)
        self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)
        self.loss = rew_loss + self.beta * p_dist
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        #Debugging stuff
        self.printer = tf.constant(0.0) 
        self.printer = tf.Print(self.printer, data=[tf.reduce_mean(std), tf.reduce_mean(logp), tf.nn.moments(ac, axes=[0,1])])
       


    def act(self, ob, sess):
        ob = np.array(ob)
        if len(ob.shape) != 2:
            ob = ob[None]
        ac, logp =  sess.run([self.ac, self.logp], feed_dict={self.ob:ob})
        return ac[0], logp[0]
    
    def optimize(self, acs, obs, advs, logps, sess):
        feed_dict= {self.adv: advs,self.ac_hist:acs, self.ob:obs, self.logp_feed:logps}
        return sess.run([self.rew_loss, self.p_dist, self.loss, self.opt], feed_dict=feed_dict)
    
    def set_opt_param(self, sess, new_lr=None, new_beta=None):
        feed_dict = dict()
        if new_beta is not None:
            feed_dict[self.beta] = new_beta
        if new_lr is not None:
            feed_dict[self.lr] = new_lr
        return sess.run([self.lr,self.beta],feed_dict=feed_dict)

    def printoo(self, obs, sess):
        return sess.run([self.printer], feed_dict={self.ob: obs})





class Critic(object):
    def __init__(self, num_ob_feat, init_lr=0.005):
        obs = tf.placeholder(shape=[None, num_ob_feat], dtype=tf.float32)
        x = tf.contrib.layers.fully_connected(inputs = obs,  activation_fn= tf.nn.relu, 
                                              weights_initializer=xavier,num_outputs=32)
        x1 = tf.contrib.layers.fully_connected(inputs = x, activation_fn= tf.nn.relu, num_outputs=32)
        x2 = tf.contrib.layers.fully_connected(inputs = x1, activation_fn= tf.nn.relu, num_outputs=16)
        v = tf.contrib.layers.fully_connected(inputs = x2, activation_fn= None, num_outputs=1)

        v_ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.square(v-v_))
        self.v, self.v_, self.obs =  v, v_, obs
        #optimization parts
        self.lr = tf.Variable(initial_value=init_lr,dtype=tf.float32, trainable=False)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
    def value(self, obs, sess):
        return sess.run(self.v, feed_dict={self.obs:obs})
    
    def optimize(self, obs, targets, sess):
        feed_dict={self.obs:obs, self.v_: targets}
        return sess.run([self.loss, self.opt], feed_dict=feed_dict)
    
    def set_opt_param(self, new_lr, sess):
        return sess.run(self.lr, feed_dict={self.lr:new_lr})



class PathAdv(object):
    def __init__(self, gamma=0.98, look_ahead=40):
        self.reset(gamma, look_ahead)
    
    def __call__(self, rews, vals, terminal):
        
        action_val =np.convolve(rews[::-1], self.kern)[len(rews)-1::-1]
        assert len(rews) == len(action_val)   
        assert len(vals) == len(rews) + 1
        max_id = len(vals) -1 if terminal else len(vals) - 2 
        advs = np.zeros(len(rews))
        for i in range(len(action_val)):
            termin_id = min(i+self.look_ahead, max_id)
            action_val[i] += np.power(self.gamma, termin_id-i) * vals[termin_id]
            advs[i] = action_val[i]- vals[i]
        return list(action_val), list(advs)        
        
    def reset(self, gamma, look_ahead):
        self.kern = [np.power(gamma, k) for k in range(look_ahead)]
        self.look_ahead = look_ahead
        self.gamma = gamma
    


def rollout(env, sess, policy, max_path_length=100, render=False):
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
        ac, logp = policy(ob, sess=sess)
        ob, rew, done, _ = env.step(ac)
        obs.append(ob)
        rews.append(rew)
        acs.append(ac)
        logps.append(logp)
    path = {'rews': rews, 'obs':obs, 'acs':acs, 'terimanted': done, 'logps':logps}
    return path



def train_ciritic(critic, sess, batch_size, repeat, obs, targets):
    assert len(obs) == len(targets)
    n = len(obs)
    perm = np.random.permutation(n)
    obs = obs[perm]
    targets = targets[perm]
    tot_loss = 0.0
    l = int(repeat*len(obs)/n+1)
    for i in range(l):
        low = (i* batch_size) % n
        high = min(low+batch_size, n)
        loss, _= critic.optimize(obs=obs[low:high], targets=targets[low:high],sess=sess)
        tot_loss += loss
    return tot_loss/ l



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
    
    return (rew_loss/l, tot_p_dist/l, tot_comb_loss/l)

LOG_FILE = 'log.txt'
ROLLS_PER_EPISODE = 10
MAX_PATH_LENGTH = 800
ITER = 40000
BATCH = 32
MULT = 5
ANIMATE = True
LOG_ROUND = 10

env = gym.make('Pendulum-v0')
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.shape[0]
actor = Actor(num_ac=ac_dim, num_ob_feat=ob_dim)
critic = Critic(num_ob_feat=ob_dim)
rew_to_advs =  PathAdv(gamma=0.97, look_ahead=30)
logger = U.Logger(logfile=LOG_FILE)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(ITER):
        ep_obs, ep_advs, ep_logps, ep_target_vals, ep_acs = [], [], [], [], []
        tot_rews = 0
        for j in range(ROLLS_PER_EPISODE):
            path = rollout(env=env, sess= sess, policy=actor.act, 
                           max_path_length=MAX_PATH_LENGTH, 
                           render=j ==0 and  i % 30 == 0 and ANIMATE)
            ep_obs += path['obs'][:-1]
            ep_logps += path['logps']
            ep_acs += path['acs']
            obs_vals = critic.value(obs=path['obs'], sess=sess)
            target_val, advs = rew_to_advs(rews=path['rews'], terminal=path['terimanted'], vals=obs_vals)
            ep_target_vals += target_val
            ep_advs += advs
            tot_rews += sum(path['rews'])
            if j ==0 and i%10 ==0:
                actor.printoo(obs=path['obs'], sess=sess)




        avg_rew = float(tot_rews)/ (ROLLS_PER_EPISODE+1e-8)  
        ep_obs, ep_advs, ep_logps, ep_target_vals, ep_acs = U.make_np(ep_obs, ep_advs, ep_logps, ep_target_vals, ep_acs)
        ep_advs = (ep_advs - np.mean(ep_advs)) / (np.std(ep_advs)+1e-8)
        # Do EVBefore and EVAFTER
        # correct np.std code
        # For now we forget about adjusting beta and gamma and stuff
        cir_loss = train_ciritic(critic=critic, sess=sess, batch_size=BATCH, repeat= MULT, obs=ep_obs, targets=ep_target_vals)
        act_loss1, act_loss2, act_loss_full = train_actor(actor=actor, sess=sess, 
                                                         batch_size=BATCH, repeat=MULT, obs=ep_obs, 
                                                          advs=ep_advs, acs=ep_acs, logps=ep_logps)

        #logz
        logger(i, act_loss1, act_loss2, act_loss_full, cir_loss, avg_rew, print_tog= (i %10) == 0)
        if i % 100 == 10:
            logger.write()

del logger
