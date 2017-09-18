import tensorflow as tf
import numpy as np
import gym
import util as U
import argparse
from scipy import signal

xavier = tf.contrib.layers.xavier_initializer()
SCALE = 0.1
def xav(*t):
    return SCALE * xavier(*t)

def variable_summaries(var, name=''):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name+'summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    #tf.summary.histogram('histogram', var)

def discount(x, gamma):
    ret = np.array(signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1])
    return ret

def lrelu(x, alpha=0.2):
    return (1-alpha) * tf.nn.relu(x) + alpha * x


def dense(name, inp, in_dim, out_dim, activation=None, initializer=xavier, summary=True):
    W = tf.get_variable(shape=[in_dim, out_dim], initializer=xavier, name=name+'weight', dtype=tf.float32)
    b = tf.get_variable(initializer= tf.constant([0.0]*out_dim), name=name+'bias', dtype=tf.float32)
    if summary:
        variable_summaries(W, name+'weight')
        variable_summaries(b, name+'bias')
    if activation is not None:
        return activation(tf.matmul(inp, W) + b)
    else:
        return tf.matmul(inp, W) + b

def var_accounted_for(target, pred):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    return 1- (np.var(target-pred)/ (np.var(target)+1e-8))
    
    #target = target /  np.sqrt(np.sum(np.square(target)))
    #pred = pred/  np.sqrt(np.sum(np.square(pred)))
    #return np.sum(target * pred)

class Actor(object):
    def __init__(self, num_ob_feat, num_ac, init_lr = 0.005, init_beta = 1, 
                       ac_scale=2.0, ob_scale=[1.0/np.sqrt(3), 1.0/np.sqrt(3), 8.0/np.sqrt(3)]):
        
        ob = tf.placeholder(shape=[None, num_ob_feat], dtype=tf.float32)
        #code for sampling a new action
        x = dense(name='first_layer', inp=ob, in_dim=num_ob_feat, out_dim=128, activation=tf.nn.relu)
        x1 = dense(name='second_layer', inp=x, in_dim=128, out_dim=64, activation=tf.nn.relu)
        mu = dense(name='third_layer', inp=x1, in_dim=64, out_dim=num_ac, initializer=xav)
        #log_std = dense(name='log', inp = ob, in_dim=num_ob_feat, out_dim=num_ac, initializer=xav)
        log_std = tf.Variable(initial_value=tf.constant([0.0]* num_ac), name='log_std')
        
        # Think about action range issue and obs range.
        std = tf.exp(log_std)+ 1e-8
        ac = mu + tf.random_normal(shape=tf.shape(mu)) * std
        logp =  tf.reduce_sum(- tf.square((ac - mu)/std)/2.0, axis=1) - tf.reduce_mean(log_std)
        self.ac, self.logp, self.ob = ac, logp, ob
        #code for computing reward loss given the advantage of each action
        adv = tf.placeholder(shape=[None], dtype=tf.float32)
        ac_hist = tf.placeholder(shape=[None, num_ac], dtype=tf.float32)
        logp_newpolicy_oldac = tf.reduce_sum(- tf.square( (ac_hist - mu) / std)/2.0, axis=1) - tf.reduce_mean(log_std)
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
        self.loss = self.rew_loss  +  self.p_dist
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        #Debugging stuff
        self.printer = tf.constant(0.0) 
        self.printer = tf.Print(self.printer, data=['Actor Data', tf.reduce_mean(std), tf.reduce_mean(logp), tf.nn.moments(ac, axes=[0,1])])
        self.printer = tf.Print(self.printer, data=['Actor layer data', tf.reduce_mean(x), tf.reduce_mean(x1), tf.reduce_mean(mu)])

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
    def __init__(self, num_ob_feat, init_lr=0.005, ac_scale=2.0, ob_scale=[1.0/np.sqrt(3), 1.0/np.sqrt(3), 8.0/np.sqrt(3)]):
        num_ob_feat = (num_ob_feat+1) * 2
        with tf.variable_scope('Critic'):
            obs = tf.placeholder(shape=[None, num_ob_feat], dtype=tf.float32)
            x = dense(name='first_layer', inp=obs,  activation= tf.nn.relu, in_dim=num_ob_feat, out_dim=64)
            x1 = dense(name='second_layer', inp=x, activation= tf.nn.relu, in_dim=64, out_dim=32)
            #x2 = dense(name='third_layer', inp=x1, activation= tf.nn.relu, in_dim=16, out_dim=16)
            v = dense(name='value', inp=x1, initializer=xav, in_dim=32, out_dim=1)
            v_ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.square(v-v_))
            self.v, self.v_, self.obs =  v, v_, obs
        #optimization parts
            self.lr = tf.Variable(initial_value=init_lr,dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.printer = tf.constant(0.0)    
            self.printer = tf.Print(self.printer, data=[tf.reduce_mean(x), tf.reduce_mean(x1), tf.reduce_mean(v)])
        
    def value(self, obs, sess):
        return sess.run(self.v, feed_dict={self.obs:obs})
    
    def optimize(self, obs, targets, sess):
        feed_dict={self.obs:obs, self.v_: targets}
        return sess.run([self.loss, self.opt], feed_dict=feed_dict)
    
    def set_opt_param(self, new_lr, sess):
        return sess.run(self.lr, feed_dict={self.lr:new_lr})
    def printoo(self, obs, sess):
        return sess.run([self.printer], feed_dict={self.obs: obs})



class PathAdv(object):
    def __init__(self, gamma=0.98, look_ahead=30):
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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--outdir", default='log.txt')
parser.add_argument("--animate", default=False, action='store_true')
args = parser.parse_args()
TB = args.outdir != 'log.txt'
LOG_FILE = args.outdir
ANIMATE = args.animate
ROLLS_PER_EPISODE = 10
MAX_PATH_LENGTH = 800
ITER = 100000
BATCH = 32
MULT = 5
LOG_ROUND = 10

env = gym.make('Pendulum-v0')
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.shape[0]
actor = Actor(num_ac=ac_dim, num_ob_feat=ob_dim)
critic = Critic(num_ob_feat=ob_dim)
rew_to_advs =  PathAdv(gamma=0.97, look_ahead=5)
logger = U.Logger(logfile=LOG_FILE)



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
        tot_rews = 0
        for j in range(ROLLS_PER_EPISODE):
            path = rollout(env=env, sess= sess, policy=actor.act, 
                           max_path_length=MAX_PATH_LENGTH, 
                           render=j ==0 and  i % 20 == 0 and ANIMATE)
            obs_aug = ob_feature_augment(path['obs'])
            ep_unproc_obs += path['obs'][:-1]
            ep_obs += obs_aug[:-1]
            ep_logps += path['logps']
            ep_acs += path['acs']
            obs_vals = critic.value(obs=obs_aug, sess=sess).reshape(-1)
            target_val, advs = rew_to_advs(rews=path['rews'], terminal=path['terimanted'], vals=obs_vals)
            #target_val = discount(path['rews'], gamma=0.97)
            #advs = target_val - obs_vals[:-1]
            ep_target_vals += list(target_val)
            ep_advs += list(advs)
            ep_rews += path['rews']
            tot_rews += sum(path['rews'])

            #if j ==0 and i%10 ==0:
                #actor.printoo(obs=path['obs'], sess=sess)
                #critic.printoo(obs=obs_aug, sess=sess)

        avg_rew = float(tot_rews)/ ROLLS_PER_EPISODE  
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
        # Do EVBefore and EVAFTER
        # correct np.std code
        # For now we forget about adjusting beta and gamma and stuff
        cir_loss, ev_before, ev_after = train_ciritic(critic=critic, sess=sess, batch_size=BATCH, repeat= MULT, obs=ep_obs, targets=ep_target_vals)
        act_loss1, act_loss2, act_loss_full = train_actor(actor=actor, sess=sess, 
                                                         batch_size=BATCH, repeat=MULT, obs=ep_unproc_obs, 
                                                          advs=ep_advs, acs=ep_acs, logps=ep_logps)
        if TB:
            summ, _, _ = sess.run([merged, actor.ac, critic.v], feed_dict={actor.ob: ep_unproc_obs[:1000], critic.obs:ep_obs[:1000]})
            writer.add_summary(summ,i)
        #logz
        logger(i, act_loss1=act_loss1, act_loss2=act_loss2,  act_loss_full=act_loss_full, circ_loss=np.sqrt(cir_loss), 
                avg_rew=avg_rew, ev_before=ev_before, ev_after=ev_after, print_tog= (i %20) == 0)
        if i % 100 == 50:
            logger.write()


del logger
