import tensorflow as tf
import numpy as np
SCALE = 0.1
ID_FN = lambda x : x
def xav(*t):
    return SCALE * xavier(*t)
xavier = tf.contrib.layers.xavier_initializer()

def lrelu(x, alpha=0.2):
    return (1-alpha) * tf.nn.relu(x) + alpha * x

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

def fancy_clip(grad, low, high):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, low, high)

def normalized_column_initializer(shape, dtype, partition_info):
    u = tf.random_normal(shape=shape,dtype=tf.float32)
    scale =  tf.sqrt(tf.reduce_sum(tf.square(u), axis=0))/0.1
    return u/scale

class Actor(object):
    """
    A MLP Actor with entropy and KL regularizations. It should recieve num_frames * env.observation_space.shape[0] features
    Works both with continuous and discrete action spaces.
    """
    def __init__(self, name, num_ob_feat, num_ac, act_type='cont', init_lr = 0.005, init_beta = 1, init_gamma=0.01,
                       ac_scale=2., ob_scaler=ID_FN,  global_actor=None, global_step=None):
        with tf.variable_scope(name):
            self.ob = tf.placeholder(shape=[None, num_ob_feat], dtype=tf.float32)
            obs_scaled = ob_scaler(self.ob)
            x = tf.layers.dense(name='first_layer', inputs=obs_scaled, units=128, activation=lrelu, kernel_initializer=xavier)
            x1 = tf.layers.dense(name='second_layer',  inputs=x, units=128, activation=lrelu, kernel_initializer=xavier)
            x2 = tf.layers.dense(name='third_layer',  inputs=x1, units=64, activation=lrelu, kernel_initializer=xavier)
            self.adv = tf.placeholder(shape=[None], dtype=tf.float32)
            self.logp_feed = tf.placeholder(shape=[None], dtype=tf.float32)
            if act_type == 'cont':            
                mu = tf.layers.dense(name='mu_layer', inputs=x2, units=num_ac, activation=tf.nn.tanh) * ac_scale
                #log_std = dense(name='log', inp = ob, in_dim=num_ob_feat, out_dim=num_ac, initializer=xav)
                log_std = tf.Variable(initial_value=tf.constant([0.0]* num_ac), name='log_std')
                log_std = tf.clip_by_value(log_std, -2.5, 2.5)
                std = tf.exp(log_std)
                dist = tf.distributions.Normal(loc=mu, scale=std)
                self.ac = dist.sample()  #at sampling we are always sampling 1
                self.logp =  tf.reduce_sum(dist.log_prob(self.ac), axis=1)
                self.entropy = entropy = tf.reduce_sum(dist.entropy(), axis=1)
                self.ac_hist = tf.placeholder(shape=[None, num_ac], dtype=tf.float32)
                logp_newpolicy_oldac = tf.reduce_sum(dist.log_prob(self.ac_hist), axis=1)
                printing_data = ['Actor Baic Data', tf.reduce_mean(std), tf.reduce_mean(self.logp), tf.nn.moments(self.ac, axes=[0,1])]
            else:
                logits = tf.layers.dense(name='logits', inputs=x2, units=num_ac, kernel_initializer=normalized_column_initializer)
                self.ac = ac = tf.cast(tf.reshape(tf.multinomial(logits=logits, num_samples=1),[-1]), dtype=tf.int32)
                logps = tf.nn.log_softmax(logits+1e-8)
                self.entropy = entropy = - tf.reduce_sum(tf.nn.softmax(logits) * logps, axis=1)
                self.logp = tf.gather_nd(params=logps, indices=tf.stack([tf.range(tf.shape(ac)[0]), ac], axis=1))
                self.ac_hist = ac_hist = tf.placeholder(shape=[None], dtype=tf.int32)
                logp_newpolicy_oldac = tf.gather_nd(params=logps, indices=tf.stack([tf.range(tf.shape(ac_hist)[0]), ac_hist], axis=1))
                mu = logits
                printing_data = ['Actor Basic Data', tf.reduce_mean(self.logp), tf.reduce_mean(logps), 
                                  tf.reduce_mean(tf.reduce_max(logps, axis=1)), tf.reduce_mean(tf.reduce_min(logps, axis=1)) ]

            self.rew_loss = -tf.reduce_mean(self.adv * logp_newpolicy_oldac) 
            self.oldnew_kl = oldnew_kl = tf.reduce_mean(tf.square(self.logp_feed-logp_newpolicy_oldac))   
            # Actual loss stuff. Can try to add action entropy here too
            self.lr = lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)
            self.gamma = gamma = tf.Variable(initial_value=init_gamma, dtype=tf.float32, trainable=False)
            self.beta = beta = tf.Variable(initial_value=init_beta, dtype=tf.float32, trainable=False)
            self.loss = loss = self.rew_loss + beta * oldnew_kl - gamma * tf.reduce_mean(entropy)

            self.my_optimizer = adam = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.my_vars = my_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            grads_and_vars = adam.compute_gradients(self.loss, var_list=my_vars)
            grads_clipped = [tf.clip_by_value(g, -.1, .1) for g,_ in grads_and_vars]

            if global_actor is not None:
                grads_and_vars = zip(grads_clipped, global_actor.my_vars)
                opt_op =adam.apply_gradients(grads_and_vars, global_step=global_step)
                def optimize(acs, obs, advs, logps, sess):
                    feed_dict= {self.adv: advs,self.ac_hist:acs, self.ob:obs, self.logp_feed:logps}
                    return sess.run([loss, opt_op], feed_dict=feed_dict)   
                self.optimize = optimize
                self._global_syncer = [v_l.assign(v_g) for v_l, v_g in zip(my_vars, global_actor.my_vars)]
                self.new_val_placeholder = tf.placeholder(shape=(), dtype=tf.float32)
                self.beta_update = beta.assign(self.new_val_placeholder)
                self.gamma_update = gamma.assign(self.new_val_placeholder)
                self.lr_update = lr.assign(self.new_val_placeholder)
                
            #Debugging stuff
            printing_data2 = ["Actor Variable data"]+ [tf.reduce_mean(v) for v in self.my_vars]
            self.printer = tf.constant(0.0) 
            self.printer = tf.Print(self.printer, data=printing_data)
            self.printer = tf.Print(self.printer, data=printing_data2)
            
    def get_kl(self, sess, logp_feeds, obs, acs):
        feed_dict = {self.logp_feed:logp_feeds, self.ac_hist:acs, self.ob:obs}
        return sess.run(self.oldnew_kl, feed_dict=feed_dict)

    def act(self, ob, sess):
        ob = np.array(ob)
        if len(ob.shape) != 2:
            ob = ob[None]
        ac, logp, ent =  sess.run([self.ac, self.logp, self.entropy], feed_dict={self.ob:ob})
        return ac[0], logp[0], ent[0]

    def sync_w_global(self, sess):
        return sess.run(self._global_syncer)
       
    def set_opt_param(self, sess, new_lr=None, new_beta=None, new_gamma=None):
        feed_dict = dict()
        if new_beta is not None:
            sess.run(self.beta_update, feed_dict={self.new_val_placeholder:new_beta})
        if new_lr is not None:
            sess.run(self.lr_update, feed_dict={self.new_val_placeholder:new_lr})
        if new_gamma is not None:
            sess.run(self.gamma_update, feed_dict={self.new_val_placeholder:new_gamma})
        return self.get_opt_param(sess)

    def get_opt_param(self, sess):
        return sess.run([self.lr, self.beta, self.gamma])

    def printoo(self, obs, sess):
        return sess.run(self.printer, feed_dict={self.ob: obs})


class Critic(object):
    """
    A simple MLP critic network whose jobs is to estimate the discounted reward the agent is going to recieve in future given 
    the current state.
    """
    def __init__(self, name, num_ob_feat, init_lr=0.001, ob_scaler=ID_FN, global_critic=None):
        with tf.variable_scope(name):
            self.obs = tf.placeholder(shape=[None, num_ob_feat], dtype=tf.float32)
            obs_scaled = ob_scaler(self.obs)
            x = tf.layers.dense(name='first_layer', inputs=obs_scaled, units=256, activation=tf.nn.relu, kernel_initializer=xavier)
            x1 = tf.layers.dense(name='second_layer',  inputs=x, units=128, activation=tf.nn.relu, kernel_initializer=xavier)
            x2 = tf.layers.dense(name='third_layer',  inputs=x1, units=128, activation=tf.nn.relu, kernel_initializer=xavier)
            #x2 = dense(name='third_layer', inp=x1, activation= tf.nn.relu, in_dim=16, out_dim=16)
            self.v = v = tf.reshape(tf.layers.dense(name='value', inputs=x2, units=1,), [-1])
            self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)
            self.v_ = v_ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.square(v-v_))
            self.lr = tf.Variable(initial_value=init_lr,dtype=tf.float32, trainable=False)
            self.my_optimizer = adam = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.my_vars = my_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            grads_and_vars = adam.compute_gradients(self.loss, var_list=my_vars)
            grads_clipped = [g for g,_ in grads_and_vars]            
            if global_critic is not None:
                grads_and_vars = zip(grads_clipped, global_critic.my_vars)
                self.opt_op = adam.apply_gradients(grads_and_vars)
                def optimize(obs, targets, sess):
                        feed_dict={self.obs:obs, self.v_: targets}
                        return sess.run([self.loss, self.opt_op], feed_dict=feed_dict)
                self.optimize = optimize 
                self._global_syncer = [v_l.assign(v_g) for v_l, v_g in zip(my_vars, global_critic.my_vars) ]

            self.printer = tf.constant(0.0)  
            self.printer = tf.Print(self.printer, data=["Critic Variable data"] + [tf.reduce_mean(v) for v in self.my_vars])  
            self.printer = tf.Print(self.printer, data=['Ciritic Basic data', tf.reduce_mean(x), tf.reduce_mean(x1), tf.reduce_mean(v)])
            self.global_critic = global_critic

    def sync_w_global(self, sess):
        return sess.run(self._global_syncer)
        
    def value(self, obs, sess):
        return sess.run(self.v, feed_dict={self.obs:obs})
    
    def set_opt_param(self, new_lr, sess):
        return sess.run(self.lr, feed_dict={self.lr:new_lr})
    def printoo(self, obs, sess):
        return sess.run([self.printer], feed_dict={self.obs: obs})