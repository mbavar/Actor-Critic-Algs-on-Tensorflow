import tensorflow as tf
import numpy as np
SCALE = 0.1
def xav(*t, dtype, partition_info):
    return 0.1 * xavier(*t)
xavier = tf.contrib.layers.xavier_initializer()

def variable_summaries(var, name=''):
  with tf.variable_scope(name+'summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.variable_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    #tf.summary.histogram('histogram', var)

def lrelu(x, alpha=0.2):
    return (1-alpha) * tf.nn.relu(x) + alpha * x

def fancy_clip(grad, low, high):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, low, high)

def normalized_column_initializer(shape, dtype, partition_info):
    u = tf.random_normal(shape=shape,dtype=tf.float32)
    scale =  tf.sqrt(tf.reduce_sum(tf.square(u), axis=0))/0.1
    return u/scale
  
class Actor(object):
    def __init__(self, num_ob_feat, num_ac, act_type='cont', init_lr = 0.005, init_beta = 1., init_gamma= .01,
                       ac_scale=2.0, ob_scale=[1.0, 1.0, 1.0]):
        with tf.variable_scope('Actor'):
            self.ob = tf.placeholder(shape=[None, num_ob_feat], dtype=tf.float32)
            x = tf.layers.dense(name='first_layer', inputs=self.ob, units=128, activation=lrelu, kernel_initializer=xavier)
            x1 = tf.layers.dense(name='second_layer',  inputs=x, units=128, activation=lrelu, kernel_initializer=xavier)
            x2 = tf.layers.dense(name='third_layer',  inputs=x1, units=64, activation=lrelu, kernel_initializer=xavier)
            self.adv = tf.placeholder(shape=[None], dtype=tf.float32)
            self.logp_feed = tf.placeholder(shape=[None], dtype=tf.float32)
            if act_type == 'cont':            
                mu = tf.layers.dense(name='mu_layer', inputs=x2, units=num_ac, kernel_initializer=xav, activation=tf.nn.tanh) * ac_scale
                log_std = tf.Variable(initial_value=[0.]*num_ac)
                log_std = tf.expand_dims(tf.clip_by_value(log_std, -2.5, 2.5), axis=0)
                std = tf.exp(log_std)
                #std = tf.layers.dense(name='std', inputs=x2, units=num_ac, kernel_initializer=xav, activation=tf.nn.softplus) + 1e-8
                dist = tf.distributions.Normal(loc=mu, scale=std)
                self.ac = dist.sample()  #at sampling we are always sampling 1
                self.logp =  tf.reduce_sum(dist.log_prob(self.ac), axis=1)
                self.entropy = entropy = tf.reduce_sum(dist.entropy(), axis=1)
                self.ac_hist = tf.placeholder(shape=[None, num_ac], dtype=tf.float32)
                logp_newpolicy_oldac = tf.reduce_sum(dist.log_prob(self.ac_hist), axis=1)
                printing_data = ['Actor Data', tf.reduce_mean(std), tf.reduce_mean(self.logp), tf.nn.moments(self.ac, axes=[0,1])]
            else:
                logits = tf.layers.dense(name='logits', inputs=x2, units=num_ac) 
                self.ac = ac = tf.cast(tf.reshape(tf.multinomial(logits=logits, num_samples=1),[-1]), dtype=tf.int32)
                logps = tf.nn.log_softmax(logits+1e-8)
                self.entropy = entropy = - tf.reduce_sum(tf.nn.softmax(logits) * logps, axis=1)
                self.logp = tf.gather_nd(params=logps, indices=tf.stack([tf.range(tf.shape(ac)[0]), ac], axis=1))
                self.ac_hist = ac_hist = tf.placeholder(shape=[None], dtype=tf.int32)
                logp_newpolicy_oldac = tf.gather_nd(params=logps, indices=tf.stack([tf.range(tf.shape(ac_hist)[0]), ac_hist], axis=1))
                mu = logits
                printing_data = ['Actor Data',  tf.reduce_mean(logits), tf.reduce_mean(self.logp), tf.reduce_mean(self.ac)]
                #self.logp = fancy_slice_2d(logps, tf.range(tf.shape(self.ac)[0]), self.ac)
                #logp_newpolicy_oldac = fancy_slice_2d(logps, tf.range(tf.shape(self.ac_hist)[0]), self.ac_hist)


            self.rew_loss = -tf.reduce_mean(self.adv * logp_newpolicy_oldac) 
            self.oldnew_kl = p_dist = tf.reduce_mean(tf.square(self.logp_feed-logp_newpolicy_oldac))   
            # Actual loss stuff. Can try to add action entropy here too
            self.beta = tf.Variable(initial_value=init_beta, dtype=tf.float32, trainable=False)
            self.gamma = tf.Variable(initial_value=init_gamma, dtype=tf.float32, trainable=False)
            self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)
            self.loss = self.rew_loss  +  self.beta * p_dist - self.gamma * tf.reduce_mean(entropy)
            adam = tf.train.AdamOptimizer(learning_rate=self.lr)
            grads_and_vars =  adam.compute_gradients(self.loss)
            grads_and_vars = [ (fancy_clip(g, -1., 1.), v) for g,v in grads_and_vars]
            self.opt = adam.apply_gradients(grads_and_vars)
            for i, gv  in enumerate(grads_and_vars):
                _, v = gv
                variable_summaries(v, 'var_{}'.format(i))
            #Debugging stuff
            self.printer = tf.constant(0.0) 
            self.printer = tf.Print(self.printer, data=printing_data)
            self.printer = tf.Print(self.printer, data=['Actor layer data', tf.reduce_mean(x), tf.reduce_mean(x1), tf.reduce_mean(mu)])


    def get_kl(self, sess, logp_feeds, obs, acs):
        feed_dict = {self.logp_feed:logp_feeds, self.ac_hist:acs, self.ob:obs}
        return sess.run(self.oldnew_kl, feed_dict=feed_dict)

    def act(self, ob, sess):
        ob = np.array(ob)
        if len(ob.shape) != 2:
            ob = ob[None]
        ac, logp, ent =  sess.run([self.ac, self.logp, self.entropy], feed_dict={self.ob:ob})
        return ac[0], logp[0], ent[0]
    
    def optimize(self, acs, obs, advs, logps, sess):
        feed_dict= {self.adv: advs,self.ac_hist:acs, self.ob:obs, self.logp_feed:logps}
        return sess.run([self.loss, self.opt], feed_dict=feed_dict)
    
    def set_opt_param(self, sess, new_lr=None, new_beta=None, new_gamma=None):
        feed_dict = dict()
        if new_beta is not None:
            sess.run(tf.assign(self.beta, new_beta))
        if new_lr is not None:
            sess.run(tf.assign(self.lr, new_lr))
        if new_gamma is not None:
            sess.run(tf.assign(self.gamma, new_gamma))
        return self.get_opt_param(sess)
        
    def get_opt_param(self, sess):
        return sess.run([self.lr, self.beta, self.gamma])

    def printoo(self, obs, sess):
        return sess.run([self.printer], feed_dict={self.ob: obs})


class Critic(object):
    def __init__(self, num_ob_feat, init_lr=0.001, ac_scale=2.0, ob_scale=[1.0, 1.0, 1.0]):
        with tf.variable_scope('Critic'):
            self.obs = tf.placeholder(shape=[None, num_ob_feat], dtype=tf.float32)
            x = tf.layers.dense(name='first_layer', inputs=self.obs, units=256, activation=tf.nn.relu, kernel_initializer=xavier)
            x1 = tf.layers.dense(name='second_layer',  inputs=x, units=128, activation=tf.nn.relu, kernel_initializer=xavier)
            x2 = tf.layers.dense(name='third_layer', inputs=x1, activation= tf.nn.relu,  units=128)
            v = tf.layers.dense(name='value', inputs=x1, units=1)
            v = tf.reshape(v, [-1])
            v_ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.square(v-v_))
            self.v, self.v_ =  v, v_
        #optimization parts
            self.lr = tf.Variable(initial_value=init_lr,dtype=tf.float32, trainable=False)
            adam = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.opt = adam.minimize(self.loss)
            self.printer = tf.constant(0.0)    
            self.printer = tf.Print(self.printer, data=['Critic data', tf.reduce_mean(x), tf.reduce_mean(x1), tf.reduce_mean(v)])
            grads_and_vars =  adam.compute_gradients(self.loss)
            for i, gv  in enumerate(grads_and_vars):
                _, v = gv
                variable_summaries(v, 'var_{}'.format(i))
        
    def value(self, obs, sess):
        return sess.run(self.v, feed_dict={self.obs:obs})
    
    def optimize(self, obs, targets, sess):
        feed_dict={self.obs:obs, self.v_: targets}
        return sess.run([self.loss, self.opt], feed_dict=feed_dict)
    
    def set_opt_param(self, new_lr, sess):
        return sess.run(self.lr, feed_dict={self.lr:new_lr})
    def printoo(self, obs, sess):
        return sess.run([self.printer], feed_dict={self.obs: obs})