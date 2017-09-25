import tensorflow as tf
import numpy as np
SCALE = 0.1
ID_FN = lambda x : x
def xav(*t):
    return SCALE * xavier(*t)
xavier = tf.contrib.layers.xavier_initializer()

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


def fancy_slice_2d(X, inds0, inds1):
    """
    like numpy X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

def categorical_sample_logits(X):
    # https://github.com/tensorflow/tensorflow/issues/456
    U = tf.random_uniform(tf.shape(X))
    return tf.argmax(X - tf.log(-tf.log(U)), axis=1)

def dense(name, inp, in_dim, out_dim, activation=None, initializer=xavier, summary=True):
    W = tf.get_variable(shape=[in_dim, out_dim], initializer=xavier, name=name+'weight', dtype=tf.float32)
    b = tf.get_variable(initializer= tf.constant([0.0]*out_dim), name=name+'bias', dtype=tf.float32)
    if summary:
        variable_summaries(W, name+'weight')
        variable_summaries(b, name+'bias')
    lin_out = tf.matmul(inp, W) #+b
    if activation is not None:
        return activation(lin_out)
    else:
        return lin_out


class Actor(object):
    def __init__(self, name, num_ob_feat, num_ac, act_type='cont', init_lr = 0.005, init_beta = 1, 
                       ac_scaler=ID_FN, ob_scaler=ID_FN, ac_activation=ID_FN, global_actor=None):
        self.name = name
        with tf.variable_scope(name):
            self.ob = tf.placeholder(shape=[None, num_ob_feat], dtype=tf.float32)
            obs_scaled = ob_scaler(self.ob)
            x = tf.layers.dense(name='first_layer', inputs=obs_scaled, units=128, activation=tf.nn.relu, kernel_initializer=xavier)
            x1 = tf.layers.dense(name='second_layer',  inputs=x, units=64, activation=tf.nn.relu, kernel_initializer=xavier)
            self.adv = tf.placeholder(shape=[None], dtype=tf.float32)
            self.logp_feed = tf.placeholder(shape=[None], dtype=tf.float32)
            if act_type == 'cont':            
                mu = ac_scaler(tf.layers.dense(name='third_layer', inputs=x1, units=num_ac, activation=ac_activation))
                #log_std = dense(name='log', inp = ob, in_dim=num_ob_feat, out_dim=num_ac, initializer=xav)
                log_std = tf.Variable(initial_value=tf.constant([0.0]* num_ac), name='log_std')
                std = tf.exp(log_std)+ 1e-8
                self.ac = mu + tf.random_normal(shape=tf.shape(mu)) * std
                self.logp =  tf.reduce_sum(- tf.square((self.ac - mu)/std)/2.0, axis=1) - tf.reduce_sum(log_std)
                self.ac_hist = tf.placeholder(shape=[None, num_ac], dtype=tf.float32)
                logp_newpolicy_oldac = tf.reduce_sum(- tf.square( (self.ac_hist - mu) / std)/2.0, axis=1) - tf.reduce_sum(log_std)
                printing_data = ['Actor Data', tf.reduce_mean(std), tf.reduce_mean(self.logp), tf.nn.moments(self.ac, axes=[0,1])]
            else:
                logits = tf.layers.dense(name='logits', inputs=x1, units=num_ac) + 1e-8
                self.ac = categorical_sample_logits(logits)
                logps = tf.nn.log_softmax(logits)
                self.logp = fancy_slice_2d(logps, tf.range(tf.shape(self.ac)[0]), self.ac)
                self.ac_hist = tf.placeholder(shape=[None], dtype=tf.int32)
                logp_newpolicy_oldac = fancy_slice_2d(logps, tf.range(tf.shape(self.ac_hist)[0]), self.ac_hist)
                mu = logits
                printing_data = ['Actor Data',  tf.reduce_mean(self.logp), tf.reduce_mean(self.ac)]

            if name == 'global_actor':
                self.optimizer = optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                dummy = tf.constant(1.)
                self.my_vars = [v for _,v in optimizer.compute_gradients(dummy)]
                def update_by_grads(grads, global_step=None):
                    grads_and_vars = zip(my_vars, grads)
                    optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)
                self.update_by_grads= update_by_grads
            else: 
                self.rew_loss = -tf.reduce_mean(self.adv * logp_newpolicy_oldac) 
                self.p_dist = tf.reduce_mean(tf.square(self.logp_feed-logp_newpolicy_oldac))   
                # Actual loss stuff. Can try to add action entropy here too
                self.beta = tf.Variable(initial_value=init_beta, dtype=tf.float32, trainable=False)
                self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)
                self.loss = self.rew_loss  +  self.p_dist
                grads_and_vars = tf.train.AdamOptimizer(learning_rate=self.lr).compute_gradients(self.loss)
                self.grads_clipped = [tf.clip_by_value(g,-1.,1.) for g,_ in grads_and_vars]
    
            #Debugging stuff
            self.printer = tf.constant(0.0) 
            self.printer = tf.Print(self.printer, data=printing_data)
            self.printer = tf.Print(self.printer, data=['Actor layer data', tf.reduce_mean(x), tf.reduce_mean(x1), tf.reduce_mean(mu)])

    def act(self, ob, sess):
        ob = np.array(ob)
        if len(ob.shape) != 2:
            ob = ob[None]
        ac, logp =  sess.run([self.ac, self.logp], feed_dict={self.ob:ob})
        return ac[0], logp[0]
    
    def get_grads(self, acs, obs, advs, logps, sess):
        feed_dict= {self.adv: advs,self.ac_hist:acs, self.ob:obs, self.logp_feed:logps}
        return sess.run([self.rew_loss, self.p_dist, self.loss, self.grads_clipped], feed_dict=feed_dict)
    
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
    def __init__(self, name, num_ob_feat, init_lr=0.005, ob_scaler=ID_FN, global_critic=None):
        self.name = name
        with tf.variable_scope(name):
            self.obs = tf.placeholder(shape=[None, num_ob_feat], dtype=tf.float32)
            obs_scaled = ob_scaler(self.obs)
            x = tf.layers.dense(name='first_layer', inputs=obs_scaled, units=64, activation=tf.nn.relu, kernel_initializer=xavier)
            x1 = tf.layers.dense(name='second_layer',  inputs=x, units=32, activation=tf.nn.relu, kernel_initializer=xavier)
            #x2 = dense(name='third_layer', inp=x1, activation= tf.nn.relu, in_dim=16, out_dim=16)
            self.v = tf.layers.dense(name='value', inputs=x1, units=1)
            if name == 'global_critic':
                self.optimizer = optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                dummy = tf.constant(1.)
                self.my_vars = [v for _,v in optimizer.compute_gradients(dummy)]
                def update_by_grads(grads, global_step=None):
                    grads_and_vars = zip(my_vars, grads, sess)
                    opt_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)
                    sess.run(opt_op)
                self.update_by_grads= update_by_grads
            else:
                self.v_ = tf.placeholder(shape=[None], dtype=tf.float32)
                self.loss = tf.reduce_mean(tf.square(v-v_))
                self.lr = tf.Variable(initial_value=init_lr,dtype=tf.float32, trainable=False)
                grads_and_vars = tf.train.AdamOptimizer(learning_rate=self.lr).compute_gradients(self.loss)
                self.grads_clipped = [tf.clip_by_value(g,-1.,1.) for g,_ in grads_and_vars]
            self.printer = tf.constant(0.0)    
            self.printer = tf.Print(self.printer, data=['Ciritic data', tf.reduce_mean(x), tf.reduce_mean(x1), tf.reduce_mean(v)])
        
    def value(self, obs, sess):
        return sess.run(self.v, feed_dict={self.obs:obs})
    
    def get_grads(self, obs, targets, sess):
        feed_dict={self.obs:obs, self.v_: targets}
        return sess.run([self.loss, self.grads_clipped], feed_dict=feed_dict)
    
    def set_opt_param(self, new_lr, sess):
        return sess.run(self.lr, feed_dict={self.lr:new_lr})
    def printoo(self, obs, sess):
        return sess.run([self.printer], feed_dict={self.obs: obs})