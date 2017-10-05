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
def fancy_clip(grad, low, high):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, low, high)

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
    def __init__(self, name, num_ob_feat, num_ac, act_type='cont', init_lr = 3e-5, init_beta = 1, 
                       ac_scaler=ID_FN, ob_scaler=ID_FN, ac_activation=ID_FN, global_actor=None, global_step=None):
        assert (global_actor == global_step == None) or ((global_actor is not None and global_step is not None))
        self.name = name
        with tf.variable_scope(name):
            self.ob = tf.placeholder(shape=[None, num_ob_feat], dtype=tf.float32)
            obs_scaled = ob_scaler(self.ob)
            x = tf.layers.dense(name='first_layer', inputs=obs_scaled, units=128, activation=tf.nn.relu, kernel_initializer=xavier)
            x1 = tf.layers.dense(name='second_layer',  inputs=x, units=64, activation=tf.nn.relu, kernel_initializer=xavier)
            x2 = tf.layers.dense(name='third_layer',  inputs=x1, units=64, activation=tf.nn.relu, kernel_initializer=xavier)
            self.adv = tf.placeholder(shape=[None], dtype=tf.float32)
            self.logp_feed = tf.placeholder(shape=[None], dtype=tf.float32)
            self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)
            if act_type == 'cont':            
                mu = ac_scaler(tf.layers.dense(name='mu_layer', inputs=x2, units=num_ac, activation=tf.nn.tanh)) * 2.
                #log_std = dense(name='log', inp = ob, in_dim=num_ob_feat, out_dim=num_ac, initializer=xav)
                log_std = tf.Variable(initial_value=tf.constant([0.0]* num_ac), name='log_std')
                #log_std = tf.clip_by_value(log_std, -2.5, 2.5)
                std = tf.exp(log_std) + 1e-8
                self.ac = mu + tf.random_normal(shape=tf.shape(mu)) * std
                self.logp =  tf.reduce_sum(- tf.square((self.ac - mu)/std)/2.0, axis=1) - tf.reduce_sum(log_std)
                self.ac_hist = tf.placeholder(shape=[None, num_ac], dtype=tf.float32)
                logp_newpolicy_oldac = tf.reduce_sum(- tf.square( (self.ac_hist - mu) / std)/2.0, axis=1) - tf.reduce_sum(log_std)
                printing_data = ['Actor Baic Data', tf.reduce_mean(std), tf.reduce_mean(self.logp), tf.nn.moments(self.ac, axes=[0,1])]
            else:
                logits = tf.layers.dense(name='logits', inputs=x2, units=num_ac) + 1e-8
                self.ac = categorical_sample_logits(logits)
                logps = tf.nn.log_softmax(logits)
                self.logp = fancy_slice_2d(logps, tf.range(tf.shape(self.ac)[0]), self.ac)
                self.ac_hist = tf.placeholder(shape=[None], dtype=tf.int32)
                logp_newpolicy_oldac = fancy_slice_2d(logps, tf.range(tf.shape(self.ac_hist)[0]), self.ac_hist)
                mu = logits
                printing_data = ['Actor Basic Data',  tf.reduce_mean(self.logp), tf.reduce_mean(self.ac)]

            self.rew_loss = -tf.reduce_mean(self.adv * logp_newpolicy_oldac) 
            self.oldnew_kl = tf.reduce_mean(tf.square(self.logp_feed-logp_newpolicy_oldac))   
            # Actual loss stuff. Can try to add action entropy here too
            self.beta = tf.Variable(initial_value=init_beta, dtype=tf.float32, trainable=False)
            self.loss = self.rew_loss   #+  self.oldnew_kl  (this used to be part of our ppo technique. But this is a bit more 
                                        #                problematic in asynchornous setting.)
            self.my_optimizer = adam = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.my_vars = my_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            grads_and_vars = adam.compute_gradients(self.loss, var_list=my_vars)
            grads_clipped = [tf.clip_by_value(g, -.1, .1) for g,_ in grads_and_vars]

            if global_actor is not None:
                print('Printing length of variables', len(global_actor.my_vars), len(my_vars))
                grads_and_vars = zip(grads_clipped, global_actor.my_vars)
                self.opt_op =adam.apply_gradients(grads_and_vars)
                def optimize(acs, obs, advs, logps, sess):
                    feed_dict= {self.adv: advs,self.ac_hist:acs, self.ob:obs, self.logp_feed:logps}
                    return sess.run([self.loss, self.opt_op], feed_dict=feed_dict)
                batch = tf.placeholder(dtype=tf.int64, shape=())
                update_global_step_op = global_step.assign_add(batch)
                def update_global_step(sess, batch_size):
                    sess.run([update_global_step_op], feed_dict={batch:batch_size})
                self.update_global_step = update_global_step    
                self.optimize = optimize
                new_lr = tf.placeholder(dtype=tf.float32, shape=())
                lr_assign = tf.assign(self.lr, new_lr)
                def _lr_update(sess, val):
                    return sess.run(lr_assign, feed_dict={new_lr:val})
                self._lr_update = _lr_update
                self._global_syncer = [tf.assign(v_l, v_g) for v_l, v_g in zip(my_vars, global_actor.my_vars) ]

            #Debugging stuff
            printing_data2 = ["Actor Variable data"]+ [tf.reduce_mean(v) for v in self.my_vars]
            self.printer = tf.constant(0.0) 
            self.printer = tf.Print(self.printer, data=printing_data)
            self.printer = tf.Print(self.printer, data=printing_data2)
            
            #self.printer = tf.Print(self.printer, data=['Actor layer data', tf.reduce_mean(x), tf.reduce_mean(x1), tf.reduce_mean(mu)])
    def get_kl(self, sess, logp_feeds, obs, acs):
        feed_dict = {self.logp_feed:logp_feeds, self.ac_hist:acs, self.ob:obs}
        return sess.run(self.oldnew_kl, feed_dict=feed_dict)

    def act(self, ob, sess):
        ob = np.array(ob)
        if len(ob.shape) != 2:
            ob = ob[None]
        ac, logp =  sess.run([self.ac, self.logp], feed_dict={self.ob:ob})
        return ac[0], logp[0]

    def sync_w_global(self, sess):
        return sess.run(self._global_syncer)
       
    def set_opt_param(self, sess, new_lr=None, new_beta=None):
        feed_dict = dict()
        if new_beta is not None:
            feed_dict[self.beta] = new_beta
        if new_lr is not None:
            self._lr_update(sess=sess, val=new_lr)
        return self.get_opt_param(sess)

    def get_opt_param(self, sess):
        return sess.run([self.lr, self.beta])

    def printoo(self, obs, sess):
        return sess.run([self.printer], feed_dict={self.ob: obs})


class Critic(object):
    def __init__(self, name, num_ob_feat, init_lr=1e-4, ob_scaler=ID_FN, global_critic=None):
        self.name = name
        with tf.variable_scope(name):
            self.obs = tf.placeholder(shape=[None, num_ob_feat], dtype=tf.float32)
            obs_scaled = ob_scaler(self.obs)
            x = tf.layers.dense(name='first_layer', inputs=obs_scaled, units=128, activation=tf.nn.relu, kernel_initializer=xavier)
            x1 = tf.layers.dense(name='second_layer',  inputs=x, units=64, activation=tf.nn.relu, kernel_initializer=xavier)
            x2 = tf.layers.dense(name='third_layer',  inputs=x1, units=64, activation=tf.nn.relu, kernel_initializer=xavier)
            #x2 = dense(name='third_layer', inp=x1, activation= tf.nn.relu, in_dim=16, out_dim=16)
            self.v = v = tf.layers.dense(name='value', inputs=x2, units=1)
            self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)
            self.v_ = v_ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.square(v-v_))
            self.lr = tf.Variable(initial_value=init_lr,dtype=tf.float32, trainable=False)
            self.my_optimizer = adam = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.my_vars = my_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            grads_and_vars = adam.compute_gradients(self.loss, var_list=my_vars)
            grads_clipped = [tf.clip_by_value(g, -.1, .1) for g,_ in grads_and_vars]            
            if global_critic is not None:    
                grads_and_vars = zip(grads_clipped, global_critic.my_vars)
                grads_and_vars = zip(grads_clipped, my_vars)
                self.opt_op = adam.apply_gradients(grads_and_vars)
                def optimize(obs, targets, sess):
                        feed_dict={self.obs:obs, self.v_: targets}
                        return sess.run([self.loss, self.opt_op], feed_dict=feed_dict)
                self.optimize = optimize 
                self._global_syncer = [tf.assign(v_l, v_g) for v_l, v_g in zip(my_vars, global_critic.my_vars) ]

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