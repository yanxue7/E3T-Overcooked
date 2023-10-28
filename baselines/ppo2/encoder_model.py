import tensorflow as tf
import functools
import pickle
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize
import numpy as np
try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None


class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """

    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm, scope, microbatch_size=None,**network_kwargs):

        self.sess = sess = get_session()
        self.scope = scope
        self.length= network_kwargs["network_kwargs"]['LENGTH']
        self.latent_dim= network_kwargs["network_kwargs"]["LATENT_DIM"]
        self.save_dir=network_kwargs["network_kwargs"]["SAVE_DIR"]
        self.RAND=network_kwargs["network_kwargs"]["RAND"]

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
                # CREATE OUR TWO MODELS
                # act_model that is used for sampling
                act_model = policy(nbatch_act, 1, sess)

                # Train model for training
                if microbatch_size is None:
                    train_model = policy(nbatch_train, nsteps, sess)
                else:
                    train_model = policy(microbatch_size, nsteps, sess)
                train_jsd=policy(300*6,nsteps,sess)
        with tf.variable_scope("ppo2_model_human", reuse=tf.AUTO_REUSE):
            act_model_human = policy(nbatch_act, 1, sess)
        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        ###jsd loss
        if 'loss' in self.save_dir:
            self.predict_action_probs = train_jsd.action_probs_h
            self.predict_action_probs = tf.reshape(self.predict_action_probs, (-1, 6, 6))
            self.hat_p = tf.reduce_mean(self.predict_action_probs, axis=1)
            self.log_hat_p = tf.log(self.hat_p)
            self.hat_ent = tf.reduce_sum(-self.hat_p * self.log_hat_p, axis=-1)  # 2000
            self.log_p = tf.log(self.predict_action_probs)
            self.ent = tf.reduce_mean(tf.reduce_sum(-self.predict_action_probs * self.log_p, axis=-1), axis=-1)
            self.jsd_loss = -tf.reduce_mean(self.hat_ent - self.ent)
            loss += 0.01 * self.jsd_loss
            self.train_jsd = train_jsd
        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        if 'joint' in self.save_dir:
            params =tf.trainable_variables(self.scope + '/ppo2_model')#tf.trainable_variables(self.scope + '/ppo2_model'+'/pi')+tf.trainable_variables(self.scope + '/ppo2_model'+'/vf')
        else:
            params =tf.trainable_variables(self.scope + '/ppo2_model' + '/pi') + tf.trainable_variables(
                self.scope + '/ppo2_model' + '/vf')
        # 2. Build our trainer
        if MPI is not None:
            self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))




        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da
        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        if 'loss' in self.save_dir:
            self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'jsd_loss']
            self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac, self.jsd_loss]
        else:
            self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
            self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        self.prob=tf.nn.softmax(act_model.logits,axis=-1)
        self.action_true = tf.placeholder(shape=(nbatch_train,), dtype=tf.int32)

        self.one_hot_action = tf.one_hot(self.action_true, 6)
        self.classification = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_model.logits,
                                                       labels=self.one_hot_action))
        context_params = tf.trainable_variables(
            self.scope + '/ppo2_model' + '/context_encoder') + tf.trainable_variables(
            self.scope + '/ppo2_model' + '/decoder')
        self.trainer_context = tf.train.AdamOptimizer(learning_rate=self.LR, epsilon=1e-5)
        self._train_op_context = self.trainer_context.minimize(self.classification,var_list=context_params)
        self.stats_list_context = [self.classification]
        if "lasp" in self.save_dir:
            self.step_human = act_model_human.step
        else:
            self.step_human=act_model_human.step_human

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        # TODO: Not sure what this next couple lines are doing
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            # print("NOT SURE WHAT THIS GUY IS DOING in model.py")
            sync_from_root(sess, global_variables)  # pylint: disable=E1101
        #self.load_context_encoder()
    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs,obs_pre,action_reward, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X: obs,
            self.train_model.pre_obs:obs_pre,
            self.train_model.action_reward:action_reward,

            self.A: actions,
            self.ADV: advs,
            self.R: returns,
            self.LR: lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC: neglogpacs,
            self.OLDVPRED: values
        }
        if 'loss' in self.save_dir:
            obs_idx=np.arange(obs.shape[0])
            np.random.shuffle(obs_idx)
            obs_less=obs[obs_idx[:300]]
            jsdobs = np.tile(np.expand_dims(obs_less, axis=1), (1, 6, 1, 1, 1)).reshape(
                (300 * 6,) + obs.shape[1:])  # 2000,1,5,4,20
            context_actions = np.eye(6)
            context_actions = np.tile(np.expand_dims(context_actions, axis=0), (obs_less.shape[0], 1, 1)).reshape(-1,6)  # 2000,6,6
            td_map[self.train_jsd.X]=jsdobs
            td_map[self.train_jsd.context_human]=context_actions

        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]
    def traincontext(self,lr,obs, actions,obs_part,actions_part):
        td_map = {
            self.train_model.pre_obs: obs,
            self.train_model.action_reward: actions,
            self.train_model.X: obs_part,
            self.action_true:actions_part,
            self.LR: lr,
        }
        return self.sess.run(
            self.stats_list_context + [self._train_op_context],
            td_map
        )[:-1]

    def eval(self, obs, actions,obs_part,actions_part):
        print(obs.shape,actions.shape)
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        # print(actions[:30])
        N=obs.shape[0]
        batch=30
        iter=N//batch
        # print("N,iter:",N,iter)
        win=0
        for i in range(iter):
            st=i*batch
            en=(i+1)*batch
            td_map = {
                self.act_model.pre_obs: obs[st:en],
                self.act_model.action_reward: actions[st:en],
                self.act_model.X:obs_part[st:en],
            }
            probability=self.sess.run(
                self.prob,
                td_map
            )
            action_pre = np.argmax(probability, axis=-1)
            action_true = actions_part[st:en]

            win += np.sum(action_pre == action_true)

        return 1.0*win/(iter*batch)
    def draw(self,raw_data,dir):
        batch=30
        results=[]
        Actions=[]
        for i in range(3):
            latents=[]
            obs=raw_data[i*2]
            actions=raw_data[i*2+1]
            obs=np.array(obs)
            actions=np.array(actions)
            N = obs.shape[0]
            iter = N // batch
            for j in range(iter):
                st = j * batch
                en = (j + 1) * batch
                td_map = {
                    self.act_model.pre_obs: obs[st:en],
                    self.act_model.action_reward: actions[st:en],
                }
                latent = self.sess.run(
                    self.act_model.human_latent_pre,
                    td_map
                )
                latents.append(latent)
            results.append(np.array(latents).reshape(iter*batch,-1))
            Actions.append(actions[:iter*batch])
        pickle.dump(results,open(dir+'latents.pickle','wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print(dir)
        pickle.dump(Actions,open(dir+'actions.pickle','wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def update_target_op(self, update_target_op):
        self.sess.run(update_target_op)

    def get_trajectory(self,pre_obs,action_reward,obs):
        print(obs.shape, obs.shape)
        trajectory=[]
        action_probs=[]

        N = obs.shape[0]
        batch = 30
        iter = N // batch
        # print("N,iter:",N,iter)
        win = 0
        for i in range(iter):
            st = i * batch
            en = (i + 1) * batch
            td_map = {
                self.act_model.pre_obs: pre_obs[st:en],
                self.act_model.action_reward: action_reward[st:en],

                self.act_model.X: obs[st:en],
            }
            [probability,traje] = self.sess.run(
                [self.prob,self.act_model.human_latent_pre],
                td_map
            )
            trajectory.append(traje)
            action_probs.append(probability)
        trajectory=np.asarray(trajectory).reshape((30,400,-1))
        action_probs=np.asarray(action_probs).reshape((30,400,-1))
        return trajectory,action_probs

