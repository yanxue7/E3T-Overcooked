import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder
import numpy as np
import gym


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations

        self.action_reward = tensors['action_reward']
        self.pre_obs = tensors['pre_obs']
        self.human_latent_pre = tensors["human_latent_pre"]
        self.logits= tensors["logits"]


        self.policy_human_latent = tensors['policy_human_latent']
        self.context_human = tensors['context_human']

        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)
        print("self.pdtype:",self.pdtype)
        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        # Actions probs
        action_probs = self.pd.mean
        self.action_probs = tf.identity(action_probs, name="action_probs")

        # Take an action
        action = self.pd.sample()
        self.action = tf.identity(action, name="action")

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:, 0]

        self.vf = tf.identity(self.vf, name="value")
        '''human policy'''
        latent_human = tensors['policy_human_latent']
        vf_latent_h = tensors['policy_human_latent']

        vf_latent_h = tf.layers.flatten(vf_latent_h)
        latent_human = tf.layers.flatten(latent_human)

        # Based on the action space, will select what probability distribution type
        self.pdtype_h = make_pdtype(env.action_space)

        self.pd_h, self.pi_h = self.pdtype_h.pdfromlatent(latent_human, init_scale=0.01)

        # Actions probs
        action_probs_h = self.pd_h.mean
        self.action_probs_h = tf.identity(action_probs_h, name="action_probs_h")

        # Take an action
        action_h = self.pd_h.sample()
        self.action_h = tf.identity(action_h, name="action_h")

        # Calculate the neg log of our probability
        self.neglogp_h = self.pd_h.neglogp(self.action_h)
        # self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q_h = fc(vf_latent_h, 'q', env.action_space.n)
            self.vf_h = self.q_h
        else:
            self.vf_h = fc(vf_latent_h, 'vf', 1)
            self.vf_h = self.vf_h[:, 0]

        self.vf_h = tf.identity(self.vf_h, name="value_h")

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys() and inpt_name != 'action_reward' and inpt_name != 'pre_obs' :
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)
        if 'pre_obs' in extra_feed.keys():
            feed_dict[self.pre_obs] = extra_feed['pre_obs']
        if 'action_reward' in extra_feed.keys():
            feed_dict[self.action_reward] = extra_feed['action_reward']
        return sess.run(variables, feed_dict)

    def step(self, observation, return_action_probs=False, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, action_probs, v, state, neglogp = self._evaluate(
            [self.action, self.action_probs, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if return_action_probs:
            return action_probs
        if state.size == 0:
            state = None
        return a, v, state, neglogp,action_probs

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

    def _evaluate_human(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name != 'context_human' and inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        if 'context_human' in extra_feed.keys():
            feed_dict[self.context_human] = extra_feed['context_human']
        return sess.run(variables, feed_dict)

    def step_human(self, observation, return_action_probs=False, **extra_feed):
        """
        Compute next action(s) given the observation(s)
        Parameters:
        ----------
        observation     observation data (either single or a batch)
        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)
        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        a, action_probs, v, state, neglogp = self._evaluate_human(
            [self.action_h, self.action_probs_h, self.vf_h, self.state, self.neglogp_h], observation,
            **extra_feed)
        if return_action_probs:
            return action_probs
        if state.size == 0:
            state = None
        return a, v, state, neglogp,action_probs


def build_policy(env, policy_network, value_network=None, normalize_observations=False, estimate_q=False,
                 **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    length = policy_kwargs['network_kwargs']["LENGTH"]  # number of transitions in history
    train_mode = policy_kwargs['network_kwargs']["TRAIN_MODE"]  # label of training,
    latent_dim = policy_kwargs['network_kwargs']["LATENT_DIM"]  # the dimension of latent
    save_dir=policy_kwargs["network_kwargs"]["SAVE_DIR"]
    context_network = get_network_builder('conv_and_mlp_embd')(**policy_kwargs)
    human_latent_prob_predict = get_network_builder('human_prob_pre')(num_layers=2,
                                                                      num_hidden= policy_kwargs['network_kwargs'][
                                                                          "LATENT_DIM"],activation=tf.nn.tanh,last_activate=tf.nn.tanh, **policy_kwargs)
    decoder_network = get_network_builder('conv_and_mlp_new')(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space
        ob_space_shape = ob_space.shape

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space,
                                                                                              batch_size=nbatch)
        X_Par = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space,
                                                                                              batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X
            encoded_x_par=X_Par

        encoded_x = encode_observation(ob_space, encoded_x)
        encoded_x_par = encode_observation(ob_space, encoded_x_par)

        ob_space_pre = ob_space_shape[:-1]
        ob_space_last = ob_space_shape[-1]
        pre_X = tf.placeholder(shape=(nbatch, length,) + ob_space_pre + (ob_space_last,), dtype=tf.float32)
        encode_pre_x = pre_X
        action_reward = tf.placeholder(shape=(nbatch, length), dtype=tf.int32)
        extra_tensors['pre_obs'] = encode_pre_x
        extra_tensors['action_reward'] = action_reward
        if 'context' in save_dir:
            Context = tf.placeholder(shape=(nbatch, latent_dim), dtype=tf.float32)
        else:
            Context = tf.placeholder(shape=(nbatch,6), dtype=tf.float32)

        extra_tensors['context_human'] = Context

        with tf.variable_scope('context_encoder', reuse=tf.AUTO_REUSE):
            encoded_context = context_network(encode_pre_x, action_reward)
            human_latent_mean_std = human_latent_prob_predict(encoded_context)
            human_latent_pre=human_latent_mean_std
            extra_tensors['human_latent_pre'] = human_latent_pre
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            policy_latent_d = decoder_network(encoded_x,human_latent_pre)# X_par = self observation
            if 'context' in save_dir:
                policy_latent_d=tf.nn.l2_normalize(policy_latent_d,axis=-1)
            action_logits=tf.layers.dense(policy_latent_d,6,activation=tf.identity, name='pi')
            extra_tensors['logits'] = action_logits
        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            if 'context' in save_dir:
                policy_latent = policy_network(encoded_x, policy_latent_d)
                Context=tf.nn.l2_normalize(Context,axis=-1)
                policy_latent_human = policy_network(encoded_x, Context)
            else:
                policy_latent = policy_network(encoded_x,tf.nn.softmax(action_logits,axis=-1))
                policy_latent_human = policy_network(encoded_x, tf.nn.softmax(Context,axis=-1))

            extra_tensors['policy_human_latent'] = policy_latent_human

            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(
                        nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)

        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms
