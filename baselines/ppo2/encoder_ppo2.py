import os
import time, tqdm
import numpy as np
import os.path as osp
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.encoder_policies import build_policy
import pickle
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.encoder_runner import Runner
from collections import defaultdict
import tensorflow as tf

def constfn(val):
    def f(_):
        return val

    return f


def encoderlearn(*, network, env, total_timesteps, early_stopping=False, eval_env=None, seed=None, nsteps=2048, ent_coef=0.0,
          lr=3e-4,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0, load_path=None, model_fn=None, scope='', **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''
    additional_params = network_kwargs["network_kwargs"]
    length = additional_params["LENGTH"]
    COPYSTEP = additional_params["COPY"]


    from baselines import logger

    # set_global_seeds(seed) We deal with seeds upstream

    if "LR_ANNEALING" in additional_params.keys():
        lr_reduction_factor = additional_params["LR_ANNEALING"]
        start_lr = lr
        lr = lambda prop: (start_lr / lr_reduction_factor) + (
                    start_lr - (start_lr / lr_reduction_factor)) * prop  # Anneals linearly from lr to lr/red factor

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    bestrew = 0
    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.encoder_model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                     max_grad_norm=max_grad_norm, scope=scope,**network_kwargs)
    nupdates = total_timesteps // nbatch
    print("TOT NUM UPDATES", nupdates)

    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam,lengthh=length,nupdates=nupdates)
    if eval_env is not None:
        eval_runner = Runner(env=eval_env, model=model, nsteps=nsteps, gamma=gamma, lam=lam,lengthh=length,nupdates=nupdates)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.perf_counter()

    best_rew_per_step = 0

    run_info = defaultdict(list)


    for update in range(1, nupdates + 1):

        assert nbatch % nminibatches == 0, "Have {} total batch size and want {} minibatches, can't split evenly".format(
            nbatch, nminibatches)
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, obs_pre,action_reward,obs_part,actions_part,mb_partner_action_distribution,states, epinfos,JSD = runner.run()  # pylint: disable=E0632

        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_obs_pre,eval_action_reward,eval_obs_part,eval_actions_part,eval_states, eval_epinfos = eval_runner.run()  # pylint: disable=E0632
        #
        eplenmean = safemean([epinfo['ep_length'] for epinfo in epinfos])
        eprewmean = safemean([epinfo['r'] for epinfo in epinfos])
        rew_per_step = eprewmean / eplenmean


        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)


        ep_sparse_rew_mean = safemean([epinfo['ep_sparse_r'] for epinfo in epinfobuf])

        if ep_sparse_rew_mean > bestrew and ep_sparse_rew_mean > 130:
            # Don't save best model if still doing some self play and it's supposed to be a BC model
            if additional_params["OTHER_AGENT_TYPE"][
               :2] == "bc" and sp_horizon != 0 and env.self_play_randomization > 0:
                pass
            else:
                from human_aware_rl.latent.encoder_ppo import save_ppo_model
                print("BEST REW", ep_sparse_rew_mean, "overwriting previous model with", bestrew)
                save_ppo_model(model, "{}seed{}/best".format(
                    additional_params["SAVE_DIR"],
                    additional_params["CURR_SEED"])
                               )
                bestrew = max(ep_sparse_rew_mean, bestrew)
            savepath = "{}seed{}/.bestsave".format(
                additional_params["SAVE_DIR"],
                additional_params["CURR_SEED"])  # osp.join(checkdir, '.test_save')
            print('Saving to', savepath)
            model.save(savepath)
        classification_acc=model.eval(obs_pre,action_reward,obs_part,actions_part)
        classification_loss=[]
        inds = np.arange(nbatch)
        for _ in range(8):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in tqdm.trange(0, nbatch, nbatch_train, desc="{}/{}".format(_, noptepochs)):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in
                          ( obs_pre, action_reward,obs_part,actions_part))
                classification_loss.append(model.traincontext(lrnow, *slices))
        classification_loss=np.mean(classification_loss)
        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None:  # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in tqdm.trange(0, nbatch, nbatch_train, desc="{}/{}".format(_, noptepochs)):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs,obs_pre,action_reward))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        else:  # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))
        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % 1 == 0:
            target_q_scope = "ppo2_model_human"  # +'/pi'
            q_scope = 'ppo_agent' + '/ppo2_model'  # +'/pi'
            # print(scope)
            t_col = tf.trainable_variables(
                target_q_scope)  # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
            q_col = tf.trainable_variables(
                q_scope)  # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
            q_dict = {}
            update_ops = []
            for var in q_col:
                name_index = var.name.find(q_scope)
                var_name = var.name[name_index + len(q_scope):]
                q_dict[var_name] = var
            for var in t_col:
                name_index = var.name.find(target_q_scope)
                var_name = var.name[name_index + len(target_q_scope):]
                # print(var_name,var)
                update_ops.append(tf.assign(var, var * (1 - COPYSTEP) + q_dict[var_name] * COPYSTEP))
            update_target_op = tf.group(*update_ops)
            model.update_target_op(update_target_op)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update * nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))

            eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
            ep_dense_rew_mean = safemean([epinfo['ep_shaped_r'] for epinfo in epinfobuf])
            eplenmean = safemean([epinfo['ep_length'] for epinfo in epinfobuf])
            run_info['eprewmean'].append(eprewmean)
            run_info['ep_dense_rew_mean'].append(ep_dense_rew_mean)
            run_info['ep_sparse_rew_mean'].append(ep_sparse_rew_mean)
            run_info['eplenmean'].append(eplenmean)
            run_info['explained_variance'].append(float(ev))
            run_info['classfication_acc'].append(classification_acc)
            run_info['classfication_loss'].append(classification_loss)
            run_info['JSD'].append(JSD)
            logger.logkv('true_eprew', safemean([epinfo['ep_sparse_r'] for epinfo in epinfobuf]))
            logger.logkv('eprewmean', eprewmean)
            logger.logkv('eplenmean', eplenmean)
            logger.logkv('classfication_acc', classification_acc)
            logger.logkv('classfication_loss', classification_loss)
            logger.logkv('JSD',JSD)


            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]))
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]))

            time_elapsed = tnow - tfirststart
            logger.logkv('time_elapsed', time_elapsed)

            time_per_update = time_elapsed / update
            time_remaining = (nupdates - update) * time_per_update
            logger.logkv('time_remaining', time_remaining / 60)

            for (lossval, lossname) in zip(lossvals, model.loss_names):
                run_info[lossname].append(lossval)

                logger.logkv(lossname, lossval)

            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()

            # Update current logs
            if additional_params["RUN_TYPE"] in ["ppo", "joint_ppo"]:
                from overcooked_ai_py.utils import save_dict_to_file
                save_dict_to_file(run_info, additional_params["SAVE_DIR"] + "logs")

                # Linear annealing of reward shaping
                if additional_params["REW_SHAPING_HORIZON"] != 0:
                    # Piecewise linear annealing schedule
                    # annealing_thresh: until when we should stop doing 100% reward shaping
                    # annealing_horizon: when we should reach doing 0% reward shaping
                    annealing_horizon = additional_params["REW_SHAPING_HORIZON"]
                    annealing_thresh = 0

                    def fn(x):
                        if annealing_thresh != 0 and annealing_thresh - (annealing_horizon / annealing_thresh) * x > 1:
                            return 1
                        else:
                            fn = lambda x: -1 * (x - annealing_thresh) * 1 / (annealing_horizon - annealing_thresh) + 1
                            return max(fn(x), 0)

                    curr_timestep = update * nbatch
                    curr_reward_shaping = fn(curr_timestep)
                    env.update_reward_shaping_param(curr_reward_shaping)
                    print("Current reward shaping", curr_reward_shaping)

                sp_horizon = additional_params["SELF_PLAY_HORIZON"]

                # Save/overwrite best model if past a certain threshold

                # If not sp run, and horizon is not None,
                # vary amount of self play over time, either with a sigmoidal feedback loop
                # or with a fixed piecewise linear schedule.
                if additional_params["OTHER_AGENT_TYPE"] != "sp" and sp_horizon is not None:
                    if type(sp_horizon) is not list:
                        # Sigmoid self-play schedule based on current performance (not recommended)
                        curr_reward = ep_sparse_rew_mean

                        rew_target = sp_horizon
                        shift = rew_target / 2
                        t = (1 / rew_target) * 10
                        fn = lambda x: -1 * (np.exp(t * (x - shift)) / (1 + np.exp(t * (x - shift)))) + 1

                        env.self_play_randomization = fn(curr_reward)
                        print("Current self-play randomization", env.self_play_randomization)
                    else:
                        assert len(sp_horizon) == 2
                        # Piecewise linear self-play schedule

                        # self_play_thresh: when we should stop doing 100% self-play
                        # self_play_timeline: when we should reach doing 0% self-play
                        self_play_thresh, self_play_timeline = sp_horizon

                        def fn(x):
                            if self_play_thresh != 0 and self_play_timeline - (
                                    self_play_timeline / self_play_thresh) * x > 1:
                                return 1
                            else:
                                fn = lambda x: -1 * (x - self_play_thresh) * 1 / (
                                            self_play_timeline - self_play_thresh) + 1
                                return max(fn(x), 0)

                        curr_timestep = update * nbatch
                        env.self_play_randomization = fn(curr_timestep)
                        print("Current self-play randomization", env.self_play_randomization)



    return model, run_info


# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


