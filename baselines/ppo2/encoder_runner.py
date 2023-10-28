import numpy as np
from baselines.common.runners import AbstractEnvRunner
import tensorflow as tf
from baselines.common.tf_util import initialize
import torch


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner
    run():
    - Make a mini batch
    """

    def __init__(self, *, env, model, nsteps, gamma, lam,lengthh,nupdates):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.latent_dim=model.latent_dim
        self.rand=model.RAND
        self.save_dir=model.save_dir
        self.nupdates=nupdates
        self.length=lengthh
        self.obs0_pre = np.tile(np.expand_dims(self.obs0, axis=1), (1, self.length, 1, 1, 1))
        self.obs1_pre = np.tile(np.expand_dims(self.obs1, axis=1), (1, self.length, 1, 1, 1))
        self.action0 = 4 + np.zeros((len(self.curr_state), self.length), dtype=int)
        self.action1 = 4 + np.zeros((len(self.curr_state), self.length), dtype=int)

        self.action_reward_test = np.zeros(shape=(len(self.curr_state), self.length), dtype=int)
        self.obs_pre_test = self.obs0_pre.copy()
        self.obs0test = np.random.normal(size=self.obs0.shape)
        self.count=0
        self.rand_new=self.rand
    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_partner_action_distribution=[]
        mb_states = self.states
        epinfos = []
        mb_JSD_rew=[]
        self.count+=1
        if "adpt" in self.save_dir:
            self.rand_new=(1.0*self.count/self.nupdates)*self.rand
        # For n in range number of steps

        import time
        tot_time = time.time()
        int_time = 0
        num_envs = len(self.curr_state)

        mb_obs_par=[]
        mb_actions_par=[]
        mb_action_reward=[]
        mb_obs_pre=[]
        # _0, _1, _2, _3 = self.model.step(self.obs0test, action_reward=self.action_reward_test,
        #                                                      pre_obs=self.obs_pre_test, S=self.states, M=self.dones)
        # print(_0,_1,_2,_3)
        '''initial'''
        self.obs0_pre = np.tile(np.expand_dims(self.obs0, axis=1), (1, self.length, 1, 1, 1))
        self.obs1_pre = np.tile(np.expand_dims(self.obs1, axis=1), (1, self.length, 1, 1, 1))
        self.action0 = 4 + np.zeros((len(self.curr_state), self.length), dtype=int)
        self.action1 = 4 + np.zeros((len(self.curr_state), self.length), dtype=int)

        if self.env.trajectory_sp:
            # Selecting which environments should run fully in self play
            sp_envs_bools = np.random.random(num_envs) < self.env.self_play_randomization
            print("SP envs: {}/{}".format(sum(sp_envs_bools), num_envs))

        other_agent_simulation_time = 0
        print("rand_new:",self.rand_new)

        from overcooked_ai_py.mdp.actions import Action

        def other_agent_action():
            if self.env.use_action_method:
                other_agent_actions = self.env.other_agent.actions(self.curr_state, self.other_agent_idx)
                return [Action.ACTION_TO_INDEX[a] for a in other_agent_actions]
            else:
                other_agent_actions = self.env.other_agent.direct_policy(self.obs1)
                return other_agent_actions
        # self.context1=np.random.normal(size=(num_envs, 6))
        self.context1 = np.random.normal(size=(num_envs, 6))
        # self.context1=self.context1/np.sum(self.context1,axis=-1,keepdims=True)
        if 'mean' in self.save_dir:
            self.context1 = np.ones( shape=(num_envs, 6))*1.0
            self.context1 = self.context1 / 6.0

        print(self.save_dir)
        if 'context' in self.save_dir:
            self.context1 = np.random.normal(size=(num_envs, self.latent_dim))

        for _ in range(self.nsteps):

            if "more" in self.save_dir:
                # self.context1 = np.random.uniform(0, 1, size=(num_envs, 6))
                self.context1 = np.random.normal(size=(num_envs, 6))
                # self.context1 = self.context1 / np.sum(self.context1, axis=-1, keepdims=True)

            if 'numb' in self.save_dir:
                self.context1 = np.zeros(shape=(num_envs,6))
                for i in range(num_envs):
                    ith_action=np.random.randint(0,6)
                    self.context1[i,ith_action]=1

            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            overcooked = 'env_name' in self.env.__dict__.keys() and self.env.env_name == "Overcooked-v0"
            if overcooked:

                actions, values, self.states, neglogpacs ,action_probs= self.model.step(self.obs0, pre_obs=self.obs0_pre,action_reward=self.action1,S=self.states, M=self.dones)
                diff_actions = []
                # for i in range(6):
                #     context_now = np.zeros(shape=(num_envs, 6))
                #     context_now[:, i] += 1.0  # np.argmax(context_now,axis=-1)
                #     context_now = context_now / np.sum(context_now, axis=-1, keepdims=True)
                #     _, _, _, _, action_probs = self.model.step_human(self.obs0, context_human=context_now)
                #     diff_actions.append(action_probs)
                # JSD = cal_JSD(diff_actions)
                mb_JSD_rew.append(0)
                import time
                current_simulation_time = time.time()

                # Randomize at either the trajectory level or the individual timestep level
                if self.env.trajectory_sp:

                    # If there are environments selected to not run in SP, generate actions
                    # for the other agent, otherwise we skip this step.
                    if sum(sp_envs_bools) != num_envs:
                        other_agent_actions_bc = other_agent_action()

                    # If there are environments selected to run in SP, generate self-play actions
                    if sum(sp_envs_bools) != 0:
                        if "lasp" in self.save_dir:
                            other_agent_actions_sp, _, _, _, action_probs = self.model.step_human(self.obs1, pre_obs=self.obs1_pre,action_reward=self.action0,S=self.states, M=self.dones)  # self.model.step(self.obs1, pre_obs=self.obs0_pre,action_reward=self.action0,S=self.states, M=self.dones)
                        else:
                            other_agent_actions_sp, _, _, _ ,action_probs=self.model.step_human(self.obs1,context_human=self.context1)#self.model.step(self.obs1, pre_obs=self.obs0_pre,action_reward=self.action0,S=self.states, M=self.dones)
                        # if "add" in self.save_dir:
                        #     add_probs=np.random.uniform(0,1,size=action_probs.shape)
                        #     add_probs=add_probs/np.sum(add_probs,axis=-1,keepdims=True)
                        #     action_probs=0.5*(action_probs+add_probs)
                        #     print(np.sum(add_probs,axis=-1))
                        #     # add_probs[:,-1]+=1-np.sum(add_probs,axis=-1)
                        #     print(add_probs[1])
                        if "addrand" in self.save_dir:
                            action_probs=(1-self.rand_new)*action_probs+self.rand_new*(np.ones(shape=(30,6))/6.0)
                        mb_partner_action_distribution.append(action_probs)
                        if "probs" in self.save_dir:
                            for i in range(num_envs):
                                other_agent_actions_sp[i]=np.random.choice(6,p=action_probs[i]/np.sum(action_probs[i]))

                        if self.rand_new>0.0 and "addrand" not in self.save_dir:
                            #mb_actions_par.append(other_agent_actions_sp)
                            # Recognize human policy action
                            for i in range(num_envs):
                                random_indicator = np.random.uniform(0, 1)
                                if random_indicator < self.rand_new:
                                    random_action = np.random.randint(0, 6)
                                    other_agent_actions_sp[i] = random_action
                        mb_actions_par.append(other_agent_actions_sp)

                    # other_agent_actions_sp, _, _, _ = self.model.step(self.obs1, pre_obs=self.obs1_pre,action_reward=self.action0,S=self.states, M=self.dones)
                    # Select other agent actions for each environment depending on whether it was selected
                    # for self play or not
                    other_agent_actions = []
                    for i in range(num_envs):
                        if sp_envs_bools[i]:
                            sp_action = other_agent_actions_sp[i]
                            other_agent_actions.append(sp_action)
                        else:
                            bc_action = other_agent_actions_bc[i]
                            other_agent_actions.append(bc_action)

                else:
                    other_agent_actions = np.zeros_like(self.curr_state)

                    if self.env.self_play_randomization < 1:
                        # Get actions through the action method of the agent
                        other_agent_actions = other_agent_action()

                    # Naive non-parallelized way of getting actions for other
                    if self.env.self_play_randomization > 0:
                        self_play_actions, _, _, _ = self.model.step(self.obs1, S=self.states, M=self.dones)
                        self_play_bools = np.random.random(num_envs) < self.env.self_play_randomization

                        for i in range(num_envs):
                            is_self_play_action = self_play_bools[i]
                            if is_self_play_action:
                                other_agent_actions[i] = self_play_actions[i]

                # NOTE: This has been discontinued as now using .other_agent_true takes about the same amount of time
                # elif self.env.other_agent_bc:
                #     # Parallelise actions with direct action, using the featurization function
                #     featurized_states = [self.env.mdp.featurize_state(s, self.env.mlp) for s in self.curr_state]
                #     player_featurizes_states = [s[idx] for s, idx in zip(featurized_states, self.other_agent_idx)]
                #     other_agent_actions = self.env.other_agent.direct_policy(player_featurizes_states, sampled=True, no_wait=True)

                other_agent_simulation_time += time.time() - current_simulation_time

                joint_action = [(actions[i], other_agent_actions[i]) for i in range(len(actions))]

                mb_obs.append(self.obs0.copy())
                mb_obs_par.append(self.obs0.copy())#self action

                mb_obs_pre.append(self.obs0_pre.copy())
                mb_action_reward.append(self.action1.copy())

            else:
                actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
                mb_obs.append(self.obs.copy())

            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            action_0_mid = np.concatenate(
                (self.action0[:, 1:], np.expand_dims(np.asarray(joint_action, dtype=np.int32)[:, 0], axis=1)), axis=1)

            self.action0 = action_0_mid.copy()
            action_1_mid = np.concatenate(
                (self.action1[:, 1:], np.expand_dims(np.asarray(joint_action, dtype=np.int32)[:, 1], axis=1)), axis=1)

            self.action1 = action_1_mid.copy()
            obs0_pre_mid = np.concatenate((self.obs0_pre[:, 1:, :, :, :], np.expand_dims(self.obs0, axis=1)), axis=1)
            self.obs0_pre = obs0_pre_mid.copy()
            obs1_pre_mid = np.concatenate((self.obs1_pre[:, 1:, :, :, :], np.expand_dims(self.obs1, axis=1)), axis=1)
            self.obs1_pre = obs1_pre_mid.copy()

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            if overcooked:
                obs, rewards, self.dones, infos = self.env.step(joint_action)
                both_obs = obs["both_agent_obs"]
                self.obs0[:] = both_obs[:, 0, :, :]
                self.obs1[:] = both_obs[:, 1, :, :]
                self.curr_state = obs["overcooked_state"]
                self.other_agent_idx = obs["other_agent_env_idx"]
            else:
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)


            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

        print("Other agent actions took", other_agent_simulation_time, "seconds")
        tot_time = time.time() - tot_time
        print("Total simulation time for {} steps: {} \t Other agent action time: {} \t {} steps/s".format(self.nsteps,
                                                                                                           tot_time,
                                                                                                           int_time,
                                                                                                           self.nsteps / tot_time))

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_JSD_rew=np.asarray(mb_JSD_rew,dtype=np.float32)
        if 'jsdrew' in self.save_dir:
            mb_rewards+=0.01*mb_JSD_rew
        print("mb_JSD_rew",0.01*np.mean(mb_JSD_rew)*400)
        mb_actions = np.asarray(mb_actions)
        mb_partner_action_distribution=np.asarray(mb_partner_action_distribution)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_action_reward=np.asarray(mb_action_reward)
        mb_obs_pre = np.asarray(mb_obs_pre, dtype=self.obs.dtype)
        mb_obs_par=np.asarray(mb_obs_par,dtype=self.obs.dtype)
        mb_actions_par=np.asarray(mb_actions_par)
        last_values = 0#self.model.value(self.obs, S=self.states, M=self.dones)
        print("mb_action_reward.shape",mb_action_reward.shape)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs,mb_obs_pre,mb_action_reward,mb_obs_par,mb_actions_par,mb_partner_action_distribution)),
                mb_states, epinfos,0.01*np.mean(mb_JSD_rew)*400)



# obs, returns, masks, actions, values, neglogpacs, states = runner.run()

def cal_entropy(p):
    logp=np.log(p)
    ent=np.mean(-np.sum(p*logp,axis=-1))
    return ent
def cal_JSD(p):
    p=np.asarray(p)
    p=p.swapaxes(0, 1)
    hatp=np.mean(p,axis=1) #30,6
    loghatp=np.log(hatp)
    hatp_ent=-np.sum(hatp*loghatp,axis=-1)
    logp=np.log(p)
    ent=np.mean(-np.sum(p*logp,axis=-1),axis=-1)
    return hatp_ent-ent
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


