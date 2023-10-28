import warnings
warnings.filterwarnings("ignore")
import pickle
import os
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow as tf
from overcooked_ai_py.utils import save_pickle, load_pickle
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Direction, Action

from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved,get_bc_agent_from_saved_added
from human_aware_rl.utils import reset_tf, set_global_seed, prepare_nested_default_dict_for_pickle
from human_aware_rl.latent.encoder_ppo  import get_ppo_agent, plot_ppo_run, PPO_DATA_DIR
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS, MediumLevelPlanner
from human_aware_rl.baselines_utils import get_vectorized_gym_env

length = 5


def plot_ppo_sp_training_curves(ppo_sp_model_paths, seeds, single=False, show=False, save=False):
    for layout, model_path in ppo_sp_model_paths.items():
        plt.figure(figsize=(8,5))
        plot_ppo_run(model_path, sparse=True, limit=None, print_config=False, single=single, seeds=seeds)
        plt.title(layout.split("_")[0])
        plt.xlabel("Environment timesteps")
        plt.ylabel("Mean episode reward")
        if save: plt.savefig("rew_ppo_sp_" + layout, bbox_inches='tight')
        if show: plt.show()
def Direction_To_Index(action,actions):
    for i in range(len(actions)):
        if actions[i]==action:
            return i
def evaluate_sp_ppo_and_bc(layout, ppo_sp_path, bc_test_path, num_rounds, seeds, best=False, display=False):
    # sp_ppo_performance = defaultdict(lambda: defaultdict(list))
    print(ppo_sp_path)
    agent_bc_test, bc_params,agent_bc_test_model = get_bc_agent_from_saved_added(bc_test_path)

    del bc_params["data_params"]
    del bc_params["mdp_fn_params"]
    evaluator = AgentEvaluator(**bc_params)


    mdp = evaluator.env.mdp#OvercookedGridworld.from_layout_name(**bc_params["mdp_params"])
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)
    encode_fn = lambda s: mdp.featurize_state(s, mlp)
    performance={}
    PP=[]
    PB=[]
    BP=[]
    agent_bc_test.set_mdp(mdp)
    for seed in seeds:

        ppo_sp_obs = []
        human1_obs = []
        human2_obs = []

        ppo_sp_a = []
        human1_a = []
        human2_a = []

        ppo_sp_obs_pred = []
        human1_obs_pred = []
        human2_obs_pred = []

        ppo_sp_a_pred = []
        human1_a_pred = []
        human2_a_pred = []
        performance[seed]=[]
        save_dir = PPO_DATA_DIR + ppo_sp_path + '/seed{}'.format(seed) + '/best'
        print("save_dir",save_dir)
        predictor = tf.contrib.predictor.from_saved_model(save_dir)
        step_fn = lambda obs,pre_obs,action_reward: predictor({"obs": obs,'pre_obs':pre_obs,"action_reward":action_reward})["action_probs"]

        rewards = []
        print("LPPO vs LPPO")
        for num in range(num_rounds):
            evaluator.env.reset()
            done = False
            flag=0
            action_reward0 = np.zeros([30,length])
            action_reward1=np.zeros([30,length])
            obs = evaluator.env.state
            tot_reward = 0
            action_reward0_mid = np.zeros([30])
            action_reward1_mid=np.zeros([30])

            while not done:
                obs_ppo0 = mdp.lossless_state_encoding(obs)[0]
                padded_obs0 = np.array([obs_ppo0] + [np.zeros(obs_ppo0.shape)] * (30 - 1))
                obs_ppo1 = mdp.lossless_state_encoding(obs)[1]
                padded_obs1 = np.array([obs_ppo1] + [np.zeros(obs_ppo1.shape)] * (30 - 1))

                if not flag:
                    pre_obs0=np.tile(np.expand_dims(padded_obs0,axis=1),(1,length,1,1,1))
                    action_reward0=np.zeros([30,length])+4.0
                    pre_obs1 = np.tile(np.expand_dims(padded_obs1, axis=1),(1, length, 1, 1, 1))
                    action_reward1 = np.zeros([30, length]) + 4.0
                    flag = 1
                ppo_probs0 = step_fn(obs=padded_obs0,pre_obs=pre_obs0,action_reward=action_reward1)[0]
                ppo_sp_obs.append(pre_obs0[0])
                ppo_sp_a.append(action_reward1[0])
                ppo_action0 = np.random.choice(len(ppo_probs0), p=ppo_probs0)

                ppo_probs = step_fn(padded_obs1, pre_obs1, action_reward0)[0]
                ppo_action1 = np.random.choice(len(ppo_probs), p=ppo_probs)
                ppo_sp_obs_pred.append(obs_ppo0)
                ppo_sp_a_pred.append(ppo_action1)
                # ppo_sp_obs.append(pre_obs0[0])
                # ppo_sp_a.append(action_reward0[0])

                pre_obs0=np.concatenate((pre_obs0[:,1:,:,:,:],np.expand_dims(padded_obs0,axis=1)),axis=1)#padded_obs0.copy()

                pre_obs1=np.concatenate((pre_obs1[:,1:,:,:,:],np.expand_dims(padded_obs1,axis=1)),axis=1)#padded_obs0.copy()


                obs, sparse_reward, done, info = evaluator.env.step(
                    (Action.INDEX_TO_ACTION[ppo_action0], Action.INDEX_TO_ACTION[ppo_action1]))
                action_reward0_mid[0]=np.array(ppo_action0)
                action_reward1_mid[0]=np.array(ppo_action1)
                action_reward0= np.concatenate( (action_reward0[:,1:],np.expand_dims(action_reward0_mid,axis=1)),axis=1 )#np.array([ppo_action0,ppo_action])
                action_reward1= np.concatenate( (action_reward1[:,1:],np.expand_dims(action_reward1_mid,axis=1)),axis=1 )#np.array([ppo_action,ppo_action0])
                tot_reward += sparse_reward
            rewards.append(tot_reward)
        print("mean epsiodes reward:", np.mean(rewards))
        print("std of epsiodes reward:", np.std(rewards))
        performance[seed].append(rewards)
        PP.append(np.mean(rewards))
        rewards = []
        print("PPO vs BC")
        for num in range(num_rounds):
            evaluator.env.reset()
            done = False
            act_reward = np.zeros([30, length])
            act_reward_mid=np.zeros([30])
            obs = evaluator.env.state
            tot_reward = 0
            agent_bc_test.set_agent_index(1)
            flag=0

            # padded_obs_pre = np.zeros([30, 5, 4, 20], dtype=np.float32)

            while not done:
                obs_ppo = mdp.lossless_state_encoding(obs)[0]
                obs_mate=mdp.lossless_state_encoding(obs)[1]
                padded_obs = np.array([obs_ppo] + [np.zeros(obs_ppo.shape)] * (30 - 1))
                padded_obs_mate = np.array([obs_mate] + [np.zeros(obs_ppo.shape)] * (30 - 1))


                if not flag:
                    pre_obs_mate=np.tile(np.expand_dims(padded_obs,axis=1),(1,length,1,1,1))
                    act_reward=np.zeros([30,length])+4.0
                    flag=1


                ppo_probs = step_fn(padded_obs,pre_obs_mate,act_reward)[0]
                human1_obs.append(pre_obs_mate[0])
                human1_a.append(act_reward[0])
                ppo_action = np.random.choice(len(ppo_probs), p=ppo_probs)
                pre_obs_mate=np.concatenate((pre_obs_mate[:,1:,:,:,:],np.expand_dims(padded_obs,axis=1)),axis=1)#padded_obs0.copy()


                bc_action=agent_bc_test.action(obs)

                obs, sparse_reward, done, info = evaluator.env.step(
                    (Action.INDEX_TO_ACTION[ppo_action], bc_action))#

                human1_obs_pred.append(obs_ppo)
                human1_a_pred.append(Direction_To_Index(bc_action,Action.INDEX_TO_ACTION))
                tot_reward += sparse_reward
                act_reward_mid[0]=np.array(Direction_To_Index(bc_action,Action.INDEX_TO_ACTION))
                act_reward=np.concatenate( (act_reward[:,1:],np.expand_dims(act_reward_mid,axis=1)),axis=1 )#np.array([ppo_action0,ppo_action])

                #act_reward[0]=np.array([ppo_action,Direction_To_Index(bc_action,Action.INDEX_TO_ACTION)])
            rewards.append(tot_reward)
        print("mean epsiodes reward:", np.mean(rewards))
        print("std of epsiodes reward:", np.std(rewards))
        performance[seed].append(rewards)
        PB.append(np.mean(rewards))
        rewards=[]
        print("BC vs PPO")
        agent_bc_test.set_agent_index(0)
        for num in range(num_rounds):
            evaluator.env.reset()
            done=False
            # pre_obs=np.zeros([30,length,5,4,40],dtype=np.float32)
            act_reward=np.zeros([30,length])
            act_reward_mid=np.zeros([30])
            obs=evaluator.env.state
            tot_reward=0
            flag = 0
            ABC=[]
            APPO=[]
            while not done:
                bc_action=agent_bc_test.action(obs)
                obs_ppo = mdp.lossless_state_encoding(obs)[1]
                padded_obs = np.array([obs_ppo] + [np.zeros(obs_ppo.shape)] * (30 - 1))
                obs_mate = mdp.lossless_state_encoding(obs)[0]
                padded_obs_mate = np.array([obs_mate] + [np.zeros(obs_ppo.shape)] * (30 - 1))

                if not flag:
                    pre_obs_mate=np.tile(np.expand_dims(padded_obs,axis=1),(1,length,1,1,1))
                    act_reward=np.zeros([30,length])+4.0
                    flag=1
                ppo_probs=step_fn(padded_obs,pre_obs_mate,act_reward)[0]
                human2_obs.append(pre_obs_mate[0])
                human2_a.append(act_reward[0])
                pre_obs_mate=np.concatenate((pre_obs_mate[:,1:,:,:,:],np.expand_dims(padded_obs,axis=1)),axis=1)#padded_obs0.copy()
                #print(ppo_probs)
                ppo_action=np.random.choice(len(ppo_probs), p=ppo_probs)
                obs, sparse_reward, done, info=\
                    evaluator.env.step((bc_action,Action.INDEX_TO_ACTION[ppo_action]))
                tot_reward+=sparse_reward
                human2_obs_pred.append(obs_ppo)
                ABC.append(Direction_To_Index(bc_action,Action.INDEX_TO_ACTION))
                APPO.append(ppo_action)
                human2_a_pred.append(Direction_To_Index(bc_action,Action.INDEX_TO_ACTION))
                act_reward_mid[0] = np.array(Direction_To_Index(bc_action,Action.INDEX_TO_ACTION))

                act_reward = np.concatenate((act_reward[:, 1:], np.expand_dims(act_reward_mid, axis=1)),
                                            axis=1)  # np.array([ppo_action0,ppo_action])

            rewards.append(tot_reward)
        print("mean epsiodes reward:",np.mean(rewards))
        print("std of epsiodes reward:",np.std(rewards))
        performance[seed].append(rewards)
        BP.append(np.mean(rewards))
        save_dir = PPO_DATA_DIR + ppo_sp_path + '/seed{}'.format(seed)
        with open(save_dir+'/evaldata.pickle', "wb") as fp:  # Pickling
            pickle.dump([ppo_sp_obs,ppo_sp_a,ppo_sp_obs_pred,ppo_sp_a_pred,human1_obs,human1_a,human1_obs_pred,human1_a_pred,human2_obs,human2_a,human2_obs_pred,human2_a_pred], fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("PP:",PP)
    print("PP mean:",np.mean(PP))
    print("PP std:",np.mean(PP))
    print("PB:", PB)
    print("PB mean:", np.mean(PB))
    print("PB std:", np.mean(PB))
    print("BP:", BP)
    print("BP mean:", np.mean(BP))
    print("BP std:", np.mean(BP))
    return performance




def evaluate_all_sp_ppo_models(ppo_sp_model_paths, bc_test_model_paths, num_rounds, seeds, best):
    ppo_sp_performance = {}
    for layout in ppo_sp_model_paths.keys():
        print(layout)
        layout_eval = evaluate_sp_ppo_and_bc(layout, ppo_sp_model_paths[layout], bc_test_model_paths[layout], num_rounds, seeds, best)
        #ppo_sp_performance.update(dict(layout_eval))
        if not os.path.exists('performance'):
            os.makedirs("performance")
        with open("./performance/layout_{}".format(ppo_sp_model_paths[layout]), "wb") as fp:  # Pickling
            pickle.dump(layout_eval, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #return prepare_nested_default_dict_for_pickle(ppo_sp_performance)

def run_all_ppo_sp_experiments(best_bc_model_paths):
    reset_tf()

    seeds = [2229,7649,7225,9807,386]
    set_global_seed(124)

    ppo_sp_model_paths = {
        'simple':'encoder_ppo_sp_simple_predactionmoreprobsaddrand',
            }


    num_rounds = 40
    evaluate_all_sp_ppo_models(ppo_sp_model_paths, best_bc_model_paths['test'], num_rounds, seeds, best=True)

if __name__ == "__main__":
    best_bc_model_paths = load_pickle("data/bc_runs/best_bc_model_paths")
    run_all_ppo_sp_experiments(best_bc_model_paths)
