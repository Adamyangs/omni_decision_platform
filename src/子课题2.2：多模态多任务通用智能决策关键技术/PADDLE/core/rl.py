from genericpath import exists
from os import path
import re
from turtle import done
from core.induction import *
from copy import deepcopy
from collections import namedtuple
import pickle
import matplotlib.pyplot as plt
import copy
from core.DRL.PPO import PPO
import torch
from torch.distributions import Categorical
import wandb


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

goto_target_map = {'key': 0, 'door': 1, 'box':2, 'goal':3, 'drop':4}

Episode = namedtuple("Episode", ["reward_history", "action_history", "action_trajectory_prob", "state_history",
               "valuation_history", "valuation_index_history", "input_vector_history",
                                 "returns", "steps", "advantages", "final_return"])

def surrounding(full_state):
    surrounding = []
    agent_pos = np.array(np.where(full_state == 10))[:2].reshape(-1).tolist()
    agent_surrounding_pos = [[agent_pos[0] - 1, agent_pos[1]],
                             [agent_pos[0] + 1, agent_pos[1]],
                             [agent_pos[0], agent_pos[1] - 1],
                             [agent_pos[0], agent_pos[1] + 1]]
    for index, value in enumerate(agent_surrounding_pos):
        surrounding.append(full_state[value[0]][value[1]][0])
    return surrounding

OBJ_MAP = {'key': 0, 'door': 1, 'goal': 2, 'drop': 3, 'box': 4}
final_return_list = []


class ReinforceLearner(object):
    def __init__(self, agent, enviornment, learning_rate, critic=None,
                 steps=300, name=None, discounting=1.0, batched=True, optimizer="RMSProp", end_by_episode=True,
                 minibatch_size=300, w2_critic=None, seed=0, guide=None, source_policies=None, similarity_table=None):
        # super(ReinforceDILP, self).__init__(rules_manager, enviornment.background)
        if isinstance(agent, RLDILP):
            self.type = "DILP"
        elif isinstance(agent, NeuralAgent):
            self.type = "NN"
        else:
            self.type = "Random"
        print("guide:", guide)
        self.guide = guide
        if guide != None:
            self.source_performance, self.target_performance = np.zeros((50,), dtype=float), np.zeros((50,), dtype=float)
        # self.info_json = []
        self.source_policies = source_policies
        self.similarity_table = similarity_table
        if self.source_policies != None and self.similarity_table != None:
            self.psi = 1.0
            self.psi_discount = 0.998
            for value in self.source_policies:
                print(value[0])
                print(value[1])
                print("-" * 100)
        self.w2_critic = w2_critic
        self.env = enviornment
        self.agent = agent
        self.state_encoding = agent.state_encoding
        self.learning_rate = learning_rate
        self._construct_train(learning_rate)
        self.critic=critic
        self.total_steps = steps
        self.name = name
        self.discounting = discounting
        print("\n", self.discounting, "\n")
        self.batched = batched
        self.end_by_episode=end_by_episode
        self.batch_size = minibatch_size
        self.optimizer = optimizer
        self.log_steps = 10
        self.episode = 0
        self.unlock = False
        self.test=False
        self.threshold = 800
        self.gamma = 5.0
        # self.guide_pron = 0.5
        self.seed = seed
        self.changed = False
        # self.model_folder = './model/unlockpickup2_{}/'.format(seed)
        self.model_folder = ('./model/' + name.split("-", -1)[1].lower() + '_{}/').format(seed)
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

    def _construct_train(self, learning_rate):
        self.tf_returns = tf.placeholder(shape=[None], dtype=tf.float32)
        #self.tf_episode_n = tf.placeholder(shape=[])
        self.tf_advantage = tf.placeholder(shape=[None], dtype=tf.float32)
        self.tf_additional_discount = tf.placeholder(shape=[None], dtype=tf.float32)
        self.tf_actions_valuation_indexes = tf.placeholder(shape=[None, self.env.action_n], dtype=tf.int32)
        self.tf_action_index = tf.placeholder(shape=[None], dtype=tf.int32)
        self.tf_gamma = tf.placeholder(shape=None, dtype=tf.float32)
        self._construct_action_prob()
        indexed_action_prob = tf.batch_gather(self.tf_action_prob, self.tf_action_index[:, None])[:, 0]
        self.tf_loss = self.loss(indexed_action_prob)
        self.tf_entropy_loss = self.tf_gamma * tf.reduce_sum(self.tf_action_prob*tf.log(self.tf_action_prob + 1e-5))
        #self.tf_loss = tf.Print(self.tf_loss, [self.tf_loss])
        self.tf_gradients = tf.gradients(self.tf_loss, self.agent.all_variables())
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        try:
            self.tf_train = self.optimizer.minimize(self.tf_loss, tf.train.get_or_create_global_step(),
                                                    var_list=self.agent.all_variables())
        except Exception as e:
            # For random agent
            pass
        self.saver = tf.train.Saver(max_to_keep=1000)
        self.h2g_drl = PPO(7, 4)
        self.guide_h2g_model = self.h2g_drl
        if self.guide != None and 'h2g_drl' in self.guide.keys():
            self.h2g_drl = self.guide['h2g_drl']
            print("change h2g model success")

    def loss(self, indexed_action_prob):
        # rl_loss = (-tf.reduce_sum(tf.log(tf.clip_by_value(indexed_action_prob, 1e-5, 1.0))
        #        )*self.tf_advantage*self.tf_additional_discount)
        rl_loss = tf.reduce_sum(-(tf.log(tf.clip_by_value(indexed_action_prob, 1e-5, 1.0))
                                  ) * self.tf_advantage * self.tf_additional_discount)
        #excess_penalty = 0.01*tf.reduce_sum(tf.nn.relu(tf.reduce_sum(self.tf_action_eval, axis=1)-1.0)**2)
        #regularization_loss = 1e-4*tf.reduce_mean(tf.stack([tf.nn.l2_loss(v) for v in self.agent.all_variables()]))
        # entropy_loss = tf.reduce_sum(self.tf_action_prob*tf.log(self.tf_action_prob + 1e-5))
        # entropy_loss = 0

        # return rl_loss + self.tf_gamma * entropy_loss
        return rl_loss

    def _construct_action_prob(self):
        """
        this method implements the function $p_a$ in the paper
        """
        if self.type == "DILP":
            # slice the action valuations from the valuation vectors
            action_eval = tf.batch_gather(self.agent.tf_result_valuation, self.tf_actions_valuation_indexes)
            self.tf_action_eval = action_eval
            sum_action_eval = tf.tile(tf.reduce_sum(action_eval, axis=1, keepdims=True), [1, self.env.action_n])
            action_prob = tf.where(sum_action_eval > 1.0,
                                   action_eval / sum_action_eval,
                                   action_eval + (1.0 - sum_action_eval) / float(self.env.action_n))
            action_prob = action_eval / sum_action_eval
            self.tf_action_prob = action_prob
            self.sum_action_eval = sum_action_eval
            # self.tf_action_prob = action_eval
        if self.type == "NN" or self.type=="Random":
            self.tf_action_prob = self.agent.tf_output

    def grad(self):
        loss_value = self.tf_loss
        weight_decay = 0.0
        regularization = 0
        for weights in self.agent.all_variables():
            weights = tf.nn.softmax(weights)
            regularization += tf.reduce_sum(tf.sqrt(weights))*weight_decay
        loss_value += regularization/len(self.agent.all_variables())
        return tf.gradients(loss_value, self.agent.all_variables())

    def sample_episode(self, sess, max_steps=50):
        action_prob_history = []
        
        action_history = []
        reward_history = []
        state_body_atoms_action_history = [[], []]
        action_trajectory_prob = []
        valuation_history = []
        state_history = []
        critic_state_history = []
        input_vector_history = []
        valuation_index_history = []
        w2_critic_history = []
        steps = []
        step = 0
        finished = False
        
        how_action_history = []
        how_reward_history = []
        how_action_trajectory_prob = []
        how_valuation_history = []
        how_state_history = []
        how_input_vector_history = []
        how_valuation_index_history = []
        how_steps = []
        how_return = []

        how2go_acc_reward = 0.0

        self.h2g_drl_buffer = []
        self.previous_w2g_body_atoms = None
        self.previous_guide_body_atoms = None
        self.previous_guide_index = None
        phase_reward, accumulated_rewards = 0, 0
        self.choose_source_policy = False
        self.old_body_atoms = None
        self.old_action = None

        while not finished:
            temp_action_history = []
            temp_reward_history = []
            temp_action_trajectory_prob = []
            temp_valuation_history = []
            temp_state_history = []
            temp_input_vector_history = []
            temp_valuation_index_history = []
            temp_steps = []
            while self.env.phase == 'how2go' and (not finished):
                step += 1
                state2atoms = self.env.state2atoms(self.env.state, self.env.phase)
                inputs = None # inputs are needed only for neural network models, so this is none

                indexes = self.agent.get_valuation_indexes(state2atoms)
                valuation = self.agent.base_valuation + self.agent.axioms2valuation(state2atoms)
                action_prob,result, sum_eval, ac_eval = sess.run([self.tf_action_prob, self.agent.tf_result_valuation, self.sum_action_eval, self.tf_action_eval], feed_dict={self.agent.tf_input_valuation: [valuation],
                                                                                                                self.tf_actions_valuation_indexes: [indexes]})
                original_action_prob = copy.deepcopy(action_prob[0])
                print('original_action_prob', original_action_prob)
                action_index = np.random.choice(range(self.env.action_n), p=original_action_prob)
                action = self.agent.all_actions[action_index]

                print(action_index, action, original_action_prob)
                temp_steps.append(step)
                temp_state_history.append(self.env.state)

                # s = list(map(int, self.env.state[1:4]))
                # s.extend(surrounding(self.env.full_state))
                # prob = self.guide_h2g_model.pi(torch.from_numpy(np.array(s)).float())
                # # prob = self.h2g_drl.pi(torch.from_numpy(np.array(s)).float())
                # original_prob = prob
                # m = Categorical(prob)
                # action_index = m.sample().item()
                # action = self.agent.all_actions[action_index]
                # print(action_index, action, prob)

                reward, finished = self.env.next_step(action)
                print(reward, finished)
                phase_reward += reward
                accumulated_rewards += reward

                # s_prime = list(map(int, self.env.state[1:4]))
                # s_prime.extend(surrounding(self.env.full_state))
                # print(original_prob, action_index, s_prime)
                # self.h2g_drl_buffer.append((s, action_index, min(reward, 0.2), s_prime, original_prob[action_index].item(), finished))

                temp_reward_history.append(min(reward,0.2))
                temp_action_history.append(action_index)
                temp_action_trajectory_prob.append(original_action_prob)
                temp_valuation_history.append(valuation)
                temp_valuation_index_history.append(indexes)
                temp_input_vector_history.append(inputs)

                how2go_acc_reward += reward

                if reward > 0 or finished:
                    if reward_history:
                        # if reward>0 and finished:reward_history[-1] = reward
                        # elif finished:reward_history[-1] = -0.1
                        if finished:reward_history[-1] = -0.05
                    temp_returns = discount(temp_reward_history, self.discounting)

                    how_return.append(temp_returns)
                    print("how return",how_return)
                    how_reward_history.extend(temp_reward_history)
                    how_action_history.extend(temp_action_history)
                    how_action_trajectory_prob.extend(temp_action_trajectory_prob)
                    how_state_history.extend(temp_state_history)
                    how_valuation_history.extend(temp_valuation_history)
                    how_valuation_index_history.extend(temp_valuation_index_history)
                    how2go_acc_reward = 0.0
                    break

            if self.env.phase != 'how2go' and (not finished):
                step += 1
                state2atoms = self.env.state2atoms(self.env.state, self.env.phase)
                inputs = None # inputs are needed only for neural network models, so this is none

                atoms_name_list, current_body_atoms_name = [], ''
                for atom in state2atoms:
                    atoms_name_list.append(atom.predicate.name)
                current_body_atoms_name = atoms_name_list[0] + ',' + atoms_name_list[1]

                indexes = self.agent.get_valuation_indexes(state2atoms)
                # print('indexes', indexes)
                valuation = self.agent.base_valuation + self.agent.axioms2valuation(state2atoms)
                action_prob,result, sum_eval, ac_eval = sess.run([self.tf_action_prob, self.agent.tf_result_valuation, self.sum_action_eval, self.tf_action_eval], feed_dict={self.agent.tf_input_valuation: [valuation],
                                                                                                                self.tf_actions_valuation_indexes: [indexes]})
                original_action_prob = copy.deepcopy(action_prob[0])
                print('original_action_prob', original_action_prob)
                action_index = np.random.choice(range(self.env.action_n), p=original_action_prob)
                action = self.agent.all_actions[action_index]

                if current_body_atoms_name == 'key_noshow,is_closed' or current_body_atoms_name == 'is_closed,key_noshow':
                    for i, a in enumerate(self.agent.all_actions):
                        if a.predicate.name == 'gt_box':
                            action_index = i
                            action = self.agent.all_actions[action_index]
                            break
                if current_body_atoms_name == 'at_box,none' or current_body_atoms_name == 'none,at_box':
                    for i, a in enumerate(self.agent.all_actions):
                        if a.predicate.name == 'toggle':
                            action_index = i
                            action = self.agent.all_actions[action_index]
                            break
                if current_body_atoms_name == 'no_key,is_closed' or current_body_atoms_name == 'is_closed,no_key':
                    for i, a in enumerate(self.agent.all_actions):
                        if a.predicate.name == 'gt_key':
                            action_index = i
                            action = self.agent.all_actions[action_index]
                            break
                if current_body_atoms_name == 'at_key,none' or current_body_atoms_name == 'none,at_key':
                    for i, a in enumerate(self.agent.all_actions):
                        if a.predicate.name == 'pick':
                            action_index = i
                            action = self.agent.all_actions[action_index]
                            break
                if current_body_atoms_name == 'is_closed,has_key' or current_body_atoms_name == 'has_key,is_closed':
                    for i, a in enumerate(self.agent.all_actions):
                        if a.predicate.name == 'gt_door':
                            action_index = i
                            action = self.agent.all_actions[action_index]
                            break
                if current_body_atoms_name == 'at_door,h_key' or current_body_atoms_name == 'h_key,at_door':
                    for i, a in enumerate(self.agent.all_actions):
                        if a.predicate.name == 'toggle':
                            action_index = i
                            action = self.agent.all_actions[action_index]
                            break

                print(action_index, action, original_action_prob)

                w2_critic_history.append(self.env.state)
                state_history.append(self.env.state)
                steps.append(step)

                if self.source_policies != None and self.similarity_table != None:
                    # choose a model from source policies for interacting with the environment
                    atoms_name, current_body_atoms = [], ''
                    for atom in state2atoms:
                        atoms_name.append(atom.predicate.name)
                    if atoms_name[0] + ',' + atoms_name[1] in self.similarity_table.keys():
                        current_body_atoms = atoms_name[0] + ',' + atoms_name[1]
                    else:
                        current_body_atoms = atoms_name[1] + ',' + atoms_name[0]
                    print('current_body_atoms:', current_body_atoms)
                    state_body_atoms_action_history[0].append(current_body_atoms)
                    source_policies_weight = [[], []]
                    current_body_atoms_table = self.similarity_table[current_body_atoms]
                    print('current_body_atoms:', current_body_atoms, 'current_body_atoms_table:', current_body_atoms_table)
                    q_value_function = self.similarity_table['value_function'][current_body_atoms]
                    print('current_q_value_function:', q_value_function)
                    max_similarity, model_num = 0, 0 if self.env.phase == 'where2go' else 1
                    for i, value in enumerate(current_body_atoms_table):
                        print('policy_index:', i, 'value:', value)
                        if max_similarity < value[1]:
                            max_similarity = value[1]
                        value_weight = []
                        for a in value[0]:
                            pron = self.source_policies[i][model_num][a][1]
                            action_space = self.source_policies[i][model_num][a][0]
                            expectation = 0
                            for j, act in enumerate(action_space):
                                for k, act_ in enumerate(q_value_function[0]):
                                    if act.name == act_.name:
                                        expectation += pron[j] * q_value_function[1][k]
                            entropy = np.sum(pron * np.log(pron))
                            value_weight.append(value[1] - max(0, min(-expectation, 0.05)) * entropy + expectation)
                            # value_weight.append(- max(0, min(-expectation, 0.05)) * entropy + expectation)
                            # value_weight.append(value[1])
                            print("similarity:", value[1], "entropy:" , entropy, 'expectation:', expectation, 'value_weight:', value[1] - max(0, min(-expectation, 0.05)) * entropy + expectation)
                        max_value_weight = max(value_weight)
                        max_value_weight_body_atoms = value[0][value_weight.index(max_value_weight)]
                        source_policies_weight[0].append(max_value_weight_body_atoms)
                        source_policies_weight[1].append(max_value_weight)
                    print('value_weight_body_atoms', source_policies_weight[0])
                    print('value_weight', source_policies_weight[1])
                    guide_max_weight = max(source_policies_weight[1])
                    guide_index = source_policies_weight[1].index(max(source_policies_weight[1]))
                    guide_body_atoms = source_policies_weight[0][guide_index]
                    print("guide_action_space:", self.source_policies[guide_index][model_num][guide_body_atoms][0])
                    print("guide_prob:", self.source_policies[guide_index][model_num][guide_body_atoms][1])

                    expectation_target = 0
                    for i, a in enumerate(self.agent.all_actions):
                        for j, a_ in enumerate(q_value_function[0]):
                            if a.predicate.name == a_.name:
                                expectation_target += original_action_prob[i] * q_value_function[1][j]
                    pron = []
                    for i in range(len(original_action_prob)):
                        if original_action_prob[i] > 0:
                            pron.append(original_action_prob[i])
                    pron = np.array(pron, dtype=float)
                    entropy = np.sum(pron * np.log(pron))
                    target_weight = max_similarity - max(0, min(-expectation_target, 0.05)) * entropy + expectation_target
                    # target_weight = - max(0, min(-expectation_target, 0.05)) * entropy + expectation_target
                    # target_weight = max_similarity
                    print('guide_max_weight', guide_max_weight, 'target_weight', target_weight)
                    print("similarity:", max_similarity, "entropy:", entropy, 'expectation:', expectation_target, 'value_weight:', target_weight)
                    print('prob:', original_action_prob)
                    if target_weight > guide_max_weight:
                        self.guide_h2g_model = self.h2g_drl
                    else:
                        pron = self.source_policies[guide_index][model_num][guide_body_atoms][1]
                        source_choose_action_index = np.random.choice(range(len(pron)), p=pron)
                        source_policy_choose_action = self.source_policies[guide_index][model_num][guide_body_atoms][0][
                            source_choose_action_index]
                        self.guide_h2g_model = self.source_policies[guide_index][2]
                        for i, a in enumerate(self.agent.all_actions):
                            if a.predicate.name == source_policy_choose_action.name:
                                action_index = i
                                action = self.agent.all_actions[action_index]
                                break
                    state_body_atoms_action_history[1].append(action)

                reward, finished = self.env.next_step(action)
                # if self.env.phase == 'where2go' and self.env.save_info:
                #     self.info_json.append(self.env.app_json)
                phase_reward += reward
                accumulated_rewards += reward

                print(f"prob: {action_prob}")
                print(f"in sampling, action: {action.predicate.name}, reward: {reward}, finished: {finished}\n")
                # if finished and reward <= 0:
                #     reward = -0.1
                reward_history.append(reward)
                action_history.append(action_index)
                action_trajectory_prob.append(original_action_prob)
                valuation_history.append(valuation)
                valuation_index_history.append(indexes)
                input_vector_history.append(inputs)

            if finished:
                print('*'*10 + 'finished' + '*'*10)
                self.env.reset(test=self.test)
                print(f"phase: {self.env.phase}, state: {self.env.state}")
                print("length of reward history: ",len(reward_history))
        
        # final_return = [np.sum(reward_history)]
        if reward_history:
            final_return = max(reward_history[-1],0.0)
        else:
            final_return = how_reward_history[-1]
        final_return_list.append(final_return)
        wandb.log({"final_return": final_return if final_return != 0.3 else 0})
        returns = discount(reward_history, self.discounting)
        G = discount(reward_history, self.discounting)
        #TODO update q_value_function
        if self.source_policies != None and self.similarity_table != None:
            print("state:", state_body_atoms_action_history[0])
            print("action:", state_body_atoms_action_history[1])
            print('G:', G)
            updated_status = []
            for i in range(len(state_body_atoms_action_history[0])):
                if state_body_atoms_action_history[0][i] not in updated_status:
                    updated_status.append(state_body_atoms_action_history[0][i])
                    actions_space = self.similarity_table['value_function'][state_body_atoms_action_history[0][i]][0]
                    for j, a in enumerate(actions_space):
                        if a.name == state_body_atoms_action_history[1][i].predicate.name:
                            self.similarity_table['value_function'][state_body_atoms_action_history[0][i]][2][j] += 1
                            print("atoms:", state_body_atoms_action_history[0][i], "action:", a.name, "K:", self.similarity_table['value_function'][state_body_atoms_action_history[0][i]][2][j], "G:", G[i])
                            print("value:", self.similarity_table['value_function'][state_body_atoms_action_history[0][i]][1][j])
                            self.similarity_table['value_function'][state_body_atoms_action_history[0][i]][1][j] = \
                                self.similarity_table['value_function'][state_body_atoms_action_history[0][i]][1][j] + ((
                                        G[i] - self.similarity_table['value_function'][state_body_atoms_action_history[0][i]][1][j]) / \
                                self.similarity_table['value_function'][state_body_atoms_action_history[0][i]][2][j])
                            print("update value:", self.similarity_table['value_function'][state_body_atoms_action_history[0][i]][1][j])
        how_returns = np.concatenate(how_return)
        returns = np.concatenate((returns,how_returns))
        if self.guide != None:
            if self.guiding == True:
                # source_performance.append(np.sum(returns))
                self.source_performance = np.append(self.source_performance[1:], np.sum(returns))
            else:
                # target_performance.append(np.sum(returns))
                self.target_performance = np.append(self.target_performance[1:], np.sum(returns))
            print("source_performance:", self.source_performance)
            print("target_performance:", self.target_performance)
        # print(self.h2g_drl_buffer)
        reward_history.extend(how_reward_history)
        action_history.extend(how_action_history)
        action_trajectory_prob.extend(how_action_trajectory_prob)
        state_history.extend(how_state_history)
        valuation_history.extend(how_valuation_history)
        valuation_index_history.extend(how_valuation_index_history)

        for i in range(len(returns)):
            print(state_history[i], self.agent.all_actions[action_history[i]], reward_history[i], returns[i])
        if self.critic and self.unlock:
            self.critic.batch_learn(critic_state_history, reward_history, sess)
            values = self.critic.get_values(critic_state_history,sess,steps).flatten()
            advantages = generalized_adv(reward_history, values, self.discounting)
        elif self.w2_critic and (not self.unlock):
            print('in w2 critic')
            self.w2_critic.batch_learn(w2_critic_history, reward_history, sess)
            values = self.w2_critic.get_values(w2_critic_history,sess,steps).flatten()
            advantages = generalized_adv(reward_history, values, self.discounting)
            # advantages = np.array(returns) - values
            # advantages[-1] = 0.0
        else:
            advantages = returns
        print("reward length: ", len(reward_history))
        
        self.episode += 1
        return Episode(reward_history, action_history, action_trajectory_prob, state_history,
               valuation_history, valuation_index_history, input_vector_history,
                       returns, steps, advantages, final_return)

    def get_minibatch_buffer(self, sess, batch_size=50, end_by_episode=True):
        empty_buffer = [[] for _ in range(10)]
        episode_buffer = deepcopy(empty_buffer)
        sample_related_indexes = range(10)

        def dump_episode2buffer(episode):
            # print("length of episode: ", len(episode))
            for i in sample_related_indexes:
                # print("sample index",i)
                episode_buffer[i].extend(episode[i])

        def split_buffer(raw_buffer, index):
            if len(episode_buffer[0]) < index:
                return raw_buffer, deepcopy(empty_buffer)
            result = []
            new_buffer = []
            for l in raw_buffer:
                result.append(l[:index])
                new_buffer.append(l[index:])
            return result, new_buffer

        while True:
            if len(episode_buffer[0]) ==0:
                if end_by_episode:
                    e = self.sample_episode(sess)
                    print("length of e: ", len(e))
                    dump_episode2buffer(e)
                    final_return = e.final_return
                else:
                    while len(episode_buffer[0]) < batch_size:
                        e = self.sample_episode(sess)
                        dump_episode2buffer(e)
                        final_return = e.final_return
            result, episode_buffer = split_buffer(episode_buffer, batch_size)
            if len(result)<=0:continue
            yield Episode(*(result+[final_return]))

    def summary_scalar(self, name, scalar):
        if self.summary_writer:
            with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar(name, scalar)

    def summary_histogram(self, name, data):
        if self.summary_writer:
            with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.histogram(name, data)

    def setup_train(self, sess, auto_load=False, checkpoints = None):
        sess.run([tf.initializers.global_variables()])
        if self.name:
            if auto_load:
                try:
                    if checkpoints:
                        # path = self.model_folder + 'MiniGrid-BoxKey-8x8-v0/'+ checkpoints
                        path = self.model_folder + self.name + '/' + checkpoints
                    else:
                        # path = self.model_folder + 'MiniGrid-BoxKey-8x8-v0/'+ 'round_1998'
                        path = self.model_folder + self.name + '/round_1998'
                    self.load(sess, path)
                    print('\n success loading')
                except Exception as e:
                    print('\n error loading',e)

            self.summary_writer = None
        else:
            self.summary_writer = None

    def evaluate(self, checkpoints =None, repeat=50):
        self.mode = 'evaluate'
        results = []
        self.test = True
        self.env.reset(unlock=True, test=self.test)
        with tf.Session() as sess:
            self.setup_train(sess, True, checkpoints)
            self.agent.log(sess)
            rules = self.agent.get_predicates_definition(sess, threshold=0.05) if self.type == "DILP" else []
            for _ in range(repeat):
                e = self.sample_episode(sess)
                reward_history, action_history, action_prob_history, state_history, \
                valuation_history, valuation_index_history, input_vector_history, returns, steps, adv, final_return = e
                results.append(final_return)
        unique, counts = np.unique(results, return_counts=True)
        distribution =  dict(zip(unique, counts))
        return {"distribution": distribution, "mean": np.mean(results), "std": np.std(results),
                "min": np.min(results), "max": np.max(results), "rules": rules}

    def train_step(self, sess):
        e = next(self.minibatch_buffer)
        #e = self.sample_episode(sess)
        reward_history, action_history, action_prob_history, state_history,\
            valuation_history, valuation_index_history, input_vector_history,\
            returns, steps, advantage, final_return = e
        #additional_discount = np.cumprod(self.discounting*np.ones_like(advnatage))
        #advantage = normalize(advantage)
        additional_discount = np.ones_like(advantage)
        log = {"return":final_return, "action_history":[str(self.env.all_actions[action_index])
                                                               for action_index in action_history],
                "advantage": advantage}

        if self.batched:
            ops = [self.tf_train, tf.contrib.summary.all_summary_ops(), self.tf_loss, self.tf_entropy_loss, self.tf_gradients, self.tf_gamma] if self.name else [self.tf_train]
            if self.type == "DILP":
                feed_dict = {self.tf_advantage:np.array(advantage),
                             self.tf_additional_discount:np.array(additional_discount),
                             self.tf_action_prob: np.array(action_prob_history),
                                self.tf_gamma: self.gamma,
                                #  self.tf_returns:final_return,
                                 self.tf_action_index:np.array(action_history),
                                 self.tf_actions_valuation_indexes: np.array(valuation_index_history),
                                 self.agent.tf_input_valuation: np.array(valuation_history)}
            elif self.type == "NN":
                feed_dict = {self.tf_advantage:np.array(advantage),
                             self.tf_additional_discount:np.array(additional_discount),
                             self.tf_returns:final_return,
                             self.tf_action_index:np.array(action_history),
                             self.agent.tf_input: np.array(input_vector_history)}
            elif self.type == "Random":
                feed_dict = {self.tf_advantage:np.array(advantage),
                             self.tf_additional_discount:np.array(additional_discount),
                             self.tf_returns:final_return,
                             self.tf_action_index:np.array(action_history),
                             self.agent.tf_input: [np.array(input_vector_history)[:, 0]]}
            result = sess.run(ops, feed_dict)
            print("self gamma: ",self.gamma,"tf gamma :",result[-1], "tf loss: ", result[2], result[3], result[3]/result[2])
            wandb.log({"loss": result[2], "entropy_loss": result[3]})
            # self.h2g_drl.data = self.h2g_drl_buffer
            # print(self.h2g_drl.data)
            # self.h2g_drl.train_net()
        else:
            first = True
            for action_index, adv, acc_discount, val, val_index in zip(action_history, advantage, additional_discount,
                                                          valuation_history,valuation_index_history):
                ops = [self.tf_train, self.tf_loss, self.tf_action_prob]
                if first == True and self.name:
                    ops += [tf.contrib.summary.all_summary_ops()]
                    first = False
                result = sess.run(ops, {self.tf_advantage: [adv],
                                 self.tf_additional_discount: [acc_discount],
                            #    self.tf_returns: final_return,
                               self.tf_action_index: [action_index],
                               self.tf_actions_valuation_indexes: [val_index],
                               self.agent.tf_input_valuation: [val]})
                # self.h2g_drl.train_net()
        return log

    def save(self, sess, path):
        self.saver.save(sess, path + "/parameters.ckpt")
        if self.critic and isinstance(self.critic, TableCritic):
            self.critic.save(path + "/critic.pl")
        torch.save(self.h2g_drl.state_dict(), path + "/h2g_ppo.pt")

    def load(self, sess, path):
        self.saver.restore(sess, path+"/parameters.ckpt")
        if self.critic and isinstance(self.critic, TableCritic):
            self.critic.load(path + "/critic.pl")
        self.h2g_drl.load_state_dict(torch.load(path + "/h2g_ppo.pt"))
        self.h2g_drl.eval()

    def train(self):
        self.mode = 'train'
        with tf.Session() as sess:
            if self.threshold == 0: auto_load = True
            else:auto_load = False
            self.setup_train(sess, auto_load=auto_load)
            self.minibatch_buffer = self.get_minibatch_buffer(sess, batch_size=self.batch_size,
                                                              end_by_episode=self.end_by_episode)
            print(self.total_steps)
            self.log_steps = 1
            # self.total_steps = 50
            for i in range(self.total_steps):
                self.currect_step = i
                if (i+1)%50 == 0:self.gamma /= 10.0
                if self.guide != None:
                    self.guide_pron = np.exp(np.mean(self.source_performance)) / (np.exp(np.mean(self.source_performance)) + np.exp(np.mean(self.target_performance)))
                if self.guide != None:
                    self.guide_pron = max(0.9 - i * 0.001, 0)
                    # self.guide_pron = 1.0
                if self.guide != None:
                    random_idx = np.random.random()
                    if random_idx < self.guide_pron:
                        self.guiding = True
                    else:
                        self.guiding = False
                    print("guide_pron:", self.guide_pron, "guiding:", self.guiding)
                log = self.train_step(sess)
                print("-"*20)
                print("in training, step "+str(i)+"return is "+str(log["return"]))
                if self.source_policies != None and self.similarity_table != None:
                    self.psi = self.psi * self.psi_discount
                    print("similarity_table", self.similarity_table)
                    for key in self.similarity_table.keys():
                        print(key)
                        for value in self.similarity_table[key]:
                            print(value)
                        print('-' * 100)

                if i%self.log_steps==0:
                    self.agent.log(sess)
                    if self.name:
                        if (i+3) >= self.total_steps:
                        # if (i+3) >= self.total_steps:
                        # if i%50 == 0:
                            path = self.model_folder + self.name
                            path = os.path.join(path,'round_'+str(i))
                            if not os.path.exists(path):os.makedirs(path)
                            self.save(sess, path)
                    if not self.unlock:
                        pprint(log)
                print("-"*20+"\n")
        print(final_return_list)
        # print(self.info_json)
        # f = open('adroit_eval.json', 'w')
        # f.write(str(self.info_json))
        # f.close()

        print(self.similarity_table)
        x = [i for i in range(0, 1900, 20)]
        mean, std = [], []
        for i in range(len(x)):
            mean.append(np.mean(final_return_list[i * 20:(i + 1) * 20]))
            std.append(np.std(final_return_list[i * 20:(i + 1) * 20]))
        mean, std = np.array(mean), np.array(std)
        plt.plot(x, mean, label=self.name, linestyle="-", linewidth=2, color="green")
        plt.fill_between(x, (mean - std), (mean + std), interpolate=True, linewidth=0.0, alpha=0.3,
                         color="green")
        plt.xlabel("episode")
        plt.ylabel("performance")
        plt.legend(loc='lower right')
        plt.show()

        return log["return"]

class PPOLearner(ReinforceLearner):
    def __init__(self, agent, enviornment, learning_rate, critic=None,
                 steps=300, name=None, discounting=1.0, optimizer="RMSProp"):
        self.epsilon = 0.2
        self.tf_previous_action_prob = tf.placeholder(tf.float32, shape=[None])
        super(PPOLearner, self).__init__(agent, enviornment, learning_rate, critic,
                                         steps, name, discounting, batched=True, optimizer="RMSProp",
                                         end_by_episode=False, minibatch_size=100)
        self.log_steps = 10


    def loss(self, new_prob):
        ratio = tf.clip_by_value(new_prob, 1e-5, 1.0) / self.tf_previous_action_prob
        return -tf.reduce_mean(tf.minimum(ratio*self.tf_advantage,
                                          tf.clip_by_value(ratio, 1.-self.epsilon, 1.+self.epsilon)*
                                         self.tf_advantage))

    def entropy_loss(self, action_probs):
        entropy = -action_probs*tf.log(tf.clip_by_value(action_probs, 1e-5, 1.0))
        return -tf.reduce_sum(entropy)

    def get_action_prob(self, states, action_indexes):
        action_probs = []
        all_action_probs = []
        if isinstance(self.agent, RLDILP):
            valuation = self.agent.deduction(self.env.state)
        for state, action_index in zip(states, action_indexes):
            if isinstance(self.agent, RLDILP):
                action_prob,_ = self.agent.valuation2action_prob(valuation, state)
            else:
                action_prob = self.agent.deduction(state)
            action_probs.append(action_prob[action_index])
            all_action_probs.append(action_prob)
        return tf.stack(action_probs), tf.stack(all_action_probs)

    def train_step(self, sess):
        e = self.minibatch_buffer.next()
        #e = self.sample_episode(sess)
        reward_history, action_history, action_prob_history, state_history,\
            valuation_history, valuation_index_history, input_vector_history,\
            returns, steps, advantage, final_return = e

        additional_discount = np.ones_like(advantage)
        log = {"return":final_return, "action_history":[str(self.agent.all_actions[action_index])
                                                          for action_index in action_history]}

        for j in range(10):
            ops = [self.tf_train, tf.contrib.summary.all_summary_ops()] if self.name else [self.tf_train]
            if self.type == "DILP":
                feed_dict = {self.tf_advantage:np.array(advantage),
                             self.tf_additional_discount:np.array(additional_discount),
                                 self.tf_returns:final_return,
                                 self.tf_previous_action_prob: np.array(action_prob_history),
                                 self.tf_action_index:np.array(action_history),
                                 self.tf_actions_valuation_indexes: np.array(valuation_index_history),
                                 self.agent.tf_input_valuation: np.array(valuation_history)}
            result = sess.run(ops, feed_dict)
        return log

class RandomAgent(object):
    def __init__(self, action_size):
        self.tf_input = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        ones = tf.ones_like(self.tf_input)/ action_size
        self.tf_output = ones * tf.ones([1, action_size])/ action_size
        self.state_encoding = "vector"

    def all_variables(self):
        return []

    def log(self, sess):
        pass

class NeuralAgent(object):
    def __init__(self, unit_list, action_size, state_size):
        self.unit_list = unit_list
        self.action_size = action_size
        self.tf_input = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        outputs = self.tf_input
        with tf.variable_scope("NN"):
            for unit_n in unit_list:
                outputs = tf.layers.dense(outputs, unit_n, activation=tf.nn.relu,)
                                      #kernel_initializer=tf.initializers.random_normal())
            outputs = tf.layers.dense(outputs, action_size, activation=tf.nn.softmax,)
                                      #kernel_initializer=tf.initializers.random_normal())
        self.tf_output = outputs
        self.state_encoding = "vector"

    def critic_loss(self, reward, current_state_value, next_state_value):
        td_error = reward - current_state_value + self.discounting*next_state_value
        loss = tf.square(td_error)
        return tf.reduce_sum(loss)

    def all_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="NN")

    def log(self, sess):
        pass

class NeuralCritic(object):
    def __init__(self, unit_list, state_size, discounting, learning_rate, state2vector,
                 involve_steps=False, scope_name='critic'):
        self.unit_list = unit_list
        self.state2vector = state2vector
        self.scope_name = scope_name
        self.tf_input = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        self.tf_steps = tf.placeholder(dtype=tf.float32, shape=[None])
        if involve_steps:
            outputs = tf.concat([self.tf_input, self.tf_steps[:, np.newaxis]], axis=1)
        else:
            outputs = self.tf_input
        with tf.variable_scope(self.scope_name):
            for unit_n in unit_list:
                outputs = tf.layers.dense(outputs, unit_n, activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, 1)
        self.involve_steps = involve_steps
        self.tf_output = outputs
        self.state_encoding = "vector"
        self.discounting = discounting
        self.tf_reward = tf.placeholder(dtype=tf.float32, shape=[None])
        self.tf_returns = tf.placeholder(dtype=tf.float32, shape=[None])
        self.tf_loss = tf.reduce_sum(tf.square(self.tf_output[:, 0] - self.tf_returns))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.tf_train = self.optimizer.minimize(self.tf_loss, tf.train.get_or_create_global_step(),
                                                var_list=self.all_variables())


    def all_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)

    def log(self, sess):
        pass

    def batch_learn(self, states, rewards, sess):
        states = [self.state2vector(s) for s in states]
        returns = discount(rewards, self.discounting)
        for i, j in zip(states, returns):
            print(i,j)
        sess.run([self.tf_train], feed_dict={self.tf_input:states, self.tf_reward: rewards,
                                             self.tf_returns: returns,
                                             self.tf_steps:np.array(range(0, len(states)),
                                                                    dtype=np.float32)})

    def get_values(self, states, sess, steps=None):
        states = [self.state2vector(s) for s in states]
        if self.involve_steps:
            return sess.run([self.tf_output], feed_dict={self.tf_input:np.array(states),
                                                     self.tf_steps: np.array(steps, dtype=np.float32)})[0]
        else:
            return sess.run([self.tf_output], feed_dict={self.tf_input:np.array(states)})[0]

class TableCritic(object):
    def __init__(self, discounting, learning_rate=0.1, involve_steps=False):
        self.__table = {}
        self.__discounting = discounting
        self.__learning_rate = learning_rate
        self.involve_steps = involve_steps

    def batch_learn(self, states, rewards, sess=None):
        for s, a, s2, step in zip(states, rewards, states[1:]+["end"], range(len(rewards))):
            if self.involve_steps:
                self.learn((s, step), a, (s2, step+1))
            else:
                self.learn(s, a, s2)

    def get_values(self, states, sess=None, steps=None):
        for i,state in enumerate(states):
            states[i] = totuple(state) if isinstance(state, np.ndarray) or isinstance(state, list) else state
        if self.involve_steps:
            return np.array([self.__table[(state, step)] for step,state in zip(steps, states)])
        else:
            return np.array([self.__table[state] for step,state in enumerate(states)])

    def save(self, path):
        with open(path, "w") as fh:
            pickle.dump(self.__table, fh)

    def load(self, path):
        with open(path) as fh:
            self.__table = pickle.load(fh)

    def learn(self, state, reward, next_state):
        state = totuple(state) if isinstance(state, np.ndarray) or isinstance(state, list) else state
        next_state = totuple(next_state) if isinstance(next_state, np.ndarray) or isinstance(next_state, list) else next_state
        if state not in self.__table:
            self.__table[state] = 0
        if next_state not in self.__table:
            self.__table[next_state] = 0
        predicated_value = reward + self.__discounting*self.__table[next_state]
        self.__table[state] += self.__learning_rate*(predicated_value-self.__table[state])


def discount(r, discounting):
    discounted_reward = np.zeros_like(r, dtype=np.float32)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * discounting + r[i]
        discounted_reward[i] = G
    return discounted_reward

def normalize(scalars):
    mean, std = np.mean(scalars), np.std(scalars)
    return (scalars - mean)/std

def generalized_adv(rewards, values, discounting, lam=0.95):
    values[-1] = rewards[-1]
    deltas = rewards[:-1] + discounting * values[1:] - values[:-1]
    return np.concatenate([discount(deltas, discounting*lam), [0]], axis=0)