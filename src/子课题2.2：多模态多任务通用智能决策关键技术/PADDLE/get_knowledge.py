from core.setup import *
from collections import OrderedDict
import argparse
import json
import tensorflow as tf
import copy


# SOURCE_MODELS_SETUP = {'MiniGrid-BallKey-8x8-v0': setup_ballkey, 'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor, 'MiniGrid-BoxDoor-6x6-v0': setup_boxdoor,
#                      'MiniGrid-DoorGoal-8x8-v0': setup_doorgoal, 'MiniGrid-GapBall-8x8-v0': setup_gapball}
# SOURCE_MODELS_SETUP = {'MiniGrid-BallKey-8x8-v0': setup_ballkey, 'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor, 'MiniGrid-BoxDoor-6x6-v0': setup_boxdoor}
## SOURCE_MODELS_SETUP = {'MiniGrid-BallKey-8x8-v0': setup_ballkey, 'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor}
# SOURCE_MODELS_SETUP = {'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor,'MiniGrid-DoorGoal-8x8-v0': setup_doorgoal, 'MiniGrid-GapBall-8x8-v0': setup_gapball}
# SOURCE_MODELS_SETUP = {'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor, 'MiniGrid-BoxDoor-6x6-v0': setup_boxdoor}
# SOURCE_MODELS_SETUP = {'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor, 'MiniGrid-GapBall-8x8-v0': setup_gapball}

# SOURCE_MODELS_SETUP = {'MiniGrid-GapBall-8x8-v0': setup_gapball, 'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor}
# SOURCE_MODELS_SETUP = {'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor, 'MiniGrid-GapBall-8x8-v0': setup_gapball}

SOURCE_MODELS_SETUP = {'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor, 'MiniGrid-BallKey-8x8-v0': setup_ballkey}
# SOURCE_MODELS_SETUP = {'MiniGrid-DoorGoal-8x8-v0': setup_doorgoal, 'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor}
# SOURCE_MODELS_SETUP = {'MiniGrid-BoxDoor-6x6-v0': setup_boxdoor, 'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor}
# SOURCE_MODELS_SETUP = {'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor, 'MiniGrid-BoxDoor-6x6-v0': setup_boxdoor}

# SOURCE_MODELS_SETUP = {'MiniGrid-BoxDoor-6x6-v0': setup_boxdoor, 'MiniGrid-BallKey-8x8-v0': setup_ballkey,
#                        'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor, 'MiniGrid-GapBall-8x8-v0': setup_gapball,
#                        'MiniGrid-DoorGoal-8x8-v0': setup_doorgoal, 'MiniGrid-BlockedBoxUnlockPickup-8x8-v0': setup_blocedboxunlockpickup}

# SOURCE_MODELS_SETUP = {'MiniGrid-BallKey-8x8-v0': setup_ballkey, 'MiniGrid-DoorGoal-8x8-v0': setup_doorgoal}
#
# SOURCE_MODELS_SETUP = {'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor, 'MiniGrid-BallKey-8x8-v0': setup_ballkey,
#                      'MiniGrid-DoorGoal-8x8-v0': setup_doorgoal}

# SOURCE_MODELS_SETUP = {'MiniGrid-BoxDoor-6x6-v0': setup_boxdoor, 'MiniGrid-BallKey-8x8-v0': setup_ballkey, 'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor,
#                      'MiniGrid-DoorGoal-8x8-v0': setup_doorgoal}
#

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


def filter_knowledge(w2g_rules, h2g_model, w2d_rules, env, actions):
    new_w2g_rules, new_w2d_rules = {}, {}
    for i in range(2):
        env.reset()
        done = False
        while not done:
            if env.phase == 'how2go' and (not done):
                s = list(map(int, env.state[1:4]))
                s.extend(surrounding(env.full_state))
                prob = h2g_model.pi(torch.from_numpy(np.array(s)).float())
                m = Categorical(prob)
                action_index = m.sample().item()
                action = actions[action_index]
                reward, done = env.next_step(action)

            if env.phase == 'what2do' and (not done):
                state2atoms = env.state2atoms(env.state, env.phase)
                atoms_name = []
                for atom in state2atoms:
                    atoms_name.append(atom.predicate.name)
                rule_form1, rule_form2 = atoms_name[0] + ',' + atoms_name[1], atoms_name[1] + ',' + atoms_name[0]
                print(rule_form1, rule_form2)
                if rule_form1 in w2d_rules.keys():
                    guide_action = w2d_rules[rule_form1][0]
                    guide_action_pron = w2d_rules[rule_form1][1]
                    new_w2d_rules[rule_form1] = w2d_rules[rule_form1]
                if rule_form2 in w2d_rules.keys():
                    guide_action = w2d_rules[rule_form2][0]
                    guide_action_pron = w2d_rules[rule_form2][1]
                    new_w2d_rules[rule_form1] = w2d_rules[rule_form2]
                action = guide_action[np.random.choice(range(len(guide_action_pron)), p=guide_action_pron)]
                for index, enum_action in enumerate(actions):
                    if enum_action.predicate.name == action.name:
                        action = enum_action
                        break
                reward, done = env.next_step(action)

            if env.phase == 'where2go' and (not done):
                state2atoms = env.state2atoms(env.state, env.phase)
                atoms_name = []
                for atom in state2atoms:
                    atoms_name.append(atom.predicate.name)
                rule_form1, rule_form2 = atoms_name[0] + ',' + atoms_name[1], atoms_name[1] + ',' + atoms_name[0]
                if rule_form1 in w2g_rules.keys():
                    guide_action = w2g_rules[rule_form1][0]
                    guide_action_pron = w2g_rules[rule_form1][1]
                    new_w2g_rules[rule_form1] = w2g_rules[rule_form1]
                if rule_form2 in w2g_rules.keys():
                    guide_action = w2g_rules[rule_form2][0]
                    guide_action_pron = w2g_rules[rule_form2][1]
                    new_w2g_rules[rule_form1] = w2g_rules[rule_form2]
                action = guide_action[np.random.choice(range(len(guide_action_pron)), p=guide_action_pron)]
                for index, enum_action in enumerate(actions):
                    if enum_action.predicate.name == action.name:
                        action = enum_action
                        break
                reward, done = env.next_step(action)

    return new_w2g_rules, new_w2d_rules


def extract_knowledge(agent=None, env=None, env_name=None, seed=None, checkpoints=None):
    learner = ReinforceLearner(agent, env, 0.05, critic=None, discounting=0.5,
                               batched=True, steps=2000, name=env_name, w2_critic=None, seed=seed)
    rules = {}
    guide_sess = tf.Session()
    learner.setup_train(guide_sess, True, checkpoints=checkpoints)
    learner.agent.log(guide_sess)
    clauses = learner.agent.rules_manager.all_clauses
    for key in clauses.keys():
        rules_weights = learner.agent.rule_weights[key]
        rules_weights = guide_sess.run([rules_weights])[0]
        value = clauses[key][0]
        for i, v in enumerate(value):
            atoms_name = []
            for body in v.body:
                atoms_name.append(body.predicate.name)
            atoms_name = ','.join(atoms_name)
            if atoms_name not in rules.keys():
                rules[atoms_name] = [[key], [rules_weights[0][i]]]
            else:
                rules[atoms_name][0].append(key)
                rules[atoms_name][1].append(rules_weights[0][i])
    for key in rules.keys():
        rules[key][1] = softmax(rules[key][1])
    tf.reset_default_graph()
    return rules, learner.h2g_drl


def get_source_policies(source_model_setup=SOURCE_MODELS_SETUP):
    SOURCE_POLICIES = []
    for guide_name in source_model_setup.keys():
        guide_size = int(guide_name.split('-')[2].split('x')[0])
        man, env = source_model_setup[guide_name](None, env_name=guide_name, size=guide_size)
        agent = RLDILP(man, env, state_encoding="atoms", seed=0)
        rules, h2g_model = extract_knowledge(agent=agent, env=env, env_name=guide_name, seed=0,
                                             checkpoints='round_1999')
        w2g_rules, w2d_rules = {}, {}
        for key in rules.keys():
            if key.split(',')[0] in W2G:
                w2g_rules[key] = rules[key]
            if key.split(',')[0] in W2D:
                w2d_rules[key] = rules[key]
        w2g_rules, w2d_rules = filter_knowledge(w2g_rules, h2g_model, w2d_rules, env, agent.all_actions)
        SOURCE_POLICIES.append([w2g_rules, w2d_rules, h2g_model])
    return SOURCE_POLICIES


def calculate_similarity(source_policies, target_policy):
    # body_atom_weight = 0.6
    # head_atom_weight = 1.0 - body_atom_weight
    similarity_table = {}
    for policy_num in range(len(source_policies)):
        source_policy = source_policies[policy_num]
        for model_num in range(len(target_policy)):
            for target_body_atoms in target_policy[model_num].keys():
                target_head_atom = target_policy[model_num][target_body_atoms][0]
                max_similarity, max_similarity_atoms = 0.0, []
                for source_body_atoms in source_policy[model_num].keys():
                    source_head_atom = source_policy[model_num][source_body_atoms][0]
                    # print(target_body_atoms, target_head_atom)
                    # print(source_body_atoms, source_head_atom)
                    source_clause_atoms_body = source_body_atoms.split(',')
                    source_clause_atoms_head = []
                    for atom in source_head_atom:
                        source_clause_atoms_head.append(atom.name)
                    target_clause_atoms_body = target_body_atoms.split(',')
                    target_clause_atoms_head = []
                    for atom in target_head_atom:
                        target_clause_atoms_head.append(atom.name)
                    # print(target_clause_atoms, source_clause_atoms)
                    # coincide_num = len(set(source_clause_atoms) & set(target_clause_atoms))
                    coincide_num_body = len(set(source_clause_atoms_body) & set(target_clause_atoms_body))
                    coincide_num_head = len(set(source_clause_atoms_head) & set(target_clause_atoms_head))
                    if 'none' in target_clause_atoms_body and ('n_key' or 'n_blockage') in source_clause_atoms_body:
                        coincide_num_body += 1
                    # print(set(source_clause_atoms) & set(target_clause_atoms), coincide_num)
                    similarity = 0.8 * (coincide_num_body / len(target_clause_atoms_body)) + 0.2 * (coincide_num_head / len(target_clause_atoms_head))
                    if max_similarity == similarity:
                        max_similarity_atoms.append(source_body_atoms)
                    elif max_similarity < similarity:
                        max_similarity_atoms = [source_body_atoms]
                        max_similarity = similarity
                # if model_num == 0 and max_similarity < 0.5:
                #     max_similarity = 0
                # if model_num == 1 and max_similarity < 0.7:
                #     max_similarity = 0
                if target_body_atoms in similarity_table.keys():
                    similarity_table[target_body_atoms].append([max_similarity_atoms, max_similarity, [0]*len(max_similarity_atoms), 0])
                else:
                    similarity_table[target_body_atoms] = [[max_similarity_atoms, max_similarity, [0]*len(max_similarity_atoms), 0]]
    return similarity_table


def set_blocedboxunlockpickup():
    source_policies = get_source_policies()
    man, env = setup_blocedboxunlockpickup(None, env_name='MiniGrid-BlockedBoxUnlockPickup-8x8-v0', size=8)
    agent = RLDILP(man, env, state_encoding="atoms", seed=0)
    rules, h2g_model = extract_knowledge(agent=agent, env=env, env_name='MiniGrid-BlockedBoxUnlockPickup-8x8-v0',
                                         seed=0)
    w2g_rules, w2d_rules = {}, {}
    for key in rules.keys():
        if key.split(',')[0] in W2G:
            w2g_rules[key] = rules[key]
        if key.split(',')[0] in W2D:
            w2d_rules[key] = rules[key]
    similarity_table = calculate_similarity(source_policies, [w2g_rules, w2d_rules])
    value_function = {}
    for key in w2g_rules.keys():
        value_function[key] = [w2g_rules[key][0], [0]*len(w2g_rules[key][0]), [0]*len(w2g_rules[key][0])]
    for key in w2d_rules.keys():
        value_function[key] = [w2d_rules[key][0], [0]*len(w2d_rules[key][0]), [0]*len(w2d_rules[key][0])]
    similarity_table['value_function'] = value_function
    return source_policies, similarity_table


def set_blockedboxplacegoal():
    source_policies = get_source_policies()
    man, env = setup_blocedboxplacegoal(None, env_name='MiniGrid-BlockedBoxPlaceGoal-8x8-v0', size=8)
    agent = RLDILP(man, env, state_encoding="atoms", seed=0)
    rules, h2g_model = extract_knowledge(agent=agent, env=env, env_name='MiniGrid-BlockedBoxPlaceGoal-8x8-v0',
                                         seed=0)
    w2g_rules, w2d_rules = {}, {}
    for key in rules.keys():
        if key.split(',')[0] in W2G:
            w2g_rules[key] = rules[key]
        if key.split(',')[0] in W2D:
            w2d_rules[key] = rules[key]
    similarity_table = calculate_similarity(source_policies, [w2g_rules, w2d_rules])
    value_function = {}
    for key in w2g_rules.keys():
        value_function[key] = [w2g_rules[key][0], [0] * len(w2g_rules[key][0]), [0] * len(w2g_rules[key][0])]
    for key in w2d_rules.keys():
        value_function[key] = [w2d_rules[key][0], [0] * len(w2d_rules[key][0]), [0] * len(w2d_rules[key][0])]
    similarity_table['value_function'] = value_function
    return source_policies, similarity_table


if __name__ == '__main__':
    source_policies, similarity_table = set_blockedboxplacegoal()
    for value in source_policies:
        print(value)
        print("-" * 100)
    for key in similarity_table.keys():
        print(key)
        if key != 'value_function':
            for value in similarity_table[key]:
                print(value)
        else:
            for key_atoms in similarity_table[key]:
                print(key_atoms, similarity_table[key][key_atoms])
        print('-'*100)



# if __name__ == "__main__":
#     man, env = setup_blocedboxunlockpickup(None, env_name='MiniGrid-BlockedBoxUnlockPickup-8x8-v0', size=8)
#     agent = RLDILP(man, env, state_encoding="atoms", seed=0)
#     rules, h2g_model = extract_knowledge(agent=agent, env=env, env_name='MiniGrid-BlockedBoxUnlockPickup-8x8-v0', seed=0)
#     w2g_rules, w2d_rules = {}, {}
#     for key in rules.keys():
#         if key.split(',')[0] in W2G:
#             w2g_rules[key] = rules[key]
#         if key.split(',')[0] in W2D:
#             w2d_rules[key] = rules[key]
#     print(w2g_rules, w2d_rules)
#     print(get_source_policies())
    # SOURCE_MODELS_SETUP = {'MiniGrid-BallKey-8x8-v0': setup_ballkey, 'MiniGrid-BlockedDoor-8x8-v0': setup_blockeddoor, 'MiniGrid-BoxDoor-6x6-v0': setup_boxdoor,
    #                  'MiniGrid-DoorGoal-8x8-v0': setup_doorgoal, 'MiniGrid-GapBall-8x8-v0': setup_gapball}
    # SOURCE_POLICIES = []
    # for guide_name in SOURCE_MODELS_SETUP.keys():
        # print(guide_name, SOURCE_MODELS_SETUP[guide_name])
        # guide_size = int(guide_name.split('-')[2].split('x')[0])
        # man, env = SOURCE_MODELS_SETUP[guide_name](None, env_name=guide_name, size=guide_size)
        # agent = RLDILP(man, env, state_encoding="atoms", seed=0)
        # rules, h2g_model = extract_knowledge(agent=agent,env=env,env_name=guide_name,seed=0,checkpoints='round_1999')
        # w2g_rules, w2d_rules = {}, {}
        # for key in rules.keys():
        #     if key.split(',')[0] in W2G:
        #         w2g_rules[key] = rules[key]
        #     if key.split(',')[0] in W2D:
        #         w2d_rules[key] = rules[key]
        # print(filter_knowledge(w2g_rules, h2g_model, w2d_rules, env, agent.all_actions))
        # w2g_rules, w2d_rules = filter_knowledge(w2g_rules, h2g_model, w2d_rules, env, agent.all_actions)
        # SOURCE_POLICIES.append([w2g_rules, w2d_rules, h2g_model])

    # guide_name = 'MiniGrid-BallKey-8x8-v0'
    # guide_size = 8
    # man, env = setup_ballkey(None, env_name=guide_name, size=guide_size)
    # agent = RLDILP(man, env, state_encoding="atoms", seed=0)
    # rules, h2g_model = extract_knowledge(agent=agent,env=env,env_name=guide_name,seed=0,checkpoints='round_1999')
    # w2g_rules, w2d_rules = {}, {}
    # for key in rules.keys():
    #     if key.split(',')[0] in W2G:
    #         w2g_rules[key] = rules[key]
    #     if key.split(',')[0] in W2D:
    #         w2d_rules[key] = rules[key]
    # print(filter_knowledge(w2g_rules, h2g_model, w2d_rules, env, agent.all_actions))
    #
    # guide_name = 'MiniGrid-BlockedDoor-8x8-v0'
    # guide_size = 8
    # man, env = setup_blockeddoor(None, env_name=guide_name, size=guide_size)
    # agent = RLDILP(man, env, state_encoding="atoms", seed=0)
    # rules, h2g_model = extract_knowledge(agent=agent, env=env, env_name=guide_name, seed=0, checkpoints='round_1999')
    # w2g_rules, w2d_rules = {}, {}
    # for key in rules.keys():
    #     if key.split(',')[0] in W2G:
    #         w2g_rules[key] = rules[key]
    #     if key.split(',')[0] in W2D:
    #         w2d_rules[key] = rules[key]
    # print(filter_knowledge(w2g_rules, h2g_model, w2d_rules, env, agent.all_actions))
    #
    # guide_name = 'MiniGrid-BoxDoor-6x6-v0'
    # guide_size = 6
    # man, env = setup_boxdoor(None, env_name=guide_name, size=guide_size)
    # agent = RLDILP(man, env, state_encoding="atoms", seed=0)
    # rules, h2g_model = extract_knowledge(agent=agent, env=env, env_name=guide_name, seed=0, checkpoints='round_1999')
    # w2g_rules, w2d_rules = {}, {}
    # for key in rules.keys():
    #     if key.split(',')[0] in W2G:
    #         w2g_rules[key] = rules[key]
    #     if key.split(',')[0] in W2D:
    #         w2d_rules[key] = rules[key]
    # print(filter_knowledge(w2g_rules, h2g_model, w2d_rules, env, agent.all_actions))
    #
    # guide_name = 'MiniGrid-DoorGoal-8x8-v0'
    # guide_size = 8
    # man, env = setup_blockeddoor(None, env_name=guide_name, size=guide_size)
    # agent = RLDILP(man, env, state_encoding="atoms", seed=0)
    # rules, h2g_model = extract_knowledge(agent=agent, env=env, env_name=guide_name, seed=0, checkpoints='round_1999')
    # w2g_rules, w2d_rules = {}, {}
    # for key in rules.keys():
    #     if key.split(',')[0] in W2G:
    #         w2g_rules[key] = rules[key]
    #     if key.split(',')[0] in W2D:
    #         w2d_rules[key] = rules[key]
    # print(filter_knowledge(w2g_rules, h2g_model, w2d_rules, env, agent.all_actions))
    #
    # guide_name = 'MiniGrid-GapBall-8x8-v0'
    # guide_size = 8
    # man, env = setup_blockeddoor(None, env_name=guide_name, size=guide_size)
    # agent = RLDILP(man, env, state_encoding="atoms", seed=0)
    # rules, h2g_model = extract_knowledge(agent=agent, env=env, env_name=guide_name, seed=0, checkpoints='round_1999')
    # w2g_rules, w2d_rules = {}, {}
    # for key in rules.keys():
    #     if key.split(',')[0] in W2G:
    #         w2g_rules[key] = rules[key]
    #     if key.split(',')[0] in W2D:
    #         w2d_rules[key] = rules[key]
    # print(filter_knowledge(w2g_rules, h2g_model, w2d_rules, env, agent.all_actions))