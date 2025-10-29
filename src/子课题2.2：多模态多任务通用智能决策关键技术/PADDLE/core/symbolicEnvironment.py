# from __future__ import all_feature_names, print_function, division, absolute_import

from time import time
from numpy.lib.function_base import percentile
from core.clause import *
from core.ilp import LanguageFrame
import copy
from random import choice, random
import gym
from gym.wrappers import Monitor
import gym_minigrid
import os
from PIL import Image
from gym_minigrid.wrappers import *
import matplotlib.pyplot as plt
# from pyvirtualdisplay import Display
from envs.Empty import EmptyEnvLeftUpp5x5, EmptyEnvRightUpp5x5, EmptyEnvLeftBottom5x5, EmptyRandom5x5
from envs.DoorKey import SimpleDoorKeyEnv
from envs.unlockpickup import UnlockPickup
from envs.BoxKey import SimpleBoxKeyEnv
from envs.multiroom import SimpleMultiRoom
from envs.BoxGoal import SimpleBoxGoalEnv
from env_transfer.DoorGoal import SimpleDoorGoalEnv
from env_transfer.BoxDoor import BoxDoorEnv
from env_transfer.GapBall import SimpleGapBallEnv
from env_transfer.BallKey import SimpleBallKeyEnv
from env_transfer.BlockedDoor import BlockedDoorEnv
from env_transfer.BlockedBoxUnlockPickup import BlockedBoxUnlockPickupEnv
from env_transfer.BlockedBoxPlaceGoal import BlockedBoxPlaceGoalEnv
from env_transfer_continuous.coffee_maze import CoffeeMaze
import time
import json
from core.mllm import ask

class SymbolicEnvironment(object):
    def __init__(self, background, initial_state, actions):
        '''
        :param language_frame
        :param background: list of atoms, the background knowledge
        :param positive: list of atoms, positive instances
        :param negative: list of atoms, negative instances
        '''
        self.background = background
        self._state = copy.deepcopy(initial_state)
        self.initial_state = copy.deepcopy(initial_state)
        self.actions = actions
        self.acc_reward = 0
        self.step = 0

    def reset(self):
        self.acc_reward = 0
        self.step = 0
        self._state = copy.deepcopy(self.initial_state)


KEY_BOOL = Predicate("key_bool", 0)
DOOR_BOOL = Predicate("door_bool", 0)

# where2go predicate
# IS_AGENT = Predicate('is_agent', 1)
IS_KEY = Predicate('is_key', 1)
IS_DOOR = Predicate('is_door', 1)
# IS_GOAL = Predicate('is_goal', 1)   # final destination
IS_OPEN = Predicate('is_open', 1)   # door is open, 以下四个看成agent的attribute, 因此arity=0
IS_CLOSED = Predicate('is_closed', 1)   # door is closed
IS_BLOCKED = Predicate('is_blocked', 1)
IS_INCOMPLETE_BLOCKED = Predicate('is_incomplete_blocked', 1)
HAS_KEY = Predicate('has_key', 1)   # agent has key
NO_KEY = Predicate('no_key', 1)   # agent doesnot have key
HAS_BLOCKAGE = Predicate('has_blockage', 1)
NO_BLOCKAGE = Predicate('no_blockage', 1)
GOTO = Predicate('goto', 1)   # object to goto
GT_KEY = Predicate('gt_key', 0)
GT_DOOR = Predicate('gt_door', 0)
GT_GOAL = Predicate('gt_goal', 0)
GT_HERE = Predicate('gt_here', 0)
GT_DROP = Predicate('gt_drop', 0)
GT_BLOCKAGE = Predicate('gt_blockage', 0)

# how2go predicate
LEFT = Predicate("left", 0)
RIGHT = Predicate("right", 0)
UP = Predicate("up", 0)
DOWN = Predicate("down", 0)

ZERO = Predicate("zero",1)
ONE = Predicate("one",1)
POS = Predicate("pos", 1) # positive number
NEG = Predicate("neg", 1) # negative number

CURRENT = Predicate("current", 2)
GOAL = Predicate("goal", 2) # get the goal coordinates from GOTO()

PXPY = Predicate("pxpy", 1)
PXNY = Predicate("pxny", 1)
PXZY = Predicate("pxzy", 1)
NXPY = Predicate("nxpy", 1)
NXNY = Predicate("nxny", 1)
NXZY = Predicate("nxzy", 1)
ZXPY = Predicate("zxpy", 1)
ZXNY = Predicate("zxny", 1)

POSX = Predicate("posx", 1)
POSY = Predicate("posy", 1)
NEGX = Predicate("negx", 1)
NEGY = Predicate("negy", 1)
ZEROX = Predicate("zerox", 1)
ZEROY = Predicate("zeroy", 1)


pn_dict = {"pp": PXPY,
           "pn": PXNY,
           "np": NXPY,
           "nn": NXNY,
           "zp": ZXPY,
           "zn": ZXNY,
           "pz": PXZY,
           "nz": NXZY}

# what2do predicate
TOGGLE = Predicate("toggle", 0)
PICK = Predicate("pick",0)
DROP = Predicate("drop",0)
# DROP = Predicate("drop",0)
AT = Predicate('at', 1) # norm(x, y) == 1, ie., adjacent to object x
# IS_DOOR\1, IS_KEY\1
AT_KEY = Predicate("at_key", 1)
AT_DOOR = Predicate("at_door", 1)
AT_GOAL = Predicate("at_goal", 1)
AT_DROP = Predicate("at_drop", 1)
H_KEY = Predicate('h_key', 1)   # agent has key
N_KEY = Predicate('n_key', 1)
H_BLOCKAGE = Predicate('h_blockage', 1)
N_BLOCKAGE = Predicate('n_blockage', 1)
NONE = Predicate('none', 1)
GT_BOX = Predicate('gt_box', 0)
KEY_SHOW = Predicate('key_show', 1)   # agent has key
KEY_NOSHOW = Predicate('key_noshow', 1)
AT_BOX = Predicate("at_box", 1)
AT_BLOCKAGE = Predicate('at_blockage', 1)
# coordinate transformation
def coor_trans(coordinate, width = 8, height = 5):
    if width == 5:
        width = 7
        height = 5
    else:
        height = width
    x = coordinate[0] - width
    y = height - coordinate[1]
    return np.array([x, y])


def trans_square(img):
    img_h, img_w, img_c = img.shape
    if img_h != img_w:
        long_side = max(img_w, img_h)
        short_side = min(img_w, img_h)
        loc = abs(img_w - img_h) // 2
        img = img.transpose((1, 0, 2)) if img_w < img_h else img  # 如果高是长边则换轴，最后再换回来
        background = np.zeros((long_side, long_side, img_c), dtype=np.uint8)  # 创建正方形背景
        background[loc: loc + short_side] = img[...]  # 数据填充在中间位置
        img = background.transpose((1, 0, 2)) if img_w < img_h else background
    return img


def adjacent_check(agent_pos, state):
    """
        return all the objects that are adjacent to agent
    """
    # state: [5*5*3]
    agent_surrounding_pos = [[agent_pos[0]-1, agent_pos[1]],
                             [agent_pos[0]+1, agent_pos[1]],
                             [agent_pos[0], agent_pos[1]-1],
                             [agent_pos[0], agent_pos[1]+1]]
    agent_surrounding_obj = [state[i[0], i[1]].tolist() for i in agent_surrounding_pos]
    object_type = [[5, 4, 0], [4, 4, 0], [4, 4, 1], [4, 4, 2]]
    object_mapping = {0:'key', 1:'door', 2:'door', 3:'door'}
    adjacent_ls = [object_mapping[i] for i in range(len(object_type)) if object_type[i] in agent_surrounding_obj]
    adjacent_obj_ls = np.unique(adjacent_ls).tolist()
    if adjacent_obj_ls == []:
        adjacent_obj_ls = ['null']
    return adjacent_obj_ls

def surrounding(agent_pos, state, goal_pos = None):
    """
        return whether there is wall in 4 directions (LRDU), 
        bird's-eye view，agnostic of agent's direction
    """
    obj_ls = [0 for i in range(4)]
    if goal_pos is None:
        agent_surrounding_pos = [[agent_pos[0]-1, agent_pos[1]],
                                [agent_pos[0]+1, agent_pos[1]],
                                [agent_pos[0], agent_pos[1]-1],
                                [agent_pos[0], agent_pos[1]+1]]
        agent_surrounding_obj = [state[i[0], i[1]].tolist() for i in agent_surrounding_pos]
        obj_ls = [1 if i==[2, 5, 0] else 0 for i in agent_surrounding_obj]
    else:
        if agent_pos[0] < goal_pos[0]:
            for i in range(agent_pos[0]+1, goal_pos[0]):
                if state[i,agent_pos[1]].tolist() == [2,5,0]:
                    obj_ls[1] = 1
                    break
        elif agent_pos[0] > goal_pos[0]:
            for i in range(agent_pos[0]-1, goal_pos[0], -1):
                if state[i, agent_pos[1]].tolist() == [2,5,0]:
                    obj_ls[0] = 1
                    break
        # elif agent_pos[0] == goal_pos[0]:
        #     pass

        if agent_pos[1] > goal_pos[1]:
            for i in range(agent_pos[1]+1, goal_pos[1]):
                if state[agent_pos[0],i].tolist() == [2,5,0]:
                    obj_ls[2] = 1
                    break
        elif agent_pos[1] < goal_pos[1]:
            for i in range(agent_pos[1]-1, goal_pos[1],-1):
                if state[agent_pos[0],i].tolist() == [2,5,0]:
                    obj_ls[3] = 1
                    break

    return "".join([str(i) for i in obj_ls])

class Empty():
    def __init__(self, env_name ='Empty' ,width=5, save=False):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]

        self.actions = h2g_actions

        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        self.save = save
        ext = h2g_ext

        self.env = EmptyRandom5x5(size = width)
        self.width = width
        self.round = -1
        self.env = FullyObsWrapper(self.env)
        self.reset()
        self.mapping = {'left':0, 'right':1, 'up':2, 'down':3}
        constants = [str(i) for i in range(-1*(width), width-1)] + ['key', 'door', 'agent']

        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        background = []

        self.background = background

    @property
    def all_actions(self):
        ac = self.actions
        return ac

    def state2vector(self, state):
        # what2do state vector
        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self,state):
        full_state = []
        obj_encoding = [0,0,0,0]
        obj_mapping = {
            'key':0,
            'door':1,
            'goal':2,
            'null':3
        }
        # obj_encoding[obj_mapping[state[-2]]] = 1
        full_state.append(int(state[0]))
        full_state.append(int(state[1]))
        full_state.append(obj_mapping[state[-2]])

        return np.array(full_state, dtype = np.float16)

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        # return current_position & current_direction
        return len(self.actions)

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # return atoms depend on which phase the agent is in

        # where2go state2atom
        if phase is 'where2go':
            if int(state[0]) == 1:
                # atoms.add(Atom(HAS_KEY, ["agent"]))
                atoms.add(Atom(HAS_KEY, ["agent"]))
            else:
                atoms.add(Atom(NO_KEY, ["agent"]))

            if int(state[1]) == 1:
                # atoms.add(Atom(IS_OPEN, ["door"]))
                atoms.add(Atom(IS_OPEN, ["agent"]))
            else:
                atoms.add(Atom(IS_CLOSED, ["agent"]))
        # how2go state2atom
        if phase is 'how2go':
            if int(state[0]) > 0: atoms.add(Atom(POSX,["agent"]))
            elif int(state[0]) < 0: atoms.add(Atom(NEGX,["agent"]))
            else: atoms.add(Atom(ZEROX,["agent"]))

            if int(state[1]) > 0: atoms.add(Atom(POSY,["agent"]))
            elif int(state[1]) < 0: atoms.add(Atom(NEGY,["agent"]))
            else: atoms.add(Atom(ZEROY,["agent"]))

        # what2do state2atom
        elif phase is 'what2do':
            for adj_obj in state[5:]:
                if adj_obj == 'key':
                    atoms.add(Atom(AT_KEY, ["agent"]))
                elif adj_obj == 'door':
                    atoms.add(Atom(AT_DOOR, ["agent"]))
            if int(state[0]) == 1:
                # atoms.add(Atom(HAS_KEY, ["agent"]))
                atoms.add(Atom(H_KEY, ["agent"]))
            else:
                atoms.add(Atom(N_KEY, ["agent"]))

        return atoms
    # how to go
    def action_refine(self, action, agent_direction):
        # action-direction
        refine_map = {'00': [1, 1, 2],
                      '01': [1, 2],
                      '02': [2],
                      '03': [0, 2],
                      '20': [0, 2],
                      '21': [0, 0, 2],
                      '22': [1, 2],
                      '23': [2],
                      '10': [2],
                      '11': [0, 2],
                      '12': [1, 1, 2],
                      '13': [1, 2],
                      '30': [1, 2],
                      '31': [2],
                      '32': [0, 2],
                      '33': [1, 1, 2]}

        return refine_map[str(action)+str(agent_direction)]

    def w2d_action_refine(self, agent_direction, agent_target_direction):
        # agent_target_direction - agent_direction
        refine_map = {'00': [1],
                      '01': [],
                      '02': [0],
                      '03': [1, 1],
                      '20': [0],
                      '21': [1, 1],
                      '22': [1],
                      '23': [],
                      '10': [],
                      '11': [0],
                      '12': [1, 1],
                      '13': [1],
                      '30': [1, 1],
                      '31': [1],
                      '32': [],
                      '33': [0]}
        agent_target_dict = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        agent_target_dir = agent_target_dict.index(agent_target_direction.tolist())

        return refine_map[str(agent_target_dir)+str(agent_direction)]

    def save_image(self,):
        if self.save:
            img = self.env.render('rgb')
            im = Image.fromarray(img)
            im.save(os.path.join(self.save_path,str(self.step)+'.png'))

    def how2go_step_learnable(self, action):
        self.step+=1
        action = self.mapping[action.predicate.name]
        actions = self.action_refine(action, self._direct)
        # print(f"action taken: {action}")
        for i in actions:
            self.full_state, reward, done, info = self.env.step(i)
        self.full_state = self.full_state['image']
        self._state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        self._direct = self.full_state[self._state[0], self._state[1]][2]
        self._state = coor_trans(self._state, self.width) - coor_trans(self.goal_pos, self.width)
        # self._direct = self.dir_mapping[self._direct]
        self._state = (str(self._state[0]), str(self._state[1]))
        print(f"present state: {self._state}")
        if not self.test:
            reward -= 0.01
        print(f"reward: {reward}")

        return reward, done

    def next_step(self, action):
        return self.how2go_step_learnable(action)

    def reset(self, test=False):
        self.test = test
        self.full_state = self.env.reset()
        self.full_state = self.full_state['image']

        # goal location
        self.goal_pos = np.array(np.where(self.full_state==8))[:2].reshape(-1)
        print(f"goal_pos: {self.goal_pos}")

        self.init_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        # relative postional vector (now cheating)
        self.init_dir_vector = coor_trans(self.init_state, self.width-2) - coor_trans(self.goal_pos, self.width-2)
        # initial direction scalar
        self.init_dir = self.full_state[self.init_state[0], self.init_state[1]][2]
        self.init_state = (str(self.init_dir_vector[0]), str(self.init_dir_vector[1]))

        self._state = copy.deepcopy(self.init_state)
        print(self._state)

        self._direct = copy.deepcopy(self.init_dir)
        self.phase = 'where2go'  # where2go, how2go, what2do
        self.round += 1
        self.agent_state = 1
        path = 'log_images_closedoor'
        if self.save:
            if not os.path.exists(os.path.join(path, str(self.round))):
                os.makedirs(os.path.join(path, str(self.round)))
            self.save_path = os.path.join(path, str(self.round))
        self.step = 0
        self.prev_door_open = False
        self.done = False
        self.phase = 'how2go'

class MiniGrid(SymbolicEnvironment):
    all_variations = ()
    def __init__(self, initial_state=("1", "1"), env_name='MiniGrid-DoorKey-8x8-v0', width=8, save = False):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]
        # w2g_actions = [GOTO]
        w2g_actions = [GT_KEY, GT_DOOR, GT_GOAL]
        w2d_actions = [TOGGLE, PICK]
        self.actions = h2g_actions + w2g_actions + w2d_actions

        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        w2g_ext = [IS_OPEN, IS_CLOSED, HAS_KEY, NO_KEY]
        # w2d_ext = [AT]
        w2d_ext = [AT_KEY, AT_DOOR, H_KEY, N_KEY]
        self.save = save
        ext = h2g_ext + w2g_ext + w2d_ext

        self.env = SimpleDoorKeyEnv(size = width)
        self.width = width
        self.round = -1
        self.env = FullyObsWrapper(self.env)
        self.reset()
        self.mapping = {'left':0, 'right':1, 'up':2, 'down':3}
        self._object_encoding = {'agent':0, 'key':1, 'door':2, 'goal':3}
        self.w2d_mapping = {'pick':3, 'toggle':5, 'drop':4}
        constants = [str(i) for i in range(-1*(width), width-1)] + ['key', 'door', 'agent']

        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        background = []

        self.background = background

    @property
    def all_actions(self):
        ac = self.actions
        return ac

    def state2vector(self, state):
        # what2do state vector
        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self,state):
        full_state = []
        obj_encoding = [0,0,0,0]
        obj_mapping = {
            'key':0,
            'door':1,
            'goal':2,
            'null':3
        }
        # obj_encoding[obj_mapping[state[-2]]] = 1
        full_state.append(int(state[0]))
        full_state.append(int(state[1]))
        full_state.append(obj_mapping[state[-2]])

        return np.array(full_state, dtype = np.float16)

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        # return current_position & current_direction
        return len(self.actions)

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # return atoms depend on which phase the agent is in

        # where2go state2atom
        if phase is 'where2go':
            if int(state[0]) == 1:
                # atoms.add(Atom(HAS_KEY, ["agent"]))
                atoms.add(Atom(HAS_KEY, ["agent"]))
            else:
                atoms.add(Atom(NO_KEY, ["agent"]))

            if int(state[1]) == 1:
                # atoms.add(Atom(IS_OPEN, ["door"]))
                atoms.add(Atom(IS_OPEN, ["agent"]))
            else:
                atoms.add(Atom(IS_CLOSED, ["agent"]))
        # how2go state2atom
        if phase is 'how2go':
            if int(state[2]) > 0: atoms.add(Atom(POSX,["agent"]))
            elif int(state[2]) < 0: atoms.add(Atom(NEGX,["agent"]))
            else: atoms.add(Atom(ZEROX,["agent"]))

            if int(state[3]) > 0: atoms.add(Atom(POSY,["agent"]))
            elif int(state[3]) < 0: atoms.add(Atom(NEGY,["agent"]))
            else: atoms.add(Atom(ZEROY,["agent"]))

        # what2do state2atom
        elif phase is 'what2do':
            for adj_obj in state[5:]:
                if adj_obj == 'key':
                    atoms.add(Atom(AT_KEY, ["agent"]))
                elif adj_obj == 'door':
                    atoms.add(Atom(AT_DOOR, ["agent"]))
            if int(state[0]) == 1:
                # atoms.add(Atom(HAS_KEY, ["agent"]))
                atoms.add(Atom(H_KEY, ["agent"]))
            else:
                atoms.add(Atom(N_KEY, ["agent"]))

        return atoms
    # how to go
    def action_refine(self, action, agent_direction):
        # action-direction
        refine_map = {'00': [1, 1, 2],
                      '01': [1, 2],
                      '02': [2],
                      '03': [0, 2],
                      '20': [0, 2],
                      '21': [0, 0, 2],
                      '22': [1, 2],
                      '23': [2],
                      '10': [2],
                      '11': [0, 2],
                      '12': [1, 1, 2],
                      '13': [1, 2],
                      '30': [1, 2],
                      '31': [2],
                      '32': [0, 2],
                      '33': [1, 1, 2]}

        return refine_map[str(action)+str(agent_direction)]

    def w2d_action_refine(self, agent_direction, agent_target_direction):
        # agent_target_direction - agent_direction
        refine_map = {'00': [1],
                      '01': [],
                      '02': [0],
                      '03': [1, 1],
                      '20': [0],
                      '21': [1, 1],
                      '22': [1],
                      '23': [],
                      '10': [],
                      '11': [0],
                      '12': [1, 1],
                      '13': [1],
                      '30': [1, 1],
                      '31': [1],
                      '32': [],
                      '33': [0]}
        agent_target_dict = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        agent_target_dir = agent_target_dict.index(agent_target_direction.tolist())

        return refine_map[str(agent_target_dir)+str(agent_direction)]

    def save_image(self,):
        if self.save:
            img = self.env.render('rgb')
            im = Image.fromarray(img)
            im.save(os.path.join(self.save_path,str(self.step)+'.png'))

    def how2go_step_learnable(self, action):
        # print(f"phase: {self.phase}")
        print(f"in how2go,step: {self.step}, state: {self._state}")
        self.step+=1
        # print(f"action taken: {action.predicate.name}")
        action = self.mapping[action.predicate.name]
        actions = self.action_refine(action, self._direct)
        # print(f"action taken: {action}")
        for i in actions:
            self.full_state, ori_reward, done, info = self.env.step(i)
        self.save_image()
        self.full_state = self.full_state['image']
        # update various state vectors
        self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width-2) - coor_trans(self.temp_goal, self.width-2)

        if np.linalg.norm(self.agent_state) == 0:
            # rotate agent to face the object when adjacent
            self.phase = 'what2do'
            self.save_image()
            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
            # reward = 0.0
            # if self.unlock and not self.test: ori_reward, done = 1.0, done
            if self.goto_target == 'door':
                if self.loc == 'left':
                    actions = self.w2d_action_refine(self._direct, np.array([-1,0]))
                    # self.agent_state = coor_trans(self.agent_state, self.width-2) - coor_trans(np.array([-1,0]), self.width-2)
                else:
                    actions = self.w2d_action_refine(self._direct, np.array([1,0]))
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                ori_reward = 0.2
            if self.test: ori_reward, done = ori_reward, done


        if self.goto_target == 'key' and np.linalg.norm(self.agent_state) == 1:
            # rotate agent to face the object when adjacent
            actions = self.w2d_action_refine(self._direct, self.agent_state)
            for i in actions:
                self.full_state, ori_reward, done, info = self.env.step(i)
            # switch phase to what2do
            self.phase = 'what2do'
            self.save_image()
            if len(actions)>0:
                self.full_state = self.full_state['image']
            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
            # reward = 0.0
            if not self.test: ori_reward, done = 0.2, done
            if self.test: ori_reward, done = ori_reward, done
        # update various state vectors
        # self.full_state = self.full_state.reshape(self.width, self.width, 3)
        self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width-2) - coor_trans(self.temp_goal, self.width-2)

        # update h2g state
        self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
        self.surround_wall = surrounding(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state, self.temp_goal)
        self._state[4] = self.surround_wall
        # update w2d state
        # if self.phase != 'what2do':
        #     self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
        #     self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
        if done and self.goto_target != 'goal': ori_reward = -0.01
        reward = self.get_reward(done, ori_reward)
        print(f"in how2go,present state: {self._state},  reward: {ori_reward}\n")

        if done and reward>0.1:
            print("reached goal!")

        return reward, done

    def next_step(self, action):
        # print(f"action: {[action.predicate.name, action.terms]}")
        # fixed where2go
        self.has_key = 1-int(np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]], np.array([5, 4, 0])))
        self.door_open = int(self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0)

        # learnable where2go
        if self.phase is 'where2go':
            # self.step+=1

            # if not self.unlock:
            self.goto_target = action.predicate.name.split('_')[-1]#action.terms[0]
            if self.goto_target == 'key':
                self.temp_goal = self.key_loc
            elif self.goto_target == 'door':
                if self.loc == 'left':
                    temp_loc = [self.door_loc[0]-1, self.door_loc[1]]
                else:temp_loc = [self.door_loc[0]+1, self.door_loc[1]]
                self.temp_goal = temp_loc
            elif self.goto_target == 'goal':
                self.temp_goal = self.goal_pos
            print(f"in where2go round: {self.round}, step: {self.step}, self.goto_target: {self.goto_target}")

            self._state[2:4] = coor_trans(np.array(np.where(self.full_state==10))[:2].reshape(-1),self.width-2) - coor_trans(self.temp_goal,self.width-2)
            if self._state[2] == 0 and self._state[3] == 0:
                if self.goto_target == 'door':
                    if self.loc == 'left':temp_state = np.array([-1,0])
                    else:temp_state = np.array([1,0])
                    actions = self.w2d_action_refine(self._direct, temp_state)
                    for i in actions:
                        self.full_state, ori_reward, done, info = self.env.step(i)
                        self.full_state = self.full_state['image']
                        self.step += 1
                        self.save_image()
                self.phase = 'what2do'
                self._state[5] = self.goto_target
                return -0.01, False

            self.phase = 'how2go'
            return -0.01, False

        if self.phase is 'how2go':
            reward, done = self.how2go_step_learnable(action)
            return reward, done

        if self.phase is 'what2do':
            self.step+=1
            # if not self.unlock:
            action_name = action.predicate.name
            action = action.predicate.name
            action = self.w2d_mapping[action]
            self.full_state, ori_reward, done, info = self.env.step(action)
            self.save_image()
            self.full_state = self.full_state['image']
            # update self._state[:2] (has_key, door_open)
            print(f"round: {self.round}, step: {self.step}")
            print("previous has key: ", self.has_key,
                    "picked up key: ", np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]], np.array([1, 0, 0])),
                    "previous door open: ", self.door_open,
                    "opened door: ", self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0,
                    "action: ", action_name)
            if (not self.has_key) and np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]], np.array([1, 0, 0])):
                self._state[0] = 1 # the key has been picked up by the agent
                print(f"success pickup!")
                # self.full_state = self.full_state.reshape(self.width, self.width, 3)
                self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
                self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
                self.phase = 'where2go'
                ori_reward = 0.3
                if self.test:ori_reward = 0.0
                return ori_reward, done,
            if (not self.door_open) and self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0:
                self._state[1] = 1 # door is open
                print(f"success open door!")
                self.full_state, ori_reward, done, info = self.env.step(2)
                self.full_state, ori_reward, done, info = self.env.step(2)
                # if self.loc == 'left':
                #     self.loc = 'right'
                # else:
                #     self.loc = 'left'
                if done:return ori_reward, done

                self.full_state = self.full_state['image']
                self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
                self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]

                self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
                self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
                self.phase = 'where2go'
                if not self.prev_door_open:
                    ori_reward = 0.3
                    self.prev_door_open = True
                    self.done = done
                if self.test:ori_reward = 0.0
                return ori_reward, done,
            self.phase = 'where2go'
            return -0.01, done,

    def get_reward(self, done, ori_reward=0.0):
        if self.test:return ori_reward
        if done and ori_reward>0.1:
            return ori_reward
        elif done:
            return ori_reward
        if ori_reward>0: return ori_reward
        return -0.01

    def vary(self):
        pass

    # def get_real_target(self, obj_loc, agent_loc):
    @property
    def critic_state(self):
        return copy.deepcopy(self._state[2:4])

    def flatten(self,):
        state = []
        state.extend(self.w2h_state['w2g'])
        temp = []
        for key, item in self.w2h_state['h2g'][0].items():
            state.extend(item)
        state.extend([int(i) for i in self.w2h_state['h2g'][1]])
        state.extend(self.w2h_state['w2d'])
        return state

    def change_state(self):
        state = self.flatten()
        full_state = []
        obj_encoding = [0,0,0]
        obj_mapping = {
            'key':0,
            'door':1,
            'goal':2
        }
        for index in range(len(state)):
            if index < len(state)-2:
                full_state.append(int(state[index]))
            else:
                if state[index] in obj_mapping:
                    obj_encoding[obj_mapping[state[index]]] = 1
        full_state.extend(obj_encoding)

        return np.array(full_state, dtype = np.float16)

    def reset(self, unlock = True, test = False):
        # self.env.render()
        self.test = test
        self.full_state = self.env.reset()
        self.full_state = self.full_state['image']
        self.unseen_background = []
        # goal location
        self.goal_pos = np.array(np.where(self.full_state==8))[:2].reshape(-1)
        print(f"goal_pos: {self.goal_pos}")
        self.unseen_background.append(Atom(GOAL, [str(self.goal_pos[0]), str(self.goal_pos[1])]))
        # key location
        self.key_loc = np.where(((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:,None,None,:]).all(3)).any(0))
        # print(f"full_state: {self.full_state.reshape(self.width,self.width,3)}")
        print(f"key_loc: {self.key_loc}")

        self.key_loc = [self.key_loc[0][0], self.key_loc[1][0]]
        # door location
        self.door_loc = np.where(((self.full_state == np.array([4, 4, 2]).reshape(1, -1)[:,None,None,:]).all(3)).any(0))
        self.door_loc = [self.door_loc[0][0], self.door_loc[1][0]]
        # initial state vector (agent location)
        self.init_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        # relative postional vector (now cheating)
        self.init_dir_vector = coor_trans(self.init_state, self.width-2) - coor_trans(self.key_loc, self.width-2)
        # initial direction scalar
        self.init_dir = self.full_state[self.init_state[0], self.init_state[1]][2]
        self.init_state = (str(self.init_dir_vector[0]), str(self.init_dir_vector[1]))
        # agent has key?
        self.has_key = str(0)
        # door is open?
        self.is_open = str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        # adjacent to?
        self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
        # surrounding wallls
        self.surround_wall = surrounding(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
        print(f"self.surround_wall: {self.surround_wall}")
        if self.key_loc[0] < self.door_loc[0]:
            # self.drop_pos = [self.goal_pos[0]-1, self.goal_pos[1]-1]
            self.loc = 'left'
        else:
            self.loc = 'right'
        # w2h state [w2g, h2g, w2d]
        self.w2h_state = ['0', str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))]
        self.w2h_state.extend(list(self.init_state))
        self.w2h_state.extend([self.surround_wall])   # add more observations to how2go state to make it markovian (#4)
        self.w2h_state.extend(self.adjacent)    # (#2)
        self.w2h_state.extend(['null'] * (7-len(self.w2h_state)))
        # b = copy.deepcopy(self.w2h_state)
        # self.init_w2h_state = copy.deepcopy(self.w2h_state)
        self._state = copy.deepcopy(self.w2h_state)
        print(self.w2h_state)

        self._direct = copy.deepcopy(self.init_dir)
        self.phase = 'where2go'  # where2go, how2go, what2do
        self.round += 1
        self.agent_state = 1
        path = 'log_images_closedoor'
        if self.save:
            if not os.path.exists(os.path.join(path, str(self.round))):
                os.makedirs(os.path.join(path, str(self.round)))
            self.save_path = os.path.join(path, str(self.round))
        self.step = 0
        self.prev_door_open = False
        self.done = False

class UnlockPickUp():
    def __init__(self, initial_state=("1", "1"), env_name='MiniGrid-UnlockPickup-8x8-v0', width=8, seed = 0, size=6, save = False):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]
        # h2g_actions = list(pn_dict.values())
        # w2g_actions = [GOTO]
        w2g_actions = [GT_KEY, GT_DOOR, GT_DROP , GT_GOAL]
        w2d_actions = [TOGGLE, PICK, DROP]
        self.actions = h2g_actions + w2g_actions + w2d_actions

        # h2g_ext = [CURRENT, PXPY, PXNY, PXZY, NXPY, NXNY, NXZY, ZXPY, ZXNY]
        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        w2g_ext = [IS_OPEN, IS_CLOSED, HAS_KEY, NO_KEY]

        w2d_ext = [AT_KEY, AT_DOOR, H_KEY, N_KEY, AT_GOAL, AT_DROP]
        ext = h2g_ext + w2g_ext + w2d_ext

        self.env = UnlockPickup(room_size=size)
        self.width = 2*size - 1
        self.height = size
        self.round = -1
        self.env = FullyObsWrapper(self.env)
        self.seed = seed
        self.save = save
        self.reset()

        self.mapping = {'left':0, 'right':1, 'up':2, 'down':3}
        self._object_encoding = {'agent':0, 'key':1, 'door':2, 'goal':3}
        self.w2d_mapping = {'pick':3, 'toggle':5, 'drop':4}
        constants = [str(i) for i in range(-1*(self.width), self.width-1)] + ['key', 'door', 'agent']

        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        self.background = []

    @property
    def all_actions(self):
        ac = self.actions
        return ac

    def state2vector(self, state):

        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self,state):
        full_state = []
        obj_encoding = [0,0,0,0]
        obj_mapping = {
            'key':0,
            'door':1,
            'goal':2,
            'null':3
        }
        # obj_encoding[obj_mapping[state[-2]]] = 1
        full_state.append(int(state[0]))
        full_state.append(int(state[1]))
        full_state.append(obj_mapping[state[-2]])
        return np.array(full_state, dtype = np.float16)

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        # return current_position & current_direction
        return len(self.actions)

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # return atoms depend on which phase the agent is in

        # where2go state2atom
        if phase is 'where2go':
            if int(state[0]) == 1:
                # atoms.add(Atom(HAS_KEY, ["agent"]))
                atoms.add(Atom(HAS_KEY, ["agent"]))
            else:
                atoms.add(Atom(NO_KEY, ["agent"]))

            if int(state[1]) == 1:
                # atoms.add(Atom(IS_OPEN, ["door"]))
                atoms.add(Atom(IS_OPEN, ["agent"]))
            else:
                atoms.add(Atom(IS_CLOSED, ["agent"]))
        # how2go state2atom
        if phase is 'how2go':
            if int(state[2]) > 0: atoms.add(Atom(POSX,["agent"]))
            elif int(state[2]) < 0: atoms.add(Atom(NEGX,["agent"]))
            else: atoms.add(Atom(ZEROX,["agent"]))

            if int(state[3]) > 0: atoms.add(Atom(POSY,["agent"]))
            elif int(state[3]) < 0: atoms.add(Atom(NEGY,["agent"]))
            else: atoms.add(Atom(ZEROY,["agent"]))

        # what2do state2atom
        elif phase is 'what2do':
            for adj_obj in state[5:]:
                if adj_obj == 'key':
                    atoms.add(Atom(AT_KEY, ["agent"]))
                elif adj_obj == 'door':
                    atoms.add(Atom(AT_DOOR, ["agent"]))
                elif adj_obj == 'goal':
                    atoms.add(Atom(AT_GOAL, ["agent"]))
                elif adj_obj == 'drop':
                    atoms.add(Atom(AT_DROP, ["agent"]))
            if int(state[0]) == 1:
                # atoms.add(Atom(HAS_KEY, ["agent"]))
                atoms.add(Atom(H_KEY, ["agent"]))
            else:
                atoms.add(Atom(N_KEY, ["agent"]))

        return atoms

    def action_refine(self, action, agent_direction):
        # action-direction
        refine_map = {'00': [1, 1, 2],
                      '01': [1, 2],
                      '02': [2],
                      '03': [0, 2],
                      '20': [0, 2],
                      '21': [0, 0, 2],
                      '22': [1, 2],
                      '23': [2],
                      '10': [2],
                      '11': [0, 2],
                      '12': [1, 1, 2],
                      '13': [1, 2],
                      '30': [1, 2],
                      '31': [2],
                      '32': [0, 2],
                      '33': [1, 1, 2]}

        return refine_map[str(action)+str(agent_direction)]

    def w2d_action_refine(self, agent_direction, agent_target_direction):
        # agent_target_direction - agent_direction
        refine_map = {'00': [1],
                      '01': [],
                      '02': [0],
                      '03': [1, 1],
                      '20': [0],
                      '21': [1, 1],
                      '22': [1],
                      '23': [],
                      '10': [],
                      '11': [0],
                      '12': [1, 1],
                      '13': [1],
                      '30': [1, 1],
                      '31': [1],
                      '32': [],
                      '33': [0]}
        agent_target_dict = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        agent_target_dir = agent_target_dict.index(agent_target_direction.tolist())

        return refine_map[str(agent_target_dir)+str(agent_direction)]# + [str(action)]

    def save_image(self,):
        if self.save:
            img = self.env.render('rgb')
            im = Image.fromarray(img)
            im.save(os.path.join(self.save_path,str(self.step)+'.png'))

    def how2go_step_learnable(self, action):
        print(f"in how2go,step: {self.step}, state: {self._state}, agent_state: {self.agent_state}, target: {self.goto_target}, loc : {self.temp_goal}")
        action = self.mapping[action.predicate.name]
        actions = self.action_refine(action, self._direct)
        # print(f"action taken: {action}")
        for i in actions:
            self.step+=1
            self.full_state, ori_reward, done, info = self.env.step(i)
            self.save_image()
        self.full_state = self.full_state['image']

        if self.test: ori_reward, done = ori_reward, done
        # update various state vectors
        self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width-2) - coor_trans(self.temp_goal, self.width-2)
        if np.linalg.norm(self.agent_state) == 0:
            if self.goto_target == 'door':
                if self.loc == 'left':temp_state = np.array([-1,0])
                else:temp_state = np.array([1,0])
                actions = self.w2d_action_refine(self._direct, temp_state)
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                    self.step += 1
                    self.save_image()
                if not self.test: ori_reward = 1.0
                else:ori_reward = 0.1
            self.phase = 'what2do'
            # self.save_image()
            self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
            self.agent_state = coor_trans(self.agent_state, self.width-2) - coor_trans(self.temp_goal, self.width-2)

            self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
            self.surround_wall = surrounding(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state, self.temp_goal)
            self._state[4] = self.surround_wall

            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
        if np.linalg.norm(self.agent_state) == 1 and self.goto_target!='door': #or (np.linalg.norm(self.agent_state) == 0 and self.goto_target=='door'):
            # rotate agent to face the object when adjacent
            if np.linalg.norm(self.agent_state) == 0: temp_state = np.array([-1, 0])
            else:temp_state = self.agent_state
            actions = self.w2d_action_refine(self._direct, temp_state)
            for i in actions:
                self.full_state, ori_reward, done, info = self.env.step(i)
                self.full_state = self.full_state['image']
                self.step += 1
                self.save_image()
            if not self.test: ori_reward = 1.0
            if self.test:ori_reward = 0.1
            # switch phase to what2do
            self.phase = 'what2do'
            # self.save_image()
            self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
            self.agent_state = coor_trans(self.agent_state, self.width-2) - coor_trans(self.temp_goal, self.width-2)

            self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
            self.surround_wall = surrounding(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state, self.temp_goal)
            self._state[4] = self.surround_wall

            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))


        # update h2g state
        self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
        self.surround_wall = surrounding(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state, self.temp_goal)
        self._state[4] = self.surround_wall
        # update w2d state
        if self.phase != 'what2do':
            self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
            self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))

        reward = self.get_reward(done, ori_reward)
        print(f"in how2go,present state: {self._state},  reward: {reward}\n")

        if done and reward>0.1:
            print("Done!")

        return reward, done

    def what2do(self,action):
        self.step+=1
        action_name = action.predicate.name
        action = action.predicate.name
        action = self.w2d_mapping[action]
        print(f"in what2do round: {self.round}, step: {self.step}, action: {action_name}, state: {self._state}")
        if self.has_key and action_name == 'drop':
            count = 0
            for i in range(3):
                self.full_state, ori_reward, done, info = self.env.step(0)
                count+=1
                self.step+=1
                # self.save_image()
                self.full_state, ori_reward, done, info = self.env.step(action)
                self.step+=1
                # self.save_image()
                self.full_state = self.full_state['image']
                if len(np.where(((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:,None,None,:]).all(3)).any(0))[0])!=0:
                    print("success drop!")
                    self.key_loc = np.where(((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:,None,None,:]).all(3)).any(0))
                    print(f"key_loc: {self.key_loc}")

                    self.key_loc = [self.key_loc[0][0], self.key_loc[1][0]]
                    self.has_key = False
                    self._state[0] = 0
                    if (not self.dropped) and ('drop' in self._state[5:7]) and (not self.test): ori_reward = 0.3

                    self.dropped = True
                    for _ in range(count):
                        self.full_state, _, done, info = self.env.step(1)
                        if done:break
                    self.full_state = self.full_state['image']
                    self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
                    self.adjacent = [self.goto_target]
                    self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
                    self.phase = 'where2go'
                    self.save_image()
                    return ori_reward, done

        self.full_state, ori_reward, done, info = self.env.step(action)
        if done and ori_reward>0.05:
            print("reached Goal!")
            if self.test: return ori_reward,done
            return ori_reward, done
        # self.save_image()
        self.full_state = self.full_state['image']

        if (not self.has_key) and np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]], np.array([1, 0, 0])):
            self._state[0] = 1 # the key has been picked up by the agent
            print(f"success pickup!")
            self.phase = 'where2go'
            if not self.prev_has_key: ori_reward = 0.3
            if self.test:ori_reward = 0.0
            self.prev_has_key = True

            return ori_reward, done,
        if (not self.is_open) and self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0:
            self._state[1] = 1 # door is open
            print(f"success open door!")
            self.full_state, ori_reward, done, info = self.env.step(2)
            self.step += 1
            self.save_image()
            self.full_state, ori_reward, done, info = self.env.step(2)
            self.step+=1
            self.save_image()
            if self.loc == 'left':
                self.loc = 'right'
            else:
                self.loc = 'left'
            # if not self.test:
            # self.full_state, ori_reward, done, info = self.env.step(0)
            # self.full_state, ori_reward, done, info = self.env.step(0)
            # self.step+=1
            # self.save_image()
            # self.full_state, ori_reward, done, info = self.env.step(5)
            # self.full_state, ori_reward, done, info = self.env.step(0)
            # self.step+=1
            # self.save_image()
            self.full_state = self.full_state['image']
            self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]

            self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
            self.adjacent = ['door']
            self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
            self.phase = 'where2go'
            if not self.prev_door_open:
                if self.test: ori_reward = ori_reward
                else: ori_reward = 0.3
                self.prev_door_open = True
                self.done = done
            # ori_reward = self.get_reward(done,ori_reward)
            return ori_reward, done,
        self.phase = 'where2go'
        if not self.test: ori_reward -= 0.01
        return ori_reward, done

    def where2go(self, action):
        self.prev_goal = self.goto_target
        self.goto_target = action.predicate.name.split('_')[-1]
        if self.goto_target == 'key':
            self.temp_goal = self.key_loc
        elif self.goto_target == 'door':
            if self.loc == 'left':
                temp_loc = [self.door_loc[0]-1, self.door_loc[1]]
            else:temp_loc = [self.door_loc[0]+1, self.door_loc[1]]
            self.temp_goal = temp_loc
        elif self.goto_target == 'goal':
            self.temp_goal = self.goal_pos
        elif self.goto_target == 'drop':
            self.temp_goal = self.drop_pos

        if self.prev_goal!=None and (self.goto_target == self.prev_goal) and self.goto_target!='door':
            self.phase = 'what2do'
            return 0.0, False
        self._state[2:4] = coor_trans(np.array(np.where(self.full_state==10))[:2].reshape(-1),self.width-2) - coor_trans(self.temp_goal,self.width-2)
        print(f"in where2go round: {self.round}, step: {self.step}, prev_target: {self.prev_goal} ,self.goto_target: {self.goto_target}, state:{self._state}")
        if self._state[2] == 0 and self._state[3] == 0:
            if self.goto_target == 'door':
                if self.loc == 'left':temp_state = np.array([-1,0])
                else:temp_state = np.array([1,0])
                actions = self.w2d_action_refine(self._direct, temp_state)
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                    self.step += 1
                    self.save_image()
            self.phase = 'what2do'
            self._state[5] = self.goto_target
            return -0.01, False
        self.phase = 'how2go'
        return -0.01, False

    def next_step(self, action):
        self.has_key = int(self._state[0])
        # door is open?
        self.is_open = int(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        # learnable where2go
        if self.phase is 'where2go':
            reward, done = self.where2go(action)

        elif self.phase is 'how2go':
            reward, done = self.how2go_step_learnable(action)

        elif self.phase is 'what2do':
            reward, done = self.what2do(action)

        return reward, done

    def get_reward(self, done, ori_reward=0.0):
        # return ori_reward
        if self.test:return ori_reward
        if done and ori_reward>0.1:
            return 1.0
        elif done:
            return ori_reward

        if ori_reward>0: return ori_reward
        return -0.01


    @property
    def critic_state(self):
        return copy.deepcopy(self._state[2:4])

    def flatten(self,):
        state = []
        state.extend(self.w2h_state['w2g'])
        temp = []
        for key, item in self.w2h_state['h2g'][0].items():
            state.extend(item)
        state.extend([int(i) for i in self.w2h_state['h2g'][1]])
        state.extend(self.w2h_state['w2d'])
        return state

    def change_state(self):
        state = self.flatten()
        full_state = []
        obj_encoding = [0,0,0]
        obj_mapping = {
            'key':0,
            'door':1,
            'goal':2
        }
        for index in range(len(state)):
            if index < len(state)-2:
                full_state.append(int(state[index]))
            else:
                if state[index] in obj_mapping:
                    obj_encoding[obj_mapping[state[index]]] = 1
        full_state.extend(obj_encoding)

        return np.array(full_state, dtype = np.float16)

    def reset(self, unlock = True, test = False):
        self.test = test
        self.full_state = self.env.reset()
        self.full_state = self.full_state['image']
        # box location
        self.goal_pos = np.array(np.where(self.full_state==7))[:2].reshape(-1)
        img = self.env.render('rgb')
        im = Image.fromarray(img)

        self.key_loc = np.where(((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:,None,None,:]).all(3)).any(0))
        # print(f"full_state: {self.full_state.reshape(self.width,self.width,3)}")

        self.key_loc = [self.key_loc[0][0], self.key_loc[1][0]]
        # door location
        self.door_loc = np.where(((self.full_state == np.array([4, 4, 2]).reshape(1, -1)[:,None,None,:]).all(3)).any(0))
        self.door_loc = [self.door_loc[0][0], self.door_loc[1][0]]

        if self.key_loc[0] < self.door_loc[0]:
            # self.drop_pos = [self.goal_pos[0]-1, self.goal_pos[1]-1]
            self.loc = 'left'
        else:
            self.loc = 'right'
            # self.drop_pos = [self.goal_pos[0]+1, self.goal_pos[1]+1]
        self.drop_pos = self.goal_pos
        print(f"key_loc: {self.key_loc}")
        print(f"door_loc: {self.door_loc}")
        print(f"drop_loc: {self.drop_pos}")
        print(f"goal_pos: {self.goal_pos}")
        # initial state vector (agent location)
        self.init_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        # relative postional vector (now cheating)
        self.init_dir_vector = coor_trans(self.init_state, self.width-2) - coor_trans(self.key_loc, self.width-2)
        # initial direction scalar
        self.init_dir = self.full_state[self.init_state[0], self.init_state[1]][2]
        self.init_state = (str(self.init_dir_vector[0]), str(self.init_dir_vector[1]))
        # agent has key?
        self.has_key = str(0)
        # door is open?
        self.is_open = str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        # adjacent to?
        self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
        # surrounding wallls
        self.surround_wall = surrounding(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
        print(f"self.surround_wall: {self.surround_wall}")

        # w2h state [w2g, h2g, w2d]
        self.w2h_state = ['0', str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))]
        self.w2h_state.extend(list(self.init_state))
        self.w2h_state.extend([self.surround_wall])   # add more observations to how2go state to make it markovian (#4)
        self.w2h_state.extend(self.adjacent)    # (#2)
        self.w2h_state.extend(['null'] * (7-len(self.w2h_state)))
        # b = copy.deepcopy(self.w2h_state)
        # self.init_w2h_state = copy.deepcopy(self.w2h_state)
        self._state = copy.deepcopy(self.w2h_state)
        print(self.w2h_state)

        self._direct = copy.deepcopy(self.init_dir)
        self.phase = 'where2go'  # where2go, how2go, what2do
        self.round += 1
        self.agent_state = 0
        if self.save:
            path = 'log_images_unlock/seed_{}'.format(self.seed)
            if not os.path.exists(os.path.join(path, str(self.round))):
                os.makedirs(os.path.join(path, str(self.round)))
            self.save_path = os.path.join(path, str(self.round))
        self.step = 0
        self.prev_door_open = False
        self.done = False
        self.prev_has_key = False
        self.dropped = False
        self.prev_goal = None
        self.goto_target = None

class BoxKey():
    def __init__(self, initial_state=("1", "1"), env_name='MiniGrid-BoxKey-8x8-v0', width=8, save=False):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]
        w2g_actions = [GT_KEY, GT_DOOR, GT_BOX, GT_GOAL]
        w2d_actions = [TOGGLE, PICK]
        self.actions = h2g_actions + w2g_actions + w2d_actions

        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        w2g_ext = [IS_OPEN, IS_CLOSED, HAS_KEY, NO_KEY, KEY_SHOW, KEY_NOSHOW]
        # w2d_ext = [AT]
        w2d_ext = [AT_KEY, AT_BOX, AT_DOOR, H_KEY, N_KEY, AT_GOAL]
        ext = h2g_ext + w2g_ext + w2d_ext
        self.env = SimpleBoxKeyEnv(size = width)
        self.env = FullyObsWrapper(self.env)
        width = 8

        self.width = width
        self.round = -1
        self.save = save
        self.reset()
        self.mapping = {'left':0, 'right':1, 'up':2, 'down':3}
        self._object_encoding = {'agent':0, 'key':1, 'door':2, 'goal':3}
        self.w2d_mapping = {'pick':3, 'toggle':5}
        constants = [str(i) for i in range(-1*(width), width-1)] + ['key', 'door', 'box', 'agent']

        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        background = []
        self.background = background

    @property
    def all_actions(self):
        ac = self.actions
        return ac

    def state2vector(self, state):
        # what2do state vector
        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self,state):
        full_state = []
        obj_encoding = [0,0,0,0]
        obj_mapping = {
            'key':0,
            'door':1,
            'goal':2,
            'box':3
        }
        # obj_encoding[obj_mapping[state[-2]]] = 1
        full_state.append(int(state[0]))
        full_state.append(int(state[1]))
        full_state.append(obj_mapping[state[-2]])
        # full_state.extend(obj_encoding)

        return np.array(full_state, dtype = np.float16)

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        # return current_position & current_direction
        return len(self.actions)

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # return atoms depend on which phase the agent is in

        # where2go state2atom
        if phase is 'where2go':
            # if int(state[0]) == 1:
            #     # atoms.add(Atom(HAS_KEY, ["agent"]))
            #     atoms.add(Atom(HAS_KEY, ["agent"]))
            # else:
            #     atoms.add(Atom(NO_KEY, ["agent"]))

            if int(state[1]) == 1:
                # atoms.add(Atom(IS_OPEN, ["door"]))
                atoms.add(Atom(IS_OPEN, ["agent"]))
            else:
                atoms.add(Atom(IS_CLOSED, ["agent"]))

            if self.key_showup:
                if int(state[0]) == 1:
                    # atoms.add(Atom(HAS_KEY, ["agent"]))
                    atoms.add(Atom(HAS_KEY, ["agent"]))
                else:
                    atoms.add(Atom(NO_KEY, ["agent"]))
                # atoms.add(Atom(KEY_SHOW, ["agent"]))
            elif not self.key_showup:
                atoms.add(Atom(KEY_NOSHOW, ["agent"]))
        # how2go state2atom
        if phase is 'how2go':
            if int(state[2]) > 0: atoms.add(Atom(POSX,["agent"]))
            elif int(state[2]) < 0: atoms.add(Atom(NEGX,["agent"]))
            else: atoms.add(Atom(ZEROX,["agent"]))

            if int(state[3]) > 0: atoms.add(Atom(POSY,["agent"]))
            elif int(state[3]) < 0: atoms.add(Atom(NEGY,["agent"]))
            else: atoms.add(Atom(ZEROY,["agent"]))

        # what2do state2atom
        elif phase is 'what2do':
            for adj_obj in state[5:]:
                if adj_obj == 'key':
                    atoms.add(Atom(AT_KEY, ["agent"]))
                elif adj_obj == 'door':
                    atoms.add(Atom(AT_DOOR, ["agent"]))
                elif adj_obj == 'box':
                    atoms.add(Atom(AT_BOX, ["agent"]))
            if int(state[0]) == 1:
                # atoms.add(Atom(HAS_KEY, ["agent"]))
                atoms.add(Atom(H_KEY, ["agent"]))
            else:
                atoms.add(Atom(N_KEY, ["agent"]))

        return atoms
        # return [Atom(CURRENT, state)]
    # how to go
    def action_refine(self, action, agent_direction):
        # action-direction
        refine_map = {'00': [1, 1, 2],
                      '01': [1, 2],
                      '02': [2],
                      '03': [0, 2],
                      '20': [0, 2],
                      '21': [0, 0, 2],
                      '22': [1, 2],
                      '23': [2],
                      '10': [2],
                      '11': [0, 2],
                      '12': [1, 1, 2],
                      '13': [1, 2],
                      '30': [1, 2],
                      '31': [2],
                      '32': [0, 2],
                      '33': [1, 1, 2]}

        return refine_map[str(action)+str(agent_direction)]

    def w2d_action_refine(self, agent_direction, agent_target_direction):
        # agent_target_direction - agent_direction
        refine_map = {'00': [1],
                      '01': [],
                      '02': [0],
                      '03': [1, 1],
                      '20': [0],
                      '21': [1, 1],
                      '22': [1],
                      '23': [],
                      '10': [],
                      '11': [0],
                      '12': [1, 1],
                      '13': [1],
                      '30': [1, 1],
                      '31': [1],
                      '32': [],
                      '33': [0]}
        agent_target_dict = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        agent_target_dir = agent_target_dict.index(agent_target_direction.tolist())

        return refine_map[str(agent_target_dir)+str(agent_direction)]# + [str(action)]

    def save_image(self,):
        if self.save:
            img = self.env.render('rgb')
            im = Image.fromarray(img)
            im.save(os.path.join(self.save_path,str(self.step)+'.png'))

    def how2go_step_learnable(self, action):
        # print(f"phase: {self.phase}")
        # print(f"in how2go,step: {self.step}, state: {self._state}")
        self.step+=1
        # print(f"action taken: {action.predicate.name}")
        action = self.mapping[action.predicate.name]
        # action = np.random.choice(4)
        actions = self.action_refine(action, self._direct)
        # print(f"action taken: {action}")
        for i in actions:
            self.full_state, ori_reward, done, info = self.env.step(i)
        self.save_image()
        self.full_state = self.full_state['image']
        # update various state vectors
        self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width-2) - coor_trans(self.temp_goal, self.width-2)

        if np.linalg.norm(self.agent_state) == 0:
            # rotate agent to face the object when adjacent
            self.phase = 'what2do'
            self.save_image()
            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
            # reward = 0.0
            # if self.unlock and not self.test: ori_reward, done = 1.0, done
            if self.goto_target == 'door':
                if self.loc == 'left':
                    actions = self.w2d_action_refine(self._direct, np.array([-1,0]))
                    # self.agent_state = coor_trans(self.agent_state, self.width-2) - coor_trans(np.array([-1,0]), self.width-2)
                else:
                    actions = self.w2d_action_refine(self._direct, np.array([1,0]))
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                ori_reward = 0.2
            if self.test: ori_reward, done = ori_reward, done

        elif (self.goto_target != 'goal') and (self.goto_target != 'door') and np.linalg.norm(self.agent_state) == 1:
            # rotate agent to face the object when adjacent
            actions = self.w2d_action_refine(self._direct, self.agent_state)
            for i in actions:
                self.full_state, ori_reward, done, info = self.env.step(i)
            # switch phase to what2do
            self.phase = 'what2do'
            self.save_image()
            if len(actions)>0:
                self.full_state = self.full_state['image']
            self.adjacent = [self.goto_target]
            # self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
            self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
            # reward = 0.0
            if self.unlock and not self.test: ori_reward, done = 0.2, done
            if self.test: ori_reward, done = ori_reward, done
        # update various state vectors
        # self.full_state = self.full_state.reshape(self.width, self.width, 3)
        self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width-2) - coor_trans(self.temp_goal, self.width-2)

        # update h2g state
        self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
        self.surround_wall = surrounding(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state, self.temp_goal)
        self._state[4] = self.surround_wall
        # update w2d state
        if done and self.goto_target != 'goal': ori_reward = -0.01
        reward = self.get_reward(done, ori_reward)
        # print(f"in how2go,present state: {self._state},  reward: {ori_reward}\n")

        if done and reward>0.5:
            print("reached goal!")

        return reward, done

    def next_step(self, action):
        # print(f"action: {[action.predicate.name, action.terms]}")
        # fixed where2go
        self.has_key = 1-int(np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]], np.array([7, 4, 0])) or np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]], np.array([5, 4, 0])))
        self.door_open = int(self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0)
        self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        # update key and box location
        self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]], np.array([7, 4, 0]))
        if self.key_showup:
            self.key_loc = self.box_loc_fix.copy()
            self.box_loc = [0, 0]
        else:
            self.key_loc = [0, 0]

        # learnable where2go
        if self.phase is 'where2go':
            self.goto_target = action.predicate.name.split('_')[-1]#action.terms[0]
            if self.goto_target == 'key':
                self.temp_goal = self.key_loc
            elif self.goto_target == 'box':
                self.temp_goal = self.box_loc
            elif self.goto_target == 'door':
                if self.loc == 'left':
                    temp_loc = [self.door_loc[0]-1, self.door_loc[1]]
                else:temp_loc = [self.door_loc[0]+1, self.door_loc[1]]
                self.temp_goal = temp_loc
            elif self.goto_target == 'goal':
                self.temp_goal = self.goal_pos

            # has_key while still going to key
            if (self._state[0] == 1) and (self.goto_target == 'key' or self.goto_target == 'box'):
                return -0.01, False
            # goto key when key no-show
            if np.array_equal(self.temp_goal, [0, 0]):
                return -0.01, False

            print(f"in where2go round: {self.round}, step: {self.step}, self.goto_target: {self.goto_target}")

            self._state[2:4] = coor_trans(np.array(np.where(self.full_state==10))[:2].reshape(-1),self.width-2) - coor_trans(self.temp_goal,self.width-2)
            if self._state[2] == 0 and self._state[3] == 0:
                if self.goto_target == 'door':
                    if self.loc == 'left':temp_state = np.array([-1,0])
                    else:temp_state = np.array([1,0])
                    actions = self.w2d_action_refine(self._direct, temp_state)
                    for i in actions:
                        self.full_state, ori_reward, done, info = self.env.step(i)
                        self.full_state = self.full_state['image']
                        self.step += 1
                        self.save_image()
                self.phase = 'what2do'
                self._state[5] = self.goto_target
                return -0.01, False
            self.phase = 'how2go'
            return -0.01, False

        if self.phase is 'how2go':
            reward, done = self.how2go_step_learnable(action)
            return reward, done

        if self.phase is 'what2do':
            self.step+=1
            # if not self.unlock:
            action_name = action.predicate.name
            action = action.predicate.name
            # else:
            #     if self.temp_goal == self.key_loc:action, action_name = 'pick', 'pick'
            #     if self.temp_goal == self.door_loc: action, action_name = 'toggle', 'toggle'
            action = self.w2d_mapping[action]
            if 'box' in self._state[5:] and action == 3:
                self.phase = 'where2go'
                return -0.01, False
            else:
                self.full_state, ori_reward, done, info = self.env.step(action)
            self.save_image()
            self.full_state = self.full_state['image']

            self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]],
                                                 np.array([7, 4, 0]))
            if self.key_showup:
                self.key_loc = self.box_loc_fix.copy()
                self.box_loc = [0, 0]
            else:
                self.key_loc = [0, 0]

            # update self._state[:2] (has_key, door_open)
            print(f"round: {self.round}, step: {self.step}")
            print("key_showup", self.key_showup)
            print("previous has key: ", self.has_key,
                    "picked up key: ", np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]], np.array([1, 0, 0])),
                    "previous door open: ", self.door_open,
                    "opened door: ", self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0,
                    "action: ", action_name)
            if (not self.has_key) and np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]], np.array([1, 0, 0])):
                self._state[0] = 1 # the key has been picked up by the agent
                print(f"success pickup!")
                # self.full_state = self.full_state.reshape(self.width, self.width, 3)
                self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
                self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
                self.phase = 'where2go'
                if self.unlock:
                    ori_reward = 1.0
                else: ori_reward = 0.3
                if self.test:ori_reward = 0.0
                return ori_reward, done,
            if (not self.door_open) and self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0:
                self._state[1] = 1 # door is open
                print(f"success open door!")
                self.full_state, ori_reward, done, info = self.env.step(2)
                self.full_state, ori_reward, done, info = self.env.step(2)
                if done:return ori_reward, done
                # if not self.test:
                #     self.full_state, ori_reward, done, info = self.env.step(0)
                #     self.full_state, ori_reward, done, info = self.env.step(0)
                #     self.full_state, ori_reward, done, info = self.env.step(5)
                self.full_state = self.full_state['image']
                self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
                self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]

                self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
                self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
                self.phase = 'where2go'
                if not self.prev_door_open:
                    if self.unlock:
                        ori_reward = 1.0
                    else: ori_reward = 0.3
                    self.prev_door_open = True
                    self.done = done
                # else: ori_reward = -0.002
                if self.test:ori_reward = 0.0
                return ori_reward, done,
            self.phase = 'where2go'
            self._state[1] = int(self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0)
            return -0.01, done,

    def get_reward(self, done, ori_reward=0.0):
        if done and ori_reward>0.1:
            return ori_reward
        elif done:
            return ori_reward-0.002
        if self.unlock == True:
            if ori_reward>0: return ori_reward
            return -0.01
        else: return -0.002

    def vary(self):
        pass

    # def get_real_target(self, obj_loc, agent_loc):
    @property
    def critic_state(self):
        return copy.deepcopy(self._state[2:4])

    def flatten(self,):
        state = []
        state.extend(self.w2h_state['w2g'])
        temp = []
        for key, item in self.w2h_state['h2g'][0].items():
            state.extend(item)
        state.extend([int(i) for i in self.w2h_state['h2g'][1]])
        state.extend(self.w2h_state['w2d'])
        return state

    def change_state(self):
        state = self.flatten()
        full_state = []
        obj_encoding = [0,0,0]
        obj_mapping = {
            'key':0,
            'door':1,
            'goal':2
        }
        for index in range(len(state)):
            if index < len(state)-2:
                full_state.append(int(state[index]))
            else:
                if state[index] in obj_mapping:
                    obj_encoding[obj_mapping[state[index]]] = 1
        full_state.extend(obj_encoding)

        return np.array(full_state, dtype = np.float16)

    def reset(self, unlock = False, test = False):
        self.test = test
        self.full_state = self.env.reset()
        self.full_state = self.full_state['image']
        self.unseen_background = []
        # goal location
        self.goal_pos = np.array(np.where(self.full_state==8))[:2].reshape(-1)
        print(f"goal_pos: {self.goal_pos}")
        self.unseen_background.append(Atom(GOAL, [str(self.goal_pos[0]), str(self.goal_pos[1])]))

        # box location
        self.box_loc = np.where(((self.full_state == np.array([7, 4, 0]).reshape(1, -1)[:,None,None,:]).all(3)).any(0))
        self.box_loc = [self.box_loc[0][0], self.box_loc[1][0]]
        self.box_loc_fix = self.box_loc.copy()

        # key location
        self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]], np.array([7, 4, 0]))
        if self.key_showup:
            self.key_loc = self.box_loc_fix.copy()
            self.box_loc = [0, 0]
        else:
            self.key_loc = [0, 0]         # goto_key and key_loc==[0, 0]则直接惩罚且return (remain where2go)
        # door location
        self.door_loc = np.where(((self.full_state == np.array([4, 4, 2]).reshape(1, -1)[:,None,None,:]).all(3)).any(0))
        self.door_loc = [self.door_loc[0][0], self.door_loc[1][0]]
        if self.box_loc[0] < self.door_loc[0]:
            # self.drop_pos = [self.goal_pos[0]-1, self.goal_pos[1]-1]
            self.loc = 'left'
        else:
            self.loc = 'right'
        # initial state vector (agent location)
        self.init_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        # relative postional vector (now cheating)
        self.init_dir_vector = coor_trans(self.init_state, self.width-2) - coor_trans(self.key_loc, self.width-2)
        # initial direction scalar
        self.init_dir = self.full_state[self.init_state[0], self.init_state[1]][2]
        self.init_state = (str(self.init_dir_vector[0]), str(self.init_dir_vector[1]))
        # agent has key?
        self.has_key = str(0)
        # door is open?
        self.is_open = str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        # adjacent to?
        self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
        # surrounding wallls
        self.surround_wall = surrounding(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
        print(f"self.surround_wall: {self.surround_wall}")

        # w2h state [w2g, h2g, w2d]
        self.w2h_state = ['0', str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))]
        self.w2h_state.extend(list(self.init_state))
        self.w2h_state.extend([self.surround_wall])   # add more observations to how2go state to make it markovian (#4)
        self.w2h_state.extend(self.adjacent)    # (#2)
        self.w2h_state.extend(['null'] * (7-len(self.w2h_state)))
        # b = copy.deepcopy(self.w2h_state)
        # self.init_w2h_state = copy.deepcopy(self.w2h_state)
        self._state = copy.deepcopy(self.w2h_state)
        print(self.w2h_state)

        self._direct = copy.deepcopy(self.init_dir)
        self.phase = 'where2go'  # where2go, how2go, what2do
        self.round += 1
        self.agent_state = 1

        path = 'log_images_boxdoor'
        if self.save:
            if not os.path.exists(os.path.join(path, str(self.round))):
                os.makedirs(os.path.join(path, str(self.round)))
            self.save_path = os.path.join(path, str(self.round))
        self.step = 0
        self.prev_door_open = False
        self.unlock = unlock
        self.done = False
# multiroom w2g actions
GT_RED = Predicate('gt_red', 0)
GT_YELLOW = Predicate('gt_yellow', 0)
GT_PURPLE = Predicate('gt_purple', 0)
GT_GREEN = Predicate('gt_green', 0)
GT_BLUE = Predicate('gt_blue', 0)

# multiroom w2g actions
NEAR_R = Predicate('near_r', 1)
NEAR_P = Predicate('near_p', 1)
NEAR_Y = Predicate('near_y', 1)
NEAR_B = Predicate('near_b', 1)
NEAR_G = Predicate('near_g', 1) # near_goal

OPEN_R = Predicate('open_r', 1)
CLOSE_R = Predicate('close_r', 1)
OPEN_P = Predicate('open_p', 1)
CLOSE_P = Predicate('close_p', 1)
OPEN_Y = Predicate('open_y', 1)
CLOSE_Y = Predicate('close_y', 1)
OPEN_B = Predicate('open_b', 1)
CLOSE_B = Predicate('close_b', 1)
class MultiRoom():

    def __init__(self, initial_state=("1", "1"), env_name='MiniGrid-BoxKey-8x8-v0', size=6, save=False):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]
        w2g_actions = [GT_RED, GT_YELLOW, GT_BLUE, GT_GOAL]
        w2d_actions = [TOGGLE, PICK]
        self.actions = h2g_actions + w2g_actions + w2d_actions

        self.DRL_actions = h2g_actions + w2d_actions

        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        w2g_ext = [NEAR_R, NEAR_Y, NEAR_B, NEAR_G, OPEN_R, CLOSE_R, OPEN_Y, CLOSE_Y, OPEN_B, CLOSE_B]
        # w2d_ext = [AT]
        w2d_ext = [AT_DOOR, IS_OPEN, IS_CLOSED]
        ext = h2g_ext + w2g_ext + w2d_ext
        # self.env = SimpleBoxKeyEnv(size = width)
        self.env = SimpleMultiRoom(maxRoomSize=size)
        self.env = FullyObsWrapper(self.env)
        # width = 8

        self.width = size
        self.round = -1
        self.save = save
        self.reset()
        self.mapping = {'left':0, 'right':1, 'up':2, 'down':3}
        self._object_encoding = {'agent':0, 'key':1, 'door':2, 'goal':3}
        self.w2d_mapping = {'pick':3, 'toggle':5}
        self.near_mapping = {'red': NEAR_R,
                              'yellow': NEAR_Y,
                              'purple': NEAR_P,
                              'blue': NEAR_B,
                              'goal': NEAR_G}
        self.door_state_map = {'o_r': OPEN_R,
                               'c_r': CLOSE_R,
                               'o_y': OPEN_Y,
                               'c_y': CLOSE_Y,
                               'o_b': OPEN_B,
                               'c_b': CLOSE_B}
        constants = [str(i) for i in range(-1*(self.width), self.width-1)] + ['key', 'door', 'box', 'agent']

        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        background = []
        self.background = background

    @property
    def all_actions(self):
        ac = self.actions
        return ac

    @property
    def DRL_Actions(self):
        return len(self.DRL_actions)

    def state2vector(self, state):
        # what2do state vector
        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self):
        self.red_open = int(self.full_state[self.red_loc[0], self.red_loc[1]][2] == 0)
        self.yellow_open = int(self.full_state[self.yellow_loc[0], self.yellow_loc[1]][2] == 0)
        self.blue_open = int(self.full_state[self.blue_loc[0], self.blue_loc[1]][2] == 0)
        agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        # agent state(x,y), red door loc (x,y), open; yellow, blue;
        agent_red = coor_trans(agent_state, self.width-2) - coor_trans(self.red_loc, self.width-2)
        agent_yellow = coor_trans(agent_state, self.width-2) - coor_trans(self.yellow_loc, self.width-2)
        agent_blue = coor_trans(agent_state, self.width-2) - coor_trans(self.blue_loc, self.width-2)
        agent_goal = coor_trans(agent_state, self.width-2) - coor_trans(self.goal_pos, self.width-2)

        full_state = []
        # full_state.extend(agent_state)
        full_state.extend(list(agent_red))
        full_state.append(self.red_open)
        full_state.extend(list(agent_yellow))
        full_state.append(self.yellow_open)
        full_state.extend(list(agent_blue))
        full_state.append(self.blue_open)
        full_state.extend(list(agent_goal))
        w2d_state = [0,0,0]
        if np.linalg.norm(agent_red) == 1 and not self.red_open:
            w2d_state[0] = 1
        if np.linalg.norm(agent_yellow) == 1 and not self.yellow_open:
            w2d_state[1] = 1
        if np.linalg.norm(agent_blue) == 1 and not self.blue_open:
            w2d_state[2] = 1
        full_state.extend(w2d_state)
        return np.array(full_state, dtype = np.float16)

    def next_step_DRL(self,action):
        self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        if action <= 3:
            action = self.DRL_actions[action]
            action = self.mapping[action.name]
            actions = self.action_refine(action, self._direct)
            for i in actions:
                self.step += 1
                self.full_state, ori_reward, done, info = self.env.step(i)
                self.full_state = self.full_state['image']
                self.save_image()
                if done: return self.w2_state2vector(),self.get_reward(done, ori_reward), done

            # self.full_state = self.full_state['image']
        else:
            agent_red = coor_trans(self.agent_state, self.width-2) - coor_trans(self.red_loc, self.width-2)
            agent_yellow = coor_trans(self.agent_state, self.width-2) - coor_trans(self.yellow_loc, self.width-2)
            agent_blue = coor_trans(self.agent_state, self.width-2) - coor_trans(self.blue_loc, self.width-2)
            if np.linalg.norm(agent_red) == 1:
                actions = self.w2d_action_refine(self._direct, agent_red)
            elif np.linalg.norm(agent_blue) == 1:
                actions = self.w2d_action_refine(self._direct, agent_blue)
            elif np.linalg.norm(agent_yellow) == 1:
                actions = self.w2d_action_refine(self._direct, agent_yellow)
            else:actions = []
            for i in actions:
                self.step += 1
                self.full_state, ori_reward, done, info = self.env.step(i)
                self.full_state = self.full_state['image']
                self.save_image()
                if done: return self.w2_state2vector(),self.get_reward(done, ori_reward), done
            action = self.DRL_actions[action]
            action = self.w2d_mapping[action.name]
            self.full_state, ori_reward, done, info = self.env.step(action)
            self.full_state = self.full_state['image']
            self.save_image()
            if ((not self.red_open) and self.full_state[self.red_loc[0], self.red_loc[1]][2] == 0) or \
            ((not self.yellow_open) and self.full_state[self.yellow_loc[0], self.yellow_loc[1]][2] == 0) or \
            ((not self.blue_open) and self.full_state[self.blue_loc[0], self.blue_loc[1]][2] == 0):
                self._state[1] = 1 # door is open
                # print(f"success open door!")
                self.full_state, ori_reward, done, info = self.env.step(2)
                self.full_state, ori_reward, done, info = self.env.step(2)
                self.full_state = self.full_state['image']
                if done:return self.w2_state2vector(),self.get_reward(done,ori_reward), done

                self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
                self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]

                self.red_open = int(self.full_state[self.red_loc[0], self.red_loc[1]][2] == 0)
                self.yellow_open = int(self.full_state[self.yellow_loc[0], self.yellow_loc[1]][2] == 0)
                self.blue_open = int(self.full_state[self.blue_loc[0], self.blue_loc[1]][2] == 0)

                self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
                self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
                if (not self.red_open_prev) and (self.full_state[self.red_loc[0], self.red_loc[1]][2] == 0):
                    ori_reward = 0.3
                    if self.test:ori_reward = 0.0
                    self.red_open_prev = True
                elif (not self.yellow_open_prev) and (self.full_state[self.yellow_loc[0], self.yellow_loc[1]][2] == 0):
                    ori_reward = 0.3
                    if self.test:ori_reward = 0.0
                    self.yellow_open_prev = True
                elif (not self.blue_open_prev) and (self.full_state[self.blue_loc[0], self.blue_loc[1]][2] == 0):
                    ori_reward = 0.3
                    if self.test:ori_reward = 0.0
                    self.blue_open_prev = True
        return self.w2_state2vector(),self.get_reward(done,ori_reward), done

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        # return current_position & current_direction
        return len(self.actions)

    # def nearest_door_goal(self, door_goal_coor_ls:np.array, agent_pos:np.array, door_goal_ls=['red', 'yellow', 'blue', 'goal']):
    #     """
    #         door_ls: ['purple', 'yellow', ...]
    #     """
    #     # door that's nearest to entry (NED)
    #     ned = door_goal_ls[np.argmin(np.linalg.norm(door_goal_coor_ls-self.init_agent_state, axis=1))]
    #     if ned is 'red':
    #         ed_state = self.red_open
    #     elif ned is 'yellow':
    #         ed_state = self.yellow_open
    #     elif ned is 'blue':
    #         ed_state = self.blue_open
    #     # if the NED is closed, it indicates that agent shall only have one nearest door
    #     if 1-int(ed_state):
    #         return [ned]
    #     else:
    #         distance_ls = list(zip(np.linalg.norm(door_goal_coor_ls[:-1]-agent_pos, axis=1), range(4)))
    #         distance_ls = sorted(distance_ls, key=lambda x: x[0])
    #         return [door_goal_ls[i[1]] for i in distance_ls[:2]]
    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3 != []

    def door_agent_exclusive(self, path):
        new_path = []
        for idx, obj in enumerate(path):
            if (idx == 0 and 10 in obj) or ((idx == 0 and 4 in obj)) or (idx == len(path) - 1 and 10 in obj) or (
                    idx == len(path) - 1 and 4 in obj):
                pass
            else:
                new_path.append(obj)
        return new_path

    def nearest_door_goal(self, door_goal_coor_ls: np.array, agent_pos: np.array,
                          door_goal_ls=['red', 'yellow', 'blue', 'goal']):
        """
            reachable door/ goal
            agent: (x1, y1)
            sub-loc: (x2, y2)
        """
        reachable_loc = []
        for i, coor in enumerate(door_goal_coor_ls):
            delta_x = coor[0] - agent_pos[0]
            delta_y = coor[1] - agent_pos[1]
            if delta_x != 0:
                # ([x1, x2], y2)
                x_range = sorted([agent_pos[0], coor[0]])
                y_path1 = [[i, coor[1]] for i in range(x_range[0], x_range[1] + 1)]  # y_path from agent to sub-location
                y_path2 = [[i, agent_pos[1]] for i in range(x_range[0], x_range[1] + 1)]
            else:
                y_path1, y_path2 = [], []
            if delta_y != 0:
                # (x1, [y1, y2])
                y_range = sorted([agent_pos[1], coor[1]])
                x_path1 = [[agent_pos[0], i] for i in
                           range(y_range[0], y_range[1] + 1)]  # x_path from agent to sub-location
                x_path2 = [[coor[0], i] for i in range(y_range[0], y_range[1] + 1)]
            else:
                x_path1, x_path2 = [], []
            # check wheter there exists brick in either y_path or x_path, if yes: unreachable
            x_path1_obj = self.door_agent_exclusive([list(self.full_state[int(i[0]), int(i[1])]) for i in x_path1])
            y_path1_obj = self.door_agent_exclusive([list(self.full_state[int(i[0]), int(i[1])]) for i in y_path1])
            x_path2_obj = self.door_agent_exclusive([list(self.full_state[int(i[0]), int(i[1])]) for i in x_path2])
            y_path2_obj = self.door_agent_exclusive([list(self.full_state[int(i[0]), int(i[1])]) for i in y_path2])
            brick_or_door = [[2, 5, 0], [4, 0, 1], [4, 0, 0], [4, 4, 1], [4, 4, 0], [4, 2, 1], [4, 2, 0]]
            # print('icoor', i, coor)
            # print(f"x_path1_obj: {x_path1_obj}")
            # print(f"y_path1_obj: {y_path1_obj}")
            # print(f"x_path2_obj: {x_path2_obj}")
            # print(f"y_path2_obj: {y_path2_obj}")
            brick_flag = ((self.intersection(brick_or_door, x_path1_obj)) or (
                self.intersection(brick_or_door, y_path1_obj))) and (
                                     (self.intersection(brick_or_door, x_path2_obj)) or (
                                 self.intersection(brick_or_door, y_path2_obj)))
            # print(f"reachable: {not brick_flag}")
            if not brick_flag:
                reachable_loc.append(i)
        # print(f"reachable: {[door_goal_ls[i] for i in reachable_loc]}")
        return [door_goal_ls[i] for i in reachable_loc]

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # return atoms depend on which phase the agent is in
        doors_open = {
            'red':int(self.red_open),
            'blue':int(self.blue_open),
            'yellow':int(self.yellow_open)
        }
        # where2go state2atom
        if phase is 'where2go':
            door_goal_ls = self.nearest_door_goal(self.door_goal_coor_ls, self.agent_state)
            # atom that grounds the color of the nearest door

            if 'goal' not in door_goal_ls and len(door_goal_ls)>=2:
                near_1, near_2 = door_goal_ls[0], door_goal_ls[1]
                if doors_open[near_1] and doors_open[near_2]:
                    door_left = set(list(doors_open.keys())) - set([near_1,near_2])
                    door_goal_ls[-1] = list(door_left)[0]

            for i in door_goal_ls:
                if i != 'goal':
                    atoms.add(Atom(self.near_mapping[i], ["agent"]))
                else:
                    # two NEAR_G ground atoms for 'near_goal'
                    atoms.add(Atom(self.near_mapping[i], ["agent"]))
                    atoms.add(Atom(self.near_mapping[i], ["agent"]))
            door_state_ls = ['c', 'o']
            for i in door_goal_ls:
                if i != 'goal':
                    color = i[0]
                    if color == 'r':
                        door_state = door_state_ls[int(self.red_open)]
                    elif color == 'y':
                        door_state = door_state_ls[int(self.yellow_open)]
                    elif color == 'b':
                        door_state = door_state_ls[int(self.blue_open)]
                    atoms.add(Atom(self.door_state_map[door_state+"_"+color], ["agent"]))

        # how2go state2atom
        if phase is 'how2go':
            if int(state[3]) > 0: atoms.add(Atom(POSX,["agent"]))
            elif int(state[3]) < 0: atoms.add(Atom(NEGX,["agent"]))
            else: atoms.add(Atom(ZEROX,["agent"]))

            if int(state[4]) > 0: atoms.add(Atom(POSY,["agent"]))
            elif int(state[4]) < 0: atoms.add(Atom(NEGY,["agent"]))
            else: atoms.add(Atom(ZEROY,["agent"]))

        # what2do state2atom
        elif phase is 'what2do':
            if set.intersection(set(state[5:]), {'red', 'yellow', 'blue'}) != []:
                atoms.add(Atom(AT_DOOR, ["agent"]))
            # check if the adjacent door is open(toggled), if yes, is_open(X)
            if state[5:][0] == 'red' and self.red_open:
                atoms.add(Atom(IS_OPEN, ["agent"]))
            elif state[5:][0] == 'yellow' and self.yellow_open:
                atoms.add(Atom(IS_OPEN, ["agent"]))
            elif state[5:][0] == 'blue' and self.blue_open:
                atoms.add(Atom(IS_OPEN, ["agent"]))
            else:
                atoms.add(Atom(IS_CLOSED, ["agent"]))

        return atoms

    # how to go
    def action_refine(self, action, agent_direction):
        # action-direction
        refine_map = {'00': [1, 1, 2],
                      '01': [1, 2],
                      '02': [2],
                      '03': [0, 2],
                      '20': [0, 2],
                      '21': [0, 0, 2],
                      '22': [1, 2],
                      '23': [2],
                      '10': [2],
                      '11': [0, 2],
                      '12': [1, 1, 2],
                      '13': [1, 2],
                      '30': [1, 2],
                      '31': [2],
                      '32': [0, 2],
                      '33': [1, 1, 2]}

        return refine_map[str(action)+str(agent_direction)]

    def w2d_action_refine(self, agent_direction, agent_target_direction):
        # agent_target_direction - agent_direction
        refine_map = {'00': [1],
                      '01': [],
                      '02': [0],
                      '03': [1, 1],
                      '20': [0],
                      '21': [1, 1],
                      '22': [1],
                      '23': [],
                      '10': [],
                      '11': [0],
                      '12': [1, 1],
                      '13': [1],
                      '30': [1, 1],
                      '31': [1],
                      '32': [],
                      '33': [0]}
        agent_target_dict = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        agent_target_dir = agent_target_dict.index(agent_target_direction.tolist())

        return refine_map[str(agent_target_dir)+str(agent_direction)]# + [str(action)]

    def save_image(self,):
        if self.save:
            img = self.env.render('rgb')
            im = Image.fromarray(img)
            im.save(os.path.join(self.save_path,str(self.step)+'.png'))

    def how2go_step_learnable(self, action):
        print(f"in how2go round: {self.round}, step: {self.step}, self.goto_target: {self.goto_target}, coor: {self._state[3:5]}")
        self.step+=1
        action = self.mapping[action.predicate.name]
        # action = np.random.choice(4)
        actions = self.action_refine(action, self._direct)
        # print(f"action taken: {action}")
        for i in actions:
            self.full_state, ori_reward, done, info = self.env.step(i)
            if done: return ori_reward-0.01, done
        self.save_image()
        self.full_state = self.full_state['image']
        # update various state vectors
        self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width-2) - coor_trans(self.temp_goal, self.width-2)

        if np.linalg.norm(self.agent_state) == 0:
            # rotate agent to face the object when adjacent
            self.phase = 'what2do'
            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
            print("round: ", self.round, " reach goal", self.goto_target)
            if self.goto_target != 'goal':
                actions = self.w2d_action_refine(self._direct, self.obj_target_dir[self.goto_target])
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                ori_reward = 0.2
            if self.test: ori_reward, done = ori_reward, done
            self.step+=1
            self.save_image()

        self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width-2) - coor_trans(self.temp_goal, self.width-2)

        # update h2g state
        self._state[3:5] = (str(self.agent_state[0]), str(self.agent_state[1]))

        # update w2d state
        if done and self.goto_target != 'goal': ori_reward = -0.01
        reward = self.get_reward(done, ori_reward)

        if done and reward>0.2:
            print("reached goal!")

        return reward, done

    def what2do(self, action):
        self.step+=1
        action_name = action.predicate.name
        action = action.predicate.name

        action = self.w2d_mapping[action]
        self.full_state, ori_reward, done, info = self.env.step(action)
        self.save_image()
        self.full_state = self.full_state['image']
        # update self._state[:2] (has_key, door_open)
        print(f"round: {self.round}, step: {self.step}")
        # print("key_showup", self.key_showup)
        print("previous red open: ", self.red_open,
                "opened red door: ", self.full_state[self.red_loc[0], self.red_loc[1]][2] == 0,
                "previous yellow open: ", self.yellow_open,
                "opened yellow door: ", self.full_state[self.yellow_loc[0], self.yellow_loc[1]][2] == 0,
                "previous blue open: ", self.blue_open,
                "opened blue door: ", self.full_state[self.blue_loc[0], self.blue_loc[1]][2] == 0,
                "action: ", action_name)

        if ((not self.red_open) and self.full_state[self.red_loc[0], self.red_loc[1]][2] == 0) or \
            ((not self.yellow_open) and self.full_state[self.yellow_loc[0], self.yellow_loc[1]][2] == 0) or \
            ((not self.blue_open) and self.full_state[self.blue_loc[0], self.blue_loc[1]][2] == 0):
            self._state[1] = 1 # door is open
            print(f"success open door!")
            self.full_state, ori_reward, done, info = self.env.step(2)
            self.full_state, ori_reward, done, info = self.env.step(2)
            if done:return self.get_reward(done,ori_reward), done

            self.full_state = self.full_state['image']
            self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]

            self.red_open = int(self.full_state[self.red_loc[0], self.red_loc[1]][2] == 0)
            self.yellow_open = int(self.full_state[self.yellow_loc[0], self.yellow_loc[1]][2] == 0)
            self.blue_open = int(self.full_state[self.blue_loc[0], self.blue_loc[1]][2] == 0)

            self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
            self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
            self.phase = 'where2go'
            if (not self.red_open_prev) and (self.full_state[self.red_loc[0], self.red_loc[1]][2] == 0):
                ori_reward = 0.3
                self.red_open_prev = True
            elif (not self.yellow_open_prev) and (self.full_state[self.yellow_loc[0], self.yellow_loc[1]][2] == 0):
                ori_reward = 0.3
                self.yellow_open_prev = True
            elif (not self.blue_open_prev) and (self.full_state[self.blue_loc[0], self.blue_loc[1]][2] == 0):
                ori_reward = 0.3
                self.blue_open_prev = True
            if self.test:ori_reward = 0.0
            return ori_reward, done,
        self.phase = 'where2go'
        # self._state[1] = int(self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0)
        return -0.01, done,

    def where2go(self,action):
        self.goto_target = action.predicate.name.split('_')[-1]#action.terms[0]
        if self.goto_target == 'red':
            self.temp_goal = self.red_loc
        elif self.goto_target == 'yellow':
            self.temp_goal = self.yellow_loc
        elif self.goto_target == 'blue':
            self.temp_goal = self.blue_loc
        elif self.goto_target == 'goal':
            self.temp_goal = self.goal_pos
        self.temp_goal = self.obj_target_loc[self.goto_target]
        print("ori loc: ",self.locs[self.goto_target], "new loc: ", self.temp_goal)
        # exception: zeroX, zeroY
        if np.linalg.norm(coor_trans(self.agent_state, self.width-2) - coor_trans(self.temp_goal, self.width-2)) == 0:
            self.phase = 'what2do'
            return -0.01, False

        self._state[3:5] = coor_trans(np.array(np.where(self.full_state==10))[:2].reshape(-1),self.width-2) - coor_trans(self.temp_goal,self.width-2)
        print(f"in where2go round: {self.round}, step: {self.step}, self.goto_target: {self.goto_target}, coor: {self._state[3:5]}")

        self.phase = 'how2go'
        self.save_image()
        return -0.01, False

    def next_step(self, action):
        self.red_open = int(self.full_state[self.red_loc[0], self.red_loc[1]][2] == 0)
        self.yellow_open = int(self.full_state[self.yellow_loc[0], self.yellow_loc[1]][2] == 0)
        self.blue_open = int(self.full_state[self.blue_loc[0], self.blue_loc[1]][2] == 0)
        self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]

        # learnable where2go
        if self.phase is 'where2go':
            return self.where2go(action)

        if self.phase is 'how2go':
            return self.how2go_step_learnable(action)

        if self.phase is 'what2do':
            return self.what2do(action)

    def get_reward(self, done, ori_reward=0.0):
        if done and ori_reward > 0:
            print(ori_reward)
        elif done:
            print(0)

        if self.test: return ori_reward
        if ori_reward > 0: return ori_reward
        else: return -0.01

    def optimal_return(self,):
        agent_nearst_door = None
        agent_loc = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        locs = {
            'agent':agent_loc,
            'red': self.red_loc,
            'yellow': self.yellow_loc,
            'blue': self.blue_loc,
            'goal': self.goal_pos
        }
        self.locs = locs
        doors = set(['red','yellow','blue'])
        sequence = ['agent']
        agent_red = np.sum(np.abs(coor_trans(agent_loc, self.width-2) - coor_trans(self.red_loc, self.width-2)))
        agent_blue = np.sum(np.abs(coor_trans(agent_loc, self.width-2) - coor_trans(self.blue_loc, self.width-2)))
        if agent_red < agent_blue:
            agent_nearst_door = agent_red
            sequence.append('red')
        else:
            agent_nearst_door = agent_blue
            sequence.append('blue')
        agent_yellow = np.sum(np.abs(coor_trans(agent_loc, self.width-2) - coor_trans(self.yellow_loc, self.width-2)))
        if agent_yellow < agent_nearst_door:
            agent_nearst_door = agent_yellow
            sequence[-1] = 'yellow'
        doors = doors - set(sequence)
        assert len(doors) == 2
        door1, door2 = doors
        if np.sum(np.abs(coor_trans(self.goal_pos, self.width-2) - coor_trans(locs[door1], self.width-2))) < \
            np.sum(np.abs(coor_trans(self.goal_pos, self.width-2) - coor_trans(locs[door2], self.width-2))):
            sequence.extend([door2, door1,'goal'])
        else:
            sequence.extend([door1,door2,'goal'])
        steps = 0
        for i in range(1,len(sequence)):
            steps+= np.sum(np.abs(coor_trans(locs[sequence[i-1]], self.width-2) - coor_trans(locs[sequence[i]], self.width-2)))


        self.sequence = sequence
        return 1 - 0.9 * ((steps+3)/self.env.max_steps)

    def get_door_target_loc(self):
        self.obj_target_loc = {}
        self.obj_target_loc['goal'] = self.locs['goal']
        self.obj_target_dir = {}
        for idx in range(1,len(self.sequence)):
            obj = self.sequence[idx]
            prev_obj = self.sequence[idx-1]
            if obj=='goal':break
            loc = self.locs[obj]
            prev_loc = self.locs[prev_obj]
            # print(loc,prev_loc)
            # X-direction is wall
            if (self.full_state[loc[0]-1][loc[1]] == [2,5,0]).all():
                # Y-direction prevobj is larger than obj
                if prev_loc[1] > loc[1]:
                    self.obj_target_loc[obj] = [loc[0],loc[1]+1]
                    self.obj_target_dir[obj] = np.array([0, -1])
                else:
                    self.obj_target_loc[obj] = [loc[0],loc[1]-1]
                    self.obj_target_dir[obj] = np.array([0, 1])
            elif (self.full_state[loc[0]][loc[1]-1] == [2,5,0]).all():
                # X-direction prevobj is larger than obj
                if prev_loc[0] > loc[0]:
                    self.obj_target_loc[obj] = [loc[0]+1,loc[1]]
                    self.obj_target_dir[obj] = np.array([1, 0])
                else:
                    self.obj_target_loc[obj] = [loc[0]-1,loc[1]]
                    self.obj_target_dir[obj] = np.array([-1, 0])

    def reset(self, unlock = False, test = False):
        self.test = test
        self.full_state = self.env.reset()

        self.full_state = self.full_state['image']
        self.unseen_background = []
        # goal location
        self.goal_pos = np.array(np.where(self.full_state==8))[:2].reshape(-1)
        # print(f"goal_pos: {self.goal_pos}")
        self.unseen_background.append(Atom(GOAL, [str(self.goal_pos[0]), str(self.goal_pos[1])]))

        # init agent coordinate
        self.init_agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1).copy()

        # door location
        self.red_loc = np.where(((self.full_state == np.array([4, 0, 1]).reshape(1, -1)[:,None,None,:]).all(3)).any(0))
        self.red_loc = [self.red_loc[0][0], self.red_loc[1][0]]
        # print(f"red_loc: {self.red_loc}")
        self.yellow_loc = np.where(((self.full_state == np.array([4, 4, 1]).reshape(1, -1)[:,None,None,:]).all(3)).any(0))
        self.yellow_loc = [self.yellow_loc[0][0], self.yellow_loc[1][0]]
        # print(f"yellow_loc: {self.yellow_loc}")
        self.blue_loc = np.where(((self.full_state == np.array([4, 2, 1]).reshape(1, -1)[:,None,None,:]).all(3)).any(0))
        self.blue_loc = [self.blue_loc[0][0], self.blue_loc[1][0]]
        # print(f"blue_loc: {self.blue_loc}")
        self.door_goal_coor_ls = [self.red_loc, self.yellow_loc, self.blue_loc, self.goal_pos]
        self.optimal_return()
        self.get_door_target_loc()

        self.subgoal_space = [self.red_loc, self.yellow_loc, self.blue_loc, self.goal_pos]

        # initial state vector (agent location)
        self.init_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)
        # relative postional vector (now cheating)
        self.init_dir_vector = coor_trans(self.init_state, self.width-2) - coor_trans(self.blue_loc, self.width-2)
        # initial direction scalar
        self.init_dir = self.full_state[self.init_state[0], self.init_state[1]][2]
        self.init_state = (str(self.init_dir_vector[0]), str(self.init_dir_vector[1]))
        # doors open?
        self.red_open = str(int(self.full_state[int(self.red_loc[0]), int(self.red_loc[1])][2] == 0))
        self.yellow_open = str(int(self.full_state[int(self.yellow_loc[0]), int(self.yellow_loc[1])][2] == 0))
        self.blue_open = str(int(self.full_state[int(self.blue_loc[0]), int(self.blue_loc[1])][2] == 0))

        self.red_open_prev = False
        self.yellow_open_prev = False
        self.blue_open_prev = False

        # adjacent to?
        self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
        # surrounding wallls
        self.surround_wall = surrounding(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
        # print(f"self.surround_wall: {self.surround_wall}")

        # w2h state [w2g, h2g, w2d]
        self.w2h_state = [str(int(self.full_state[int(self.red_loc[0]), int(self.red_loc[1])][2] == 0)),
                          str(int(self.full_state[int(self.yellow_loc[0]), int(self.yellow_loc[1])][2] == 0)),
                          str(int(self.full_state[int(self.blue_loc[0]), int(self.blue_loc[1])][2] == 0))]
        self.w2h_state.extend(list(self.init_state))
        # self.w2h_state.extend([self.surround_wall])   # add more observations to how2go state to make it markovian (#4)
        self.w2h_state.extend(self.adjacent)    # (#2)
        self.w2h_state.extend(['null'] * (7-len(self.w2h_state)))
        # b = copy.deepcopy(self.w2h_state)
        # self.init_w2h_state = copy.deepcopy(self.w2h_state)
        self._state = copy.deepcopy(self.w2h_state)
        # print(self.w2h_state)

        self._direct = copy.deepcopy(self.init_dir)
        self.phase = 'where2go'  # where2go, how2go, what2do
        self.round += 1
        self.agent_state = np.array(np.where(self.full_state==10))[:2].reshape(-1)

        path = 'log_images_multiroom_test'
        if self.save:
            if not os.path.exists(os.path.join(path, str(self.round))):
                os.makedirs(os.path.join(path, str(self.round)))
            self.save_path = os.path.join(path, str(self.round))
        self.step = 0
        self.save_image()
        self.prev_red_open = False
        self.red_unlock = unlock
        self.prev_yellow_open = False
        self.yellow_unlock = unlock
        self.prev_blue_open = False
        self.blue_unlock = unlock
        self.done = False

        return self.w2_state2vector()


class DoorGoal():
    all_variation = ()
    def __init__(self, initial_state=("1", "1"), env_name='MiniGrid-DoorGoal-8x8-v0', width=8, save=False):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]
        w2g_actions = [GT_KEY, GT_DOOR, GT_GOAL]
        w2d_actions = [TOGGLE, PICK, DROP]
        self.actions = h2g_actions + w2g_actions + w2d_actions
        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        w2g_ext = [IS_OPEN, IS_CLOSED, HAS_KEY, NO_KEY]
        w2d_ext = [AT_KEY, AT_DOOR, H_KEY, N_KEY]
        ext = h2g_ext + w2g_ext + w2d_ext

        self.save = save
        self.env = FullyObsWrapper(SimpleDoorGoalEnv(size = width))
        self.width = width
        self.round = -1
        self.reset()
        self.mapping = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        self._object_encoding = {'agent': 0, 'key': 1, 'door': 2, 'goal': 3}
        self.w2d_mapping = {'pick': 3, 'toggle': 5, 'drop': 4}
        constants = [str(i) for i in range(-1*(width), width-1)] + ['key', 'door', 'agent']
        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        background = []

        self.background = background

    @property
    def all_actions(self):
        return self.actions

    def state2vector(self, state):
        # what2do state vector
        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self, state):
        full_state = []
        obj_encoding = [0, 0, 0, 0]
        obj_mapping = {
            'key': 0,
            'door': 1,
            'goal': 2,
            'null': 3
        }
        full_state.append(int(state[0]))
        full_state.append(int(state[1]))
        full_state.append(obj_mapping[state[-2]])

        return np.array(full_state, dtype=np.float16)

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        # return current_position & current_direction
        return len(self.actions)

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # return atoms depend on which phase the agent is in

        # where2go state2atom
        if phase is 'where2go':
            if int(state[0]) == 1:
                atoms.add(Atom(HAS_KEY, ["agent"]))
            else:
                atoms.add(Atom(NO_KEY, ["agent"]))

            if int(state[1]) == 1:
                atoms.add(Atom(IS_OPEN, ["agent"]))
            else:
                atoms.add(Atom(IS_CLOSED, ["agent"]))
        # how2go state2atom
        if phase is 'how2go':
            if int(state[2]) > 0: atoms.add(Atom(POSX,["agent"]))
            elif int(state[2]) < 0: atoms.add(Atom(NEGX,["agent"]))
            else: atoms.add(Atom(ZEROX,["agent"]))

            if int(state[3]) > 0: atoms.add(Atom(POSY,["agent"]))
            elif int(state[3]) < 0: atoms.add(Atom(NEGY,["agent"]))
            else: atoms.add(Atom(ZEROY,["agent"]))
        # what2do state2atom
        elif phase is 'what2do':
            for adj_obj in state[5:]:
                if adj_obj == 'door':
                    atoms.add(Atom(AT_DOOR, ["agent"]))
                elif adj_obj == 'key':
                    atoms.add(Atom(AT_KEY, ["agent"]))
            if int(state[0]) == 1:
                atoms.add(Atom(H_KEY, ["agent"]))
            else:
                atoms.add(Atom(N_KEY, ["agent"]))
        return atoms

    # how to go
    def action_refine(self, action, agent_direction):
        # action-direction
        refine_map = {'00': [1, 1, 2],
                      '01': [1, 2],
                      '02': [2],
                      '03': [0, 2],
                      '20': [0, 2],
                      '21': [0, 0, 2],
                      '22': [1, 2],
                      '23': [2],
                      '10': [2],
                      '11': [0, 2],
                      '12': [1, 1, 2],
                      '13': [1, 2],
                      '30': [1, 2],
                      '31': [2],
                      '32': [0, 2],
                      '33': [1, 1, 2]}

        return refine_map[str(action)+str(agent_direction)]

    def w2d_action_refine(self, agent_direction, agent_target_direction):
        # agent_target_direction - agent_direction
        refine_map = {'00': [1],
                      '01': [],
                      '02': [0],
                      '03': [1, 1],
                      '20': [0],
                      '21': [1, 1],
                      '22': [1],
                      '23': [],
                      '10': [],
                      '11': [0],
                      '12': [1, 1],
                      '13': [1],
                      '30': [1, 1],
                      '31': [1],
                      '32': [],
                      '33': [0]}
        agent_target_dict = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        agent_target_dir = agent_target_dict.index(agent_target_direction.tolist())

        return refine_map[str(agent_target_dir)+str(agent_direction)]

    def save_image(self, ):
        if self.save:
            img = self.env.render('rgb')
            im = Image.fromarray(img)
            im.save(os.path.join(self.save_path, str(self.step) + '.png'))

    def how2go_step_learnable(self, action):
        # print(f"phase: {self.phase}")
        print(f"in how2go,step: {self.step}, state: {self._state}")
        self.step += 1
        # print(f"action taken: {action.predicate.name}")
        action = self.mapping[action.predicate.name]
        actions = self.action_refine(action, self._direct)
        # print(f"action taken: {action}")
        for i in actions:
            self.full_state, ori_reward, done, info = self.env.step(i)
        self.save_image()
        self.full_state = self.full_state['image']
        # update various state vectors
        self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

        if np.linalg.norm(self.agent_state) == 0:
            # rotate agent to face the object when adjacent
            self.phase = 'what2do'
            self.save_image()
            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
            # reward = 0.0
            # if self.unlock and not self.test: ori_reward, done = 1.0, done
            if self.goto_target == 'door':
                if self.loc == 'left':
                    actions = self.w2d_action_refine(self._direct, np.array([-1, 0]))
                else:
                    actions = self.w2d_action_refine(self._direct, np.array([1, 0]))
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                ori_reward = 0.2
            if self.test: ori_reward, done = ori_reward, done

        if self.goto_target == 'key' and np.linalg.norm(self.agent_state) == 1:
            # rotate agent to face the object when adjacent
            actions = self.w2d_action_refine(self._direct, self.agent_state)
            for i in actions:
                self.full_state, ori_reward, done, info = self.env.step(i)
            # switch phase to what2do
            self.phase = 'what2do'
            self.save_image()
            if len(actions) > 0:
                self.full_state = self.full_state['image']
            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
            # reward = 0.0
            if not self.test: ori_reward, done = 0.2, done
            if self.test: ori_reward, done = ori_reward, done

        # update various state vectors
        # self.full_state = self.full_state.reshape(self.width, self.width, 3)
        self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

        # update h2g state
        self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state, self.temp_goal)
        self._state[4] = self.surround_wall
        # update w2d state
        # if self.phase != 'what2do':
        #     self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
        #     self._state[5:7] = self.adjacent + ['null'] * (2-len(self.adjacent))
        if done and self.goto_target != 'goal': ori_reward = -0.01
        reward = self.get_reward(done, ori_reward)
        print(f"in how2go,present state: {self._state},  reward: {ori_reward}\n")

        if done and reward > 0.1:
            print("reached goal!")

        return reward, done

    def next_step(self, action):
        # print(f"action: {[action.predicate.name, action.terms]}")
        # fixed where2go
        self.has_key = 1-int(np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]], np.array([5, 4, 0])))
        self.door_open = int(self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0)

        # learnable where2go
        if self.phase is 'where2go':
            # self.step+=1
            # if not self.unlock:
            self.goto_target = action.predicate.name.split('_')[-1]  # action.terms[0]
            if self.goto_target == 'key':
                self.temp_goal = self.key_loc
            elif self.goto_target == 'door':
                if self.loc == 'left':
                    temp_loc = [self.door_loc[0] - 1, self.door_loc[1]]
                else:
                    temp_loc = [self.door_loc[0] + 1, self.door_loc[1]]
                self.temp_goal = temp_loc
            elif self.goto_target == 'goal':
                self.temp_goal = self.goal_pos
            print(f"in where2go round: {self.round}, step: {self.step}, self.goto_target: {self.goto_target}")

            self._state[2:4] = coor_trans(np.array(np.where(self.full_state == 10))[:2].reshape(-1),
                                          self.width - 2) - coor_trans(self.temp_goal, self.width - 2)
            if self._state[2] == 0 and self._state[3] == 0:
                if self.goto_target == 'door':
                    if self.loc == 'left':
                        temp_state = np.array([-1, 0])
                    else:
                        temp_state = np.array([1, 0])
                    actions = self.w2d_action_refine(self._direct, temp_state)
                    for i in actions:
                        self.full_state, ori_reward, done, info = self.env.step(i)
                        self.full_state = self.full_state['image']
                        self.step += 1
                        self.save_image()
                self.phase = 'what2do'
                self._state[5] = self.goto_target
                return -0.01, False

            self.phase = 'how2go'
            return -0.01, False

        if self.phase is 'how2go':
            reward, done = self.how2go_step_learnable(action)
            return reward, done

        if self.phase is 'what2do':
            self.step += 1
            # if not self.unlock:
            action_name = action.predicate.name
            action = action.predicate.name
            action = self.w2d_mapping[action]
            self.full_state, ori_reward, done, info = self.env.step(action)
            self.save_image()
            self.full_state = self.full_state['image']
            # update self._state[:2] (has_key, door_open)
            print(f"round: {self.round}, step: {self.step}")
            print("previous has key: ", self.has_key,
                  "picked up key: ", np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]], np.array([1, 0, 0])),
                  "previous door open: ", self.door_open,
                  "opened door: ", self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0,
                  "action: ", action_name)
            if (not self.has_key) and np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]],
                                                     np.array([1, 0, 0])):
                self._state[0] = 1  # the key has been picked up by the agent
                print(f"success pickup!")
                # self.full_state = self.full_state.reshape(self.width, self.width, 3)
                self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                               self.full_state)
                self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                self.phase = 'where2go'
                ori_reward = -0.01
                if self.test: ori_reward = 0.0
                return ori_reward, done,
            if (not self.door_open) and self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0:
                self._state[1] = 1  # door is open
                print(f"success open door!")
                self.full_state, ori_reward, done, info = self.env.step(2)
                self.full_state, ori_reward, done, info = self.env.step(2)
                # if self.loc == 'left':
                #     self.loc = 'right'
                # else:
                #     self.loc = 'left'
                if done: return ori_reward, done

                self.full_state = self.full_state['image']
                self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
                self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]

                self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                               self.full_state)
                self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                self.phase = 'where2go'
                if not self.prev_door_open:
                    ori_reward = 0.3
                    self.prev_door_open = True
                    self.done = done
                if self.test: ori_reward = 0.0
                return ori_reward, done,
            self.phase = 'where2go'
            return -0.01, done,

    def get_reward(self, done, ori_reward=0.0):
        if self.test: return ori_reward
        if done and ori_reward > 0.1:
            return ori_reward
        elif done:
            return ori_reward
        if ori_reward > 0: return ori_reward
        return -0.01

    def vary(self):
        pass

    @property
    def critic_state(self):
        return copy.deepcopy(self._state[2:4])

    def flatten(self, ):
        state = []
        state.extend(self.w2h_state['w2g'])
        for key, item in self.w2h_state['h2g'][0].items():
            state.extend(item)
        state.extend([int(i) for i in self.w2h_state['h2g'][1]])
        state.extend(self.w2h_state['w2d'])
        return state

    def change_state(self):
        state = self.flatten()
        full_state = []
        obj_encoding = [0, 0, 0]
        obj_mapping = {
            'key': 0,
            'door': 1,
            'goal': 2
        }
        for index in range(len(state)):
            if index < len(state) - 2:
                full_state.append(int(state[index]))
            else:
                if state[index] in obj_mapping:
                    obj_encoding[obj_mapping[state[index]]] = 1
        full_state.extend(obj_encoding)

        return np.array(full_state, dtype=np.float16)

    def reset(self, unlock=True, test=False):
        # self.env.render()
        self.test = test
        self.full_state = self.env.reset()
        self.full_state = self.full_state['image']
        self.unseen_background = []
        # goal location
        self.goal_pos = np.array(np.where(self.full_state == 8))[:2].reshape(-1)
        print(f"goal_pos: {self.goal_pos}")
        self.unseen_background.append(Atom(GOAL, [str(self.goal_pos[0]), str(self.goal_pos[1])]))
        # key location
        self.key_loc = np.where(
            ((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        # print(f"full_state: {self.full_state.reshape(self.width,self.width,3)}")
        print(f"key_loc: {self.key_loc}")
        self.key_loc = [self.key_loc[0][0], self.key_loc[1][0]]
        # door location
        self.door_loc = np.where(
            ((self.full_state == np.array([4, 4, 1]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.door_loc = [self.door_loc[0][0], self.door_loc[1][0]]
        # initial state vector (agent location)
        self.init_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        # relative postional vector (now cheating)
        self.init_dir_vector = coor_trans(self.init_state, self.width - 2) - coor_trans(self.init_state, self.width - 2)
        # initial direction scalar
        self.init_dir = self.full_state[self.init_state[0], self.init_state[1]][2]
        self.init_state = (str(self.init_dir_vector[0]), str(self.init_dir_vector[1]))
        # agent has key?
        self.has_key = str(0)
        # door is open?
        self.is_open = str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        # adjacent to?
        self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                       self.full_state)
        # surrounding wallls
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state)
        print(f"self.surround_wall: {self.surround_wall}")
        if self.goal_pos[0] > self.door_loc[0]:
            # self.drop_pos = [self.goal_pos[0]-1, self.goal_pos[1]-1]
            self.loc = 'left'
        else:
            self.loc = 'right'
        # w2h state [w2g, h2g, w2d]
        self.w2h_state = ['0', str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))]
        self.w2h_state.extend(list(self.init_state))
        self.w2h_state.extend([self.surround_wall])  # add more observations to how2go state to make it markovian (#4)
        self.w2h_state.extend(self.adjacent)  # (#2)
        self.w2h_state.extend(['null'] * (7 - len(self.w2h_state)))
        # b = copy.deepcopy(self.w2h_state)
        # self.init_w2h_state = copy.deepcopy(self.w2h_state)
        self._state = copy.deepcopy(self.w2h_state)
        print(self.w2h_state)

        self._direct = copy.deepcopy(self.init_dir)
        self.phase = 'where2go'  # where2go, how2go, what2do
        self.round += 1
        self.agent_state = 1
        path = 'log_images_doorgoal'
        if self.save:
            if not os.path.exists(os.path.join(path, str(self.round))):
                os.makedirs(os.path.join(path, str(self.round)))
            self.save_path = os.path.join(path, str(self.round))
        self.step = 0
        self.prev_door_open = False
        self.done = False


class BoxDoor():
    def __init__(self, initial_state=("1", "1"), env_name='MiniGrid-BoxDoor-8x8-v0', width=8, save=False):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]
        w2g_actions = [GT_KEY, GT_DOOR, GT_BOX]
        w2d_actions = [TOGGLE, PICK, DROP]
        self.actions = h2g_actions + w2g_actions + w2d_actions

        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        w2g_ext = [IS_OPEN, IS_CLOSED, HAS_KEY, NO_KEY, KEY_SHOW, KEY_NOSHOW]
        w2d_ext = [AT_KEY, AT_BOX, AT_DOOR, H_KEY, N_KEY]
        ext = h2g_ext + w2g_ext + w2d_ext

        self.env = FullyObsWrapper(BoxDoorEnv(room_size=width))
        self.width = 2*width - 1
        self.height = width
        self.round = -1
        self.save = save
        self.reset()

        self.mapping = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        self._object_encoding = {'agent': 0, 'key': 1, 'door': 2, 'goal': 3}
        self.w2d_mapping = {'pick': 3, 'toggle': 5, 'drop':4}
        constants = [str(i) for i in range(-1 * (width), width - 1)] + ['key', 'door', 'box', 'agent']

        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        background = []
        self.background = background

    @property
    def all_actions(self):
        return self.actions

    def state2vector(self, state):
        # what2do state vector
        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self,state):
        full_state = []
        obj_encoding = [0,0,0,0]
        obj_mapping = {
            'key':0,
            'door':1,
            'goal':2,
            'box':3
        }
        # obj_encoding[obj_mapping[state[-2]]] = 1
        full_state.append(int(state[0]))
        full_state.append(int(state[1]))
        full_state.append(obj_mapping[state[-2]])
        # full_state.extend(obj_encoding)

        return np.array(full_state, dtype = np.float16)

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        return len(self.actions)

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # where2go state2atom
        if phase is 'where2go':
            if int(state[1]) == 1:
                atoms.add(Atom(IS_OPEN, ["agent"]))
            else:
                atoms.add(Atom(IS_CLOSED, ["agent"]))

            if self.key_showup:
                if int(state[0]) == 1:
                    atoms.add(Atom(HAS_KEY, ["agent"]))
                else:
                    atoms.add(Atom(NO_KEY, ["agent"]))
            elif not self.key_showup:
                atoms.add(Atom(KEY_NOSHOW, ["agent"]))
        # how2go state2atom
        if phase is 'how2go':
            if int(state[2]) > 0:
                atoms.add(Atom(POSX, ["agent"]))
            elif int(state[2]) < 0:
                atoms.add(Atom(NEGX, ["agent"]))
            else:
                atoms.add(Atom(ZEROX, ["agent"]))

            if int(state[3]) > 0:
                atoms.add(Atom(POSY, ["agent"]))
            elif int(state[3]) < 0:
                atoms.add(Atom(NEGY, ["agent"]))
            else:
                atoms.add(Atom(ZEROY, ["agent"]))
        # what2do state2atom
        elif phase is 'what2do':
            for adj_obj in state[5:]:
                if adj_obj == 'key':
                    atoms.add(Atom(AT_KEY, ["agent"]))
                elif adj_obj == 'door':
                    atoms.add(Atom(AT_DOOR, ["agent"]))
                elif adj_obj == 'box':
                    atoms.add(Atom(AT_BOX, ["agent"]))
            if int(state[0]) == 1:
                atoms.add(Atom(H_KEY, ["agent"]))
            else:
                atoms.add(Atom(N_KEY, ["agent"]))

        return atoms

    def action_refine(self, action, agent_direction):
        # action-direction
        refine_map = {'00': [1, 1, 2],
                      '01': [1, 2],
                      '02': [2],
                      '03': [0, 2],
                      '20': [0, 2],
                      '21': [0, 0, 2],
                      '22': [1, 2],
                      '23': [2],
                      '10': [2],
                      '11': [0, 2],
                      '12': [1, 1, 2],
                      '13': [1, 2],
                      '30': [1, 2],
                      '31': [2],
                      '32': [0, 2],
                      '33': [1, 1, 2]}

        return refine_map[str(action) + str(agent_direction)]

    def w2d_action_refine(self, agent_direction, agent_target_direction):
        # agent_target_direction - agent_direction
        refine_map = {'00': [1],
                      '01': [],
                      '02': [0],
                      '03': [1, 1],
                      '20': [0],
                      '21': [1, 1],
                      '22': [1],
                      '23': [],
                      '10': [],
                      '11': [0],
                      '12': [1, 1],
                      '13': [1],
                      '30': [1, 1],
                      '31': [1],
                      '32': [],
                      '33': [0]}
        agent_target_dict = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        agent_target_dir = agent_target_dict.index(agent_target_direction.tolist())

        return refine_map[str(agent_target_dir) + str(agent_direction)]  # + [str(action)]

    def save_image(self, ):
        if self.save:
            img = self.env.render('rgb')
            im = Image.fromarray(img)
            im.save(os.path.join(self.save_path, str(self.step) + '.png'))

    def how2go_step_learnable(self, action):
        action = self.mapping[action.predicate.name]
        actions = self.action_refine(action, self._direct)
        # print(f"action taken: {action}")
        for i in actions:
            self.step += 1
            self.full_state, ori_reward, done, info = self.env.step(i)
            self.save_image()
        self.full_state = self.full_state['image']
        # update various state vectors
        self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

        if np.linalg.norm(self.agent_state) == 0:
            if self.goto_target == 'door':
                if self.loc == 'left':
                    actions = self.w2d_action_refine(self._direct, np.array([-1, 0]))
                else:
                    actions = self.w2d_action_refine(self._direct, np.array([1, 0]))
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                    self.step += 1
                    self.save_image()
            self.phase = 'what2do'
            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
            if not self.test: ori_reward = 0.2

        elif (self.goto_target != 'door') and np.linalg.norm(self.agent_state) == 1:
            # rotate agent to face the object when adjacent
            actions = self.w2d_action_refine(self._direct, self.agent_state)
            for i in actions:
                self.full_state, ori_reward, done, info = self.env.step(i)
                self.full_state = self.full_state['image']
                self.step += 1
                self.save_image()
            # switch phase to what2do
            self.phase = 'what2do'
            self.adjacent = [self.goto_target]
            # self.adjacent = adjacent_check(np.array(np.where(self.full_state==10))[:2].reshape(-1).tolist(), self.full_state)
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
            # reward = 0.0
            if not self.test: ori_reward = 0.2
        # update various state vectors
        # self.full_state = self.full_state.reshape(self.width, self.width, 3)
        self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

        # update h2g state
        self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state, self.temp_goal)
        self._state[4] = self.surround_wall
        # update w2d state
        # if done and self.goto_target != 'door': ori_reward = -0.01
        reward = self.get_reward(done, ori_reward)
        # print(f"in how2go,present state: {self._state},  reward: {ori_reward}\n")

        return reward, done

    def next_step(self, action):
        # print(f"action: {[action.predicate.name, action.terms]}")
        # fixed where2go
        self.has_key = 1 - int(
            np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]], np.array([7, 4, 0])) or np.array_equal(
                self.full_state[self.key_loc[0], self.key_loc[1]], np.array([5, 4, 0])))
        self.door_open = int(self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0)
        self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        # update key and box location
        self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]],
                                             np.array([7, 4, 0]))
        if self.key_showup:
            self.key_loc = self.box_loc_fix.copy()
            self.box_loc = [0, 0]
        else:
            self.key_loc = [0, 0]

        # learnable where2go
        if self.phase is 'where2go':
            self.goto_target = action.predicate.name.split('_')[-1]  # action.terms[0]
            if self.goto_target == 'key':
                self.temp_goal = self.key_loc
            elif self.goto_target == 'box':
                self.temp_goal = self.box_loc
            elif self.goto_target == 'door':
                if self.loc == 'left':
                    temp_loc = [self.door_loc[0] - 1, self.door_loc[1]]
                else:
                    temp_loc = [self.door_loc[0] + 1, self.door_loc[1]]
                self.temp_goal = temp_loc

            # has_key while still going to key
            if (self._state[0] == 1) and (self.goto_target == 'key' or self.goto_target == 'box'):
                return -0.01, False
            # goto key when key no-show
            if np.array_equal(self.temp_goal, [0, 0]):
                return -0.01, False

            print(f"in where2go round: {self.round}, step: {self.step}, self.goto_target: {self.goto_target}")

            self._state[2:4] = coor_trans(np.array(np.where(self.full_state == 10))[:2].reshape(-1),
                                          self.width - 2) - coor_trans(self.temp_goal, self.width - 2)
            if self._state[2] == 0 and self._state[3] == 0:
                if self.goto_target == 'door':
                    if self.loc == 'left':
                        temp_state = np.array([-1, 0])
                    else:
                        temp_state = np.array([1, 0])
                    actions = self.w2d_action_refine(self._direct, temp_state)
                    for i in actions:
                        self.full_state, ori_reward, done, info = self.env.step(i)
                        self.full_state = self.full_state['image']
                        self.step += 1
                        self.save_image()
                self.phase = 'what2do'
                self._state[5] = self.goto_target
                return -0.01, False
            self.phase = 'how2go'
            return -0.01, False

        if self.phase is 'how2go':
            reward, done = self.how2go_step_learnable(action)
            return reward, done

        if self.phase is 'what2do':
            self.step += 1
            action_name = action.predicate.name
            action = action.predicate.name
            action = self.w2d_mapping[action]
            if 'box' in self._state[5:] and action == 3:
                self.phase = 'where2go'
                return -0.01, False
            else:
                self.full_state, ori_reward, done, info = self.env.step(action)
            self.save_image()
            self.full_state = self.full_state['image']

            self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]],
                                                 np.array([7, 4, 0]))
            if self.key_showup:
                self.key_loc = self.box_loc_fix.copy()
                self.box_loc = [0, 0]
            else:
                self.key_loc = [0, 0]
            # update self._state[:2] (has_key, door_open)
            print(f"round: {self.round}, step: {self.step}")
            print("key_showup", self.key_showup)
            print("previous has key: ", self.has_key,
                  "picked up key: ",
                  np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]], np.array([1, 0, 0])),
                  "previous door open: ", self.door_open,
                  "opened door: ", self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0,
                  "action: ", action_name)
            if (not self.has_key) and np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]],
                                                     np.array([1, 0, 0])):
                self._state[0] = 1  # the key has been picked up by the agent
                print(f"success pickup!")
                # self.full_state = self.full_state.reshape(self.width, self.width, 3)
                self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                               self.full_state)
                self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                self.phase = 'where2go'
                if self.unlock:
                    ori_reward = 1.0
                else:
                    ori_reward = 0.3
                if self.test: ori_reward = 0.0
                return ori_reward, done,
            if (not self.door_open) and self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0:
                self._state[1] = 1  # door is open
                print(f"success open door!")
                # self.full_state, ori_reward, done, info = self.env.step(2)
                # self.full_state, ori_reward, done, info = self.env.step(2)
                # if done: return ori_reward, done
                # self.full_state = self.full_state['image']
                self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
                self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]

                self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                               self.full_state)
                self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                self.phase = 'where2go'
                if not self.prev_door_open:
                    self.prev_door_open = True
                    self.done = done
                if self.test: ori_reward = 0.0
                return ori_reward, done,
            self.phase = 'where2go'
            self._state[1] = int(self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0)
            return -0.01, done,

    def get_reward(self, done, ori_reward=0.0):
        if done and ori_reward > 0.1:
            return ori_reward
        elif done:
            return -0.01
        if ori_reward > 0: return ori_reward
        return -0.01

    def vary(self):
        pass

    @property
    def critic_state(self):
        return copy.deepcopy(self._state[2:4])

    def flatten(self, ):
        state = []
        state.extend(self.w2h_state['w2g'])
        temp = []
        for key, item in self.w2h_state['h2g'][0].items():
            state.extend(item)
        state.extend([int(i) for i in self.w2h_state['h2g'][1]])
        state.extend(self.w2h_state['w2d'])
        return state

    def change_state(self):
        state = self.flatten()
        full_state = []
        obj_encoding = [0, 0, 0]
        obj_mapping = {
            'key': 0,
            'door': 1,
            'goal': 2
        }
        for index in range(len(state)):
            if index < len(state) - 2:
                full_state.append(int(state[index]))
            else:
                if state[index] in obj_mapping:
                    obj_encoding[obj_mapping[state[index]]] = 1
        full_state.extend(obj_encoding)

        return np.array(full_state, dtype=np.float16)

    def reset(self, unlock=False, test=False):
        self.test = test
        self.full_state = self.env.reset()
        self.full_state = self.full_state['image']
        self.unseen_background = []

        # box location
        self.box_loc = np.where(
            ((self.full_state == np.array([7, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.box_loc = [self.box_loc[0][0], self.box_loc[1][0]]
        self.box_loc_fix = self.box_loc.copy()

        # key location
        self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]],
                                             np.array([7, 4, 0]))
        if self.key_showup:
            self.key_loc = self.box_loc_fix.copy()
            self.box_loc = [0, 0]
        else:
            self.key_loc = [0, 0]  # goto_key and key_loc==[0, 0]则直接惩罚且return (remain where2go)
        # door location
        self.door_loc = np.where(
            ((self.full_state == np.array([4, 4, 2]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.door_loc = [self.door_loc[0][0], self.door_loc[1][0]]
        if self.box_loc[0] < self.door_loc[0]:
            self.loc = 'left'
        else:
            self.loc = 'right'
        # initial state vector (agent location)
        self.init_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        # relative postional vector (now cheating)
        self.init_dir_vector = coor_trans(self.init_state, self.width - 2) - coor_trans(self.key_loc, self.width - 2)
        # initial direction scalar
        self.init_dir = self.full_state[self.init_state[0], self.init_state[1]][2]
        self.init_state = (str(self.init_dir_vector[0]), str(self.init_dir_vector[1]))
        # agent has key?
        self.has_key = str(0)
        # door is open?
        self.is_open = str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        # adjacent to?
        self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                       self.full_state)
        # surrounding wallls
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state)
        print(f"self.surround_wall: {self.surround_wall}")

        # w2h state [w2g, h2g, w2d]
        self.w2h_state = ['0', str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))]
        self.w2h_state.extend(list(self.init_state))
        self.w2h_state.extend([self.surround_wall])  # add more observations to how2go state to make it markovian (#4)
        self.w2h_state.extend(self.adjacent)  # (#2)
        self.w2h_state.extend(['null'] * (7 - len(self.w2h_state)))
        # b = copy.deepcopy(self.w2h_state)
        # self.init_w2h_state = copy.deepcopy(self.w2h_state)
        self._state = copy.deepcopy(self.w2h_state)
        print(self.w2h_state)

        self._direct = copy.deepcopy(self.init_dir)
        self.phase = 'where2go'  # where2go, how2go, what2do
        self.round += 1
        self.agent_state = 1

        path = 'log_images_boxdoor'
        if self.save:
            if not os.path.exists(os.path.join(path, str(self.round))):
                os.makedirs(os.path.join(path, str(self.round)))
            self.save_path = os.path.join(path, str(self.round))
        self.step = 0
        self.prev_door_open = False
        self.unlock = unlock
        self.done = False


class GapBall():
    def __init__(self, initial_state=("1", "1"), env_name='MiniGrid-GapBall-8x8-v0', width=8, save=False):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]
        w2g_actions = [GT_KEY, GT_BOX, GT_GOAL]
        w2d_actions = [TOGGLE, PICK, DROP]
        self.actions = h2g_actions + w2g_actions + w2d_actions
        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        w2g_ext = [IS_OPEN, IS_CLOSED, HAS_KEY, NO_KEY, KEY_SHOW, KEY_NOSHOW]
        w2d_ext = [AT_KEY, AT_BOX, H_KEY, N_KEY, AT_GOAL]
        ext = h2g_ext + w2g_ext + w2d_ext
        self.env = FullyObsWrapper(SimpleGapBallEnv(size=8))
        self.width = width
        self.round = -1
        self.save = save
        self.reset()

        self.mapping = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        self._object_encoding = {'agent': 0, 'key': 1, 'door': 2, 'goal': 3}
        self.w2d_mapping = {'pick': 3, 'toggle': 5, 'drop':4}
        constants = [str(i) for i in range(-1 * (width), width - 1)] + ['key', 'door', 'box', 'agent']
        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        background = []
        self.background = background

    @property
    def all_actions(self):
        return self.actions

    def state2vector(self, state):
        # what2do state vector
        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self, state):
        full_state = []
        obj_encoding = [0, 0, 0, 0]
        obj_mapping = {
            'key': 0,
            'door': 1,
            'goal': 2,
            'box': 3
        }
        # obj_encoding[obj_mapping[state[-2]]] = 1
        full_state.append(int(state[0]))
        full_state.append(int(state[1]))
        full_state.append(obj_mapping[state[-2]])
        # full_state.extend(obj_encoding)

        return np.array(full_state, dtype=np.float16)

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        # return current_position & current_direction
        return len(self.actions)

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # where2go state2atom
        if phase is 'where2go':
            if int(state[1]) == 1:
                atoms.add(Atom(IS_OPEN, ["agent"]))
            else:
                atoms.add(Atom(IS_CLOSED, ["agent"]))

            if self.key_showup:
                if int(state[0]) == 1:
                    atoms.add(Atom(HAS_KEY, ["agent"]))
                else:
                    atoms.add(Atom(NO_KEY, ["agent"]))
            elif not self.key_showup:
                atoms.add(Atom(KEY_NOSHOW, ["agent"]))
        # how2go state2atom
        if phase is 'how2go':
            if int(state[2]) > 0:
                atoms.add(Atom(POSX, ["agent"]))
            elif int(state[2]) < 0:
                atoms.add(Atom(NEGX, ["agent"]))
            else:
                atoms.add(Atom(ZEROX, ["agent"]))

            if int(state[3]) > 0:
                atoms.add(Atom(POSY, ["agent"]))
            elif int(state[3]) < 0:
                atoms.add(Atom(NEGY, ["agent"]))
            else:
                atoms.add(Atom(ZEROY, ["agent"]))

        # what2do state2atom
        elif phase is 'what2do':
            for adj_obj in state[5:]:
                if adj_obj == 'key':
                    atoms.add(Atom(AT_KEY, ["agent"]))
                elif adj_obj == 'goal':
                    atoms.add(Atom(AT_GOAL, ["agent"]))
                elif adj_obj == 'box':
                    atoms.add(Atom(AT_BOX, ["agent"]))
            if int(state[0]) == 1:
                # atoms.add(Atom(HAS_KEY, ["agent"]))
                atoms.add(Atom(H_KEY, ["agent"]))
            else:
                atoms.add(Atom(N_KEY, ["agent"]))
        return atoms

    # how to go
    def action_refine(self, action, agent_direction):
        # action-direction
        refine_map = {'00': [1, 1, 2],
                      '01': [1, 2],
                      '02': [2],
                      '03': [0, 2],
                      '20': [0, 2],
                      '21': [0, 0, 2],
                      '22': [1, 2],
                      '23': [2],
                      '10': [2],
                      '11': [0, 2],
                      '12': [1, 1, 2],
                      '13': [1, 2],
                      '30': [1, 2],
                      '31': [2],
                      '32': [0, 2],
                      '33': [1, 1, 2]}

        return refine_map[str(action)+str(agent_direction)]

    def w2d_action_refine(self, agent_direction, agent_target_direction):
        # agent_target_direction - agent_direction
        refine_map = {'00': [1],
                      '01': [],
                      '02': [0],
                      '03': [1, 1],
                      '20': [0],
                      '21': [1, 1],
                      '22': [1],
                      '23': [],
                      '10': [],
                      '11': [0],
                      '12': [1, 1],
                      '13': [1],
                      '30': [1, 1],
                      '31': [1],
                      '32': [],
                      '33': [0]}
        agent_target_dict = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        agent_target_dir = agent_target_dict.index(agent_target_direction.tolist())

        return refine_map[str(agent_target_dir)+str(agent_direction)]# + [str(action)]

    def save_image(self, ):
        if self.save:
            img = self.env.render('rgb')
            im = Image.fromarray(img)
            im.save(os.path.join(self.save_path, str(self.step) + '.png'))

    def how2go_step_learnable(self, action):
        action = self.mapping[action.predicate.name]
        actions = self.action_refine(action, self._direct)
        for i in actions:
            self.step += 1
            self.full_state, ori_reward, done, info = self.env.step(i)
            self.save_image()
        self.full_state = self.full_state['image']

        # update various state vectors
        self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)
        if np.linalg.norm(self.agent_state) == 1:  # or (np.linalg.norm(self.agent_state) == 0 and self.goto_target=='door'):
            # rotate agent to face the object when adjacent
            temp_state = self.agent_state
            actions = self.w2d_action_refine(self._direct, temp_state)
            for i in actions:
                self.full_state, ori_reward, done, info = self.env.step(i)
                self.full_state = self.full_state['image']
                self.step += 1
                self.save_image()
            if not self.test: ori_reward = 0.2
            # switch phase to what2do
            self.phase = 'what2do'
            # self.save_image()
            self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
            self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

            self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
            self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                             self.full_state, self.temp_goal)
            self._state[4] = self.surround_wall

            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))

        # update h2g state
        self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(), self.full_state, self.temp_goal)
        self._state[4] = self.surround_wall
        # update w2d state
        if self.phase != 'what2do':
            self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(), self.full_state)
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))

        reward = self.get_reward(done, ori_reward)
        print(f"in how2go,present state: {self._state},  reward: {reward}\n")

        return reward, done

    def what2do(self, action):
        self.step += 1
        action_name = action.predicate.name
        action = action.predicate.name
        action = self.w2d_mapping[action]
        if self.has_key and action_name == 'drop':
            count = 0
            for i in range(3):
                self.full_state, ori_reward, done, info = self.env.step(0)
                count += 1
                self.step += 1
                self.save_image()
                self.full_state, ori_reward, done, info = self.env.step(action)
                self.step += 1
                self.save_image()
                self.full_state = self.full_state['image']
                if len(np.where(((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[0]) != 0:
                    print("success drop!")
                    self.key_loc = np.where(((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
                    print(f"key_loc: {self.key_loc}")
                    self.key_loc = [self.key_loc[0][0], self.key_loc[1][0]]
                    self.has_key = False
                    self._state[0] = 0
                    if not self.dropped and (not self.test): ori_reward = 0.3

                    self.dropped = True
                    for _ in range(count):
                        self.full_state, _, done, info = self.env.step(1)
                        if done: break
                    self.full_state = self.full_state['image']
                    self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                                   self.full_state)
                    self.adjacent = [self.goto_target]
                    self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                    self.phase = 'where2go'
                    self.save_image()
                    return ori_reward, done

        # self.full_state, ori_reward, done, info = self.env.step(action)
        if 'box' in self._state[5:] and action == 3:
            self.phase = 'where2go'
            return -0.01, False
        else:
            self.full_state, ori_reward, done, info = self.env.step(action)
        if done and ori_reward > 0.05:
            print("reached Goal!")
            return ori_reward, done
        self.save_image()
        self.full_state = self.full_state['image']
        self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]],
                                             np.array([7, 4, 0]))
        if self.key_showup:
            if self.key_loc == [0, 0]:
                self.key_loc = self.box_loc_fix.copy()
            self.box_loc = [0, 0]
        else:
            self.key_loc = [0, 0]

        if (not self.has_key) and np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]],
                                                 np.array([1, 0, 0])):
            self._state[0] = 1  # the key has been picked up by the agent
            print(f"success pickup!")
            self.phase = 'where2go'
            if not self.prev_has_key:
                ori_reward = 0.3
                self._state[1] = 1
            self.prev_has_key = True
            return ori_reward, done

        self.phase = 'where2go'
        if not self.test: ori_reward -= 0.01
        return ori_reward, done

    def where2go(self, action):
        self.prev_goal = self.goto_target
        self.goto_target = action.predicate.name.split('_')[-1]
        if self.goto_target == 'key':
            self.temp_goal = self.key_loc
        elif self.goto_target == 'goal':
            self.temp_goal = self.goal_pos
        elif self.goto_target == 'box':
            self.temp_goal = self.box_loc

        # if self.prev_goal != None and (self.goto_target == self.prev_goal):
        #     self.phase = 'what2do'
        #     return -0.01, False
        # has_key while still going to key
        if (self._state[0] == 1) and (self.goto_target == 'key' or self.goto_target == 'box'):
            return -0.01, False
        # goto key when key no-show
        if np.array_equal(self.temp_goal, [0, 0]):
            return -0.01, False
        else:
            if self.prev_goal != None and (self.goto_target == self.prev_goal):
                self.phase = 'what2do'
                return -0.01, False

        self._state[2:4] = coor_trans(np.array(np.where(self.full_state == 10))[:2].reshape(-1), self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

        print(f"in where2go round: {self.round}, step: {self.step}, prev_target: {self.prev_goal} ,self.goto_target: {self.goto_target}, state:{self._state}")
        self.phase = 'how2go'
        return -0.01, False

    def next_step(self, action):
        self.has_key = int(self._state[0])
        # door is open?
        self.is_open = int(self._state[1])
        # learnable where2go
        if self.phase is 'where2go':
            reward, done = self.where2go(action)

        elif self.phase is 'how2go':
            reward, done = self.how2go_step_learnable(action)

        elif self.phase is 'what2do':
            reward, done = self.what2do(action)

        return reward, done

    def get_reward(self, done, ori_reward=0.0):
        # return ori_reward
        if self.test: return ori_reward
        if done and ori_reward > 0.1:
            return ori_reward
        elif done:
            return ori_reward

        if ori_reward > 0: return ori_reward
        return -0.01

    @property
    def critic_state(self):
        return copy.deepcopy(self._state[2:4])

    def flatten(self, ):
        state = []
        state.extend(self.w2h_state['w2g'])
        temp = []
        for key, item in self.w2h_state['h2g'][0].items():
            state.extend(item)
        state.extend([int(i) for i in self.w2h_state['h2g'][1]])
        state.extend(self.w2h_state['w2d'])
        return state

    def change_state(self):
        state = self.flatten()
        full_state = []
        obj_encoding = [0, 0, 0]
        obj_mapping = {
            'key': 0,
            'box': 1,
            'goal': 2
        }
        for index in range(len(state)):
            if index < len(state) - 2:
                full_state.append(int(state[index]))
            else:
                if state[index] in obj_mapping:
                    obj_encoding[obj_mapping[state[index]]] = 1
        full_state.extend(obj_encoding)

        return np.array(full_state, dtype=np.float16)

    def reset(self, unlock=True, test=False):
        self.test = test
        self.full_state = self.env.reset()
        self.full_state = self.full_state['image']
        # goal location
        self.goal_pos = np.array(np.where(self.full_state == 6))[:2].reshape(-1)

        # box location
        self.box_loc = np.where(((self.full_state == np.array([7, 4, 0]).reshape(1, -1)[:,None,None,:]).all(3)).any(0))
        self.box_loc = [self.box_loc[0][0], self.box_loc[1][0]]
        self.box_loc_fix = self.box_loc.copy()

        # key location
        self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]],
                                             np.array([7, 4, 0]))
        if self.key_showup:
            self.key_loc = self.box_loc_fix.copy()
            self.box_loc = [0, 0]
        else:
            self.key_loc = [0, 0]  # goto_key and key_loc==[0, 0]则直接惩罚且return (remain where2go)

        print(f"key_loc: {self.key_loc}")
        print(f"box_loc: {self.box_loc}")
        print(f"goal_pos: {self.goal_pos}")
        # initial state vector (agent location)
        self.init_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        # relative postional vector (now cheating)
        self.init_dir_vector = coor_trans(self.init_state, self.width - 2) - coor_trans(self.key_loc, self.width - 2)
        # initial direction scalar
        self.init_dir = self.full_state[self.init_state[0], self.init_state[1]][2]
        self.init_state = (str(self.init_dir_vector[0]), str(self.init_dir_vector[1]))
        # agent has key?
        self.has_key = str(0)
        # door is open?
        self.is_open = str(0)
        # adjacent to?
        self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                       self.full_state)
        # surrounding wallls
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state)
        print(f"self.surround_wall: {self.surround_wall}")

        # w2h state [w2g, h2g, w2d]
        self.w2h_state = ['0', '0']
        self.w2h_state.extend(list(self.init_state))
        self.w2h_state.extend([self.surround_wall])  # add more observations to how2go state to make it markovian (#4)
        self.w2h_state.extend(self.adjacent)  # (#2)
        self.w2h_state.extend(['null'] * (7 - len(self.w2h_state)))
        # b = copy.deepcopy(self.w2h_state)
        # self.init_w2h_state = copy.deepcopy(self.w2h_state)
        self._state = copy.deepcopy(self.w2h_state)
        print(self.w2h_state)

        self._direct = copy.deepcopy(self.init_dir)
        self.phase = 'where2go'  # where2go, how2go, what2do
        self.round += 1
        self.agent_state = 0
        if self.save:
            path = 'log_images_gapball/seed_{}'.format(self.seed)
            if not os.path.exists(os.path.join(path, str(self.round))):
                os.makedirs(os.path.join(path, str(self.round)))
            self.save_path = os.path.join(path, str(self.round))
        self.step = 0
        self.done = False
        self.prev_has_key = False
        self.dropped = False
        self.prev_goal = None
        self.goto_target = None


class BlockedDoor():
    def __init__(self, initial_state=("1", "1"), env_name='MiniGrid-BlockedDoor-8x8-v0', width=8, save=False):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]
        w2g_actions = [GT_KEY, GT_DOOR, GT_BLOCKAGE]
        w2d_actions = [TOGGLE, PICK, DROP]
        self.actions = h2g_actions + w2g_actions + w2d_actions

        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        w2g_ext = [IS_OPEN, IS_CLOSED, IS_BLOCKED, IS_INCOMPLETE_BLOCKED, HAS_KEY, NO_KEY]
        w2d_ext = [AT_KEY, AT_BLOCKAGE, AT_DOOR, H_KEY, H_BLOCKAGE, NONE] # H_KEY,H_BLOCKAGE, NONE
        ext = h2g_ext + w2g_ext + w2d_ext

        self.env = FullyObsWrapper(BlockedDoorEnv(room_size=width))
        self.width = 2 * width - 1
        self.height = width
        self.round = -1
        self.save = save
        self.reset()

        self.mapping = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        self._object_encoding = {'agent': 0, 'key': 1, 'door': 2, 'goal': 3}
        self.w2d_mapping = {'pick': 3, 'toggle': 5, 'drop': 4}
        constants = [str(i) for i in range(-1 * (width), width - 1)] + ['key', 'door', 'blockage', 'agent']

        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        background = []
        self.background = background

    @property
    def all_actions(self):
        return self.actions

    def state2vector(self, state):
        # what2do state vector
        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self, state):
        full_state = []
        obj_encoding = [0, 0, 0, 0]
        obj_mapping = {
            'key': 0,
            'door': 1,
            'blockage': 2,
        }
        # obj_encoding[obj_mapping[state[-2]]] = 1
        full_state.append(int(state[0]))
        full_state.append(int(state[1]))
        full_state.append(obj_mapping[state[-2]])
        # full_state.extend(obj_encoding)

        return np.array(full_state, dtype=np.float16)

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        return len(self.actions)

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # where2go state2atom
        if phase is 'where2go':
            if self.not_blocked:
                if self.has_blockage:
                    atoms.add(Atom(IS_INCOMPLETE_BLOCKED, ["agent"]))
                elif int(state[1]) == 1:
                    atoms.add(Atom(IS_OPEN, ["agent"]))
                else:
                    atoms.add(Atom(IS_CLOSED, ["agent"]))
            else:
                atoms.add(Atom(IS_BLOCKED, ["agent"]))
            if int(state[0]) == 1:
                atoms.add(Atom(HAS_KEY, ["agent"]))
            else:
                atoms.add(Atom(NO_KEY, ["agent"]))
        # how2go state2atom
        if phase is 'how2go':
            if int(state[2]) > 0:
                atoms.add(Atom(POSX, ["agent"]))
            elif int(state[2]) < 0:
                atoms.add(Atom(NEGX, ["agent"]))
            else:
                atoms.add(Atom(ZEROX, ["agent"]))

            if int(state[3]) > 0:
                atoms.add(Atom(POSY, ["agent"]))
            elif int(state[3]) < 0:
                atoms.add(Atom(NEGY, ["agent"]))
            else:
                atoms.add(Atom(ZEROY, ["agent"]))
        # what2do state2atom
        elif phase is 'what2do':
            for adj_obj in state[5:]:
                if adj_obj == 'key':
                    atoms.add(Atom(AT_KEY, ["agent"]))
                elif adj_obj == 'door':
                    atoms.add(Atom(AT_DOOR, ["agent"]))
                elif adj_obj == 'blockage':
                    atoms.add(Atom(AT_BLOCKAGE, ["agent"]))
            if int(state[0]) == 1:
                atoms.add(Atom(H_KEY, ["agent"]))
            elif self.has_blockage:
                atoms.add(Atom(H_BLOCKAGE, ['agent']))
            else:
                atoms.add(Atom(NONE, ["agent"]))
        return atoms

    def action_refine(self, action, agent_direction):
        # action-direction
        refine_map = {'00': [1, 1, 2],
                      '01': [1, 2],
                      '02': [2],
                      '03': [0, 2],
                      '20': [0, 2],
                      '21': [0, 0, 2],
                      '22': [1, 2],
                      '23': [2],
                      '10': [2],
                      '11': [0, 2],
                      '12': [1, 1, 2],
                      '13': [1, 2],
                      '30': [1, 2],
                      '31': [2],
                      '32': [0, 2],
                      '33': [1, 1, 2]}

        return refine_map[str(action) + str(agent_direction)]

    def w2d_action_refine(self, agent_direction, agent_target_direction):
        # agent_target_direction - agent_direction
        refine_map = {'00': [1],
                      '01': [],
                      '02': [0],
                      '03': [1, 1],
                      '20': [0],
                      '21': [1, 1],
                      '22': [1],
                      '23': [],
                      '10': [],
                      '11': [0],
                      '12': [1, 1],
                      '13': [1],
                      '30': [1, 1],
                      '31': [1],
                      '32': [],
                      '33': [0]}
        agent_target_dict = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        agent_target_dir = agent_target_dict.index(agent_target_direction.tolist())

        return refine_map[str(agent_target_dir) + str(agent_direction)]  # + [str(action)]

    def save_image(self, ):
        if self.save:
            img = self.env.render('rgb')
            im = Image.fromarray(img)
            im.save(os.path.join(self.save_path, str(self.step) + '.png'))

    def how2go_step_learnable(self, action):
        print(f"in how2go,step: {self.step}, state: {self._state}, agent_state: {self.agent_state}, target: {self.goto_target}, loc : {self.temp_goal}")
        action = self.mapping[action.predicate.name]
        actions = self.action_refine(action, self._direct)
        # print(f"action taken: {action}")
        for i in actions:
            self.step += 1
            self.full_state, ori_reward, done, info = self.env.step(i)
            self.save_image()
        self.full_state = self.full_state['image']
        # update various state vectors
        self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)
        if np.linalg.norm(self.agent_state) == 0:
            if self.goto_target == 'door':
                if self.loc == 'left':
                    temp_state = np.array([-1, 0])
                else:
                    temp_state = np.array([1, 0])
                actions = self.w2d_action_refine(self._direct, temp_state)
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                    self.step += 1
                    self.save_image()
                if not self.test:
                    ori_reward = 1.0
                else:
                    ori_reward = 0.1
            self.phase = 'what2do'
            self.save_image()
            self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
            self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

            self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
            self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                             self.full_state, self.temp_goal)
            self._state[4] = self.surround_wall

            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
        if np.linalg.norm(self.agent_state) == 1 and self.goto_target != 'door':  # or (np.linalg.norm(self.agent_state) == 0 and self.goto_target=='door'):
            # rotate agent to face the object when adjacent
            temp_state = self.agent_state
            actions = self.w2d_action_refine(self._direct, temp_state)
            for i in actions:
                self.full_state, ori_reward, done, info = self.env.step(i)
                self.full_state = self.full_state['image']
                self.step += 1
                self.save_image()
            if not self.test: ori_reward = 1.0
            if self.test: ori_reward = 0.1
            # switch phase to what2do
            self.phase = 'what2do'
            # self.save_image()
            self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
            self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

            self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
            self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                             self.full_state, self.temp_goal)
            self._state[4] = self.surround_wall

            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))

        # update h2g state
        self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state, self.temp_goal)
        self._state[4] = self.surround_wall
        # update w2d state
        if self.phase != 'what2do':
            self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                           self.full_state)
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))

        reward = self.get_reward(done, ori_reward)
        print(f"in how2go,present state: {self._state},  reward: {reward}\n")

        return reward, done

    def what2do(self, action):
        self.step += 1
        action_name = action.predicate.name
        action = action.predicate.name
        action = self.w2d_mapping[action]
        print(f"in what2do round: {self.round}, step: {self.step}, action: {action_name}, state: {self._state}")
        if (self.has_key or self.has_blockage) and action_name == 'drop':
            count = 0
            for i in range(3):
                self.full_state, ori_reward, done, info = self.env.step(0)
                count += 1
                self.step += 1
                self.save_image()
                self.full_state, ori_reward, done, info = self.env.step(action)
                self.step += 1
                self.save_image()
                self.full_state = self.full_state['image']
                if self.has_key and len(np.where(((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[0]) != 0:
                    print("success drop key!")
                    self.key_loc = np.where(((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
                    print(f"key_loc: {self.key_loc}")

                    self.key_loc = [self.key_loc[0][0], self.key_loc[1][0]]
                    self.has_key = False
                    self._state[0] = 0
                    for _ in range(count):
                        self.full_state, _, done, info = self.env.step(1)
                        if done: break
                    self.full_state = self.full_state['image']
                    self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                                   self.full_state)
                    self.adjacent = [self.goto_target]
                    self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                    self.phase = 'where2go'
                    self.save_image()
                    return ori_reward, done
                if self.has_blockage and len(np.where(((self.full_state == np.array([6, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[0]) != 0:
                    print("success drop blockage!")
                    self.has_blockage = False
                    if (not self.dropped) and (not self.test): ori_reward = 0.3
                    self.dropped = True
                    for _ in range(count):
                        self.full_state, _, done, info = self.env.step(1)
                        if done: break
                    self.full_state = self.full_state['image']
                    self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                                   self.full_state)
                    self.adjacent = [self.goto_target]
                    self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                    self.phase = 'where2go'
                    self.save_image()
                    return ori_reward, done

        self.full_state, ori_reward, done, info = self.env.step(action)
        if done and ori_reward > 0.05:
            print("reached Goal!")
            if self.test: return ori_reward, done
            return ori_reward, done
        self.save_image()
        self.full_state = self.full_state['image']
        if (not self.has_key) and np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]],
                                                 np.array([1, 0, 0])):
            self._state[0] = 1  # the key has been picked up by the agent
            print(f"success pickup!")
            self.phase = 'where2go'
            if not self.prev_has_key: ori_reward = 0.3
            if self.test: ori_reward = 0.0
            self.prev_has_key = True
            return ori_reward, done,
        if (not self.has_blockage) and len(np.where(((self.full_state == np.array([6, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[0]) == 0:
            print(f"success pickup blockage!")
            self.phase = 'where2go'
            self.has_blockage = True
            self.not_blocked = True
            self.blockage_loc = [0,0]
            ori_reward = 0.3
            if self.test: ori_reward = 0.0

        self.phase = 'where2go'
        if not self.test: ori_reward -= 0.01
        return ori_reward, done

    def where2go(self, action):
        self.prev_goal = self.goto_target
        self.goto_target = action.predicate.name.split('_')[-1]
        if self.goto_target == 'key':
            self.temp_goal = self.key_loc
        elif self.goto_target == 'door':
            if self.loc == 'left':
                temp_loc = [self.door_loc[0] - 1, self.door_loc[1]]
            else:
                temp_loc = [self.door_loc[0] + 1, self.door_loc[1]]
            self.temp_goal = temp_loc
        elif self.goto_target == 'blockage':
            self.temp_goal = self.blockage_loc

        if self.prev_goal != None and (self.goto_target == self.prev_goal) and self.goto_target!='door':
            self.phase = 'what2do'
            return 0.0, False
        self._state[2:4] = coor_trans(np.array(np.where(self.full_state == 10))[:2].reshape(-1),
                                      self.width - 2) - coor_trans(self.temp_goal, self.width - 2)
        print(f"in where2go round: {self.round}, step: {self.step}, prev_target: {self.prev_goal} ,self.goto_target: {self.goto_target}, state:{self._state}")
        if self._state[2] == 0 and self._state[3] == 0:
            if self.goto_target == 'door':
                if self.loc == 'left':
                    temp_state = np.array([-1, 0])
                else:
                    temp_state = np.array([1, 0])
                actions = self.w2d_action_refine(self._direct, temp_state)
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                    self.step += 1
                    self.save_image()
            self.phase = 'what2do'
            self._state[5] = self.goto_target
            return -0.01, False
        self.phase = 'how2go'
        return -0.01, False

    def next_step(self, action):
        self.has_key = int(self._state[0])
        # door is open?
        self.is_open = int(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        # learnable where2go
        if self.phase is 'where2go':
            reward, done = self.where2go(action)

        elif self.phase is 'how2go':
            reward, done = self.how2go_step_learnable(action)

        elif self.phase is 'what2do':
            reward, done = self.what2do(action)
        return reward, done

    def get_reward(self, done, ori_reward=0.0):
        # return ori_reward
        if self.test: return ori_reward
        if done and ori_reward > 0.1:
            return 1.0
        elif done:
            return ori_reward

        if ori_reward > 0: return ori_reward
        return -0.01

    @property
    def critic_state(self):
        return copy.deepcopy(self._state[2:4])

    def flatten(self, ):
        state = []
        state.extend(self.w2h_state['w2g'])
        temp = []
        for key, item in self.w2h_state['h2g'][0].items():
            state.extend(item)
        state.extend([int(i) for i in self.w2h_state['h2g'][1]])
        state.extend(self.w2h_state['w2d'])
        return state

    def change_state(self):
        state = self.flatten()
        full_state = []
        obj_encoding = [0, 0, 0]
        obj_mapping = {
            'key': 0,
            'door': 1,
            'goal': 2
        }
        for index in range(len(state)):
            if index < len(state) - 2:
                full_state.append(int(state[index]))
            else:
                if state[index] in obj_mapping:
                    obj_encoding[obj_mapping[state[index]]] = 1
        full_state.extend(obj_encoding)

        return np.array(full_state, dtype=np.float16)

    def reset(self, unlock=True, test=False):
        self.test = test
        self.full_state = self.env.reset()
        self.full_state = self.full_state['image']

        self.key_loc = np.where(((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.key_loc = [self.key_loc[0][0], self.key_loc[1][0]]
        # door location
        self.door_loc = np.where(((self.full_state == np.array([4, 4, 2]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.door_loc = [self.door_loc[0][0], self.door_loc[1][0]]
        if self.key_loc[0] < self.door_loc[0]:
            self.loc = 'left'
            if np.array_equal(self.full_state[self.door_loc[0]-1, self.door_loc[1]], np.array([6, 4, 0])):
                self.not_blocked = False
                self.blockage_loc = [self.door_loc[0]-1, self.door_loc[1]]
            else:
                self.not_blocked = True
                self.blockage_loc = [0, 0]
        else:
            self.loc = 'right'
            if np.array_equal(self.full_state[self.door_loc[0]+1, self.door_loc[1]], np.array([6, 4, 0])):
                self.not_blocked = False
                self.blockage_loc = [self.door_loc[0]+1, self.door_loc[1]]
            else:
                self.not_blocked = True
                self.blockage_loc = [0, 0]

        print(f"key_loc: {self.key_loc}")
        print(f"door_loc: {self.door_loc}")
        print(f"blockage_loc: {self.blockage_loc}")
        # initial state vector (agent location)
        self.init_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        # relative postional vector (now cheating)
        self.init_dir_vector = coor_trans(self.init_state, self.width - 2) - coor_trans(self.key_loc, self.width - 2)
        # initial direction scalar
        self.init_dir = self.full_state[self.init_state[0], self.init_state[1]][2]
        self.init_state = (str(self.init_dir_vector[0]), str(self.init_dir_vector[1]))
        # agent has key?
        self.has_key = str(0)
        self.has_blockage = False
        # door is open?
        self.is_open = str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        # adjacent to?
        self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                       self.full_state)
        # surrounding wallls
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state)
        print(f"self.surround_wall: {self.surround_wall}")

        # w2h state [w2g, h2g, w2d]
        self.w2h_state = ['0', str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))]
        self.w2h_state.extend(list(self.init_state))
        self.w2h_state.extend([self.surround_wall])  # add more observations to how2go state to make it markovian (#4)
        self.w2h_state.extend(self.adjacent)  # (#2)
        self.w2h_state.extend(['null'] * (7 - len(self.w2h_state)))
        # b = copy.deepcopy(self.w2h_state)
        # self.init_w2h_state = copy.deepcopy(self.w2h_state)
        self._state = copy.deepcopy(self.w2h_state)
        print(self.w2h_state)

        self._direct = copy.deepcopy(self.init_dir)
        self.phase = 'where2go'  # where2go, how2go, what2do
        self.round += 1
        self.agent_state = 0
        if self.save:
            path = 'log_images_unlock/seed_{}'.format(self.seed)
            if not os.path.exists(os.path.join(path, str(self.round))):
                os.makedirs(os.path.join(path, str(self.round)))
            self.save_path = os.path.join(path, str(self.round))
        self.step = 0
        self.done = False
        self.prev_has_key = False
        self.dropped = False
        self.prev_goal = None
        self.goto_target = None


class BallKey():
    def __init__(self, initial_state=("1", "1"), env_name='MiniGrid-BallKey-8x8-v0', width=8, save=False):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]
        w2g_actions = [GT_KEY, GT_DOOR, GT_GOAL]
        w2d_actions = [TOGGLE, PICK, DROP]
        self.actions = h2g_actions + w2g_actions + w2d_actions

        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        w2g_ext = [IS_OPEN, IS_CLOSED, HAS_KEY, NO_KEY]
        w2d_ext = [AT_KEY, AT_DOOR, H_KEY, N_KEY, AT_GOAL]
        ext = h2g_ext + w2g_ext + w2d_ext

        self.env = FullyObsWrapper(SimpleBallKeyEnv(size=width))
        self.width = width
        self.round = -1
        self.save = save
        self.reset()
        self.mapping = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        self._object_encoding = {'agent': 0, 'key': 1, 'door': 2, 'goal': 3}
        self.w2d_mapping = {'pick': 3, 'toggle': 5, 'drop': 4}
        constants = [str(i) for i in range(-1 * (self.width), self.width - 1)] + ['key', 'door', 'agent']

        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        self.background = []

    @property
    def all_actions(self):
        return self.actions

    def state2vector(self, state):
        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self, state):
        full_state = []
        obj_encoding = [0, 0, 0, 0]
        obj_mapping = {
            'key': 0,
            'door': 1,
            'goal': 2,
            'null': 3
        }
        # obj_encoding[obj_mapping[state[-2]]] = 1
        full_state.append(int(state[0]))
        full_state.append(int(state[1]))
        full_state.append(obj_mapping[state[-2]])
        return np.array(full_state, dtype=np.float16)

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        # return current_position & current_direction
        return len(self.actions)

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # where2go state2atom
        if phase is 'where2go':
            if int(state[0]) == 1:
                atoms.add(Atom(HAS_KEY, ["agent"]))
            else:
                atoms.add(Atom(NO_KEY, ["agent"]))

            if int(state[1]) == 1:
                atoms.add(Atom(IS_OPEN, ["agent"]))
            else:
                atoms.add(Atom(IS_CLOSED, ["agent"]))
        # how2go state2atom
        if phase is 'how2go':
            if int(state[2]) > 0:
                atoms.add(Atom(POSX, ["agent"]))
            elif int(state[2]) < 0:
                atoms.add(Atom(NEGX, ["agent"]))
            else:
                atoms.add(Atom(ZEROX, ["agent"]))

            if int(state[3]) > 0:
                atoms.add(Atom(POSY, ["agent"]))
            elif int(state[3]) < 0:
                atoms.add(Atom(NEGY, ["agent"]))
            else:
                atoms.add(Atom(ZEROY, ["agent"]))
        # what2do state2atom
        elif phase is 'what2do':
            for adj_obj in state[5:]:
                if adj_obj == 'key':
                    atoms.add(Atom(AT_KEY, ["agent"]))
                elif adj_obj == 'door':
                    atoms.add(Atom(AT_DOOR, ["agent"]))
                elif adj_obj == 'goal':
                    atoms.add(Atom(AT_GOAL, ["agent"]))
            if int(state[0]) == 1:
                atoms.add(Atom(H_KEY, ["agent"]))
            else:
                atoms.add(Atom(N_KEY, ["agent"]))
        return atoms

    def action_refine(self, action, agent_direction):
        # action-direction
        refine_map = {'00': [1, 1, 2],
                      '01': [1, 2],
                      '02': [2],
                      '03': [0, 2],
                      '20': [0, 2],
                      '21': [0, 0, 2],
                      '22': [1, 2],
                      '23': [2],
                      '10': [2],
                      '11': [0, 2],
                      '12': [1, 1, 2],
                      '13': [1, 2],
                      '30': [1, 2],
                      '31': [2],
                      '32': [0, 2],
                      '33': [1, 1, 2]}

        return refine_map[str(action) + str(agent_direction)]

    def w2d_action_refine(self, agent_direction, agent_target_direction):
        # agent_target_direction - agent_direction
        refine_map = {'00': [1],
                      '01': [],
                      '02': [0],
                      '03': [1, 1],
                      '20': [0],
                      '21': [1, 1],
                      '22': [1],
                      '23': [],
                      '10': [],
                      '11': [0],
                      '12': [1, 1],
                      '13': [1],
                      '30': [1, 1],
                      '31': [1],
                      '32': [],
                      '33': [0]}
        agent_target_dict = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        agent_target_dir = agent_target_dict.index(agent_target_direction.tolist())
        return refine_map[str(agent_target_dir) + str(agent_direction)]  # + [str(action)]

    def save_image(self, ):
        if self.save:
            img = self.env.render('rgb')
            im = Image.fromarray(img)
            im.save(os.path.join(self.save_path, str(self.step) + '.png'))

    def how2go_step_learnable(self, action):
        print(f"in how2go,step: {self.step}, state: {self._state}, agent_state: {self.agent_state}, target: {self.goto_target}, loc : {self.temp_goal}")
        action = self.mapping[action.predicate.name]
        actions = self.action_refine(action, self._direct)
        for i in actions:
            self.step += 1
            self.full_state, ori_reward, done, info = self.env.step(i)
            self.save_image()
        self.full_state = self.full_state['image']

        # update various state vectors
        self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)
        if np.linalg.norm(self.agent_state) == 0:
            if self.goto_target == 'door':
                if self.loc == 'left':
                    temp_state = np.array([-1, 0])
                else:
                    temp_state = np.array([1, 0])
                actions = self.w2d_action_refine(self._direct, temp_state)
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                    self.step += 1
                    self.save_image()
                if not self.test:
                    ori_reward = 1.0
                else:
                    ori_reward = 0.1
            self.phase = 'what2do'
            # self.save_image()
            self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
            self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

            self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
            self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                             self.full_state, self.temp_goal)
            self._state[4] = self.surround_wall

            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
        if np.linalg.norm(self.agent_state) == 1 and self.goto_target != 'door':  # or (np.linalg.norm(self.agent_state) == 0 and self.goto_target=='door'):
            # rotate agent to face the object when adjacent
            temp_state = self.agent_state
            actions = self.w2d_action_refine(self._direct, temp_state)
            for i in actions:
                self.full_state, ori_reward, done, info = self.env.step(i)
                self.full_state = self.full_state['image']
                self.step += 1
                self.save_image()
            if not self.test: ori_reward = 1.0
            if self.test: ori_reward = 0.1
            # switch phase to what2do
            self.phase = 'what2do'
            # self.save_image()
            self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
            self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

            self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
            self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                             self.full_state, self.temp_goal)
            self._state[4] = self.surround_wall

            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))

        # update h2g state
        self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state, self.temp_goal)
        self._state[4] = self.surround_wall
        # update w2d state
        if self.phase != 'what2do':
            self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                           self.full_state)
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))

        reward = self.get_reward(done, ori_reward)
        print(f"in how2go,present state: {self._state},  reward: {reward}\n")

        return reward, done

    def what2do(self, action):
        self.step += 1
        action_name = action.predicate.name
        action = action.predicate.name
        action = self.w2d_mapping[action]
        print(f"in what2do round: {self.round}, step: {self.step}, action: {action_name}, state: {self._state}")
        if self.has_key and action_name == 'drop':
            count = 0
            for i in range(3):
                self.full_state, ori_reward, done, info = self.env.step(0)
                count += 1
                self.step += 1
                self.save_image()
                self.full_state, ori_reward, done, info = self.env.step(action)
                self.step += 1
                self.save_image()
                self.full_state = self.full_state['image']
                if len(np.where(((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[0]) != 0:
                    print("success drop!")
                    self.key_loc = np.where(((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
                    print(f"key_loc: {self.key_loc}")

                    self.key_loc = [self.key_loc[0][0], self.key_loc[1][0]]
                    self.has_key = False
                    self._state[0] = 0
                    if (not self.dropped) and ('drop' in self._state[5:7]) and (not self.test): ori_reward = 0.3
                    self.dropped = True
                    for _ in range(count):
                        self.full_state, _, done, info = self.env.step(1)
                        if done: break
                    self.full_state = self.full_state['image']
                    self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                                   self.full_state)
                    self.adjacent = [self.goto_target]
                    self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                    self.phase = 'where2go'
                    self.save_image()
                    return ori_reward, done

        self.full_state, ori_reward, done, info = self.env.step(action)
        if done and ori_reward > 0.05:
            print("reached Goal!")
            if self.test: return ori_reward, done
            return ori_reward, done
        self.save_image()
        self.full_state = self.full_state['image']
        if (not self.has_key) and np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]],
                                                 np.array([1, 0, 0])):
            self._state[0] = 1  # the key has been picked up by the agent
            print(f"success pickup!")
            self.phase = 'where2go'
            if not self.prev_has_key: ori_reward = 0.3
            if self.test: ori_reward = 0.0
            self.prev_has_key = True

            return ori_reward, done,
        if (not self.is_open) and self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0:
            self._state[1] = 1  # door is open
            print(f"success open door!")
            self.full_state, ori_reward, done, info = self.env.step(2)
            self.step += 1
            self.save_image()
            self.full_state, ori_reward, done, info = self.env.step(2)
            self.step += 1
            self.save_image()
            if self.loc == 'left':
                self.loc = 'right'
            else:
                self.loc = 'left'
            self.full_state = self.full_state['image']
            self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]

            self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                           self.full_state)
            self.adjacent = ['door']
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
            self.phase = 'where2go'
            if not self.prev_door_open:
                if self.test:
                    ori_reward = ori_reward
                else:
                    ori_reward = 0.3
                self.prev_door_open = True
                self.done = done
            # ori_reward = self.get_reward(done,ori_reward)
            return ori_reward, done,
        self.phase = 'where2go'
        if not self.test: ori_reward -= 0.01
        return ori_reward, done

    def where2go(self, action):
        self.prev_goal = self.goto_target
        self.goto_target = action.predicate.name.split('_')[-1]
        if self.goto_target == 'key':
            self.temp_goal = self.key_loc
        elif self.goto_target == 'door':
            if self.loc == 'left':
                temp_loc = [self.door_loc[0] - 1, self.door_loc[1]]
            else:
                temp_loc = [self.door_loc[0] + 1, self.door_loc[1]]
            self.temp_goal = temp_loc
        elif self.goto_target == 'goal':
            self.temp_goal = self.goal_pos

        if self.prev_goal != None and (self.goto_target == self.prev_goal) and self.goto_target != 'door':
            self.phase = 'what2do'
            return 0.0, False
        self._state[2:4] = coor_trans(np.array(np.where(self.full_state == 10))[:2].reshape(-1),
                                      self.width - 2) - coor_trans(self.temp_goal, self.width - 2)
        print(f"in where2go round: {self.round}, step: {self.step}, prev_target: {self.prev_goal} ,self.goto_target: {self.goto_target}, state:{self._state}")
        if self._state[2] == 0 and self._state[3] == 0:
            if self.goto_target == 'door':
                if self.loc == 'left':
                    temp_state = np.array([-1, 0])
                else:
                    temp_state = np.array([1, 0])
                actions = self.w2d_action_refine(self._direct, temp_state)
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                    self.step += 1
                    self.save_image()
            self.phase = 'what2do'
            self._state[5] = self.goto_target
            return -0.01, False
        self.phase = 'how2go'
        return -0.01, False

    def next_step(self, action):
        self.has_key = int(self._state[0])
        # door is open?
        self.is_open = int(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        # learnable where2go
        if self.phase is 'where2go':
            reward, done = self.where2go(action)

        elif self.phase is 'how2go':
            reward, done = self.how2go_step_learnable(action)

        elif self.phase is 'what2do':
            reward, done = self.what2do(action)
        return reward, done

    def get_reward(self, done, ori_reward=0.0):
        # return ori_reward
        if self.test: return ori_reward
        if done and ori_reward > 0.1:
            return 1.0
        elif done:
            return ori_reward

        if ori_reward > 0: return ori_reward
        return -0.01

    @property
    def critic_state(self):
        return copy.deepcopy(self._state[2:4])

    def flatten(self, ):
        state = []
        state.extend(self.w2h_state['w2g'])
        temp = []
        for key, item in self.w2h_state['h2g'][0].items():
            state.extend(item)
        state.extend([int(i) for i in self.w2h_state['h2g'][1]])
        state.extend(self.w2h_state['w2d'])
        return state

    def change_state(self):
        state = self.flatten()
        full_state = []
        obj_encoding = [0, 0, 0]
        obj_mapping = {
            'key': 0,
            'door': 1,
            'goal': 2
        }
        for index in range(len(state)):
            if index < len(state) - 2:
                full_state.append(int(state[index]))
            else:
                if state[index] in obj_mapping:
                    obj_encoding[obj_mapping[state[index]]] = 1
        full_state.extend(obj_encoding)

        return np.array(full_state, dtype=np.float16)

    def reset(self, unlock=True, test=False):
        self.test = test
        self.full_state = self.env.reset()
        self.full_state = self.full_state['image']
        # ball location
        self.goal_pos = np.array(np.where(self.full_state == 6))[:2].reshape(-1)
        self.key_loc = np.where(
            ((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.key_loc = [self.key_loc[0][0], self.key_loc[1][0]]
        # door location
        self.door_loc = np.where(((self.full_state == np.array([4, 4, 2]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.door_loc = [self.door_loc[0][0], self.door_loc[1][0]]
        if self.key_loc[0] < self.door_loc[0]:
            self.loc = 'left'
        else:
            self.loc = 'right'
        # initial state vector (agent location)
        self.init_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        # relative postional vector (now cheating)
        self.init_dir_vector = coor_trans(self.init_state, self.width - 2) - coor_trans(self.key_loc, self.width - 2)
        # initial direction scalar
        self.init_dir = self.full_state[self.init_state[0], self.init_state[1]][2]
        self.init_state = (str(self.init_dir_vector[0]), str(self.init_dir_vector[1]))
        # agent has key?
        self.has_key = str(0)
        # door is open?
        self.is_open = str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        # adjacent to?
        self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                       self.full_state)
        # surrounding wallls
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state)
        print(f"self.surround_wall: {self.surround_wall}")

        # w2h state [w2g, h2g, w2d]
        self.w2h_state = ['0', str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))]
        self.w2h_state.extend(list(self.init_state))
        self.w2h_state.extend([self.surround_wall])  # add more observations to how2go state to make it markovian (#4)
        self.w2h_state.extend(self.adjacent)  # (#2)
        self.w2h_state.extend(['null'] * (7 - len(self.w2h_state)))
        # b = copy.deepcopy(self.w2h_state)
        # self.init_w2h_state = copy.deepcopy(self.w2h_state)
        self._state = copy.deepcopy(self.w2h_state)
        print(self.w2h_state)

        self._direct = copy.deepcopy(self.init_dir)
        self.phase = 'where2go'  # where2go, how2go, what2do
        self.round += 1
        self.agent_state = 0
        if self.save:
            path = 'log_images_ballkey/seed_{}'.format(self.seed)
            if not os.path.exists(os.path.join(path, str(self.round))):
                os.makedirs(os.path.join(path, str(self.round)))
            self.save_path = os.path.join(path, str(self.round))
        self.step = 0
        self.prev_door_open = False
        self.done = False
        self.prev_has_key = False
        self.dropped = False
        self.prev_goal = None
        self.goto_target = None


class BlockedBoxUnlockPickup():
    def __init__(self, initial_state=("1", "1"), env_name='MiniGrid-BlockedBoxUnlockPickup-8x8-v0', width=8, save=False):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]
        w2g_actions = [GT_KEY, GT_DOOR, GT_BOX, GT_BLOCKAGE, GT_GOAL]
        w2d_actions = [TOGGLE, PICK, DROP]
        self.actions = h2g_actions + w2g_actions + w2d_actions

        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        w2g_ext = [IS_OPEN, IS_CLOSED, IS_BLOCKED, IS_INCOMPLETE_BLOCKED, HAS_KEY, NO_KEY, KEY_NOSHOW]
        w2d_ext = [AT_KEY, AT_BLOCKAGE, AT_DOOR, AT_BOX, AT_GOAL, H_KEY, H_BLOCKAGE, NONE]  # H_KEY,H_BLOCKAGE, NONE
        ext = h2g_ext + w2g_ext + w2d_ext
        self.env = FullyObsWrapper(BlockedBoxUnlockPickupEnv(room_size=width))
        # self.env = Monitor(FullyObsWrapper(BlockedBoxUnlockPickupEnv(room_size=width)), './video', force=True)
        self.width = 2 * width - 1
        self.height = width
        self.round = -1
        self.save = save
        self.reset()

        self.mapping = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        self._object_encoding = {'agent': 0, 'key': 1, 'door': 2, 'goal': 3}
        self.w2d_mapping = {'pick': 3, 'toggle': 5, 'drop': 4}
        constants = [str(i) for i in range(-1 * (width), width - 1)] + ['key', 'door', 'box', 'blockage', 'agent']

        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        background = []
        self.background = background

    @property
    def all_actions(self):
        return self.actions

    def state2vector(self, state):
        # what2do state vector
        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self, state):
        full_state = []
        obj_encoding = [0, 0, 0, 0]
        obj_mapping = {
            'key': 0,
            'door': 1,
            'blockage': 2,
            'box': 3,
            'goal':4,
            'null':5
        }
        # obj_encoding[obj_mapping[state[-2]]] = 1
        full_state.append(int(state[0]))
        full_state.append(int(state[1]))
        full_state.append(obj_mapping[state[-2]])
        # full_state.extend(obj_encoding)

        return np.array(full_state, dtype=np.float16)

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        return len(self.actions)

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # where2go state2atom
        if phase is 'where2go':
            if self.not_blocked:
                if self.has_blockage:
                    atoms.add(Atom(IS_INCOMPLETE_BLOCKED, ["agent"]))
                elif int(state[1]) == 1:
                    atoms.add(Atom(IS_OPEN, ["agent"]))
                else:
                    atoms.add(Atom(IS_CLOSED, ["agent"]))
            else:
                atoms.add(Atom(IS_BLOCKED, ["agent"]))

            if self.key_showup:
                if int(state[0]) == 1:
                    atoms.add(Atom(HAS_KEY, ["agent"]))
                else:
                    atoms.add(Atom(NO_KEY, ["agent"]))
            else:
                atoms.add(Atom(KEY_NOSHOW, ["agent"]))
        # how2go state2atom
        if phase is 'how2go':
            if int(state[2]) > 0:
                atoms.add(Atom(POSX, ["agent"]))
            elif int(state[2]) < 0:
                atoms.add(Atom(NEGX, ["agent"]))
            else:
                atoms.add(Atom(ZEROX, ["agent"]))

            if int(state[3]) > 0:
                atoms.add(Atom(POSY, ["agent"]))
            elif int(state[3]) < 0:
                atoms.add(Atom(NEGY, ["agent"]))
            else:
                atoms.add(Atom(ZEROY, ["agent"]))
        # what2do state2atom
        elif phase is 'what2do':
            for adj_obj in state[5:]:
                if adj_obj == 'key':
                    atoms.add(Atom(AT_KEY, ["agent"]))
                elif adj_obj == 'door':
                    atoms.add(Atom(AT_DOOR, ["agent"]))
                elif adj_obj == 'blockage':
                    atoms.add(Atom(AT_BLOCKAGE, ["agent"]))
                elif adj_obj == 'box':
                    atoms.add(Atom(AT_BOX, ["agent"]))
                elif adj_obj == 'goal':
                    atoms.add(Atom(AT_GOAL, ["agent"]))
            if int(state[0]) == 1:
                atoms.add(Atom(H_KEY, ["agent"]))
            elif self.has_blockage:
                atoms.add(Atom(H_BLOCKAGE, ['agent']))
            else:
                atoms.add(Atom(NONE, ["agent"]))
        return atoms

    def action_refine(self, action, agent_direction):
        # action-direction
        refine_map = {'00': [1, 1, 2],
                      '01': [1, 2],
                      '02': [2],
                      '03': [0, 2],
                      '20': [0, 2],
                      '21': [0, 0, 2],
                      '22': [1, 2],
                      '23': [2],
                      '10': [2],
                      '11': [0, 2],
                      '12': [1, 1, 2],
                      '13': [1, 2],
                      '30': [1, 2],
                      '31': [2],
                      '32': [0, 2],
                      '33': [1, 1, 2]}

        return refine_map[str(action) + str(agent_direction)]

    def w2d_action_refine(self, agent_direction, agent_target_direction):
        # agent_target_direction - agent_direction
        refine_map = {'00': [1],
                      '01': [],
                      '02': [0],
                      '03': [1, 1],
                      '20': [0],
                      '21': [1, 1],
                      '22': [1],
                      '23': [],
                      '10': [],
                      '11': [0],
                      '12': [1, 1],
                      '13': [1],
                      '30': [1, 1],
                      '31': [1],
                      '32': [],
                      '33': [0]}
        agent_target_dict = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        agent_target_dir = agent_target_dict.index(agent_target_direction.tolist())

        return refine_map[str(agent_target_dir) + str(agent_direction)]  # + [str(action)]

    def save_image(self, ):
        if self.save:
            img = self.env.render('rgb')
            im = Image.fromarray(img)
            im.save(os.path.join(self.save_path, str(self.step) + '.png'))

    def how2go_step_learnable(self, action):
        print(f"in how2go,step: {self.step}, state: {self._state}, agent_state: {self.agent_state}, target: {self.goto_target}, loc : {self.temp_goal}")
        action = self.mapping[action.predicate.name]
        actions = self.action_refine(action, self._direct)
        # print(f"action taken: {action}")
        for i in actions:
            self.step += 1
            self.full_state, ori_reward, done, info = self.env.step(i)
            self.save_image()
        self.full_state = self.full_state['image']
        # update various state vectors
        self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)
        if np.linalg.norm(self.agent_state) == 0:
            if self.goto_target == 'door':
                if self.loc == 'left':
                    temp_state = np.array([-1, 0])
                else:
                    temp_state = np.array([1, 0])
                actions = self.w2d_action_refine(self._direct, temp_state)
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                    self.step += 1
                    # self.save_image()
                if not self.test:
                    ori_reward = 0.3
                else:
                    ori_reward = 0.1
            self.phase = 'what2do'
            # self.save_image()
            self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
            self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

            self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
            self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                             self.full_state, self.temp_goal)
            self._state[4] = self.surround_wall

            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
        if np.linalg.norm(self.agent_state) == 1 and self.goto_target != 'door':  # or (np.linalg.norm(self.agent_state) == 0 and self.goto_target=='door'):
            # rotate agent to face the object when adjacent
            temp_state = self.agent_state
            actions = self.w2d_action_refine(self._direct, temp_state)
            for i in actions:
                self.full_state, ori_reward, done, info = self.env.step(i)
                self.full_state = self.full_state['image']
                self.step += 1
                self.save_image()
            if not self.test:
                ori_reward = 0.3
            else:
                ori_reward = 0.1
            # switch phase to what2do
            self.phase = 'what2do'
            # self.save_image()
            self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
            self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

            self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
            self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                             self.full_state, self.temp_goal)
            self._state[4] = self.surround_wall

            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))

        # update h2g state
        self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state, self.temp_goal)
        self._state[4] = self.surround_wall
        # update w2d state
        if self.phase != 'what2do':
            self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                           self.full_state)
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))

        # reward = self.get_reward(done, ori_reward)
        reward = ori_reward
        return reward, done

    def what2do(self, action):
        self.step += 1
        action_name = action.predicate.name
        action = action.predicate.name
        action = self.w2d_mapping[action]
        print(f"in what2do round: {self.round}, step: {self.step}, action: {action_name}, state: {self._state}")
        if (self.has_key or self.has_blockage) and action_name == 'drop':
            count = 0
            for i in range(3):
                self.full_state, ori_reward, done, info = self.env.step(0)
                count += 1
                self.step += 1
                self.save_image()
                self.full_state, ori_reward, done, info = self.env.step(action)
                self.step += 1
                self.save_image()
                self.full_state = self.full_state['image']
                if self.has_key and len(np.where(
                        ((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[
                                            0]) != 0:
                    print("success drop key!")
                    self.key_loc = np.where(
                        ((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
                    print(f"key_loc: {self.key_loc}")

                    self.key_loc = [self.key_loc[0][0], self.key_loc[1][0]]
                    self.has_key = False
                    self._state[0] = 0
                    if (not self.dropped_key) and self.is_open and (not self.test): ori_reward = 0.3
                    self.dropped_key = True
                    for _ in range(count):
                        self.full_state, _, done, info = self.env.step(1)
                        if done: break
                    self.full_state = self.full_state['image']
                    self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                                   self.full_state)
                    self.adjacent = [self.goto_target]
                    self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                    self.phase = 'where2go'
                    self.save_image()
                    return ori_reward, done
                if self.has_blockage and len(np.where(
                        ((self.full_state == np.array([6, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[
                                                 0]) != 0:
                    print("success drop blockage!")
                    self.has_blockage = False
                    if (not self.dropped_blockage) and (not self.test): ori_reward = 0.3
                    self.dropped_blockage = True
                    if self.loc == 'left' and np.array_equal(self.full_state[self.door_loc[0]-1, self.door_loc[1]], np.array([6, 4, 0])):
                        self.blockage_loc = [self.door_loc[0] - 1, self.door_loc[1]]
                        self.not_blocked = False
                    elif self.loc == 'right' and np.array_equal(self.full_state[self.door_loc[0]+1, self.door_loc[1]], np.array([6, 4, 0])):
                        self.blockage_loc = [self.door_loc[0] + 1, self.door_loc[1]]
                        self.not_blocked = False
                    for _ in range(count):
                        self.full_state, _, done, info = self.env.step(1)
                        if done: break
                    self.full_state = self.full_state['image']
                    self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                                   self.full_state)
                    self.adjacent = [self.goto_target]
                    self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                    self.phase = 'where2go'
                    self.save_image()
                    return ori_reward, done

        # self.full_state, ori_reward, done, info = self.env.step(action)
        # if 'box' in self._state[5:] and action == 3:
        #     self.phase = 'where2go'
        #     return 0, False
        # else:
        self.full_state, ori_reward, done, info = self.env.step(action)
        self.save_image()
        self.full_state = self.full_state['image']
        pre_key_showup = self.key_showup
        self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]],
                                             np.array([7, 4, 0]))
        if pre_key_showup != self.key_showup:
            self.key_loc = self.box_loc_fix.copy()
            self.box_loc = [0, 0]
            self.goto_target = ''
            if not self.prev_has_key: ori_reward = 0.3
            self.phase = 'where2go'
            return ori_reward, done
        if done and ori_reward > 0.05:
            print("reached Goal!")
            if self.test: return ori_reward, done
            return ori_reward, done
        if (not self.has_key) and np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]],
                                                 np.array([1, 0, 0])):
            self._state[0] = 1  # the key has been picked up by the agent
            print(f"success pickup!")
            self.phase = 'where2go'
            self.key_loc = [0, 0]
            self.goto_target = ''
            if not self.prev_has_key: ori_reward = 0.3
            if self.test: ori_reward = 0.0
            self.prev_has_key = True
            return ori_reward, done,
        if (not self.has_blockage) and len(
                np.where(((self.full_state == np.array([6, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[
                    0]) == 0:
            print(f"success pickup blockage!")
            self.phase = 'where2go'
            self.has_blockage = True
            self.not_blocked = True
            self.blockage_loc = [0, 0]
            self.goto_target = ''
            ori_reward = 0.3
            if self.test: ori_reward = 0.0
            return ori_reward, done
        if (not self.is_open) and self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0:
            self._state[1] = 1  # door is open
            print(f"success open door!")
            self.full_state, ori_reward, done, info = self.env.step(2)
            self.step += 1
            self.save_image()
            self.full_state, ori_reward, done, info = self.env.step(2)
            self.step += 1
            self.save_image()
            if self.loc == 'left':
                self.loc = 'right'
            else:
                self.loc = 'left'
            self.goto_target = ''
            self.full_state = self.full_state['image']
            self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]

            self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                           self.full_state)
            self.adjacent = ['door']
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
            self.phase = 'where2go'
            if not self.prev_door_open:
                ori_reward = 0.3
                self.prev_door_open = True
                self.done = done
            return ori_reward, done,
        self.phase = 'where2go'
        if not self.test: ori_reward -= 0.01
        return ori_reward, done

    def where2go(self, action):
        self.prev_goal = self.goto_target
        self.goto_target = action.predicate.name.split('_')[-1]
        if self.goto_target == 'key':
            self.temp_goal = self.key_loc
        elif self.goto_target == 'door':
            if self.loc == 'left':
                temp_loc = [self.door_loc[0] - 1, self.door_loc[1]]
            else:
                temp_loc = [self.door_loc[0] + 1, self.door_loc[1]]
            self.temp_goal = temp_loc
        elif self.goto_target == 'blockage':
            self.temp_goal = self.blockage_loc
        elif self.goto_target == 'box':
            self.temp_goal = self.box_loc
        elif self.goto_target == 'goal':
            self.temp_goal = self.goal_pos
        self.env.step(6)

        if self.prev_goal != None and (self.goto_target == self.prev_goal) and self.goto_target != 'door':
            self.phase = 'what2do'
            return 0, False
        self._state[2:4] = coor_trans(np.array(np.where(self.full_state == 10))[:2].reshape(-1),
                                      self.width - 2) - coor_trans(self.temp_goal, self.width - 2)
        print(
            f"in where2go round: {self.round}, step: {self.step}, prev_target: {self.prev_goal} ,self.goto_target: {self.goto_target}, state:{self._state}")
        if self._state[2] == 0 and self._state[3] == 0:
            if self.goto_target == 'door':
                if self.loc == 'left':
                    temp_state = np.array([-1, 0])
                else:
                    temp_state = np.array([1, 0])
                actions = self.w2d_action_refine(self._direct, temp_state)
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                    self.step += 1
                    # self.save_image()
            self.phase = 'what2do'
            self._state[5] = self.goto_target
            return 0, False
        self.phase = 'how2go'
        return 0, False

    def next_step(self, action):
        self.has_key = int(self._state[0])
        # door is open?
        self.is_open = int(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]],
                                             np.array([7, 4, 0]))
        # if self.key_showup:
        #     self.key_loc = self.box_loc_fix.copy()
        #     self.box_loc = [0, 0]
        # else:
        #     self.key_loc = [0, 0]
        # learnable where2go
        if self.phase is 'where2go':
            reward, done = self.where2go(action)

        elif self.phase is 'how2go':
            reward, done = self.how2go_step_learnable(action)

        elif self.phase is 'what2do':
            reward, done = self.what2do(action)
        return reward, done

    def get_reward(self, done, ori_reward=0.0):
        # return ori_reward
        if self.test: return ori_reward
        if done and ori_reward > 0.1:
            return ori_reward
        elif done:
            return -0.01

        if ori_reward > 0: return ori_reward
        return -0.01

    @property
    def critic_state(self):
        return copy.deepcopy(self._state[2:4])

    def flatten(self, ):
        state = []
        state.extend(self.w2h_state['w2g'])
        temp = []
        for key, item in self.w2h_state['h2g'][0].items():
            state.extend(item)
        state.extend([int(i) for i in self.w2h_state['h2g'][1]])
        state.extend(self.w2h_state['w2d'])
        return state

    def change_state(self):
        state = self.flatten()
        full_state = []
        obj_encoding = [0, 0, 0]
        obj_mapping = {
            'key': 0,
            'door': 1,
            'goal': 2
        }
        for index in range(len(state)):
            if index < len(state) - 2:
                full_state.append(int(state[index]))
            else:
                if state[index] in obj_mapping:
                    obj_encoding[obj_mapping[state[index]]] = 1
        full_state.extend(obj_encoding)

        return np.array(full_state, dtype=np.float16)

    def reset(self, unlock=True, test=False):
        self.test = test
        self.full_state = self.env.reset()
        self.full_state = self.full_state['image']
        # goal location
        self.goal_pos = np.where(
            ((self.full_state == np.array([6, 1, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.goal_pos = [self.goal_pos[0][0], self.goal_pos[1][0]]
        # box location
        self.box_loc = np.where(
            ((self.full_state == np.array([7, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.box_loc = [self.box_loc[0][0], self.box_loc[1][0]]
        self.box_loc_fix = self.box_loc.copy()
        # key location
        self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]],
                                             np.array([7, 4, 0]))
        if self.key_showup:
            self.key_loc = self.box_loc_fix.copy()
            self.box_loc = [0, 0]
        else:
            self.key_loc = [0, 0]  # goto_key and key_loc==[0, 0]则直接惩罚且return (remain where2go)
        # door location
        self.door_loc = np.where(((self.full_state == np.array([4, 4, 2]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.door_loc = [self.door_loc[0][0], self.door_loc[1][0]]
        if self.box_loc[0] < self.door_loc[0]:
            self.loc = 'left'
            if np.array_equal(self.full_state[self.door_loc[0] - 1, self.door_loc[1]], np.array([6, 4, 0])):
                self.not_blocked = False
                self.blockage_loc = [self.door_loc[0] - 1, self.door_loc[1]]
            else:
                self.not_blocked = True
                self.blockage_loc = [0, 0]
        else:
            self.loc = 'right'
            if np.array_equal(self.full_state[self.door_loc[0] + 1, self.door_loc[1]], np.array([6, 4, 0])):
                self.not_blocked = False
                self.blockage_loc = [self.door_loc[0] + 1, self.door_loc[1]]
            else:
                self.not_blocked = True
                self.blockage_loc = [0, 0]
        print(f"key_loc: {self.key_loc}")
        print(f"door_loc: {self.door_loc}")
        print(f"blockage_loc: {self.blockage_loc}")
        print(f"goal_pos: {self.goal_pos}")
        print(f"box_loc: {self.box_loc}")
        # initial state vector (agent location)
        self.init_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        # relative postional vector (now cheating)
        self.init_dir_vector = coor_trans(self.init_state, self.width - 2) - coor_trans(self.key_loc, self.width - 2)
        # initial direction scalar
        self.init_dir = self.full_state[self.init_state[0], self.init_state[1]][2]
        self.init_state = (str(self.init_dir_vector[0]), str(self.init_dir_vector[1]))
        # agent has key?
        self.has_key = str(0)
        self.has_blockage = False
        # door is open?
        self.is_open = str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        # adjacent to?
        self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                       self.full_state)
        # surrounding wallls
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state)
        print(f"self.surround_wall: {self.surround_wall}")

        # w2h state [w2g, h2g, w2d]
        self.w2h_state = ['0', str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))]
        self.w2h_state.extend(list(self.init_state))
        self.w2h_state.extend([self.surround_wall])  # add more observations to how2go state to make it markovian (#4)
        self.w2h_state.extend(self.adjacent)  # (#2)
        self.w2h_state.extend(['null'] * (7 - len(self.w2h_state)))
        # b = copy.deepcopy(self.w2h_state)
        # self.init_w2h_state = copy.deepcopy(self.w2h_state)
        self._state = copy.deepcopy(self.w2h_state)
        print(self.w2h_state)

        self._direct = copy.deepcopy(self.init_dir)
        self.phase = 'where2go'  # where2go, how2go, what2do
        self.round += 1
        self.agent_state = 0
        if self.save:
            path = 'log_images_unlock/seed_{}'.format(self.seed)
            if not os.path.exists(os.path.join(path, str(self.round))):
                os.makedirs(os.path.join(path, str(self.round)))
            self.save_path = os.path.join(path, str(self.round))
        self.step = 0
        self.prev_door_open = False
        self.done = False
        self.prev_has_key = False
        self.dropped_key = False
        self.dropped_blockage = False
        self.prev_goal = None
        self.goto_target = None


GT_FLOOR = Predicate('gt_floor', 0)
PICK_GOAL = Predicate('pick_goal', 1)   # agent pick goal
H_GOAL = Predicate('h_goal', 1)   # agent has goal
AT_FLOOR = Predicate("at_floor", 1)
IS_NOT_PLACE = Predicate("is_not_place", 1)
# IS_INCOMPLETE_BLOCKED = Predicate('is_incomplete_blocked', 1)
GT_BLOCKEDPLACE = Predicate('gt_blockedplace', 0)
AT_BLCOKEDPLACE = Predicate("at_blockedplace", 1)
class BlockedBoxPlaceGoal():
    def __init__(self, initial_state=("1", "1"), env_name='MiniGrid-BlockedBoxPlaceGoal-8x8-v0', width=8, save=False):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]
        w2g_actions = [GT_KEY, GT_DOOR, GT_BOX, GT_BLOCKAGE, GT_GOAL, GT_FLOOR, GT_BLOCKEDPLACE]
        w2d_actions = [TOGGLE, PICK, DROP]
        self.actions = h2g_actions + w2g_actions + w2d_actions

        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        w2g_ext = [IS_OPEN, IS_CLOSED, IS_BLOCKED, IS_INCOMPLETE_BLOCKED, IS_NOT_PLACE, HAS_KEY, NO_KEY, KEY_NOSHOW, PICK_GOAL]
        w2d_ext = [AT_KEY, AT_BLOCKAGE, AT_DOOR, AT_BOX, AT_FLOOR, AT_GOAL, AT_BLCOKEDPLACE, H_KEY, H_BLOCKAGE, NONE, H_GOAL]  # H_KEY,H_BLOCKAGE, NONE
        ext = h2g_ext + w2g_ext + w2d_ext

        self.env = FullyObsWrapper(BlockedBoxPlaceGoalEnv(room_size=width))
        self.width = 2 * width - 1
        self.height = width
        self.round = -1
        self.save = save
        self.reset()

        self.mapping = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
        self._object_encoding = {'agent': 0, 'key': 1, 'door': 2, 'goal': 3}
        self.w2d_mapping = {'pick': 3, 'toggle': 5, 'drop': 4}
        constants = [str(i) for i in range(-1 * (width), width - 1)] + ['key', 'door', 'box', 'blockage', 'agent']

        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        background = []
        self.background = background

    @property
    def all_actions(self):
        return self.actions

    def state2vector(self, state):
        # what2do state vector
        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self, state):
        full_state = []
        obj_encoding = [0, 0, 0, 0]
        obj_mapping = {
            'key': 0,
            'door': 1,
            'blockage': 2,
            'box': 3,
            'goal': 4,
            'null': 5
        }
        # obj_encoding[obj_mapping[state[-2]]] = 1
        full_state.append(int(state[0]))
        full_state.append(int(state[1]))
        full_state.append(obj_mapping[state[-2]])
        # full_state.extend(obj_encoding)

        return np.array(full_state, dtype=np.float16)

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        return len(self.actions)

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # where2go state2atom
        if phase is 'where2go':
            if self.not_blocked:
                if self.has_blockage:
                    atoms.add(Atom(IS_INCOMPLETE_BLOCKED, ["agent"]))
                elif not self.place_blockage:
                    atoms.add(Atom(IS_NOT_PLACE, ["agent"]))
                elif int(state[1]) == 1:
                    atoms.add(Atom(IS_OPEN, ["agent"]))
                else:
                    atoms.add(Atom(IS_CLOSED, ["agent"]))
            else:
                atoms.add(Atom(IS_BLOCKED, ["agent"]))

            if self.key_showup:
                if int(state[0]) == 1:
                    atoms.add(Atom(HAS_KEY, ["agent"]))
                elif self.has_goal:
                    atoms.add(Atom(PICK_GOAL, ["agent"]))
                else:
                    atoms.add(Atom(NO_KEY, ["agent"]))
            else:
                atoms.add(Atom(KEY_NOSHOW, ["agent"]))
        # how2go state2atom
        if phase is 'how2go':
            if int(state[2]) > 0:
                atoms.add(Atom(POSX, ["agent"]))
            elif int(state[2]) < 0:
                atoms.add(Atom(NEGX, ["agent"]))
            else:
                atoms.add(Atom(ZEROX, ["agent"]))

            if int(state[3]) > 0:
                atoms.add(Atom(POSY, ["agent"]))
            elif int(state[3]) < 0:
                atoms.add(Atom(NEGY, ["agent"]))
            else:
                atoms.add(Atom(ZEROY, ["agent"]))
        # what2do state2atom
        elif phase is 'what2do':
            for adj_obj in state[5:]:
                if adj_obj == 'key':
                    atoms.add(Atom(AT_KEY, ["agent"]))
                elif adj_obj == 'door':
                    atoms.add(Atom(AT_DOOR, ["agent"]))
                elif adj_obj == 'blockage':
                    atoms.add(Atom(AT_BLOCKAGE, ["agent"]))
                elif adj_obj == 'box':
                    atoms.add(Atom(AT_BOX, ["agent"]))
                elif adj_obj == 'goal':
                    atoms.add(Atom(AT_GOAL, ["agent"]))
                elif adj_obj == 'floor':
                    atoms.add(Atom(AT_FLOOR, ["agent"]))
                elif adj_obj == 'blockedplace':
                    atoms.add(Atom(AT_BLCOKEDPLACE, ["agent"]))
            if int(state[0]) == 1:
                atoms.add(Atom(H_KEY, ["agent"]))
            elif self.has_blockage:
                atoms.add(Atom(H_BLOCKAGE, ['agent']))
            elif self.has_goal:
                atoms.add(Atom(H_GOAL, ['agent']))
            else:
                atoms.add(Atom(NONE, ["agent"]))
        return atoms

    def action_refine(self, action, agent_direction):
        # action-direction
        refine_map = {'00': [1, 1, 2],
                      '01': [1, 2],
                      '02': [2],
                      '03': [0, 2],
                      '20': [0, 2],
                      '21': [0, 0, 2],
                      '22': [1, 2],
                      '23': [2],
                      '10': [2],
                      '11': [0, 2],
                      '12': [1, 1, 2],
                      '13': [1, 2],
                      '30': [1, 2],
                      '31': [2],
                      '32': [0, 2],
                      '33': [1, 1, 2]}

        return refine_map[str(action) + str(agent_direction)]

    def w2d_action_refine(self, agent_direction, agent_target_direction):
        # agent_target_direction - agent_direction
        refine_map = {'00': [1],
                      '01': [],
                      '02': [0],
                      '03': [1, 1],
                      '20': [0],
                      '21': [1, 1],
                      '22': [1],
                      '23': [],
                      '10': [],
                      '11': [0],
                      '12': [1, 1],
                      '13': [1],
                      '30': [1, 1],
                      '31': [1],
                      '32': [],
                      '33': [0]}
        agent_target_dict = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        agent_target_dir = agent_target_dict.index(agent_target_direction.tolist())

        return refine_map[str(agent_target_dir) + str(agent_direction)]  # + [str(action)]

    def save_image(self, ):
        if self.save:
            img = self.env.render('rgb')
            im = Image.fromarray(img)
            im.save(os.path.join(self.save_path, str(self.step) + '.png'))

    def how2go_step_learnable(self, action):
        print(
            f"in how2go,step: {self.step}, state: {self._state}, agent_state: {self.agent_state}, target: {self.goto_target}, loc : {self.temp_goal}")
        action = self.mapping[action.predicate.name]
        actions = self.action_refine(action, self._direct)
        # print(f"action taken: {action}")
        for i in actions:
            self.step += 1
            self.full_state, ori_reward, done, info = self.env.step(i)
            self.save_image()
        self.full_state = self.full_state['image']
        # update various state vectors
        self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
        self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)
        self.at_floor = False
        self.at_blockedplace = False
        if np.linalg.norm(self.agent_state) == 0:
            if self.goto_target == 'door':
                if self.loc == 'left':
                    temp_state = np.array([-1, 0])
                else:
                    temp_state = np.array([1, 0])
                actions = self.w2d_action_refine(self._direct, temp_state)
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                    self.step += 1
                    self.save_image()
                if not self.test:
                    ori_reward = 0.3
                else:
                    ori_reward = 0.1
            self.phase = 'what2do'
            self.save_image()
            self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
            self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

            self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
            self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                             self.full_state, self.temp_goal)
            self._state[4] = self.surround_wall

            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
        if np.linalg.norm(
                self.agent_state) == 1 and self.goto_target != 'door':  # or (np.linalg.norm(self.agent_state) == 0 and self.goto_target=='door'):
            # rotate agent to face the object when adjacent
            temp_state = self.agent_state
            actions = self.w2d_action_refine(self._direct, temp_state)
            for i in actions:
                self.full_state, ori_reward, done, info = self.env.step(i)
                self.full_state = self.full_state['image']
                self.step += 1
                self.save_image()
            if not self.test:
                ori_reward = 0.3
            else:
                ori_reward = 0.1
            # switch phase to what2do
            if self.goto_target == 'floor':
                self.at_floor = True
            if self.goto_target == 'blockedplace':
                self.at_blockedplace = True
            self.phase = 'what2do'
            # self.save_image()
            self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]
            self.agent_state = coor_trans(self.agent_state, self.width - 2) - coor_trans(self.temp_goal, self.width - 2)

            self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
            self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                             self.full_state, self.temp_goal)
            self._state[4] = self.surround_wall

            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))

        # update h2g state
        self._state[2:4] = (str(self.agent_state[0]), str(self.agent_state[1]))
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state, self.temp_goal)
        self._state[4] = self.surround_wall
        # update w2d state
        if self.phase != 'what2do':
            self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                           self.full_state)
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))

        # reward = self.get_reward(done, ori_reward)
        reward = ori_reward
        return reward, done

    def what2do(self, action):
        action_name = action.predicate.name
        action = action.predicate.name
        action = self.w2d_mapping[action]
        print(f"in what2do round: {self.round}, step: {self.step}, action: {action_name}, state: {self._state}")
        print("self.at_floor", self.at_floor, 'self.at_blockedplace', self.at_blockedplace)
        if self.has_goal and self.at_floor and action_name == 'drop':
            self.full_state, ori_reward, done, info = self.env.step(action)
            self.step += 1
            self.save_image()
            self.full_state = self.full_state['image']
            print("mission accomplished!")
            self.has_goal = False
            self.goal_pos = np.where(
                ((self.full_state == np.array([6, 1, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
            print(f"goal_pos: {self.goal_pos}")
            self.goal_pos = [self.goal_pos[0][0], self.goal_pos[1][0]]
            self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                           self.full_state)
            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
            self.phase = 'where2go'
            self.save_image()
            return ori_reward, done
        elif self.has_blockage and self.at_blockedplace and action_name == 'drop':
            self.full_state, ori_reward, done, info = self.env.step(action)
            if (not self.test): ori_reward = 0.3
            self.place_blockage = True
            self.step += 1
            self.save_image()
            self.full_state = self.full_state['image']
            print("success drop blockage!")
            self.has_blockage = False
            self.blockage_loc = np.where(
                        ((self.full_state == np.array([6, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
            self.blockage_loc = [self.blockage_loc[0][0], self.blockage_loc[1][0]]
            self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                           self.full_state)
            self.adjacent = [self.goto_target]
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
            self.phase = 'where2go'
            self.save_image()
            return ori_reward, done
        elif (self.has_key or self.has_blockage or self.has_goal) and action_name == 'drop':
            count = 0
            for i in range(3):
                self.full_state, ori_reward, done, info = self.env.step(0)
                count += 1
                self.step += 1
                self.save_image()
                self.full_state, ori_reward, done, info = self.env.step(action)
                self.step += 1
                self.save_image()
                self.full_state = self.full_state['image']
                if self.has_key and len(np.where(
                        ((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[
                                            0]) != 0:
                    print("success drop key!")
                    self.key_loc = np.where(
                        ((self.full_state == np.array([5, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
                    print(f"key_loc: {self.key_loc}")

                    self.key_loc = [self.key_loc[0][0], self.key_loc[1][0]]
                    self.has_key = False
                    self._state[0] = 0
                    if (not self.dropped_key) and self.is_open and (not self.test):
                        ori_reward = 0.3
                        self.dropped_key = True
                    # if (not self.dropped_key) and self.is_open and (not self.test): ori_reward = 0.3
                    for _ in range(count):
                        self.full_state, _, done, info = self.env.step(1)
                        if done: break
                    self.full_state = self.full_state['image']
                    self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                                   self.full_state)
                    self.adjacent = [self.goto_target]
                    self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                    self.phase = 'where2go'
                    self.save_image()
                    return ori_reward, done
                if self.has_blockage and len(np.where(
                        ((self.full_state == np.array([6, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[
                                                 0]) != 0:
                    print("drop blockage!")
                    self.has_blockage = False
                    # if (not self.dropped_blockage) and (not self.test): ori_reward = 0.3
                    self.dropped_blockage = True
                    if self.loc == 'left' and np.array_equal(self.full_state[self.door_loc[0] - 1, self.door_loc[1]],
                                                             np.array([6, 4, 0])):
                        self.blockage_loc = [self.door_loc[0] - 1, self.door_loc[1]]
                        self.not_blocked = False
                    elif self.loc == 'right' and np.array_equal(self.full_state[self.door_loc[0] + 1, self.door_loc[1]],
                                                                np.array([6, 4, 0])):
                        self.blockage_loc = [self.door_loc[0] + 1, self.door_loc[1]]
                        self.not_blocked = False
                    else:
                        self.blockage_loc = np.where(
                            ((self.full_state == np.array([6, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
                        self.blockage_loc = [self.blockage_loc[0][0], self.blockage_loc[1][0]]
                    for _ in range(count):
                        self.full_state, _, done, info = self.env.step(1)
                        if done: break
                    self.full_state = self.full_state['image']
                    self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                                   self.full_state)
                    self.adjacent = [self.goto_target]
                    self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                    self.phase = 'where2go'
                    self.save_image()
                    return ori_reward, done
                if self.has_goal and len(np.where(
                        ((self.full_state == np.array([6, 1, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[
                                                 0]) != 0:
                    print("success drop goal!")
                    self.has_goal = False
                    self.goal_pos = np.where(
                        ((self.full_state == np.array([6, 1, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
                    print(f"goal_pos: {self.goal_pos}")
                    self.goal_pos = [self.goal_pos[0][0], self.goal_pos[1][0]]
                    for _ in range(count):
                        self.full_state, _, done, info = self.env.step(1)
                        if done: break
                    self.full_state = self.full_state['image']
                    self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                                   self.full_state)
                    self.adjacent = [self.goto_target]
                    self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
                    self.phase = 'where2go'
                    self.save_image()
                    return ori_reward, done

        self.full_state, ori_reward, done, info = self.env.step(action)
        self.step += 1
        self.save_image()
        self.full_state = self.full_state['image']
        pre_key_showup = self.key_showup
        self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]],
                                             np.array([7, 4, 0]))
        if pre_key_showup != self.key_showup:
            self.key_loc = self.box_loc_fix.copy()
            self.box_loc = [0, 0]
            self.goto_target = ''
            if not self.prev_has_key: ori_reward = 0.3
            self.phase = 'where2go'
            return ori_reward, done

        if (not self.has_key) and np.array_equal(self.full_state[self.key_loc[0], self.key_loc[1]],
                                                 np.array([1, 0, 0])):
            self._state[0] = 1  # the key has been picked up by the agent
            print(f"success pickup key!")
            self.phase = 'where2go'
            self.key_loc = [0, 0]
            self.goto_target = ''
            if not self.prev_has_key and self.place_blockage:
                ori_reward = 0.3
                self.prev_has_key = True
            if self.test: ori_reward = 0.0
            return ori_reward, done,

        if(not self.has_goal) and len(
                np.where(((self.full_state == np.array([6, 1, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[
                    0]) == 0:
            print(f"success pickup goal!")
            self.phase = 'where2go'
            self.goal_pos = [0, 0]
            self.goto_target = ''
            if not self.prev_has_goal: ori_reward = 0.3
            if self.test: ori_reward = 0.0
            self.prev_has_goal = True
            self.has_goal = True
            return ori_reward, done,

        if (not self.has_blockage) and len(
                np.where(((self.full_state == np.array([6, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))[
                    0]) == 0:
            print(f"success pickup blockage!")
            self.phase = 'where2go'
            self.has_blockage = True
            self.not_blocked = True
            self.blockage_loc = [0, 0]
            self.goto_target = ''
            if not self.prev_has_blockage: ori_reward = 0.3
            self.prev_has_blockage = True
            if self.test: ori_reward = 0.0
            return ori_reward, done

        if (not self.is_open) and self.full_state[self.door_loc[0], self.door_loc[1]][2] == 0:
            self._state[1] = 1  # door is open
            print(f"success open door!")
            self.full_state, ori_reward, done, info = self.env.step(2)
            self.step += 1
            self.save_image()
            self.full_state, ori_reward, done, info = self.env.step(2)
            self.step += 1
            self.save_image()
            if self.loc == 'left':
                self.loc = 'right'
            else:
                self.loc = 'left'
            self.goto_target = ''
            self.full_state = self.full_state['image']
            self.agent_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
            self._direct = self.full_state[int(self.agent_state[0]), int(self.agent_state[1])][2]

            self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                           self.full_state)
            self.adjacent = ['door']
            self._state[5:7] = self.adjacent + ['null'] * (2 - len(self.adjacent))
            self.phase = 'where2go'
            if not self.prev_door_open:
                ori_reward = 0.3
                self.prev_door_open = True
                self.done = done
            return ori_reward, done,
        self.phase = 'where2go'
        # if not self.test: ori_reward -= 0.01
        return ori_reward, done

    def where2go(self, action):
        self.prev_goal = self.goto_target
        self.goto_target = action.predicate.name.split('_')[-1]
        if self.goto_target == 'key':
            self.at_blockedplace = False
            self.temp_goal = self.key_loc
        elif self.goto_target == 'door':
            self.at_blockedplace = False
            if self.loc == 'left':
                temp_loc = [self.door_loc[0] - 1, self.door_loc[1]]
            else:
                temp_loc = [self.door_loc[0] + 1, self.door_loc[1]]
            self.temp_goal = temp_loc
        elif self.goto_target == 'blockage':
            self.at_blockedplace = False
            self.temp_goal = self.blockage_loc
        elif self.goto_target == 'box':
            self.at_blockedplace = False
            self.temp_goal = self.box_loc
        elif self.goto_target == 'goal':
            self.at_blockedplace = False
            self.temp_goal = self.goal_pos
        elif self.goto_target == 'floor':
            self.at_blockedplace = False
            self.temp_goal = self.floor_loc
        elif self.goto_target == 'blockedplace':
            self.temp_goal = self.blockedplace_loc
        self.env.step(6)

        if self.prev_goal != None and (self.goto_target == self.prev_goal) and self.goto_target != 'door':
            self.phase = 'what2do'
            return 0, False
        self._state[2:4] = coor_trans(np.array(np.where(self.full_state == 10))[:2].reshape(-1),
                                      self.width - 2) - coor_trans(self.temp_goal, self.width - 2)
        print(
            f"in where2go round: {self.round}, step: {self.step}, prev_target: {self.prev_goal} ,self.goto_target: {self.goto_target}, state:{self._state}")
        if self._state[2] == 0 and self._state[3] == 0:
            if self.goto_target == 'door':
                if self.loc == 'left':
                    temp_state = np.array([-1, 0])
                else:
                    temp_state = np.array([1, 0])
                actions = self.w2d_action_refine(self._direct, temp_state)
                for i in actions:
                    self.full_state, ori_reward, done, info = self.env.step(i)
                    self.full_state = self.full_state['image']
                    self.step += 1
                    self.save_image()
            self.phase = 'what2do'
            self._state[5] = self.goto_target
            return 0, False
        self.phase = 'how2go'
        return 0, False

    def next_step(self, action):
        self.has_key = int(self._state[0])
        # door is open?
        self.is_open = int(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]],
                                             np.array([7, 4, 0]))
        # if self.key_showup:
        #     self.key_loc = self.box_loc_fix.copy()
        #     self.box_loc = [0, 0]
        # else:
        #     self.key_loc = [0, 0]
        # learnable where2go
        if self.phase is 'where2go':
            reward, done = self.where2go(action)

        elif self.phase is 'how2go':
            reward, done = self.how2go_step_learnable(action)

        elif self.phase is 'what2do':
            reward, done = self.what2do(action)
        return reward, done

    def get_reward(self, done, ori_reward=0.0):
        # return ori_reward
        if self.test: return ori_reward
        if done and ori_reward > 0.1:
            return ori_reward
        elif done:
            return -0.01

        if ori_reward > 0: return ori_reward
        return -0.01

    @property
    def critic_state(self):
        return copy.deepcopy(self._state[2:4])

    def flatten(self, ):
        state = []
        state.extend(self.w2h_state['w2g'])
        temp = []
        for key, item in self.w2h_state['h2g'][0].items():
            state.extend(item)
        state.extend([int(i) for i in self.w2h_state['h2g'][1]])
        state.extend(self.w2h_state['w2d'])
        return state

    def change_state(self):
        state = self.flatten()
        full_state = []
        obj_encoding = [0, 0, 0]
        obj_mapping = {
            'key': 0,
            'door': 1,
            'goal': 2
        }
        for index in range(len(state)):
            if index < len(state) - 2:
                full_state.append(int(state[index]))
            else:
                if state[index] in obj_mapping:
                    obj_encoding[obj_mapping[state[index]]] = 1
        full_state.extend(obj_encoding)

        return np.array(full_state, dtype=np.float16)

    def reset(self, unlock=True, test=False):
        self.test = test
        self.full_state = self.env.reset()
        self.full_state = self.full_state['image']

        path = 'log_images_target_task2/seed_{}'.format(0)
        if not os.path.exists(os.path.join(path, str(self.round))):
            os.makedirs(os.path.join(path, str(self.round)))
        self.save_path = os.path.join(path, str(self.round))
        img = self.env.render('rgb')
        im = Image.fromarray(img)
        im.save(os.path.join(self.save_path, 'reset.png'))

        # goal location
        self.goal_pos = np.where(
            ((self.full_state == np.array([6, 1, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.goal_pos = [self.goal_pos[0][0], self.goal_pos[1][0]]
        # floor location
        self.floor_loc = np.where(
            ((self.full_state == np.array([3, 0, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.floor_loc = [self.floor_loc[0][0], self.floor_loc[1][0]]
        # blockedplace location
        self.blockedplace_loc = np.where(
            ((self.full_state == np.array([3, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.blockedplace_loc = [self.blockedplace_loc[0][0], self.blockedplace_loc[1][0]]
        # box location
        self.box_loc = np.where(
            ((self.full_state == np.array([7, 4, 0]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.box_loc = [self.box_loc[0][0], self.box_loc[1][0]]
        self.box_loc_fix = self.box_loc.copy()
        # key location
        self.key_showup = 1 - np.array_equal(self.full_state[self.box_loc_fix[0], self.box_loc_fix[1]],
                                             np.array([7, 4, 0]))
        if self.key_showup:
            self.key_loc = self.box_loc_fix.copy()
            self.box_loc = [0, 0]
        else:
            self.key_loc = [0, 0]  # goto_key and key_loc==[0, 0]则直接惩罚且return (remain where2go)
        # door location
        self.door_loc = np.where(
            ((self.full_state == np.array([4, 4, 2]).reshape(1, -1)[:, None, None, :]).all(3)).any(0))
        self.door_loc = [self.door_loc[0][0], self.door_loc[1][0]]
        if self.box_loc[0] < self.door_loc[0]:
            self.loc = 'left'
            if np.array_equal(self.full_state[self.door_loc[0] - 1, self.door_loc[1]], np.array([6, 4, 0])):
                self.not_blocked = False
                self.blockage_loc = [self.door_loc[0] - 1, self.door_loc[1]]
            else:
                self.not_blocked = True
                self.blockage_loc = [0, 0]
        else:
            self.loc = 'right'
            if np.array_equal(self.full_state[self.door_loc[0] + 1, self.door_loc[1]], np.array([6, 4, 0])):
                self.not_blocked = False
                self.blockage_loc = [self.door_loc[0] + 1, self.door_loc[1]]
            else:
                self.not_blocked = True
                self.blockage_loc = [0, 0]
        print(f"key_loc: {self.key_loc}")
        print(f"door_loc: {self.door_loc}")
        print(f"blockage_loc: {self.blockage_loc}")
        print(f"goal_pos: {self.goal_pos}")
        print(f"box_loc: {self.box_loc}")
        print(f"floor_loc: {self.floor_loc}")
        print(f"blockedplace_loc: {self.blockedplace_loc}")


        # initial state vector (agent location)
        self.init_state = np.array(np.where(self.full_state == 10))[:2].reshape(-1)
        # relative postional vector (now cheating)
        self.init_dir_vector = coor_trans(self.init_state, self.width - 2) - coor_trans(self.key_loc, self.width - 2)
        # initial direction scalar
        self.init_dir = self.full_state[self.init_state[0], self.init_state[1]][2]
        self.init_state = (str(self.init_dir_vector[0]), str(self.init_dir_vector[1]))
        # agent has key?
        self.has_key = str(0)
        self.has_blockage = False
        self.place_blockage = False
        self.has_goal = False
        self.at_floor = False
        self.at_blockedplace = False
        # door is open?
        self.is_open = str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))
        # adjacent to?
        self.adjacent = adjacent_check(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                       self.full_state)
        # surrounding wallls
        self.surround_wall = surrounding(np.array(np.where(self.full_state == 10))[:2].reshape(-1).tolist(),
                                         self.full_state)
        print(f"self.surround_wall: {self.surround_wall}")

        # w2h state [w2g, h2g, w2d]
        self.w2h_state = ['0', str(int(self.full_state[int(self.door_loc[0]), int(self.door_loc[1])][2] == 0))]
        self.w2h_state.extend(list(self.init_state))
        self.w2h_state.extend([self.surround_wall])  # add more observations to how2go state to make it markovian (#4)
        self.w2h_state.extend(self.adjacent)  # (#2)
        self.w2h_state.extend(['null'] * (7 - len(self.w2h_state)))
        # b = copy.deepcopy(self.w2h_state)
        # self.init_w2h_state = copy.deepcopy(self.w2h_state)
        self._state = copy.deepcopy(self.w2h_state)
        print(self.w2h_state)

        self._direct = copy.deepcopy(self.init_dir)
        self.phase = 'where2go'  # where2go, how2go, what2do
        self.round += 1
        self.agent_state = 0
        if self.save:
            path = 'log_images_unlock/seed_{}'.format(self.seed)
            if not os.path.exists(os.path.join(path, str(self.round))):
                os.makedirs(os.path.join(path, str(self.round)))
            self.save_path = os.path.join(path, str(self.round))
        self.step = 0
        self.prev_door_open = False
        self.done = False
        self.prev_has_key = False
        self.prev_has_goal = False
        self.prev_has_blockage = False
        self.dropped_key = False
        self.dropped_blockage = False
        self.prev_goal = None
        self.goto_target = None


GT_COFFEE = Predicate('gt_coffee', 0)
GT_OFFIC = Predicate('gt_coffee', 0)
DO = Predicate("do",0)
COFFEE_NOT = Predicate('coffee_not', 1)
COFFEE_HAVE = Predicate('coffee_have', 1)
WAIT_COFFEE = Predicate('wait_coffee', 1)
ACCEPT_COOFFEE = Predicate('accept_coffee', 1)
AT_COFFEE = Predicate("at_coffee", 1)
AT_OFFICE = Predicate("at_office", 1)
H_COFFEE = Predicate("h_coffee", 1)
N_COFFEE = Predicate("n_coffee", 1)
class CoffeeMazeEnv():
    def __init__(self):
        h2g_actions = [LEFT, RIGHT, UP, DOWN]
        w2g_actions = [GT_COFFEE, GT_OFFIC, GT_GOAL]
        # w2d_actions = [DO]
        # self.actions = h2g_actions + w2g_actions + w2d_actions
        self.actions = h2g_actions + w2g_actions

        h2g_ext = [POSX, POSY, NEGX, NEGY, ZEROX, ZEROY]
        w2g_ext = [COFFEE_NOT, COFFEE_HAVE, WAIT_COFFEE, ACCEPT_COOFFEE]
        # w2d_ext = [AT_COFFEE, AT_OFFICE, H_COFFEE, N_COFFEE, AT_GOAL]
        # ext = h2g_ext + w2g_ext + w2d_ext
        ext = h2g_ext + w2g_ext

        self.env = CoffeeMaze()
        self.round = -1
        self.reset()
        constants = [str(i) for i in range(-1 * 12, 12 - 1)] + ['coffee', 'office', 'goal', 'agent']

        self.language = LanguageFrame(self.actions, extensional=ext,
                                      constants=constants)
        self.background = []

    @property
    def all_actions(self):
        return self.actions

    def state2vector(self, state):
        return np.array([float(state[0]), float(state[1])])

    def w2_state2vector(self, state):
        full_state = []
        obj_encoding = [0, 0, 0, 0]
        obj_mapping = {
            'coffee': 0,
            'office': 1,
            'goal': 2,
            'null': 3
        }
        # obj_encoding[obj_mapping[state[-2]]] = 1
        full_state.append(int(state[0]))
        full_state.append(int(state[1]))
        full_state.append(obj_mapping[state[-2]])
        return np.array(full_state, dtype=np.float16)

    @property
    def state(self):
        # return current_position & current_direction
        return copy.deepcopy(self._state)

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2atom(self, action_index):
        return Atom(self.actions[action_index], ["-1", "-1"])

    @property
    def action_n(self):
        # return current_position & current_direction
        return self.env.action_space.shape[0]

    def state2atoms(self, state, phase='where2go'):
        atoms = set()
        # where2go state2atom
        if phase is 'where2go':
            if int(state[0]) == 1:
                atoms.add(Atom(COFFEE_HAVE, ["agent"]))
            else:
                atoms.add(Atom(COFFEE_NOT, ["agent"]))

            if int(state[1]) == 1:
                atoms.add(Atom(ACCEPT_COOFFEE, ["agent"]))
            else:
                atoms.add(Atom(WAIT_COFFEE, ["agent"]))
        # how2go state2atom
        if phase is 'how2go':
            if int(state[2]) > 0:
                atoms.add(Atom(POSX, ["agent"]))
            elif int(state[2]) < 0:
                atoms.add(Atom(NEGX, ["agent"]))
            else:
                atoms.add(Atom(ZEROX, ["agent"]))

            if int(state[3]) > 0:
                atoms.add(Atom(POSY, ["agent"]))
            elif int(state[3]) < 0:
                atoms.add(Atom(NEGY, ["agent"]))
            else:
                atoms.add(Atom(ZEROY, ["agent"]))
        # what2do state2atom
        # elif phase is 'what2do':
        #     for adj_obj in state[5:]:
        #         if adj_obj == 'coffee':
        #             atoms.add(Atom(AT_COFFEE, ["agent"]))
        #         elif adj_obj == 'office':
        #             atoms.add(Atom(AT_OFFICE, ["agent"]))
        #         elif adj_obj == 'goal':
        #             atoms.add(Atom(AT_GOAL, ["agent"]))
        #     if int(state[0]) == 1:
        #         atoms.add(Atom(H_COFFEE, ["agent"]))
        #     else:
        #         atoms.add(Atom(N_COFFEE, ["agent"]))
        return atoms

    def how2go(self, action):
        state, reward, done, _ = self.env.step(action)
        self.full_state = state
        self._state[0], self._state[1] = state[4], state[5]
        self._state[2], self._state[3] = state[0] - self.goto_target[0], state[1] - self.goto_target[1]
        if np.linalg.norm(state[0:2] - self.goto_target) <= 0.4:
            # self.phase = "what2do"
            self.phase = 'where2go'
            self.goto_reward = reward
            self._state[4] = self.goto_target_str
        reward = np.exp(-np.linalg.norm(state[0:2] - self.goto_target))
        return reward, done

    # def what2do(self, action):
    #     self.phase = "where2go"
    #     return 0, False

    def where2go(self, action):
        self.goto_target_str = action.predicate.name.split('_')[-1]
        if self.goto_target_str == 'coffee':
            self.goto_target = self.coffee_pos
        if self.goto_target_str == 'office':
            self.goto_target = self.office_pos
        if self.goto_target_str == 'goal':
            self.goto_target = self.goal_pos
        reward = self.goto_reward
        self.goto_reward = 0
        self.phase = 'how2go'
        self._state[2], self._state[3] = self.full_state[0] - self.goto_target[0], self.full_state[1] - self.goto_target[1]
        return reward, False

    def next_step(self, action):
        if self.phase is 'where2go':
            reward, done = self.where2go(action)

        if self.phase is 'how2go':
            reward, done = self.how2go(action)

        # if self.phase is 'what2do':
        #     reward, done = self.what2do(action)
        return reward, done

    @property
    def critic_state(self):
        return copy.deepcopy(self._state[2:4])

    def reset(self):
        self.full_state = self.env.reset()
        self.coffee_pos = self.env.coffee_pos
        self.office_pos = self.env.office_pos
        self.goal_pos = self.env.get_target()
        self._state = [0] * 5
        self._state[0], self._state[1] = self.full_state[4], self.full_state[5]
        self._state[2], self._state[3] = self.full_state[0], self.full_state[1]
        self._state[4] = 'goal'
        self.phase = 'where2go'



