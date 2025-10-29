from __future__ import print_function, division, absolute_import
from tkinter import N
from core.clause import *
from core.ilp import *
# import ray
from core.rules import *
from core.induction import *
from core.rl import *
from core.clause import str2atom,str2clause
from core.symbolicEnvironment import *


def setup_predecessor():
    constants = [str(i) for i in range(10)]
    background = [Atom(Predicate("succ", 2), [constants[i], constants[i + 1]]) for i in range(9)]
    positive = [Atom(Predicate("predecessor", 2), [constants[i], constants[i+2]]) for i in range(8)]
    all_atom = [Atom(Predicate("predecessor", 2), [constants[i], constants[j]]) for i in range(10) for j in range(10)]
    negative = list(set(all_atom) - set(positive))

    language = LanguageFrame(Predicate("predecessor",2), [Predicate("succ",2)], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([], {Predicate("predecessor", 2): [RuleTemplate(1, False), RuleTemplate(0, False)]},
                                   4)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_fizz():
    constants = [str(i) for i in range(10)]
    succ = Predicate("succ", 2)
    zero = Predicate("zero", 1)
    fizz = Predicate("fizz", 1)
    pred1 = Predicate("pred1", 2)
    pred2 = Predicate("pred2", 2)

    background = [Atom(succ, [constants[i], constants[i + 1]]) for i in range(9)]
    background.append(Atom(zero, "0"))
    positive = [Atom(fizz, [constants[i]]) for i in range(0, 10, 3)]
    all_atom = [Atom(fizz, [constants[i]]) for i in range(10)]
    negative = list(set(all_atom) - set(positive))
    language = LanguageFrame(fizz, [zero, succ], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([pred1, pred2], {fizz: [RuleTemplate(1, True), RuleTemplate(1, False)],
                                                    pred1: [RuleTemplate(1, True),],
                                                    pred2: [RuleTemplate(1, True),],},
                                   10)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_even():
    constants = [str(i) for i in range(10)]
    succ = Predicate("succ", 2)
    zero = Predicate("zero", 1)
    target = Predicate("even", 1)
    pred = Predicate("pred", 2)
    background = [Atom(succ, [constants[i], constants[i + 1]]) for i in range(9)]
    background.append(Atom(zero, "0"))
    positive = [Atom(target, [constants[i]]) for i in range(0, 10, 2)]
    all_atom = [Atom(target, [constants[i]]) for i in range(10)]
    negative = list(set(all_atom) - set(positive))
    language = LanguageFrame(target, [zero, succ], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([pred], {target: [RuleTemplate(1, True), RuleTemplate(1, False)],
                                            pred: [RuleTemplate(1, True),RuleTemplate(1, False)],
                                            },
                                   10)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_minigrid(variation=None, invented=False, env_name = None, size=8):
    env = MiniGrid(env_name = env_name, width=size)
    temp0 = [RuleTemplate(0, False)]
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 2)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 2)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
                                                    invented2: temp_2extential,
                                                    invented: temp_2extential,
                                                    invented3: temp_2extential,
                                                    invented4: temp_2extential,
                                                    UP: temp_2extential,
                                                    DOWN: temp_2extential,
                                                    LEFT: temp_2extential,
                                                    RIGHT: temp_2extential,
                                                    TOGGLE: temp1,
                                                    PICK: temp1},
                                       4)
    else:
        program_temp = ProgramTemplate([], {UP: temp_2extential,
                                            DOWN: temp_2extential,
                                            LEFT: temp_2extential,
                                            RIGHT: temp_2extential,
                                            GT_KEY: temp1,
                                            GT_DOOR: temp1,
                                            GT_GOAL: temp1,
                                            TOGGLE: temp1,
                                            PICK: temp1,}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_empty(variation=None, invented=False, env_name = None, size=8):
    env = Empty(env_name = env_name, width=size)
    temp0 = [RuleTemplate(0, False)]
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 2)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 2)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
                                                    invented2: temp_2extential,
                                                    invented: temp_2extential,
                                                    invented3: temp_2extential,
                                                    invented4: temp_2extential,
                                                    UP: temp_2extential,
                                                    DOWN: temp_2extential,
                                                    LEFT: temp_2extential,
                                                    RIGHT: temp_2extential,
                                                    TOGGLE: temp1,
                                                    PICK: temp1},
                                       4)
    else:
        program_temp = ProgramTemplate([], {UP: temp_2extential,
                                            DOWN: temp_2extential,
                                            LEFT: temp_2extential,
                                            RIGHT: temp_2extential,
                                            GT_KEY: temp1,
                                            GT_DOOR: temp1,
                                            GT_GOAL: temp1,
                                            TOGGLE: temp1,
                                            PICK: temp1,}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_unlockpickup(variation=None, invented=False, env_name = None, size=None, seed = 0):
    env = UnlockPickUp(env_name = env_name, size=size, seed=seed)
    temp0 = [RuleTemplate(0, False)]
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 2)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 2)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
                                                    invented2: temp_2extential,
                                                    invented: temp_2extential,
                                                    invented3: temp_2extential,
                                                    invented4: temp_2extential,
                                                    UP: temp_2extential,
                                                    DOWN: temp_2extential,
                                                    LEFT: temp_2extential,
                                                    RIGHT: temp_2extential,
                                                    TOGGLE: temp1,
                                                    PICK: temp1},
                                       4)
    else:
        program_temp = ProgramTemplate([], {UP: temp_2extential,
                                            DOWN: temp_2extential,
                                            LEFT: temp_2extential,
                                            RIGHT: temp_2extential,
                                            GT_KEY: temp1,
                                            GT_DOOR: temp1,
                                            GT_GOAL: temp1,
                                            GT_DROP: temp1,
                                            TOGGLE: temp1,
                                            PICK: temp1,
                                            DROP: temp1}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env


def setup_boxkey(variation=None, invented=False, env_name = None, size=8):
    env = BoxKey(env_name = env_name, width=size)
    temp0 = [RuleTemplate(0, False)]
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 2)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 2)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
                                                    invented2: temp_2extential,
                                                    invented: temp_2extential,
                                                    invented3: temp_2extential,
                                                    invented4: temp_2extential,
                                                    UP: temp_2extential,
                                                    DOWN: temp_2extential,
                                                    LEFT: temp_2extential,
                                                    RIGHT: temp_2extential,
                                                    TOGGLE: temp1,
                                                    PICK: temp1},
                                       4)
    else:
        program_temp = ProgramTemplate([], {UP: temp_2extential,
                                            DOWN: temp_2extential,
                                            LEFT: temp_2extential,
                                            RIGHT: temp_2extential,
                                            GT_KEY: temp1,
                                            GT_DOOR: temp1,
                                            GT_GOAL: temp1,
                                            GT_BOX: temp1,
                                            KEY_SHOW: temp1,
                                            KEY_NOSHOW: temp1,
                                            TOGGLE: temp1,
                                            PICK: temp1}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env


def setup_doorgoal(variation=None, invented=False, env_name = None, size=8):
    env = DoorGoal(width=size, env_name=env_name)
    temp0 = [RuleTemplate(0, False)]
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 2)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 2)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
            invented2: temp_2extential,
            invented: temp_2extential,
            invented3: temp_2extential,
            invented4: temp_2extential,
            UP: temp_2extential,
            DOWN: temp_2extential,
            LEFT: temp_2extential,
            RIGHT: temp_2extential,
            TOGGLE: temp1,
            PICK: temp1}, 4)
    else:
        program_temp = ProgramTemplate([], {UP: temp_2extential,
                                            DOWN: temp_2extential,
                                            LEFT: temp_2extential,
                                            RIGHT: temp_2extential,
                                            GT_KEY: temp1,
                                            GT_DOOR: temp1,
                                            GT_GOAL: temp1,
                                            TOGGLE: temp1,
                                            PICK: temp1,
                                            DROP: temp1}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env


def setup_boxdoor(variation=None, invented=False, env_name = None, size=8):
    env = BoxDoor(env_name=env_name, width=size)
    temp0 = [RuleTemplate(0, False)]
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 2)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 2)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
            invented2: temp_2extential,
            invented: temp_2extential,
            invented3: temp_2extential,
            invented4: temp_2extential,
            UP: temp_2extential,
            DOWN: temp_2extential,
            LEFT: temp_2extential,
            RIGHT: temp_2extential,
            TOGGLE: temp1,
            PICK: temp1},
                                       4)
    else:
        program_temp = ProgramTemplate([], {UP: temp_2extential,
                                            DOWN: temp_2extential,
                                            LEFT: temp_2extential,
                                            RIGHT: temp_2extential,
                                            GT_KEY: temp1,
                                            GT_DOOR: temp1,
                                            GT_BOX: temp1,
                                            KEY_SHOW: temp1,
                                            KEY_NOSHOW: temp1,
                                            TOGGLE: temp1,
                                            PICK: temp1,
                                            DROP: temp1}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env


def setup_gapball(variation=None, invented=False, env_name = None, size=8):
    env = GapBall(env_name=env_name, width=size)
    temp0 = [RuleTemplate(0, False)]
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 2)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 2)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
            invented2: temp_2extential,
            invented: temp_2extential,
            invented3: temp_2extential,
            invented4: temp_2extential,
            UP: temp_2extential,
            DOWN: temp_2extential,
            LEFT: temp_2extential,
            RIGHT: temp_2extential,
            TOGGLE: temp1,
            PICK: temp1},
                                       4)
    else:
        program_temp = ProgramTemplate([], {UP: temp_2extential,
                                            DOWN: temp_2extential,
                                            LEFT: temp_2extential,
                                            RIGHT: temp_2extential,
                                            GT_KEY: temp1,
                                            GT_BOX: temp1,
                                            GT_GOAL: temp1,
                                            KEY_SHOW: temp1,
                                            KEY_NOSHOW: temp1,
                                            TOGGLE: temp1,
                                            PICK: temp1,
                                            DROP: temp1}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env


def setup_blockeddoor(variation=None, invented=False, env_name = None, size=None):
    env = BlockedDoor(env_name=env_name, width=size)
    temp0 = [RuleTemplate(0, False)]
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 2)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 2)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
            invented2: temp_2extential,
            invented: temp_2extential,
            invented3: temp_2extential,
            invented4: temp_2extential,
            UP: temp_2extential,
            DOWN: temp_2extential,
            LEFT: temp_2extential,
            RIGHT: temp_2extential,
            TOGGLE: temp1,
            PICK: temp1},
                                       4)
    else:
        program_temp = ProgramTemplate([], {UP: temp_2extential,
                                            DOWN: temp_2extential,
                                            LEFT: temp_2extential,
                                            RIGHT: temp_2extential,
                                            GT_KEY: temp1,
                                            GT_DOOR: temp1,
                                            GT_BLOCKAGE: temp1,
                                            TOGGLE: temp1,
                                            PICK: temp1,
                                            DROP: temp1}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env


def setup_ballkey(variation=None, invented=False, env_name = None, size=None):
    env = BallKey(env_name=env_name, width=size)
    temp0 = [RuleTemplate(0, False)]
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 2)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 2)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
            invented2: temp_2extential,
            invented: temp_2extential,
            invented3: temp_2extential,
            invented4: temp_2extential,
            UP: temp_2extential,
            DOWN: temp_2extential,
            LEFT: temp_2extential,
            RIGHT: temp_2extential,
            TOGGLE: temp1,
            PICK: temp1},
                                       4)
    else:
        program_temp = ProgramTemplate([], {UP: temp_2extential,
                                            DOWN: temp_2extential,
                                            LEFT: temp_2extential,
                                            RIGHT: temp_2extential,
                                            GT_KEY: temp1,
                                            GT_DOOR: temp1,
                                            GT_GOAL: temp1,
                                            TOGGLE: temp1,
                                            PICK: temp1,
                                            DROP: temp1}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env


def setup_blocedboxunlockpickup(variation=None, invented=False, env_name = None, size=None):
    env = BlockedBoxUnlockPickup(env_name=env_name, width=size)
    temp0 = [RuleTemplate(0, False)]
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 2)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 2)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
            invented2: temp_2extential,
            invented: temp_2extential,
            invented3: temp_2extential,
            invented4: temp_2extential,
            UP: temp_2extential,
            DOWN: temp_2extential,
            LEFT: temp_2extential,
            RIGHT: temp_2extential,
            TOGGLE: temp1,
            PICK: temp1},
                                       4)
    else:
        program_temp = ProgramTemplate([], {UP: temp_2extential,
                                            DOWN: temp_2extential,
                                            LEFT: temp_2extential,
                                            RIGHT: temp_2extential,
                                            GT_KEY: temp1,
                                            GT_DOOR: temp1,
                                            GT_GOAL: temp1,
                                            GT_BOX: temp1,
                                            GT_BLOCKAGE: temp1,
                                            KEY_SHOW: temp1,
                                            KEY_NOSHOW: temp1,
                                            TOGGLE: temp1,
                                            PICK: temp1,
                                            DROP: temp1}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env

def setup_blocedboxplacegoal(variation=None, invented=False, env_name = None, size=None):
    env = BlockedBoxPlaceGoal(env_name=env_name, width=size)
    temp0 = [RuleTemplate(0, False)]
    temp1 = [RuleTemplate(1, False)]
    temp1_main = [RuleTemplate(2, False)]
    temp2_main = [RuleTemplate(3, True)]
    temp2_invent = [RuleTemplate(1, True)]
    temp_2extential = [RuleTemplate(2, False)]
    if invented:
        invented = Predicate("invented", 2)
        invented2 = Predicate("invented2", 2)
        invented3 = Predicate("invented3", 2)
        invented4 = Predicate("invented4", 2)
        program_temp = ProgramTemplate([invented, invented2, invented3, invented4], {
            invented2: temp_2extential,
            invented: temp_2extential,
            invented3: temp_2extential,
            invented4: temp_2extential,
            UP: temp_2extential,
            DOWN: temp_2extential,
            LEFT: temp_2extential,
            RIGHT: temp_2extential,
            TOGGLE: temp1,
            PICK: temp1},
                                       4)
    else:
        program_temp = ProgramTemplate([], {UP: temp_2extential,
                                            DOWN: temp_2extential,
                                            LEFT: temp_2extential,
                                            RIGHT: temp_2extential,
                                            GT_KEY: temp1,
                                            GT_DOOR: temp1,
                                            GT_GOAL: temp1,
                                            GT_BOX: temp1,
                                            GT_BLOCKAGE: temp1,
                                            GT_FLOOR: temp1,
                                            GT_BLOCKEDPLACE: temp1,
                                            KEY_SHOW: temp1,
                                            KEY_NOSHOW: temp1,
                                            TOGGLE: temp1,
                                            PICK: temp1,
                                            DROP: temp1}, 4)
    man = RulesManager(env.language, program_temp)
    return man, env