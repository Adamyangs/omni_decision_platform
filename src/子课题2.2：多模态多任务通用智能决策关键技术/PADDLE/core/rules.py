from __future__ import print_function, division, absolute_import
import numpy as np
from itertools import product
from core.ilp import *
from core.clause import *
from collections import defaultdict

H2G = ['left','right','up','down','posx','posy','negx','negy','zerox','zeroy']
W2G = ['gt_key','gt_door','gt_goal','gt_drop','gt_box','gt_blockage','gt_floor','gt_blockedplace','is_open','is_closed','is_blocked','is_incomplete_blocked','has_key','no_key','no_blockage','has_blockage','has_box','no_box', 'key_show', 'key_noshow', 'has_blockage', 'no_blockage','pick_goal', 'is_not_place']
W2D = ['at_key','at_door','at_goal','at_drop','at_box','at_blockage','at_blockedplace','h_key','n_key','h_box','n_box','toggle','pick','drop', 'h_blockage', 'n_blockage', 'none', 'h_goal','at_floor']

try:
    from itertools import izip_longest
except Exception:
    from itertools import zip_longest as izip_longest

class RulesManager():
    def __init__(self, language_frame, program_template, independent_clause=True):
        self.__language = language_frame
        self.program_template = program_template
        self.independent_clause = independent_clause
        self.predicate_phase = {'left': 'h2g',
                                'right': 'h2g',
                                'up': 'h2g',
                                'down': 'h2g',
                                'posx': 'h2g',
                                'posy': 'h2g',
                                'negx': 'h2g',
                                'negy': 'h2g',
                                'zerox': 'h2g',
                                'zeroy': 'h2g',
                                'gt_key': 'w2g',
                                'gt_door': 'w2g',
                                'gt_goal': 'w2g',
                                'gt_drop':'w2g',
                                'gt_box':'w2g',
                                'gt_blockage': 'w2g',
                                'gt_blockedplace': 'w2g',
                                'gt_floor': 'w2g',
                                'is_open': 'w2g',
                                'is_closed': 'w2g',
                                'is_blocked': 'w2g',
                                'is_incomplete_blocked': 'w2g',
                                'is_not_place': 'w2g',
                                'has_key': 'w2g',
                                'no_key': 'w2g',
                                'has_box': 'w2g',
                                'has_blockage': 'w2g',
                                'no_blockage': 'w2g',
                                'pick_goal': 'w2g',
                                'no_box': 'w2g',
                                'key_show': 'w2g',
                                'key_noshow': 'w2g',
                                'at_key': 'w2d',
                                'at_door': 'w2d',
                                'at_goal': 'w2d',
                                'at_drop': 'w2d',
                                'at_box': 'w2d',
                                'at_blockage': 'w2d',
                                'at_floor': 'w2d',
                                'at_blockedplace': 'w2d',
                                'h_key': 'w2d',
                                'n_key': 'w2d',
                                'h_box': 'w2d',
                                'n_box': 'w2d',
                                'h_blockage': 'w2d',
                                'n_blockage': 'w2d',
                                'h_goal': 'w2d',
                                'none': 'w2d',
                                'toggle': 'w2d',
                                'pick': 'w2d',
                                'drop': 'w2d'
                                }

        self.__predicate_mapping = {} # map from predicate to ground atom indices
        self.all_grounds = []
        self.__generate_grounds()
        self.all_clauses = defaultdict(list) # dictionary of predicate to list(2d) of lists of clause.
        self.__init_all_clauses()
        self.deduction_matrices =defaultdict(list) # dictionary of predicate to list of lists of deduction matrices.
        self.__init_deduction_matrices()

    def __init_all_clauses(self):
        intensionals = self.__language.target + self.program_template.auxiliary
        if self.independent_clause:
            for intensional in intensionals:
                for i in range(len(self.program_template.rule_temps[intensional])):
                    # print(i, intensional, self.program_template.rule_temps[intensional][i])
                    self.all_clauses[intensional].append(self.generate_clauses(intensional,
                                                                               self.program_template.rule_temps[
                                                                                   intensional][i]))
            # assert 1==0
        else:
            for intensional in intensionals:
                self.all_clauses[intensional].append(self.generate_clauses(intensional,
                                                                           self.program_template.rule_temps[intensional][0]))
                self.all_clauses[intensional].append(self.generate_clauses(intensional,
                                                                           self.program_template.rule_temps[intensional][1]))
        for key in self.all_clauses:
            for clauses in self.all_clauses[key]:
                for clause in clauses:
                    print(key, clause)
        print("\n")
    def __init_deduction_matrices(self):
        for intensional, clauses in self.all_clauses.items():

            for row in clauses:
                row_matrices = []
                for clause in row:
                    row_matrices.append(self.generate_induction_matrix(clause))
                self.deduction_matrices[intensional].append(row_matrices)


    def generate_clauses(self, intensional, rule_template):
        base_variable = tuple(range(intensional.arity))
        head = (Atom(intensional,base_variable),)

        body_variable = tuple(range(intensional.arity+rule_template.variables_n))
        if rule_template.allow_intensional:
            predicates = list(set(self.program_template.auxiliary).union((self.__language.extensional)))
            # .union(set([intensional])))
        else:
            predicates = self.__language.extensional
        terms = []
        for predicate in predicates:
            body_variables = [body_variable for _ in range(predicate.arity)]
            terms += self.generate_body_atoms(predicate, *body_variables)
        result_tuples = product(head, terms, terms)
        result_tuples = list(result_tuples)

        result_tuples = self.phase_constraint(result_tuples)
        return self.prune([Clause(result[0], result[1:]) for result in result_tuples])

    def phase_constraint(self, result_tuples):
        constrained_ls = []
        for i in result_tuples:
            # head phase
            if i[0].predicate.name in list(self.predicate_phase.keys()):
                head_phase = self.predicate_phase[i[0].predicate.name]
            else:
                head_phase = 'inv'

            # body term_0 phase
            if i[1].predicate.name in list(self.predicate_phase.keys()):
                term0_phase = self.predicate_phase[i[1].predicate.name]
            else:
                term0_phase = 'inv'

            # body term_1 phase
            if i[2].predicate.name in list(self.predicate_phase.keys()):
                term1_phase = self.predicate_phase[i[2].predicate.name]
            else:
                term1_phase = 'inv'

            if head_phase == 'h2g':
                # if i[1].predicate.name != 'current' and i[2].predicate.name != 'current':
                #     continue
                if i[1].predicate.name == 'current' and i[2].predicate.name == 'current':
                    continue
            
            if (term0_phase == 'h2g' and int(i[1].terms[0])==1) or (term1_phase == 'h2g' and int(i[2].terms[0])==1):
                # print(i[1].predicate.name, i[2].predicate.name)
                continue

            if (term0_phase == 'h2g' and term1_phase == 'h2g') and (i[1].predicate.name[-1] == i[2].predicate.name[-1]):
                # print(i[1].predicate.name, i[2].predicate.name)
                continue

            if (len(set([head_phase, term0_phase, term1_phase]).intersection({'w2g', 'h2g', 'w2d'}))==1 and i[1].predicate.name == 'posx' and i[2].predicate.name == 'posy') or \
             (len(set([head_phase, term0_phase, term1_phase]).intersection({'w2g', 'h2g', 'w2d'}))==1 and i[1].predicate.name == 'negx' and i[2].predicate.name == 'negy'):
                constrained_ls.append(i)

            if len(set([head_phase, term0_phase, term1_phase]).intersection({'w2g', 'h2g', 'w2d'}))==1 and i[1].predicate.name!=i[2].predicate.name and i[1].predicate.name[:2]!=i[2].predicate.name[:2]:
                is_predicates = ['is_key', 'is_door', 'is_goal', 'is_open', 'is_closed', 'is_blocked']
                has_predicates = ['has_key', 'no_key','h_key', 'n_key','has_box','no_box','h_box','n_box', 'has_blockage', 'no_blockage', 'h_blockage', 'n_blockage', 'none']
                if (i[1].predicate.name in is_predicates) and (i[2].predicate.name in is_predicates):
                    pass
                elif (i[1].predicate.name in has_predicates) and (i[2].predicate.name in has_predicates):
                    pass
                # elif (i[1].predicate.name in ['is_open', 'no_key']) and (i[2].predicate.name in ['is_open', 'no_key']):
                #     pass
                elif (i[1].predicate.name=='current' and i[1].terms[0]==i[1].terms[1]) or (i[2].predicate.name=='current' and i[2].terms[0]==i[2].terms[1]):
                    pass
                elif (len(i[1].terms)>1 and int(i[1].terms[0])==1) or (len(i[2].terms)>1 and int(i[2].terms[0])==1):
                    pass
                else:
                    constrained_ls.append(i)
            elif head_phase == 'w2d' and i[1].predicate.name in ['at_key', 'at_door', 'h_key', 'n_key'] and i[2].predicate.name in ['at_key', 'at_door', 'h_key', 'n_key']:
                if i[1].predicate.name[:2]!=i[2].predicate.name[:2]:
                    constrained_ls.append(i)
            # elif head_phase == 'h2g' and len(set([head_phase, term0_phase, term1_phase]).intersection({'w2g', 'h2g', 'w2d'}))==1:
            #     constrained_ls.append(i)
        return constrained_ls

    def find_index(self,atom):
        '''
        find index for a ground atom
        :param atom:
        :return:
        '''
        for term in atom.terms:
            assert isinstance(term, str)
        all_indexes = self.__predicate_mapping[atom.predicate]
        for index in all_indexes:
            if self.all_grounds[index] == atom:
                return index
        raise ValueError("didn't find {} in all ground atoms".format(atom))

    def generate_induction_matrix(self, clause):
        '''
        :param cluase:
        :return: array of size (number_of_ground_atoms, max_satisfy_paris, 2)
        '''
        #TODO: genrate matrix n
        satisfy = []
        for atom in self.all_grounds:
            if clause.head.predicate == atom.predicate:
                satisfy.append(self.find_satisfy_by_head(clause, atom))
                # print(f"satisfy: {satisfy}")
            else:
                satisfy.append([])
        X = np.empty(find_shape(satisfy), dtype=np.int32)
        fill_array(X, satisfy)
        return X

    def find_satisfy_by_head(self, clause, head):
        '''
        find combination of ground atoms that can trigger the clause to get a specific conclusion (head atom)
        :param clause:
        :param head:
        :return: list of tuples of indexes
        '''
        result = [] #list of paris of indexes
        free_body = clause.replace_by_head(head).body
        free_variables = list(free_body[0].variables.union(free_body[1].variables))
        # constrain constant according to respective phase 'head' predicate belong to
        if head.predicate.name in list(self.predicate_phase.keys()):
            if self.predicate_phase[head.predicate.name] == 'h2g':
                # if head.predicate.name=='current':
                #     constant = self.__language.constants[:-3]
                # else:
                constant = self.__language.constants[-1:]
            elif self.predicate_phase[head.predicate.name] == 'w2d':
                constant = self.__language.constants[-1:]
            else:
                constant = self.__language.constants[-1:]
        # print(f"head: {head.predicate.name}")
        # repeat_constatns = [self.__language.constants for _ in free_variables]
        repeat_constatns = [constant for _ in free_variables]
        all_constants_combination = product(*repeat_constatns)
        all_match = []
        for combination in all_constants_combination:
            all_match.append({free_variables[i]:constant for i,constant in enumerate(combination)})
        for match in all_match:
            result.append((self.find_index(free_body[0].replace_terms(match)),
                           self.find_index(free_body[1].replace_terms(match))))
        return result

    def __generate_grounds(self):
        self.all_grounds.append(Atom(Predicate("Empty", 0), []))
        self.__predicate_mapping[Predicate("Empty", 0)] = [0]
        all_predicates = self.__language.extensional+self.__language.target+self.program_template.auxiliary
        for predicate in all_predicates:
            if predicate.name in list(self.predicate_phase.keys()):
                pred_phase = self.predicate_phase[predicate.name]
                if pred_phase == 'h2g':
                    # if predicate.name=='current':
                    # if predicate.name in ['pos','neg','zero']:
                        # constant = self.__language.constants[:-3]
                    constant = self.__language.constants[-1:]
                else:
                    constant = self.__language.constants[-3:]
            else:   
                constant = self.__language.constants
            # # constant = self.__language.constants
            constants = [constant for _ in range(predicate.arity)]
            grounds = self.generate_body_atoms(predicate, *constants)
            start = len(self.all_grounds)
            self.all_grounds += grounds
            end = len(self.all_grounds)
            self.__predicate_mapping[predicate] = list(range(start, end))
        # for i, atom in enumerate(self.all_grounds):
        #     print("in axioms atom: ", i, atom)
    @staticmethod
    def prune(clauses):
        pruned = []
        def not_unsafe(clause):
            head_variables = set(clause.head.terms)
            body_variables = set(clause.body[0].terms+clause.body[1].terms)
            return head_variables.issubset(body_variables)

        def not_circular(clause):
            return clause.head not in clause.body

        def not_duplicated(clause):
            for pruned_caluse in pruned:
                if tuple(reversed(pruned_caluse.body)) == clause.body:
                    return False
                if str(clause)==str(pruned_caluse):
                    return False
            return True

        def not_recursive(clause):
            for body_atom in clause.body:
                if body_atom.predicate == clause.head.predicate:
                    return False
            return True

        def follow_order(clause):
            symbols = OrderedSet()
            for atom in clause.atoms:
                for term in atom.terms:
                    symbols.add(term)
            max_v = 0
            for term in symbols:
                if isinstance(term, int):
                    if term>=max_v:
                        max_v = term
                    else:
                        return False
            return True

        def no_insertion(clause):
            symbols = OrderedSet()
            for atom in clause.atoms:
                for term in atom.terms:
                    symbols.add(term)
            symbols = list(symbols)
            if len(symbols) == max(symbols) - min(symbols)+1:
                return True
            else:
                return False


        for clause in clauses:
            if follow_order(clause) and not_unsafe(clause) and no_insertion(clause) \
                    and not_circular(clause) and not_duplicated(clause) and not_recursive(clause):
                pruned.append(clause)
        return pruned


    @staticmethod
    def generate_body_atoms(predicate, *variables):
        '''
        :param predict_candidate: string, candiate of predicate
        :param variables: iterable of tuples of integers, candidates of variables at each position
        :return: tuple of atoms
        '''
        result_tuples = product((predicate,), *variables)
        atoms = [Atom(result[0], result[1:]) for result in result_tuples]
        return atoms

# from https://stackoverflow.com/questions/27890052
def find_shape(seq):
    try:
        len_ = len(seq)
    except TypeError:
        return ()
    shapes = [find_shape(subseq) for subseq in seq]
    return (len_,) + tuple(max(sizes) for sizes in izip_longest(*shapes,
                                                                fillvalue=1))

def fill_array(arr, seq):
    if arr.ndim == 1:
        try:
            len_ = len(seq)
        except TypeError:
            len_ = 0
        arr[:len_] = seq
        arr[len_:] = 0
    else:
        for subarr, subseq in izip_longest(arr, seq, fillvalue=()):
            fill_array(subarr, subseq)

import collections

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)