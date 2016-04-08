#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import classifiers
from collections import namedtuple
import sys
import os
import copy
import numpy
from features import extract_features_eng

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn import grid_search

description = """(ArcStandard Transition-based) Statistical Dependency Parser.

INPUT: a file with to-be-parsed sentences in CoNLL06 format.
OUTPUT: a file with parsed sentences in CoNLL06 format.

Replaces 'head' and 'deprel' fields of input tokens with
predicted head and label when writing them to output file.

ASSUMES: 'id' fields of tokens in a sentence start counting
         from 1 and are convertible to integers.

USAGE: python3 sdp.py <training_file> <development_file> <input_file> <output_file>
<path_to_model> <path_to_vectorizer>
(also see acceptance-tests/T_* folders for different modes)
UNIT TESTS: py.test sdp.py (py.test has to be installed)"""

# fixme changed signature silently, all tests will fail

# =================
# Constants:


# =================
# Data definitions:


Token = namedtuple('Token', ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats',
                             'head', 'deprel', 'phead', 'pdeprel'])
# Token is Token(Integer, String, String, String, String,
#                String, Integer, String, Integer, String)
# interp. one word of a sentence with information in CoNLL06 format

ROOT = Token(0, 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 0, 'ROOT', 0, 'ROOT')  # default for root node

T_1 = Token(1, 'John', 'john', 'NNP', '_', '_', 2, 'subj', 2, 'subj')  # gold standard with an 'alternative'
# projective arc (=head&label)

T_2 = Token(1, 'John', 'john', 'NNP', '_', '_', 2, 'subj', '_', '_')  # gold standard without an alternative
# projective arc (=head&label)

T_3 = Token(1, 'John', 'john', 'NNP', '_', '_', '_', '_', '_', '_')  # no gold standard arc given

T_4 = Token(1, 'John', '_', '_', '_', '_', '_', '_', '_', '_')  # bare minimum of information for
# a sentence to be parsable given

"""
def fn_for_token(t):
    ... t.id         # Integer
        t.form       # String
        t.lemma      # String
        t.cpostag    # String
        t.postag     # String
        t.feats      # String
        t.head       # Integer
        t.deprel     # String
        t.phead      # Integer
        t.pdeprel    # String
"""
# Template rules:
#  - compound: 10 fields


# Sentence is (listof Token) of arbitrary size
# interp. a sentence to be parsed/used for training

S_0 = []  # base case

S_1 = [ROOT,  # projective
       Token(1, 'John', 'john', 'NNP', '_', '_', 2, 'subj', '_', '_', ),
       Token(2, 'sees', 'see', 'VBZ', '_', '_', 0, 'root', '_', '_', ),
       Token(3, 'a', 'a', 'DT', '_', '_', 4, 'nmod', '_', '_'),
       Token(4, 'dog', 'dog', 'NN', '_', '_', 2, 'obj', '_', '_')]

S_2 = [ROOT,  # non-projective
       Token(1, 'It', 'it', '_', '_', '_', 2, '_', '_', '_'),
       Token(2, 'is', 'is', '_', '_', '_', 0, '_', '_', '_'),
       Token(3, 'what', 'what', '_', '_', '_', 9, '_', '_', '_'),
       Token(4, 'federal', 'federal', '_', '_', '_', 5, '_', '_', '_'),
       Token(5, 'support', 'support', '_', '_', '_', 6, '_', '_', '_'),
       Token(6, 'should', 'should', '_', '_', '_', 2, '_', '_', '_'),
       Token(7, 'try', 'try', '_', '_', '_', 6, '_', '_', '_'),
       Token(8, 'to', 'to', '_', '_', '_', 7, '_', '_', '_'),
       Token(9, 'achieve', 'achieve', '_', '_', '_', 8, '_', '_', '_')]

"""
def fn_for_sentence(s):
    if not s:
        ...
    else:
        for token in sentence:
            fn_for_token(token)
"""
# Template rules:
#  - one of: 2 cases
#  - atomic distinct: empty
#  - compound: (cons Token Sentence)
#  - reference: (first s) is Token
#  - self-reference: (rest s) is Sentence


Arc = namedtuple('Arc', ['h', 'l', 'd'])
# Arc is Arc(Integer, String, Integer)
# interp. a dependency arc from the token with id h to the token with id d, labeled as l

A_1 = Arc(2, 'subj', 1)  # labeled
A_2 = Arc(2, '_', 1)  # unlabeled

"""
def fn_for_arc(a):
    ... a.h  # Integer
        a.l  # String
        a.d  # Integer
"""
# Template rules:
#  - compound: 3 fields


Configuration = namedtuple('Configuration', ['stack', 'buffer', 'sentence', 'arcs'])
# Configuration is Configuration(Stack, Buffer, Sentence, (setof Arc))
# interp. a state in the parsing process representing (id's of) partially processed tokens,
#         (id's of) input tokens, the sentence being parsed and the set of created arcs

C_1 = Configuration([0], [1, 2, 3, 4], S_1, set())  # start configuration

C_2 = Configuration([0], [], S_1, {Arc(2, 'subj', 1),  # terminal configuration
                                   Arc(0, 'root', 2),
                                   Arc(4, 'nmod', 3),
                                   Arc(2, 'obj', 4)})

"""
def fn_for_configuration(c):
    ... c.stack      # Stack
        c.buffer     # Buffer
        c.sentence   # Sentence
        c.arcs       # (setof Arc)
"""
# Template rules:
#  - compound: 4 fields
#  - reference: c.stack is Stack
#  - reference: c.buffer is Buffer
#  - reference: c.sentence is Sentence
#  - reference: c.arcs is (setof Arc)


# Stack is (listof Integer)
# interp. a LIFO queue with id's of partially processed tokens

ST_1 = [0]  # stack in the start and, optimally, in the end configuration
ST_2 = [0, 2, 4]  # stack in an intermediate configuration; 'top of stack' is 4

# Buffer is (listof Integer)
# interp. a FIFO queue with id's of input tokens

B_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # buffer in the start configuration; 'front of buffer' is 1
B_2 = [2, 6, 7, 8, 9]  # buffer in an intermediate configuration
B_3 = []  # buffer in the end configuration

Transition = namedtuple('Transition', ['op', 'l'])
# Transition is Transition(String, String)
# interp. transition (operation and label) from one configuration to the next

TR_1 = Transition('sh', '_')  # shift
TR_2 = Transition('la', '_')  # unlabeled left arc transition
TR_3 = Transition('la', 'subj')  # labeled left arc transition
TR_4 = Transition('ra', '_')  # unlabeled right arc transition
TR_5 = Transition('ra', 'nmod')  # labeled right arc transition
"""
def fn_for_transition(tr):
    if tr.op == 'sh':
        ...
    elif tr.op == 'la':
        ... tr.op
            tr.l
    elif tr.op == 'ra':
        ... tr.op
            tr.l
"""


# Template rules:
#  - compound: 2 fields
#  - tr.op is one of 3 cases:
#     - atomic distinct: 'sh'
#     - atomic distinct: 'la'
#     - atomic distinct: 'rg'
#  - tr.l is one of 2 cases:
#     - atomic distinct: '_'
#     - atomic non-distinct: String


# =================
# Functions:


# Sentence Function -> Configuration
def parse(s, oracle_or_guide):
    """Given a sentence and a next transition predictor, parse the sentence."""
    c = initialize_configuration(s)
    while c.buffer:
        tr = oracle_or_guide(c)
        if tr.op == 'sh':
            c = shift(c)
        elif tr.op == 'la':
            try:
                c = left_arc(c, tr.l)
            except IndexError:
                c = shift(c)
        elif tr.op == 'ra':
            try:
                c = right_arc(c, tr.l)
            except IndexError:
                c = shift(c)
    return c


def test_parse():
    assert parse(S_1, oracle) == Configuration([0], [], S_1, {Arc(2, 'subj', 1),
                                                              Arc(0, 'root', 2),
                                                              Arc(4, 'nmod', 3),
                                                              Arc(2, 'obj', 4)})


# Filename -> (generator (tupleof Sentence, (listof (tupleof FeatureVector, String))))
def generate_training_data(train_conll, as_dict=False):
    """Generate sentence and a list of (feature vector, expected label) tuples (representing
    configuration and correct transition operation) out of the training sentences.
    """
    for s in read_sentences(train_conll):
        c = initialize_configuration(s)
        fvecs_and_labels = []
        while c.buffer:
            tr = oracle(c)
            fvecs_and_labels.append((extract_features_eng(c, as_dict), tr.op))
            if tr.op == 'sh':
                c = shift(c)
            elif tr.op == 'la':
                c = left_arc(c, tr.l)
            elif tr.op == 'ra':
                c = right_arc(c, tr.l)
        yield (s, fvecs_and_labels)


def test_generate_training_data(tmpdir):
    f = tmpdir.join('f.conll06')
    f.write('1\tJohn\tjohn\tNNP\t_\t_\t2\tsubj\t_\t_\n'
            '2\tsees\tsee\tVBZ\t_\t_\t0\troot\t_\t_\n'
            '3\ta\ta\tDT\t_\t_\t4\tnmod\t_\t_\n'
            '4\tdog\tdog\tNN\t_\t_\t2\tobj\t_\t_\n'
            ' \n ')
    assert list(generate_training_data(str(f))) == [(S_1,
                                                     [(extract_features_eng(Configuration([0], [1, 2, 3, 4], S_1,
                                                                                          set())),
                                                       "sh"),
                                                      (extract_features_eng(Configuration([0, 1], [2, 3, 4], S_1,
                                                                                          set())),
                                                       "la"),
                                                      (extract_features_eng(Configuration([0], [2, 3, 4], S_1,
                                                                                          {Arc(2, '_', 1)})),
                                                       "sh"),
                                                      (extract_features_eng(Configuration([0, 2], [3, 4], S_1,
                                                                                          {Arc(2, '_', 1)})),
                                                       "sh"),
                                                      (extract_features_eng(Configuration([0, 2, 3], [4], S_1,
                                                                                          {Arc(2, '_', 1)})),
                                                       "la"),
                                                      (extract_features_eng(Configuration([0, 2], [4], S_1,
                                                                                          {Arc(2, '_', 1),
                                                                                           Arc(4, '_', 3)})),
                                                       "ra"),
                                                      (extract_features_eng(Configuration([0], [2], S_1,
                                                                                          {Arc(2, '_', 1),
                                                                                           Arc(4, '_', 3),
                                                                                           Arc(2, '_', 4)})),
                                                       "ra"),
                                                      (extract_features_eng(Configuration([], [0], S_1,
                                                                                          {Arc(2, '_', 1),
                                                                                           Arc(4, '_', 3),
                                                                                           Arc(2, '_', 4),
                                                                                           Arc(0, '_', 2)})),
                                                       "sh")])]


# Filename -> Function
def train_internal_classifier(train_conll):
    """Train a classifier on gold standard sentences and return a guide function
    which predicts transitions for given configurations using that classifier.
    """
    training_collection = []
    for s, fvecs_labels in generate_training_data(train_conll):
        training_collection.extend(fvecs_labels)
    dev_sents = list(read_sentences('data/en-ud-dev.conllu'))  # fixme but why hardcode?

    classifier = classifiers.MulticlassPerceptron(['sh', 'ra', 'la'])
    best_classifier = copy.deepcopy(classifier)
    best_uas = 0.0
    uas_after = 0.0
    iter = 0

    while iter < 15:  # fixme hardcoded number of iterations
        uas_before = uas_after
        classifier.train_one_iteration(training_collection)
        uas_after = micro_uas(
            [(c2s(parse(s, lambda c: Transition(classifier.classify(extract_features_eng(c)), '_'))),
              s[1:]) for s in dev_sents])
        print('Iteration : ', iter)
        print('    UAS on dev before: ', uas_before)
        print('    UAS on dev after:  ', uas_after)
        iter += 1
        if uas_after > best_uas:
            best_classifier = copy.deepcopy(classifier)
            best_uas = uas_after

    # Configuration -> Transition
    def guide(c):
        return Transition(best_classifier.classify(extract_features_eng(c)), '_')

    return guide


def train(training_path, development_path):
    """Train a classifier on gold standard sentences and return a guide function
    which predicts transitions for given configurations using that classifier.
    :param training_path: path to training file in CONLL06 format
    :param development_path: path to development file in CONLL06 format
    """
    training_collection = []    # a list of dicts containing features
    labels = []                 # a list of target transition labels
    development_collection = [] # do I need features for these? um, yes.
    dev_labels = []

    for s, fvecs_labels in generate_training_data(training_path, as_dict=True):
        for item in fvecs_labels:
            training_collection.append(item[0])
            labels.append(item[-1])  # todo test this, not sure if correct

    for s, fvecs_labels in generate_training_data(development_path, as_dict=True):

        for item in fvecs_labels:
            development_collection.append(item[0])
            dev_labels.append(item[-1])

    # transform string features via one-hot encoding
    vec = DictVectorizer()
    data = vec.fit_transform(training_collection)
    target = numpy.array(labels)
    data_test = vec.transform(development_collection)
    target_test = numpy.array(dev_labels)

    # set parameters for grid search
    tuned_parameters = [{'loss': ['hinge', 'log'], 'shuffle': [True],
                         'learning_rate': ['constant'], 'eta0': [2**(-8)], 'average': [True, False],
                         'penalty': ['l1', 'l2', 'elasticnet'],
                         'alpha': [0.001, 0.0001, 0.00001, 0.000001]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = grid_search.GridSearchCV(SGDClassifier(), tuned_parameters, cv=5,
                           scoring='%s_weighted' % score, verbose=2)
        clf.fit(data, target)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = target_test, clf.predict(data_test)
        print(classification_report(y_true, y_pred))
        best = clf.best_estimator_
        print(clf.best_score_)

        joblib.dump(best, 'best_model_for_%s.pkl' % (os.path.basename(training_path)))
        print()

    joblib.dump(vec, 'vectorizer_for_%s.pkl' % (os.path.basename(training_path)))

    # Configuration -> Transition
    def guide(c):
        vector = vec.transform(extract_features_eng(c, as_dict=True))
        return Transition(best.predict(vector), '_')  # fixme this should assign a dependency label
    return guide


def load_model(clf_path, vec_path):
    """
    Load a pre-trained model instead of training a new one.
    Model and vectorizer files are saved during training.
    :param clf_path: path to .pkl model file
    :param vec_path: path to vectorizer file
    """
    clf = joblib.load(clf_path)
    vec = joblib.load(vec_path)

    def guide(c):
        vector = vec.transform(extract_features_eng(c, as_dict=True))
        return Transition(clf.predict(vector), '_')

    return guide

# -----------------
# Evaluation


# (listof (Sentence, Sentence)) -> Float
def macro_uas(pred_gold):
    """Given a list of predicted and gold standard sentences, return macro-averaged unlabeled attachment score."""
    return sum(
        [nbr_of_tokens_with_correct_head(s_pred, s_gold) / len(s_pred) for s_pred, s_gold in pred_gold]) / \
           len(sentences)


# (listof (Sentence, Sentence)) -> Float
def micro_uas(pred_gold):
    """Given a list of predicted and gold standard sentences, return micro-averaged unlabeled attachment score."""
    all_tokens = 0
    correct = 0
    for s_pred, s_gold in pred_gold:
        all_tokens += len(s_pred)
        correct += nbr_of_tokens_with_correct_head(s_pred, s_gold)
    return correct / all_tokens


# Configuration -> Float
def uas(c):
    """Return Unlabeled Attachment Score for the predicted parse tree
    represented by the given final configuration."""

    return nbr_of_tokens_with_correct_head(c2s(c), c.sentence[1:]) / len(c.sentence[1:])


def test_uas():
    assert abs(uas(Configuration([0, 1, 2, 3, 4, 5, 6, 7], [],
                                 [ROOT,
                                  Token(1, 'John', '_', '_', '_', '_', 2, 'subj', '_', '_'),
                                  Token(2, 'meets', '_', '_', '_', '_', 0, 'root', '_', '_', ),
                                  Token(3, 'Mary', '_', '_', '_', '_', 2, 'obj', '_', '_'),
                                  Token(4, 'at', '_', '_', '_', '_', 2, 'adv', '_', '_'),
                                  Token(5, 'the', '_', '_', '_', '_', 7, 'nmod', '_', '_'),
                                  Token(6, 'bus', '_', '_', '_', '_', 7, 'nmod', '_', '_'),
                                  Token(7, 'station', '_', '_', '_', '_', 4, 'pmod', '_', '_')],
                                 {Arc(2, 'obj', 1), Arc(0, 'root', 2), Arc(2, 'subj', 3),
                                  Arc(3, 'adv', 4), Arc(6, 'nmod', 5), Arc(4, 'pmod', 7), }))) - 5 / 7 < 0.0001


# Sentence Sentence -> Integer
def nbr_of_tokens_with_correct_head(s_pred, s_gold):
    """Given a sentence with predicted arcs/labels and the same sentence with gold standard arcs/sentence,
    return the number of correct attachments."""
    return len([t_pred for t_pred, t_gold in zip(s_pred, s_gold) if t_pred.head == t_gold.head])


# -----------------
# Oracle


# Configuration -> Transition
def oracle(c):
    """Given a configuration with gold standard sentence in it, return the correct transition.
    ASSUME: - buffer is not empty
    """
    correct_arcs = get_arcs(c.sentence)
    if can_left_arc(c, correct_arcs):
        return Transition('la', c.sentence[c.stack[-1]].deprel)
    elif can_right_arc(c, correct_arcs):
        return Transition('ra', c.sentence[c.buffer[0]].deprel)
    else:
        return Transition('sh', '_')


def test_oracle():
    def get_all_correct_transitions(start_configuration):
        transitions = []
        c = start_configuration
        while c.buffer:
            tr = oracle(c)
            if tr.op == 'sh':
                c = shift(c)
            elif tr.op == 'la':
                c = left_arc(c, tr.l)
            elif tr.op == 'ra':
                c = right_arc(c, tr.l)
            transitions.append(tr)
        return transitions

    s = [ROOT,
         Token(1, 'In', '_', '_', '_', '_', 0, '_', '_', '_', ),
         Token(2, 'France', '_', '_', '_', '_', 1, '_', '_', '_', ),
         Token(3, '?', '_', '_', '_', '_', 1, '_', '_', '_'),
         Token(4, '?', '_', '_', '_', '_', 1, '_', '_', '_'),
         Token(5, '!', '_', '_', '_', '_', 1, '_', '_', '_'),
         Token(6, '!', '_', '_', '_', '_', 1, '_', '_', '_')]
    assert get_all_correct_transitions(initialize_configuration(s)) == \
           [('sh', '_'), ('ra', '_'), ('sh', '_'), ('ra', '_'), ('sh', '_'), ('ra', '_'),
            ('sh', '_'), ('ra', '_'), ('sh', '_'), ('ra', '_'), ('ra', '_'), ('sh', '_')]

    s = [ROOT,
         Token(1, 'Is', '_', '_', '_', '_', 0, '_', '_', '_', ),
         Token(2, 'this', '_', '_', '_', '_', 1, '_', '_', '_', ),
         Token(3, 'the', '_', '_', '_', '_', 4, '_', '_', '_'),
         Token(4, 'future', '_', '_', '_', '_', 1, '_', '_', '_'),
         Token(5, 'of', '_', '_', '_', '_', 4, '_', '_', '_'),
         Token(6, 'chamber', '_', '_', '_', '_', 7, '_', '_', '_'),
         Token(7, 'music', '_', '_', '_', '_', 5, '_', '_', '_'),
         Token(8, '?', '_', '_', '_', '_', 1, '_', '_', '_')]
    assert get_all_correct_transitions(initialize_configuration(s)) == \
           [('sh', '_'), ('ra', '_'), ('sh', '_'), ('sh', '_'), ('la', '_'), ('sh', '_'),
            ('sh', '_'), ('sh', '_'), ('la', '_'), ('ra', '_'), ('ra', '_'), ('ra', '_'),
            ('sh', '_'), ('ra', '_'), ('ra', '_'), ('sh', '_')]


# Configuration (setof Arc) -> Boolean
def can_left_arc(c, correct_arcs):
    """Return True if given configuration allows left_arc transition.
    ASSUME: - correct arcs are unlabeled
    """
    try:
        return Arc(c.buffer[0], c.sentence[c.stack[-1]].deprel, c.stack[-1]) in correct_arcs
    except IndexError:
        return False


def test_can_left_arc():
    assert can_left_arc(Configuration([0, 1], [2, 3, 4], S_1, set()),
                        {Arc(0, 'root', 2), Arc(2, 'subj', 1)}) == True
    assert can_left_arc(Configuration([0, 1], [2, 3, 4], S_1, set()),
                        {Arc(4, 'nmod', 3), Arc(0, 'root', 2)}) == False


# Configuration (setof Arc) -> Boolean
def can_right_arc(c, correct_arcs):
    """Return True if given configuration allows right_arc transition."""
    try:
        return Arc(c.stack[-1], c.sentence[c.buffer[0]].deprel, c.buffer[0]) in correct_arcs \
               and has_all_children(c.buffer[0], c, correct_arcs)
    except IndexError:
        return False


def test_can_right_arc():
    assert can_right_arc(Configuration([0], [2, 3, 4], S_1, set()),
                         {Arc(2, 'subj', 1), Arc(2, 'obj', 4), Arc(4, 'nmod', 3)}) == False
    assert can_right_arc(Configuration([0], [2, 3, 4], S_1, set()),
                         {Arc(0, 'root', 2), Arc(2, 'subj', 1), Arc(2, 'obj', 4), Arc(4, 'nmod', 3)}) == False
    assert can_right_arc(Configuration([0], [2, 3, 4], S_1, {Arc(2, 'subj', 1)}),
                         {Arc(0, 'root', 2), Arc(2, 'subj', 1), Arc(2, 'obj', 4), Arc(4, 'nmod', 3)}) == False
    assert can_right_arc(Configuration([0], [2, 3, 4], S_1, {Arc(2, 'subj', 1), Arc(2, 'obj', 4)}),
                         {Arc(0, 'root', 2), Arc(2, 'subj', 1), Arc(2, 'obj', 4), Arc(4, 'nmod', 3)}) == True


# Integer Configuration (setof Arc) -> Boolean
def has_all_children(t_id, c, correct_arcs):
    """Produce True if in the configuration all children of the token with id 't_id' were collected."""
    return {arc for arc in correct_arcs if arc.h == t_id} <= c.arcs


def test_has_all_children():
    # token is not a head of anything
    assert has_all_children(3, Configuration([0], [], S_1, set()), get_arcs(S_1)) == True
    # token has two children, one was found, one was not
    assert has_all_children(2, Configuration([0], [], S_1, {Arc(2, 'subj', 1)}), get_arcs(S_1)) == False
    # token has 2 children, both were found
    assert has_all_children(2, Configuration([0], [], S_1, {Arc(2, 'subj', 1), Arc(2, 'obj', 4)}),
                            get_arcs(S_1)) == True


# -------------------------------------
# Next state (configuration) generators


# Configuration -> Configuration
def shift(c):
    """Take the first token from the front of the buffer and push it onto the stack.
    ASSUME: - buffer is not empty
    """
    return Configuration(c.stack + [c.buffer[0]], c.buffer[1:], c.sentence, c.arcs)


def test_shift():
    assert shift(Configuration([], [0], S_1, set())) == Configuration([0], [], S_1, set())
    assert shift(Configuration([0, 1, 2], [3, 4], S_1, {Arc(0, 'root', 1)})) == \
           Configuration([0, 1, 2, 3], [4], S_1, {Arc(0, 'root', 1)})


# Configuration String -> Configuration
def left_arc(c, l):
    """Introduce an arc with label l from the front of the buffer to the top-most token on the stack
    and remove the top-most token from the stack.
    ASSUME: - stack is not empty
            – top of the stack has no head already
            - top of the stack is not root
    """
    return Configuration(c.stack[:-1], c.buffer, c.sentence, c.arcs | {Arc(c.buffer[0], l, c.stack[-1])})


def test_left_arc():
    assert left_arc(Configuration([0, 1, 2], [3, 4], S_1, {Arc(0, 'root', 1)}),
                    'nmod') == \
           Configuration([0, 1], [3, 4], S_1, {Arc(0, 'root', 1), Arc(3, 'nmod', 2)})


# Configuration String -> Configuration
def right_arc(c, l):
    """Introduce an arc with label l from the top-most token on the stack to the front of the buffer,
    remove the front of the buffer, and move the top-most token from the stack back onto the buffer.
    ASSUME: - stack is not empty
            – front of the buffer has no head already
    """
    return Configuration(c.stack[:-1], [c.stack[-1]] + c.buffer[1:], c.sentence,
                         c.arcs | {Arc(c.stack[-1], l, c.buffer[0])})


def test_right_arc():
    assert right_arc(Configuration([0, 1, 2], [3, 4], S_1, {Arc(0, 'root', 1)}),
                     'nmod') == \
           Configuration([0, 1], [2, 4], S_1, {Arc(0, 'root', 1), Arc(2, 'nmod', 3)})


# ---------------------
# Feature extractors
# moved to features.py
def test_extract_features():
    vector = extract_features_eng(Configuration([0, 1], [3, 4, 5],
                                              [ROOT,
                                               Token(1, 'The', 'the', 'DT', '_', '_', 3, '_', '_', '_'),
                                               Token(2, 'cute', 'cute', 'JJ', '_', '_', 3, 'amod', '_', '_',),
                                               Token(3, 'dog', 'dog', 'NN', '_', '_', 4, '_', '_', '_'),
                                               Token(4, 'likes', 'like', 'VBZ', '_', '_', 0, '_', '_', '_'),
                                               Token(5, 'apples', 'apple', 'NNS', '_', '_', 4, '_', '_', '_')],
                                              {Arc(3, 'amod', 2)}))
    assert vector == \
           ['bias', 'b0.form=dog', 'b0.pos=NN', 's0.form=The', 's0.pos=DT',
            'b1.pos=VBZ', 's1.pos=ROOT', 'ld(b0).pos=JJ',
            's0.pos b0.pos=DT NN', 's0.pos b0.form=DT dog', 's0.form b0.pos=The NN', 's0.form b0.form=The dog',
            's0.lemma=the', 'b0.lemma=dog', 'b1.form=likes', 'b2.pos=NNS', 'b3.pos=None',
            'rd(s0).deprel=None', 'ld(s0).deprel=None', 'rd(b0).deprel=amod', 'ld(b0).deprel=amod']

# ---------------------
# Helper functions


# Sentence -> Configuration
def initialize_configuration(s):
    """Initialize a configuration with root in stack and all other tokens in buffer."""
    return Configuration([0], [t.id for t in s[1:]], s, set())


def test_initialize_configuration():
    assert initialize_configuration(S_1) == C_1


# Sentence -> (setof Arc)
def get_arcs(s):
    """Return arcs from a gold standard sentence s."""
    return {Arc(t.head, t.deprel, t.id) for t in s[1:]}


def test_get_arcs():
    assert get_arcs(S_1) == {Arc(2, 'subj', 1), Arc(0, 'root', 2), Arc(4, 'nmod', 3), Arc(2, 'obj', 4)}


# - - - - - - - - -
# Input/output


# Configuration -> Sentence
def c2s(c):
    """Return the parsed sentence out of the (final) configuration.
    TODO ASSUME: - tree represented by c.arcs is a valid tree (each token
                   in c.sentence was assigned a head)
    """
    # fixme well it doesn't always get a head...

    # (setof Arc) -> (dictionaryof Integer:(tupleof Integer, String)
    def invert_arcs(arcs):
        """Return a dictionary which maps dependents to (head, label) tuples."""
        return {a.d: (a.h, a.l) for a in arcs}

    d2h_l = invert_arcs(c.arcs)

    # Token -> Integer
    def head(t):
        try:
            pred_head = d2h_l[t.id][0]
        except KeyError:
            # pred_head = "_"
            pred_head = 0  # fixme this is a dumb fix to make MaltParser happy in the few cases
                           # make it a real fix
        return pred_head

    # Token -> String
    def label(t):
        try:
            pred_label = d2h_l[t.id][1]
        except KeyError:
            pred_label = "_"
        return pred_label

    return [Token(t.id, t.form, t.lemma, t.cpostag, t.postag, t.feats, head(t), label(t),
                  t.phead, t.pdeprel) for t in c.sentence[1:]]


def test_c2s(): \
        assert c2s(Configuration([0], [], S_1, {Arc(2, 'subj', 1),  # arcs are deliberately wrong so that
                                                Arc(0, 'root', 2),  # we get something different from the
                                                Arc(2, 'obj', 3),  # input sentence
                                                Arc(3, 'nmod', 4)})) == \
               [Token(1, 'John', 'john', 'NNP', '_', '_', 2, 'subj', '_', '_', ),
                Token(2, 'sees', 'see', 'VBZ', '_', '_', 0, 'root', '_', '_', ),
                Token(3, 'a', 'a', 'DT', '_', '_', 2, 'obj', '_', '_'),
                Token(4, 'dog', 'dog', 'NN', '_', '_', 3, 'nmod', '_', '_')]


# Sentence -> String
def s2string(s):
    """Produce a one-line string with forms from s.
    ASSUME: - ROOT has already been removed from the sentence"""
    return ' '.join(t.form for t in s)


def test_s2string():
    assert s2string(S_0[1:]) == ''
    assert s2string(S_1[1:]) == 'John sees a dog'


# Sentence -> String
def s2conll(s):
    """Produce a string representing the sentence in CoNLL06 format.
    ASSUME: - ROOT has already been removed from the sentence"""
    string = ''
    for t in s:
        string += '\t'.join([str(t.id), t.form, t.lemma, t.cpostag, t.postag, t.feats,
                             str(t.head), t.deprel, str(t.phead), t.pdeprel]) + '\n'
    return string


def test_s2conll():
    assert s2conll(S_0[1:]) == ''
    assert s2conll(S_1[1:]) == '1\tJohn\tjohn\tNNP\t_\t_\t2\tsubj\t_\t_\n' \
                               '2\tsees\tsee\tVBZ\t_\t_\t0\troot\t_\t_\n' \
                               '3\ta\ta\tDT\t_\t_\t4\tnmod\t_\t_\n' \
                               '4\tdog\tdog\tNN\t_\t_\t2\tobj\t_\t_\n'


# Filename -> (generator Sentence)
def read_sentences(f):
    """Return Sentences from a file in CoNLL06 format."""
    with open(f, 'r') as conll_file:
        s = [ROOT]
        for line in conll_file:
            if line.strip():
                s.append(read_token(line))
            elif len(s) != 1:
                yield s
                s = [ROOT]
        if len(s) != 1:  # file ended without a new line at the end
            yield s


def test_read_sentences(tmpdir):
    f = tmpdir.join('f.conll06')
    f.write('1\tJohn\tjohn\tNNP\t_\t_\t2\tsubj\t_\t_\n'
            '2\tsees\tsee\tVBZ\t_\t_\t0\troot\t_\t_\n'
            '3\ta\ta\tDT\t_\t_\t4\tnmod\t_\t_\n'
            '4\tdog\tdog\tNN\t_\t_\t2\tobj\t_\t_\n'
            ' \n '
            '1\tJohn\tjohn\tNNP\t_\t_\t2\tsubj\t_\t_\n'
            '2\tsees\tsee\tVBZ\t_\t_\t0\troot\t_\t_\n'
            '3\ta\ta\tDT\t_\t_\t4\tnmod\t_\t_\n'
            '4\tdog\tdog\tNN\t_\t_\t2\tobj\t_\t_\n')
    assert list(read_sentences(str(f))) == [S_1, S_1]
    f2 = tmpdir.join('f2.conll06')
    f2.write('1\tJohn\tjohn\tNNP\t_\t_\t2\tsubj\t_\t_\n'
             '2\tsees\tsee\tVBZ\t_\t_\t0\troot\t_\t_\n'
             '3\ta\ta\tDT\t_\t_\t4\tnmod\t_\t_\n'
             '4\tdog\tdog\tNN\t_\t_\t2\tobj\t_\t_\n'
             ' \n ')
    assert list(read_sentences(str(f2))) == [S_1]


# Strting -> Token
def read_token(line):
    """Parse a line of the file in CoNLL06 format and return a Token."""
    token = line.strip().split('\t')
    if len(token) == 6:
        token += ['_', '_', '_', '_']
    id, form, lemma, cpostag, postag, feats, head, deprel, phead, pdeprel = token
    try:
        head = int(head)
    except ValueError:
        head = '_'
    try:
        phead = int(phead)
    except ValueError:
        phead = '_'
    return Token(int(id), form, lemma, cpostag, postag, feats, head, deprel, phead, pdeprel)


def test_read_token():
    assert read_token('1\tJohn\tjohn\tNNP\t_\t_\t2\tsubj\t2\tsubj\n') == T_1
    assert read_token('1\tJohn\tjohn\tNNP\t_\t_\t2\tsubj\t_\t_\n') == T_2
    assert read_token('1\tJohn\tjohn\tNNP\t_\t_\t_\t_\t_\t_\n') == T_3
    assert read_token('1\tJohn\t_\t_\t_\t_\t_\t_\t_\t_\n') == T_4


# ---------------------
# Runner

class AbsPath(argparse.Action):

    def __call__(self, parser, namespace, path, option_string=None):
        cwd = os.getcwd()
        # cwd = os.path.dirname(os.path.realpath(__file__))  # use for debugging and config launches
        if not os.path.isabs(path):
            path = os.path.join(cwd, path)
        setattr(namespace, self.dest, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-t', '--train', action=AbsPath)
    parser.add_argument('-dev', '--development', action=AbsPath)
    parser.add_argument('input_file', action=AbsPath)
    parser.add_argument('output_file', action=AbsPath)
    parser.add_argument('-m', '--model', action=AbsPath)
    parser.add_argument('-vec', '--vectorizer', action=AbsPath)

    args = parser.parse_args()

    # resolve argument pairs
    if args.train and args.development:
        if args.model and args.vectorizer:
            raise argparse.ArgumentError(args.train,
                                         "Please provide either training and development sets, or a pre-trained model and a vectorizer, but not both.")

    elif args.train:
        raise argparse.ArgumentError(args.development, "Please provide a development set in addition to the training set.")

    elif args.development:
        raise argparse.ArgumentError(args.train, "Please provide a training set in addition to the development set.")

    elif args.model and args.vectorizer:
        pass

    elif args.model:
        raise argparse.ArgumentError(args.vectorizer, "Please provide a vectorizer generated during training.")

    elif args.vectorizer:
        raise argparse.ArgumentError(args.model, "Please provide a training model with the vectorizer.")

    # get guide function
    if args.model and args.vectorizer:
        model_path = args.model
        vec_path = args.vectorizer
        guide_function = load_model(model_path, vec_path)

    else:
        guide_function = train(args.train, args.development)

    # parse input file
    cwd = os.getcwd()
    with open(os.path.join(cwd, args.output_file), 'w') as output_file:
        for s in read_sentences(args.input_file):
            final_config = parse(s, guide_function)
            output_file.write(s2conll(c2s(final_config)) + '\n')