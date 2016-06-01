#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
from joint_features import extract_features
from metrics import *
from conllz import read_conllz_for_joint, SurfaceToken

from sklearn.externals import joblib

description = """Use jdp.py to launch joint parsing.
This file should not be run independently."""

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

def load_model(clf_path, vec_path):
    """
    Load a pre-trained model instead of training a new one.
    Model and vectorizer files are saved during training.
    :param clf_path: path to .pkl model file
    :param vec_path: path to vectorizer file
    """
    clf = joblib.load(clf_path)
    vec = joblib.load(vec_path)

    def guide(c, feats=None):
        vector = vec.transform(extract_features(c, feats))
        try:
            transition, label = clf.predict(vector)[0].split('_')
        except ValueError:
            transition = clf.predict(vector)[0].split('_')[0]
            label = '_'
        return Transition(transition, label)

    return guide


def load_tagging_model(clf_path, vec_path):
    """
    Load a pre-trained morphological tagging model.
    Do not use load_model for tagging models; they
    return different guide functions.
    """
    clf = joblib.load(clf_path)
    vec = joblib.load(vec_path)

    def guide(c, feats):
        """
        Given a Configuration and a set of training features, disambiguate buffer
        front of this configuration and return a list of tokens that make up the
        best analysis.
        Assume buffer front is not empty and is in (SurfaceToken, [analyses]) format.
        """
        vector = vec.transform(extract_features(c, feats))
        predicted_tags = sorted([i for i in zip(clf.best_estimator_.predict_proba(vector), clf.best_estimator_.classes_)], reverse=True)

        # get a list of tags allowed for this configuration
        possible_tags = get_morph_label(c).split('$')
        index_of_best_tag = 0  # nothing found case

        for tag in predicted_tags:
            if tag in possible_tags:
                index_of_best_tag = possible_tags.index(tag)
                break

        # return the analysis corresponding to the best tagset
        analyses = c.sentence[c.buffer[0]][1]
        return analyses[index_of_best_tag]

    return guide


def joint_parse(s, parsing_guide, parsing_feats, tagging_guide, tagging_feats):
    """
    Jointly parse a sentence.
    :param parsing_guide: a dependency parsing guide function obtained during training
    :param parsing_feats: a list of features for dependency parsing
    :param tagging_guide: a morphological tagging guide function
    :param tagging_feats: a list of features for morphological tagging
    """
    c = initialize_configuration(s)
    while c.buffer:
        c = disambiguate_buffer_front(c, tagging_guide, tagging_feats)
        tr = parsing_guide(c, parsing_feats)
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
# Helper functions


# Sentence -> Configuration
def initialize_configuration(s):
    """Initialize a configuration with root in stack and all other tokens in buffer."""
    try:
        return Configuration([0], [t.id for t in s[1:]], s, set())
    except AttributeError:
        c = initialize_configuration_joint(s)
        return c


def test_initialize_configuration():
    assert initialize_configuration(S_1) == C_1


def initialize_configuration_joint(s):
    return Configuration([0], [i for i in range(1, len(s))], s, set())


# Sentence -> (setof Arc)
def get_arcs(s):
    """Return arcs from a gold standard sentence s."""
    try:
        return {Arc(t.head, t.deprel, t.id) for t in s[1:]}
    except AttributeError:
        arcs = set([])

        # add the rest of the arcs
        for item in s[1:]:
            try:  # case item is a Token
                arcs.add(Arc(item.head, item.deprel, item.id))
            except AttributeError:  # item is an ambiguous token, add all arcs
                for t in item[1][0]:
                    arcs.add(Arc(t.head, t.deprel, t.id))
        return arcs


def test_get_arcs():
    assert get_arcs(S_1) == {Arc(2, 'subj', 1), Arc(0, 'root', 2), Arc(4, 'nmod', 3), Arc(2, 'obj', 4)}

# - - - - - - - - -
# Morphological disambiguator and helper functions
def disambiguate_buffer_front(c, guide=None, feats=None):
    """
    Given a configuration, check if buffer front needs disambiguation and perform it.
    :param guide: a disambiguating guide function obtained at training
    """
    if not isinstance(c.sentence[c.buffer[0]][1], list):
        return c  # was already disambiguated, do nothing

    if not guide:
        guide = morph_oracle

    best_analysis = guide(c, feats)
    last_id = get_span_id(c.sentence[c.buffer[0]])
    diff = last_id - best_analysis[-1].id  # difference between the old and the new number of tokens

    # make new sentence and buffer to match
    new_sentence = expand_sentence(c.sentence, best_analysis, diff)
    new_buffer = expand_buffer(c.buffer, new_sentence, best_analysis)

    return Configuration(c.stack, new_buffer, new_sentence, c.arcs)


def expand_sentence(s, analyses, diff=0):
    """
    Replace surface token in the sentence with disambiguated tokens.
    Assume the tokens before it have already been disambiguated.
    :param diff: a number indicating by how much the ids should shift
    in the resulting sentence. Happens if not the whole range is selected,
    e.g. 1-3 -> 1 (diff=2)
    """
    i = analyses[0].id  # find index of token to be removed
    if diff:
        sentence = enumerate_tokens(s[i+1:], diff)
        return s[:i] + analyses + sentence
    return s[:i] + analyses + s[i+1:]


def expand_buffer(b, s, analyses):
    """
    Update buffer to match the ids of the newly disambiguated tokens
    :param b: buffer
    :param s: sentence
    :param analyses: best analysis selected in disambiguation (may contain several tokens)
    """
    idx = analyses[0].id
    return [i for i in range(idx, len(s))]


def get_span_id(b0):
    """
    Return the last id of the span, or the only id of a simple token
    """
    try:
        last_id = int(b0[0].id.split('-')[1])
    except IndexError:
        last_id = int(b0[0].id)
    except AttributeError:
        last_id = b0.id

    return last_id


def morph_oracle(c, feats=None):
    """
    Returns the first analysis of buffer front as a list of tokens.
    In case of the training corpus, this is the correct analysis.
    """
    return c.sentence[c.buffer[0]][1][0]


def enumerate_tokens(s, d):
    """
    In case a range token was interpreted as only part of the range,
    e.g. 1-2 -> 1, all token ids should shift
    :param s: part of sentence that has to shift
    :param d: difference by which to shift
    """
    new_sentence = []
    for item in s:
        try:  # move a disambiguated token
            i = item.id
            new_sentence.append(Token(
                i-d,
                item[1],
                item[2],
                item[3],
                item[4],
                item[5],
                item[6],
                item[7],
                item[8],
                item[9],
            ))
        except AttributeError:  # move an ambiguous token
            surface_token = item[0]
            try:
                surface_id = int(surface_token.id)-d
            except ValueError:
                surface_id = '-'.join([str(int(j)-d) for j in surface_token.id.split('-')])
            new_surface_token = SurfaceToken(str(surface_id), surface_token.form)
            analyses = []
            for analysis in item[1]:
                new_analysis = []
                for token in analysis:
                    i = token.id
                    new_analysis.append(Token(
                        i-d,
                        token[1],
                        token[2],
                        token[3],
                        token[4],
                        token[5],
                        token[6],
                        token[7],
                        token[8],
                        token[9],
                    ))
                analyses.append(new_analysis)
            new_sentence.append((new_surface_token, analyses))
    return new_sentence


def get_morph_label(c):
    """
    Given a configuration c, return morphological label of buffer front
    Label is a concatenation of POS tag with all morphological information.
    In case the buffer front is ambiguous, return the possible analyses joined through $.
    On training, it should always return something like n||fem|sg or n||fem|sg&v||1sg,
    first case for one token, second for multiple.

    """
    try:
        return c.sentence[c.buffer[0]].postag + '||' + c.sentence[c.buffer[0]].feats
    except AttributeError:
        analyses = c.sentence[c.buffer[0]][1]
        labels = []
        for analysis in analyses:
            labels.append('&'.join([token.postag + '||' + token.feats for token in analysis]))
        return '$'.join(labels)


# - - - - - - - - -
# Input/output


# Configuration -> Sentence
def c2s(c):
    """Return the parsed sentence out of the (final) configuration.
    TODO ASSUME: - tree represented by c.arcs is a valid tree (each token
                   in c.sentence was assigned a head)
    """

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
            pred_head = 0
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


if __name__ == '__main__':
    print(description)