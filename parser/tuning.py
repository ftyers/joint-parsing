"""
Contains functions for tuning parser performance through
different classifiers and feature sets
"""
from sdp import *
from metrics import las
import numpy
import json
from scripts.morph_precision import compare
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def train_parser():
    train = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki.conllx_train'
    # train = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki_ambig_train'
    dev = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki.conllx_dev'
    # dev = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki_ambig_dev'
    test = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki.conllx_test_reduced'
    gold = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki.conllx_test'
    feature_set = json.load(open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/feature_config_english'))

    classifiers = [
        SGDClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier()
    ]

    params = [
        [{'loss': ['hinge'], 'shuffle': [True],
         'learning_rate': ['constant'], 'eta0': [2**(-8)], 'average': [True, False],
         'penalty': ['l1', 'l2', 'elasticnet'],
         'alpha': [0.001, 0.0001, 0.00001, 0.000001]
        }],
        [{
            'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
            'class_weight': ['balanced', None]
        }],
        [{
            'weights': ['uniform', 'distance'], 'leaf_size': [5, 10, 30, 50],

        }]
    ]

    # output_file = open('/Users/Sereni/PycharmProjects/Joint Parsing/tuning_corpus', 'w')
    for clf, param in zip(classifiers, params):
        for set_name, feats in feature_set:

            # train the classifier
            guide = train_with_classifier(train, dev, clf, param, feats)

            # parse the test corpus and calculate LAS
            gold_sentences = read_sentences(gold)
            lasses = []

            for s in read_sentences(test):
                final_config = parse_with_feats(s, guide, feats)

                parsed_sentence = c2s(final_config)
                gold_sentence = next(gold_sentences)
                lasses.append(las(parsed_sentence, gold_sentence[1:]))

                # output_file.write(s2conll(c2s(final_config)) + '\n')

            print('Feature set: %s' % set_name)
            print('LAS: %.3f' % float(sum(lasses)/len(lasses)))
            print()

    # output_file.close()


def train_tagger():
    train = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki_ambig_train'
    dev = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki_ambig_dev'
    feature_set = json.load(open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/feature_config_full'))

    tuvan = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/tuvan/puupankki.tyv.x-best.conllz'
    crimean = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/crimean/puupankki.crh.x-best.conllz'

    classifiers = [
        SGDClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier()
    ]

    params = [
        [{'loss': ['hinge'], 'shuffle': [True],
         'learning_rate': ['constant'], 'eta0': [2**(-8)], 'average': [True, False],
         'penalty': ['l1', 'l2', 'elasticnet'],
         'alpha': [0.001, 0.0001, 0.00001, 0.000001]
        }],
        [{
            'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
            'class_weight': ['balanced', None]
        }],
        [{
            'weights': ['uniform', 'distance'], 'leaf_size': [5, 10, 30, 50],

        }]
    ]

    # output_file = open('/Users/Sereni/PycharmProjects/Joint Parsing/tuning_corpus', 'w')
    for clf, param in zip(classifiers, params):
        for set_name, feats in feature_set:

            print('Feature set: %s' % set_name, end='')
            print('\t', end='')

            # train the classifier
            guide = train_morph_classifier(train, dev, clf, param, feats)

if __name__ == '__main__':

    train_tagger()