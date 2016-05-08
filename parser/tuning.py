"""
Contains functions for tuning parser performance through
different classifiers and feature sets
"""
from sdp import *
from metrics import las
import numpy
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn import grid_search
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def train_with_classifier(training_path, development_path, classifier, parameters, features):
    """
    Train the model using the classifier and its parameters supplied.
    :param features: a set of features to use in this round of training
    :param training_path: path to training corpus
    :param development_path: path to development corpus
    :param classifier: a sklearn classifier object
    :param parameters: a list of parameters to use for grid search
    """

    training_collection = []
    labels = []
    development_collection = []
    dev_labels = []

    for s, fvecs_labels in generate_training_data(training_path, feature_config=features):
        for item in fvecs_labels:
            training_collection.append(item[0])
            labels.append(item[-1])

    for s, fvecs_labels in generate_training_data(development_path, feature_config=features):

        for item in fvecs_labels:
            development_collection.append(item[0])
            dev_labels.append(item[-1])

    # transform string features via one-hot encoding
    vec = DictVectorizer()
    data = vec.fit_transform(training_collection)
    target = numpy.array(labels)
    data_test = vec.transform(development_collection)
    target_test = numpy.array(dev_labels)

    score = 'precision'  # todo can you tune for LAS instead?

    clf = grid_search.GridSearchCV(classifier, parameters, cv=5, scoring='%s_weighted' % score, verbose=0)
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
    y_true, y_pred = target_test, clf.predict(data_test)
    print(classification_report(y_true, y_pred))
    print(clf.best_score_)

    def guide(c, feats):

        # c = disambiguate_buffer_front(c)
        vector = vec.transform(extract_features(c, config=feats))
        try:
            transition, label = clf.best_estimator_.predict(vector)[0].split('_')
        except ValueError:
            transition = clf.best_estimator_.predict(vector)[0].split('_')[0]
            label = '_'
        return Transition(transition, label)

    return guide


def parse_with_feats(s, oracle_or_guide, feats):
    """Given a sentence and a next transition predictor, parse the sentence."""
    c = initialize_configuration(s)
    while c.buffer:
        tr = oracle_or_guide(c, feats)
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

if __name__ == '__main__':

    train = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki.conllx_train'
    dev = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki.conllx_dev'
    test = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki.conllx_test_reduced'
    gold = '/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki.conllx_test'
    feature_set = json.load(open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/feature_config_more_deprel'))

    # feature_set = [["no_deprel",
    #                 ['b0.form',
    # 'b0.pos',
    # 's0.form',
    # 's0.pos',
    # 'b1.pos',
    # 's1.pos',
    # 'ld(b0).pos',
    # 's0.pos b0.pos',
    # 's0.pos b0.form',
    # 's0.form b0.pos',
    # 's0.form b0.form',
    # 's0.lemma',
    # 'b0.lemma',
    # 'b1.form',
    # 'b2.pos',
    # 'b3.pos',
    # # 'rd(s0).deprel',
    # # 'ld(s0).deprel',
    # # 'rd(b0).deprel',
    # # 'ld(b0).deprel',
    # # 's0.deprel',
    # 's0_head.form']
    #                 ]]
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
        # [{
        #     'criterion': ['gini'], 'splitter': ['random'],
        #     'class_weight': [None]
        # }],
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