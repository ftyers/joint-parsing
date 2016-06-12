"""Model training script for the combined parser.

Trains a dependency parsing or a morphological tagging classifier
on the provided training and development corpora.
"""
import argparse
import json
import numpy
import os

from sklearn import grid_search
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from sdp import *
from conllz import read_conllz_for_joint

description = """Model training script for the combined parser.

Trains a dependency parsing or a morphological tagging classifier
on the provided training and development corpora.

USAGE:
  python3 train.py <path_to_training_corpus> <path_to_development_corpus>
  <path_to_feature_set> --morph --synt
"""


# Training functions
def train_with_classifier(training_path, development_path, classifier, parameters, features):
    """
    Train the dependency parsing model using the classifier and its parameters supplied.
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

    # Transform string features via one-hot encoding.
    vec = DictVectorizer()
    data = vec.fit_transform(training_collection)
    target = numpy.array(labels)
    data_test = vec.transform(development_collection)
    target_test = numpy.array(dev_labels)

    score = 'precision'

    clf = grid_search.GridSearchCV(classifier, parameters, cv=5, scoring='%s_weighted' % score, verbose=0)
    clf.fit(data, target)

    # Classification report.
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

    # Save the model
    joblib.dump(clf, 'model_for_%s.pkl' % (os.path.basename(training_path)))
    joblib.dump(vec, 'vectorizer_for_%s.pkl' % (os.path.basename(training_path)))
    print('Classifier files created in current directory.')

    # The same function will be created when the model loads from dump files.
    # No need to pass this anywhere. It remains from the times when training
    # happened on each parser launch.
    def guide(c, feats):

        vector = vec.transform(extract_features(c, feats))
        try:
            transition, label = clf.best_estimator_.predict(vector)[0].split('_')
        except ValueError:
            transition = clf.best_estimator_.predict(vector)[0].split('_')[0]
            label = '_'
        return Transition(transition, label)

    return guide


def train_morph_classifier(training_path, development_path, classifier, parameters, features):
    """
    Trains the morphological classifier.
    :param training_path: path to training corpus
    :param development_path: path to development corpus
    :param classifier: an sklearn classifier object to use in training
    :param parameters: parameters of the classifier
    :param features: a set of features to use in training
    Returns a guide function that determines the best morphological analysis
    """

    training_collection = []
    labels = []
    development_collection = []
    dev_labels = []

    for s, fvecs_labels in generate_training_data_morph(training_path, feature_config=features):
        for item in fvecs_labels:
            training_collection.append(item[0])
            labels.append(item[-1])

    for s, fvecs_labels in generate_training_data_morph(development_path, feature_config=features):

        for item in fvecs_labels:
            development_collection.append(item[0])
            dev_labels.append(item[-1])

    # transform string features via one-hot encoding
    vec = DictVectorizer()
    data = vec.fit_transform(training_collection)
    target = numpy.array(labels)
    data_test = vec.transform(development_collection)
    target_test = numpy.array(dev_labels)

    score = 'precision'

    clf = grid_search.GridSearchCV(classifier, parameters, cv=3, scoring='%s_weighted' % score, verbose=0)
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
    print('%.3f' % clf.best_score_, end='')
    print('\t', end='')
    print(clf.best_params_)

    joblib.dump(clf, 'morph_model_for_%s.pkl' % (os.path.basename(training_path)))
    joblib.dump(vec, 'morph_vectorizer_for_%s.pkl' % (os.path.basename(training_path)))
    print('Classifier files created in current directory.')

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


# Training data generators
def generate_training_data(train_conll, feature_config=None):
    """Generate sentence and a list of (feature vector, expected label) tuples (representing
    configuration and correct transition operation) out of the training sentences.
    """
    for s in read_sentences(train_conll):
        c = initialize_configuration(s)
        fvecs_and_labels = []
        while c.buffer:
            tr = oracle(c)

            fvecs_and_labels.append((extract_features(c, feature_config), tr.op+'_'+tr.l))

            if tr.op == 'sh':
                c = shift(c)
            elif tr.op == 'la':
                c = left_arc(c, tr.l)
            elif tr.op == 'ra':
                c = right_arc(c, tr.l)
        yield (s, fvecs_and_labels)


def generate_training_data_morph(train_conll, feature_config=None):

    for s in read_conllz_for_joint(train_conll):

        c = initialize_configuration(s)
        fvecs_and_labels = []

        while c.buffer:

            fvecs_and_labels.append((extract_features(c, feature_config), get_morph_label(c)))
            c = disambiguate_buffer_front(c)  # convert buffer front to a flat token

            tr = oracle(c)  # determine the next transition

            # change configurations in parallel and continue
            if tr.op == 'sh':
                c = shift(c)
            elif tr.op == 'la':
                c = left_arc(c, tr.l)
            elif tr.op == 'ra':
                c = right_arc(c, tr.l)

        yield (s, fvecs_and_labels)


# Utility functions for syntactic training
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


# Filename -> (generator Sentence)
def read_sentences(f):
    """Return Sentences from a file in CoNLL06 format."""
    with open(f, 'r') as conll_file:
        s = [ROOT]
        for line in conll_file:
            if line.strip() and not line.startswith('#'):
                s.append(read_token(line))
            elif len(s) != 1:
                yield s
                s = [ROOT]
        if len(s) != 1:  # file ended without a new line at the end
            yield s


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


# Absolute path resolver for command-line arguments.
class AbsPath(argparse.Action):

    def __call__(self, parser, namespace, path, option_string=None):
        cwd = os.getcwd()
        if not os.path.isabs(path):
            path = os.path.join(cwd, path)
        setattr(namespace, self.dest, path)

# runner
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # resolve command-line arguments
    parser.add_argument('training_corpus', action=AbsPath,
                        help="Path to the training corpus.")
    parser.add_argument('development_corpus', action=AbsPath,
                        help="Path to the development corpus.")
    parser.add_argument('features', action=AbsPath,
                        help="Path to feature set file.")
    parser.add_argument('-m', '--morph', action='store_true',
                        help="A flag to train the morphological model.")
    parser.add_argument('-s', '--synt', action='store_true',
                        help="A flag to train the syntactic model.")

    args = parser.parse_args()
    features = json.load(open(args.features))

    # Define your classifier here. For DecisionTree docs, see
    # http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    # For other classifiers, see http://scikit-learn.org/stable/supervised_learning.html
    classifier = DecisionTreeClassifier()

    # Specify parameters to tune in classifier training. If you want a
    # specific classifier, fill in one option in each field.
    parameters = {'criterion': ['gini'],
                  'splitter': ['best', 'random']}

    if not args.morph and not args.synt:
        print('WARNING: mode not specified. Training the syntactic model.')
        print('Training the syntactic model...')
        train_with_classifier(args.training_corpus, args.development_corpus, classifier, parameters, features)

    elif args.morph:
        print('Training the morphological model...')
        train_morph_classifier(args.training_corpus, args.development_corpus, classifier, parameters, features)

    elif args.synt:
        print('Training the syntactic model...')
        train_with_classifier(args.training_corpus, args.development_corpus, classifier, parameters, features)

    else:  # todo and if both? Should we allow to train both?
        pass
