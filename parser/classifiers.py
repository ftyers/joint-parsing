#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## A multiclass perceptron-based classifier (to begin with).
##
## Consumes (listof (tupleof FeatureVector, Label)) for training
## (where Label is a String) and FeatureVector for classification.
##
## USAGE: supposed to be imported into other modules to be provided
##        with data described above.
## UNIT TESTS: py.test classifier.py (py.test has to be installed)


## =================
## Constants:


INITIAL_WEIGHT = 0.0
MAX_NUMBER_OF_ITERATIONS = 10
STEP_SIZE = 1.0


## =================
## Data definitions:


## FeatureVector is (listof String) of arbitrary size with string "bias" as 0th element.
## interp. a list of features extracted from an object to be classified. 'bias' corresponds
##         to -threshold in the { f(x)=1, if dot_product(x, w) - threshold >= 0
##                                     0, if dot_product(x, w) - threshold < 0 } formula

FV_1 = ['bias', 's0.form=They', 'b0.form=are', 's0.cpos=PRP', 'b0.cpos=VB']


## Collection is (listof (tupleof FeatureVector, String))
## interp. a list of features vectors and expected labels for them to be learned from

COLL_1 = [(['bias', 's0.form=They', 'b0.form=are', 's0.cpos=PRP', 'b0.cpos=VB'], 'la'),
          (['bias', 's0.form=at', 'b0.form=home', 's0.cpos=IN', 'b0.cpos=NN'], 'ra')]


## Weights is a Dictionary which maps Strings to Floats.
## interp. weights associated with each feature


## Score is Float.
## interp. sum of weights of features in a feature vector (corresponds to dot product of features
##         with weights since in a FeatureVector we store only features which would have a value
##         of 1 in a normal binary feature vector)


## Perceptron is Perceptron(String)
## interp. a perceptron classifier targeted at the given label (=class) containing a dictionary
##         with weights for each feature
class Perceptron:

    def __init__(self, label):
        self.label = label
        self.weights = {}


## MulticlassPerceptron is MulticlassPerceptron((listof String))
## interp. a multiclass perceptron targeted at labels (=classes) in the given list of labels
class MulticlassPerceptron:

    def __init__(self, labels):
        self.perceptrons = {}
        for label in labels:
            self.perceptrons[label] = Perceptron(label)

    ## Collection -> Integer
    def train_one_iteration(self, train_instances):
        """Given a list of training instances represented as feature vectors and their labels,
        do one iteration of training. Return number of errors made.
        MUTATES: - 'weights' dictionaries of individual perceptrons"""
        errors = 0
        for feature_vector, expected_label in train_instances:
            pred_label = self.classify(feature_vector)
            if pred_label != expected_label:
                errors += 1
                adjust_weights(feature_vector, self.perceptrons[pred_label].weights, -STEP_SIZE)
                adjust_weights(feature_vector, self.perceptrons[expected_label].weights, STEP_SIZE)
        return errors

    ## Collection -> Integer
    def train(self, train_instances):
        """Given a list of training instances represented as feature vectors and their labels,
        learn to classify them correctly. Return number of iterations in training.
        MUTATES: - 'weights' dictionaries of individual perceptrons"""
        errors = 1
        iter = 0
        number_of_instances = len(train_instances)
        while errors != 0 and iter < MAX_NUMBER_OF_ITERATIONS:
            iter += 1
            errors = self.train_one_iteration(train_instances)
            print('Accuracy of predictions: ', errors/number_of_instances, '\n')
        return iter

    ## FeatureVector -> String
    def classify(self, feature_vector):
        """Given a feature vector, predict the most likely label for it."""
        return max(self.perceptrons.keys(),
                   key=lambda label: dot_product(feature_vector, self.perceptrons[label].weights))

def test_multiclass_perceptron():
    # learning OR function
    mcp = MulticlassPerceptron(["0", "1"])
    mcp.train([(["bias", "first_boolean=0", "second_boolean=0"], "0"),
               (["bias", "first_boolean=0", "second_boolean=1"], "1"),
               (["bias", "first_boolean=1", "second_boolean=0"], "1"),
               (["bias", "first_boolean=1", "second_boolean=1"], "1")])
    assert len(mcp.perceptrons) == 2
    assert mcp.perceptrons["0"].label == "0"
    assert mcp.perceptrons["1"].label == "1"
    assert len(mcp.perceptrons["0"].weights) == 5
    assert len(mcp.perceptrons["1"].weights) == 5
    assert mcp.classify(["bias", "first_boolean=0", "second_boolean=0"]) == "0"
    assert mcp.classify(["bias", "first_boolean=0", "second_boolean=1"]) == "1"
    assert mcp.classify(["bias", "first_boolean=1", "second_boolean=0"]) == "1"
    assert mcp.classify(["bias", "first_boolean=1", "second_boolean=1"]) == "1"

    # learning XOR function (not linearly separable -> max number of iterations expected)
    mcp = MulticlassPerceptron(["0", "1"])
    iter = mcp.train([(["bias", "first_boolean=0", "second_boolean=0"], "0"),
                      (["bias", "first_boolean=0", "second_boolean=1"], "1"),
                      (["bias", "first_boolean=1", "second_boolean=0"], "1"),
                      (["bias", "first_boolean=1", "second_boolean=1"], "0")])
    assert iter == MAX_NUMBER_OF_ITERATIONS


## =================
## Functions:


## FeatureVector Weights Float -> Void
def adjust_weights(feature_vector, weights, step):
    """Adjust weights associated with features in feature_vector by step.
    MUTATES: - weights"""
    for feature in feature_vector:
        try:
            weights[feature] += step
        except KeyError:
            weights[feature] = INITIAL_WEIGHT + step


def test_adjust_weights():
    INITIAL_WEIGHT = 0.0

    # features in weights are a subset of features in feature vector
    weights = {}
    adjust_weights(["bias", "prefix=inc", "suffix=ing", "length=10"], weights, 1.0)
    assert weights == {"bias": 1.0, "prefix=inc": 1.0, "suffix=ing": 1.0, "length=10": 1.0}

    # features in feature vector are a subset of features in weights
    weights = {"bias": 1.0, "prefix=inc": 0.0, "suffix=ing": -1.0, "caps": 3.0}
    adjust_weights(["bias", "prefix=inc", "suffix=ing"], weights, 1.0)
    assert weights == {"bias": 2.0, "prefix=inc": 1.0, "suffix=ing": 0.0, "caps": 3.0}

    # common features are a subset of both
    weights = {"bias": 1.0, "prefix=inc": 0.0, "suffix=ing": -1.0, "caps": 3.0}
    adjust_weights(["bias", "prefix=inc", "suffix=ing", "pos=NN"], weights, -1.0)
    assert weights == {"bias": 0.0, "prefix=inc": -1.0, "suffix=ing": -2.0, "caps": 3.0, "pos=NN": -1.0}


## FeatureVector Weights -> Float
def dot_product(feature_vector, weights):
    """Return sum of the weights (=score) of features stored in the feature vector.
    MUTATES: - weights: sets the weight of newly encountered features to INITIAL_WEIGHT"""
    sum_of_weights = 0.0
    for feature in feature_vector:
        try:
            weights[feature]
        except KeyError:
            weights[feature] = INITIAL_WEIGHT
        sum_of_weights += weights[feature]
    return sum_of_weights

def test_dot_product():
    INITIAL_WEIGHT = 0.0

    weights = {}
    assert dot_product(["bias", "first_boolean=0", "second_boolean=1"], weights) == \
           INITIAL_WEIGHT + INITIAL_WEIGHT + INITIAL_WEIGHT
    assert weights == {"bias": INITIAL_WEIGHT, "first_boolean=0": INITIAL_WEIGHT, "second_boolean=1": INITIAL_WEIGHT}

    weights = {"bias": 1.0, "first_boolean=0": -1.0, "second_boolean=1": 1.0}
    assert dot_product(["bias", "first_boolean=0", "second_boolean=1"], weights) == 1.0 - 1.0 + 1.0
    assert weights == {"bias": 1.0, "first_boolean=0": -1.0, "second_boolean=1": 1.0}
