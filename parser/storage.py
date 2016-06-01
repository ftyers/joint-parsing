"""
This file stores code that used to be part of sdp.py.
sdp.py was stripped of all code not related to joint parsing
with a pre-trained model.

Some of this code will be moved to the training module.
Some will be removed completely.

No functions should be imported from this file; most of them
depend on other stuff still in sdp.py and will not work.
"""

# ----------------
# TRAINING FUNCTIONS
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

    # print("Best parameters set found on development set:")
    # print()
    # print(clf.best_params_)
    # print()
    # print("Grid scores on development set:")
    # print()
    # for params, mean_score, scores in clf.grid_scores_:
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean_score, scores.std() * 2, params))
    # print()
    y_true, y_pred = target_test, clf.predict(data_test)
    # print(classification_report(y_true, y_pred))
    print('%.3f' % clf.best_score_, end='')
    print('\t', end='')
    print(clf.best_params_)


    joblib.dump(clf, 'morph_model_for_%s.pkl' % (os.path.basename(training_path)))
    joblib.dump(vec, 'morph_vectorizer_for_%s.pkl' % (os.path.basename(training_path)))

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

    # transform string features via one-hot encoding
    vec = DictVectorizer()
    data = vec.fit_transform(training_collection)
    target = numpy.array(labels)
    data_test = vec.transform(development_collection)
    target_test = numpy.array(dev_labels)

    score = 'precision'

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

    joblib.dump(clf, 'model_for_%s.pkl' % (os.path.basename(training_path)))
    joblib.dump(vec, 'vectorizer_for_%s.pkl' % (os.path.basename(training_path)))

    def guide(c, feats):

        vector = vec.transform(extract_features(c, feats))
        try:
            transition, label = clf.best_estimator_.predict(vector)[0].split('_')
        except ValueError:
            transition = clf.best_estimator_.predict(vector)[0].split('_')[0]
            label = '_'
        return Transition(transition, label)

    return guide

# old training function with hardcoded classifier. replaced by train_with_classifier().
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

    for s, fvecs_labels in generate_training_data(training_path):
        for item in fvecs_labels:
            training_collection.append(item[0])
            labels.append(item[-1])

    for s, fvecs_labels in generate_training_data(development_path):

        for item in fvecs_labels:
            development_collection.append(item[0])
            dev_labels.append(item[-1])

    # transform string features via one-hot encoding
    vec = DictVectorizer()
    data = vec.fit_transform(training_collection)
    target = numpy.array(labels)
    data_test = vec.transform(development_collection)
    target_test = numpy.array(dev_labels)

    # that's a lot of code for training different classifiers
    # smaller set
    tuned_parameters = [{'loss': ['hinge'], 'shuffle': [True],
                         'learning_rate': ['constant'], 'eta0': [2**(-8)], 'average': [True, False],
                         'penalty': ['l1', 'l2', 'elasticnet'],
                         'alpha': [0.000001]}]

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
        vector = vec.transform(extract_features(c))
        try:
            transition, label = best.predict(vector)[0].split('_')
        except ValueError:
            transition = best.predict(vector)[0].split('_')[0]
            label = '_'
        return Transition(transition, label)

    return guide

# old training function that trains a non-sklearn perceptron. deprecated
def train_internal_classifier(train_conll):
    """Train a classifier on gold standard sentences and return a guide function
    which predicts transitions for given configurations using that classifier.
    """
    training_collection = []
    for s, fvecs_labels in generate_training_data(train_conll):
        training_collection.extend(fvecs_labels)
    dev_sents = list(read_sentences('data/en-ud-dev.conllu'))

    classifier = classifiers.MulticlassPerceptron(['sh', 'ra', 'la'])
    best_classifier = copy.deepcopy(classifier)
    best_uas = 0.0
    uas_after = 0.0
    iter = 0

    while iter < 15:
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

# ----------------
# Training data generators and helpers

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



# ----------------
# non-joint dependency parsing functions
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

# ----------------
# CoNLL-06 processing functions

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


def test_read_token():
    assert read_token('1\tJohn\tjohn\tNNP\t_\t_\t2\tsubj\t2\tsubj\n') == T_1
    assert read_token('1\tJohn\tjohn\tNNP\t_\t_\t2\tsubj\t_\t_\n') == T_2
    assert read_token('1\tJohn\tjohn\tNNP\t_\t_\t_\t_\t_\t_\n') == T_3
    assert read_token('1\tJohn\t_\t_\t_\t_\t_\t_\t_\t_\n') == T_4

# ----------------
# OTHER STUFF
class AbsPath(argparse.Action):
    """
    This is a helper class to resolve command line arguments in dependency_runner().
    """

    def __call__(self, parser, namespace, path, option_string=None):
        cwd = os.getcwd()
        # cwd = os.path.dirname(os.path.realpath(__file__))  # use for debugging and config launches
        if not os.path.isabs(path):
            path = os.path.join(cwd, path)
        setattr(namespace, self.dest, path)

def dependency_runner():
    """
    This function was used to run dependency parsing experiments.
    """
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
            print('Loading model...', end='')
            guide_function = load_model(model_path, vec_path)
            print('done')

        else:
            print('Training the classifier...')
            guide_function = train_with_classifier(args.train, args.development,
                                                   # DecisionTreeClassifier(),
                                                   # [{'criterion': ['gini'], 'splitter': ['random'], 'class_weight': [None]}],
                                                   SGDClassifier(),
                                                   [{'alpha': [1e-05], 'average': [False], 'learning_rate': ['constant'],
                                                     'eta0': [0.00390625], 'shuffle': [True], 'loss': ['hinge'], 'penalty':
                                                         ['l2']}],
                                                   ['b0.form',
                                                    'b0.pos',
                                                    's0.form',
                                                    's0.pos',
                                                    'b1.pos',
                                                    's1.pos',
                                                    'ld(b0).pos',
                                                    's0.pos b0.pos',
                                                    's0.pos b0.form',
                                                    's0.form b0.pos',
                                                    's0.form b0.form',
                                                    'b1.form',
                                                    'b2.pos',
                                                    'b3.pos',
                                                    's0_head.form',
                                                    'morph'])

        # parse input file
        cwd = os.getcwd()
        counter = 1
        print('Parsing sentences...')
        with open(os.path.join(cwd, args.output_file), 'w') as output_file:

            for s in read_conllz_for_joint(args.input_file):
                if counter % 20 == 0:
                    print('Parsing sentence %d' % counter)
                final_config = parse_with_feats(s, guide_function, ['b0.form',
                                                                    'b0.pos',
                                                                    's0.form',
                                                                    's0.pos',
                                                                    'b1.pos',
                                                                    's1.pos',
                                                                    'ld(b0).pos',
                                                                    's0.pos b0.pos',
                                                                    's0.pos b0.form',
                                                                    's0.form b0.pos',
                                                                    's0.form b0.form',
                                                                    'b1.form',
                                                                    'b2.pos',
                                                                    'b3.pos',
                                                                    's0_head.form',
                                                                    'morph'])
                output_file.write(s2conll(c2s(final_config)) + '\n')
                counter += 1

def parse_with_feats(s, oracle_or_guide, feats):
    """Given a sentence and a next transition predictor, parse the sentence.
    This function was used in external calls from tuning scripts.
    """
    c = initialize_configuration(s)
    while c.buffer:
        c = disambiguate_buffer_front(c)
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

