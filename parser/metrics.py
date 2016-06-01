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


def las(parsed, gold):
    """
    Calculate labeled attachment score for a given sentence
    """
    if len(parsed) != len(gold):
        return 0
    if not parsed:
        return 0
    return len([t_pred for t_pred, t_gold in zip(parsed, gold) if t_pred.head == t_gold.head and
                t_pred.deprel == t_gold.deprel]) / float(len(parsed))


def get_wordform_dependencies(s):
    """
    Create a set of dependencies from a sentence.
    Dependencies are tuples of (head.form, label, dependent.form)
    """
    deps = set([])
    for token in s:
        head_id = token.head - 1
        head = s[head_id].form
        deps.add((head, token.deprel, token.form))
    return deps


def wordform_las(parsed, gold):
    """
    Calculate labeled attachment score using word forms as reference.
    In regular LAS, token ids are used to compare parsed dependencies
    with the gold output. In the case where the number of tokens in
    gold and parsed do not match, LAS cannot be calculated.
    As a workaround, create a set of dependencies (head, label, dependent)
    where head and dependent are word forms.
    Return the number of dependencies that coincide for gold and parsed,
    over the number of dependencies in gold.
    """
    if not parsed:
        return 0
    parsed_deps = get_wordform_dependencies(parsed)
    gold_deps = get_wordform_dependencies(gold)
    correct = len(parsed_deps.intersection(gold_deps))

    return correct / len(gold)