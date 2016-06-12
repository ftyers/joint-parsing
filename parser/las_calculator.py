import sys
from metrics import wordform_las


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


def las(parsed, gold):
    """
    Calculate labeled attachment score for a given sentence
    """
    if len(parsed) != len(gold):
        return None
    if not parsed:
        return 0
    return len([t_pred for t_pred, t_gold in zip(parsed, gold) if t_pred.head == t_gold.head and
                t_pred.deprel == t_gold.deprel]) / float(len(parsed))


def get_las(parsed, gold):
    gold_sentences = read_sentences(gold)
    lasses = []

    parsed_sentences = read_sentences(parsed)

    for s in parsed_sentences:
        gold_sentence = next(gold_sentences)

        if len(s[1:]) == len(gold_sentence[1:]):
            las_function = las
        else:
            las_function = wordform_las

        lasses.append(las_function(s[1:], gold_sentence[1:]))

    good_scores = [i for i in lasses if i is not None]
    ignored = [str(i) for i in range(len(lasses)) if lasses[i] is None]

    print('LAS: %.3f' % float(sum(good_scores)/len(good_scores)))
    print('Ignored %d sentences:' % len(ignored))
    print(', '.join(ignored))


if __name__ == '__main__':

    gold = sys.argv[1]
    parsed = sys.argv[2]

    get_las(parsed, gold)