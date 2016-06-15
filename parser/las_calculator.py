import sys
from sdp import *
from metrics import wordform_las


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

def get_forms(s): #{
	o = '';
	for w in s: #{
		o = o + ' ' + w[1];
	#}
	return o.strip();
#}

def get_las(parsed, gold):
    gold_sentences = read_sentences(gold)
    lasses = []

    parsed_sentences = read_sentences(parsed)
    scount = 0;
    for s in parsed_sentences:
        gold_sentence = next(gold_sentences)

        if len(s[1:]) == len(gold_sentence[1:]):
            las_function = las
        else:
            print('ERR[%d][%d]\t%s' % (scount, len(s), get_forms(s)), file=sys.stderr);
            print('ERR[%d][%d]\t%s' % (scount, len(gold_sentence), get_forms(gold_sentence)), file=sys.stderr);
            las_function = wordform_las

        lasses.append(las_function(s[1:], gold_sentence[1:]))
        scount = scount + 1;
    

    good_scores = [i for i in lasses if i is not None]
    ignored = [str(i) for i in range(len(lasses)) if lasses[i] is None]

    print('LAS: %.3f' % float(sum(good_scores)/len(good_scores)))
    print('Ignored %d sentences:' % len(ignored))
    print(', '.join(ignored))


if __name__ == '__main__':

    gold = sys.argv[1]
    parsed = sys.argv[2]

    get_las(parsed, gold)
