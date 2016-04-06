"""
Parse ambiguous CONLL-Z input into a sequence of sentences
"""
from collections import deque
from copy import copy
from sdp import ROOT, read_token, Token


def collect_tokens(lines):
    """
    Collect tokens from a list of CONLL-Z lines.
    The list contains tiers of tokens, where each tier
    is made of all possible choices for a token at a
    given step. The list is then used to make all
    possible sentences given ambiguous analyses.
    """

    # helper functions
    def new_tier(t):
        t.append([])

    def write_token(t, token_list):
        token_list[-1].append([read_token(t)])

    def write_group(g, token_list):
        token_list[-1].append(g)

    # initialize variables
    last_id = None
    in_range = False
    tokens = [[]]

    while lines:

        token = lines.popleft()  # fetch new token
        raw_id = token.split('\t')[0]

        try:  # get numerical token id
            token_id = int(raw_id)

        except ValueError:  # if id fails conversion to string, it is a range marker

            # set range variables and continue to the next line
            in_range = True
            last_id, range_max = raw_id.split('-')
            last_id = int(last_id)
            range_max = int(range_max)
            new_tier(tokens)
            continue

        if in_range:  # if previously seen a range token, tokens should be processed as a group

            group = [read_token(token)]

            while lines:
                next_token = lines.popleft()
                next_id_raw = next_token.split('\t')[0]

                try:
                    next_id = int(next_id_raw)
                except ValueError:  # encountered a range marker
                    write_group(group, tokens)
                    last_id, range_max = next_id_raw.split('-')
                    last_id = int(last_id)
                    range_max = int(range_max)
                    new_tier(tokens)
                    break

                if next_id > range_max:  # range is over, next independent token starts
                    write_group(group, tokens)
                    new_tier(tokens)
                    write_token(next_token, tokens)
                    last_id = next_id
                    break

                else:
                    if last_id != next_id:
                        group.append(read_token(next_token))
                    else:
                        write_group(group, tokens)
                        group = [read_token(next_token)]

        else:  # if range marker is not set, process the token individually
            if token_id != last_id:  # if id changed, make new tier of tokens
                new_tier(tokens)
            write_token(token, tokens)

    return [item for item in tokens if item]



def make_sentences(lines):
    """
    Construct all possible sentences out of ambiguous CONLL analyses
    :type lines: collections.deque
    :returns list of sentences, where sentence is a list of Tokens
    """
    tokens = collect_tokens(lines)
    sentences = [[ROOT]]

    for i in range(len(tokens)):
        new_sentences = []
        for sentence in sentences:

            # at each new layer, copy all other sentences and add one of the new possible tokens tokens
            for n in range(len(tokens[i])):
                new_sentences.append(copy(sentence) + tokens[i][n])
        sentences = new_sentences
    return sentences


def read_sentences(filename):
    """
    Generates sentences from CONLLZ file
    :param filename: path to conllz file
    """
    with open(filename, 'r') as f:

        sentence_buffer = deque()
        while True:

            # if sentence buffer is empty, get more sentences from file
            if not sentence_buffer:

                # store lines from one sentence in a buffer to be processed in batches
                line_buffer = deque()
                new_line = f.readline()

                # add lines to buffer until an empty line
                while new_line != '\n':
                    line_buffer.append(new_line)
                    new_line = f.readline()

                # process lines in buffer
                sentences = make_sentences(line_buffer)

                # store sentences
                for item in sentences:
                    sentence_buffer.append(item)

            # return first item on the buffer
            try:
                yield sentence_buffer.popleft()
            except IndexError:  # buffer empty
                return


S1 = [
    ROOT,
    Token(1, 'Азамат', 'Азамат', 'PROPN', '_', 'Gender=Masc|Case=Nom', '_', '_', '_', '_'),
    Token(3, 'пен', 'мен', 'CONJ', '_', '_', '_', '_', '_', '_'),
    Token(4, 'Айгүл', 'Айгүл', 'PROPN', '_', 'Gender=Fem|Case=Nom', '_', '_', '_', '_'),
    Token(6, 'бақшада', 'бақша', 'NOUN', '_', 'Case=Loc', '_', '_', '_', '_'),
    Token(8, '.', '.', 'PUNCT', '_', '_', '_', '_', '_', '_'),
]

S2 = [
    ROOT,
    Token(1, 'Азамат', 'Азамат', 'PROPN', '_', 'Gender=Masc|Case=Nom', '_', '_', '_', '_'),
    Token(3, 'пен', 'мен', 'CONJ', '_', '_', '_', '_', '_', '_'),
    Token(4, 'Айгүл', 'Айгүл', 'PROPN', '_', 'Gender=Fem|Case=Nom', '_', '_', '_', '_'),
    Token(6, 'бақшада', 'бақша', 'NOUN', '_', 'Case=Loc', '_', '_', '_', '_'),
    Token(7, '_', 'е', 'VERB', '_', 'Tense=Aor|Person=3|Number=Sing', '_', '_', '_', '_'),
    Token(8, '.', '.', 'PUNCT', '_', '_', '_', '_', '_', '_'),
]


def test_read_sentences():
    filename = 'kaz.conllz'
    sentences = read_sentences(filename)
    s1 = next(sentences)
    s2 = next(sentences)
    assert s1 == S1
    assert s2 == S2

if __name__ == '__main__':
    test_read_sentences()