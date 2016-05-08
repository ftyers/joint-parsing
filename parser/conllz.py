"""
Parse ambiguous CONLL-Z input into a sequence of sentences
"""
from collections import deque, namedtuple
from copy import copy

SurfaceToken = namedtuple('SurfaceToken', ['id', 'form'])
Token = namedtuple('Token', ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats',
                             'head', 'deprel', 'phead', 'pdeprel'])
ROOT = Token(0, 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 0, 'ROOT', 0, 'ROOT')


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


def enumerate_tokens(sentences):
    """
    Fix token ids after some of the tokens have been left out
    """
    max_length = sentences[0][-1].id + 1
    for sentence in sentences:
        if len(sentence) != max_length:
            for i in range(len(sentence)):
                sentence[i] = Token(i,
                                    sentence[i][1],
                                    sentence[i][2],
                                    sentence[i][3],
                                    sentence[i][4],
                                    sentence[i][5],
                                    sentence[i][6],
                                    sentence[i][7],
                                    sentence[i][8],
                                    sentence[i][9],
                                    )


def collect_surface_tokens(lines):
    """
    From a list of lines, collect all surface form tokens.
    Surface tokens mean the tokens that have spanned ids,
    or just the token itself if it is the only token with
    its id.
    """
    max_id = 0
    tokens = []

    for line in lines:
        index = line.split('\t')[0]

        try:
            index = int(index)
            if index > max_id:  # reached the next independent token
                tokens.append(SurfaceToken(line.split('\t')[0], line.split('\t')[1]))

        except ValueError:
            ids = index.split('-')
            max_id = int(ids[-1])
            tokens.append(SurfaceToken(line.split('\t')[0], line.split('\t')[1]))

    return tokens


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
                    in_range = False
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

    # remove empty tokens
    tokens = [item for item in tokens if item]

    # I plug in here to save token counts, which are used to calculate ambiguity
    # writes number of tokens \t number of analyses
    # with open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/token_counts', 'a') as f:
    #     f.write('\t'.join((str(len(tokens)), str(sum([len(tier) for tier in tokens]))))+'\n')

    return tokens


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

    enumerate_tokens(sentences)

    return sentences


def read_conllz(filename, signals=False):
    """
    Generates sentences from CONLLZ file
    :param filename: path to conllz file
    :param signals: whether or not to send signals at the end of each macro-sentence.
    If True, the generator will return None after each group of alternatives obtained
    from one sentence.
    """
    with open(filename, 'r') as f:

        first = True

        sentence_buffer = deque()
        while True:

            # if sentence buffer is empty, get more sentences from file
            if not sentence_buffer:

                if signals and not first:
                    sentence_buffer.append(None)

                # store lines from one sentence in a buffer to be processed in batches
                line_buffer = deque()
                new_line = f.readline()

                # add lines to buffer until an empty line
                while new_line != '\n' and new_line != '':

                    if new_line.startswith('#'):  # skip comments
                        new_line = f.readline()
                        continue
                    line_buffer.append(new_line)
                    new_line = f.readline()

                # process lines in buffer
                if line_buffer:
                    sentences = make_sentences(line_buffer)
                else:  # eof
                    try:
                        yield sentence_buffer.popleft()
                    except IndexError:  # buffer empty
                        return

                print('Extracted %d sentences' % len(sentences))

                # store sentences
                for item in sentences:
                    sentence_buffer.append(item)
                first = False

            # return first item on the buffer
            try:
                yield sentence_buffer.popleft()
            except IndexError:  # buffer empty
                return


def read_conllz_for_joint(corpus):
    """
    Reads sentences from conllz file for joint disambiguation.
    The difference from read_conllz is that each original sentence
    gets only one representation, and it is not a flat sequence of
    tokens, but a sequence of (token, [analyses]).
    """
    with open(corpus) as f:

        sentences = deque()
        line_buffer = deque()
        new_line = f.readline()

        while True:

            # read all lines pertaining to one sentence
            while new_line != '\n' and new_line != '':

                if new_line.startswith('#'):  # skip comments
                    new_line = f.readline()
                    continue
                line_buffer.append(new_line)
                new_line = f.readline()

            new_line = f.readline()

            if line_buffer:
                other_buffer = copy(line_buffer)  # because we need to get title tokens from there as well

                tokens = collect_tokens(line_buffer)
                title_tokens = collect_surface_tokens(other_buffer)

                sentences.append([ROOT]+[i for i in zip(title_tokens, tokens)])

            else:  # eof
                try:
                    yield sentences.popleft()
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
    sentences = read_conllz(filename)
    s1 = next(sentences)
    s2 = next(sentences)
    assert s1 == S1
    assert s2 == S2


if __name__ == '__main__':
    # test_read_sentences()
    for i in read_conllz_for_joint('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/allmorph_short'):
        for token in i:
            print(token)
        print()