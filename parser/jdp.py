import os
import argparse
import json
from sdp import *

description = """

"""
# todo write description for the parser

# a full set of features to use as a fallback
FEATURES = [
    'b0.form', 'b0.pos', 's0.form', 's0.pos', 'b1.pos', 's1.pos', 'ld(b0).pos', 's0.pos b0.pos', 's0.pos b0.form',
    's0.form b0.pos', 's0.form b0.form', 's0.lemma', 'b0.lemma', 'b1.form', 'b2.pos', 'b3.pos', 'rd(s0).deprel',
    'ld(s0).deprel', 'rd(b0).deprel', 'ld(b0).deprel', 's0.deprel', 's0_head.form'
]


class AbsPath(argparse.Action):

    def __call__(self, parser, namespace, path, option_string=None):
        # cwd = os.getcwd()
        cwd = os.path.dirname(os.path.realpath(__file__))  # use for debugging and config launches
        if not os.path.isabs(path):
            path = os.path.join(cwd, path)
        setattr(namespace, self.dest, path)

# runner
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('input_file', action=AbsPath)
    parser.add_argument('output_file', action=AbsPath)
    parser.add_argument('-pm', '--parsing_model', action=AbsPath)
    parser.add_argument('-pvec', '--parsing_vectorizer', action=AbsPath)
    parser.add_argument('-tm', '--tagging_model', action=AbsPath)
    parser.add_argument('-tvec', '--tagging_vectorizer', action=AbsPath)  # todo make everything required
    parser.add_argument('-pf', '--parsing_features', action=AbsPath)
    parser.add_argument('-tf', '--tagging_features', action=AbsPath)

    args = parser.parse_args()
    parsing_features = json.load(open(args.parsing_features))
    tagging_features = json.load(open(args.tagging_features))

    # load parsing model
    model_path = args.parsing_model
    vec_path = args.parsing_vectorizer
    print('Loading parsing model...', end='')
    guide_function = load_model(model_path, vec_path)
    print('done')

    # load morphological model
    model_path = args.tagging_model
    vec_path = args.tagging_vectorizer
    print('Loading tagging model...', end='')
    morph_guide_function = load_tagging_model(model_path, vec_path)
    print('done')

    cwd = os.getcwd()
    counter = 1
    print('Parsing sentences...')
    with open(os.path.join(cwd, args.output_file), 'w') as output_file:

        for s in read_conllz_for_joint(args.input_file):
            if counter % 20 == 0:
                print('Parsing sentence %d' % counter)
            final_config = joint_parse(s, guide_function, parsing_features, morph_guide_function, tagging_features)
            output_file.write(s2conll(c2s(final_config)) + '\n')
            counter += 1