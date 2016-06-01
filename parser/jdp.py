import os
import argparse
import json
from sdp import *

description = """
Joint Syntactic and Morphological Parser

This parser performs joint morphological and syntactic disambiguation.

INPUT: file in CoNLL-U format (first 6 fields required)
OUTPUT: file in CoNLL-U format with head and deprel fields filled in,
best morphological analysis selected for ambiguous tokens.

USAGE: python3 jdp.py <input_file> <output_file> -pm <parsing_model> -pvec <parsing_vectorizer>
-pf <parsing_features> -tm <tagging_model> -tvec <tagging_vectorizer>
-tf <tagging_features>

Model and vectorizer files are obtained during model training.
For more information, see README.
"""


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
    parser.add_argument('input_file', action=AbsPath)
    parser.add_argument('output_file', action=AbsPath)
    parser.add_argument('-pm', '--parsing_model', action=AbsPath, required=True,
                        help="path to the first file of the dependency parsing model")
    parser.add_argument('-pvec', '--parsing_vectorizer', action=AbsPath, required=True,
                        help="path to the vectorizer that goes with the dependency parsing model")
    parser.add_argument('-tm', '--tagging_model', action=AbsPath, required=True,
                        help="path to the first file of the morphological tagging model")
    parser.add_argument('-tvec', '--tagging_vectorizer', action=AbsPath, required=True,
                        help="path to the vectorizer that goes with the morphological tagging model")
    parser.add_argument('-pf', '--parsing_features', action=AbsPath, required=True,
                        help="path to feature config used in training the parsing model")
    parser.add_argument('-tf', '--tagging_features', action=AbsPath, required=True,
                        help="path to feature config used in training the tagging model")

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

    # begin parsing
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