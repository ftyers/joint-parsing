#!/usr/bin/bash

source acceptance-tests/helper_functions.sh

# Feature: Read in gold standard sentences as Sentence data types,
#          parse them using the oracle function,
#          and write the result to another file.
# Input and output files should not differ.


# Scenario: input file contains only projective trees
  # When
    python3 sdp.py --oracle ../data/english/train/wsj_train.only-projective.conll06 output.conll09
  # Then
    if diff <(cat ../data/english/train/wsj_train.only-projective.conll06 | rmv_trailing_empty_lines) \
            <(cat output.conll09 | rmv_trailing_empty_lines)
    then
      echo "PASSED: $0"
      exit 0
    else
      echo "FAILED: $0"
      exit 1
    fi
