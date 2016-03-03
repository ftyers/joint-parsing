#!/usr/bin/bash

source acceptance-tests/helper_functions.sh


# Feature: Given:[1] a gold standard training conll file, [2] a file with to be parsed sentences
#          also in conll format, and [3] the name of the output file, learn a parsing model,
#          parse sentences in [2] using newly learned model and output parsed sentences to [3].



# Scenario 1:
  # When
  # training on and parsing the same file
    python3 sdp.py --train-and-parse=../data/english/train/wsj_train.only-projective.first-1k.conll06 \
    ../data/english/train/wsj_train.only-projective.first-1k.conll06 output.conll09

  # Then
  # all fields except the last four (head, deprel, phead, pdeprel should stay the same
  # when transforming input file to output file
    if diff <(cut -f1-6 -d$'\t' ../data/english/train/wsj_train.only-projective.first-1k.conll06) \
            <(cut -f1-6 -d$'\t' output.conll09) &&
  # but the last four fields should change (unless we settle for memorizing everything)
       ! diff <(cut -f7-10 -d$'\t' ../data/english/train.wsj_train.only-projective.first-1k.conll06) \
              <(cut -f7-10 -d$'\t' output.conll09)
    then
      echo "PASSED: $0"
      exit 0
    else
      echo "FAILED: $0"
      exit 1
    fi
