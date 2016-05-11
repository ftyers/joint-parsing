# out = open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/en-ud-test-reduced-small.conllu', 'w')
#
# with open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/en-ud-test.conllu_reduced') as f:
#     counter = 1
#     for line in f:
#         if line == '\n':
#             counter += 1
#         if counter % 5 == 0:
#             out.write(line)
#
# out.close()
#
# with open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/xbest.conllz_test') as test:
#     ids = []
#     for line in test:
#         if line.startswith('#'):
#             ids.append(int(line.strip('#\n ')))
# out = open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki_ambig_full', 'w')
# with open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki.conllu') as full:
#     sentences = full.read().split('\n\n')
#     for i in range(len(sentences)):
#         if i in ids:
#             continue
#         out.write(sentences[i-1] + '\n\n')
#
# out.close()

train = open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki_ambig_train', 'w')
dev = open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki_ambig_dev', 'w')

with open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/kazakh/puupankki_ambig_full') as full:
    sentences = full.read().split('\n\n')
    counter = 0
    for s in sentences:
        if counter % 5 == 0:
            dev.write(s+'\n\n')
        else:
            train.write(s+'\n\n')
        counter += 1

train.close()
dev.close()