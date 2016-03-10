#!/usr/bin/python3

import sys;

lookup = {};

for line in open(sys.argv[1]).readlines(): #{
	if line.strip() == '': #{
		continue;
	#}
	row = line.strip().split('\t'); 
#	print(row);
	if row[0] not in lookup: #{
		lookup[row[0]] = set();
	#}
	lookup[row[0]].add((row[1], row[2], row[3]));
#}

for line in sys.stdin.readlines(): #{

	if line.strip() == '': #{
		sys.stdout.write(line);
		continue;
	#}
	row = line.split('\t');

	num = int(row[0]);
	sur = row[1];
	lem = row[2];
	cat = row[4];
	cha = row[5];

	# 1	Al	Al	PROPN	NNP	Number=Sing	3	name	_	SpaceAfter=No
	if sur in lookup: #{
		for analys in lookup[sur]: #{
			print('%d\t%s\t%s\t%s\t_\t%s' % (num, sur, analys[0], analys[1], analys[2]));
		#}
	else: #{
		print('WARNING: "%s" not found in dictionary.', file=sys.stderr);
		print('%d\t%s\t%s\t%s\t_\t%s' % (num, sur, lem, cat, cha));
	#}
#}
