import sys;

#  # Gold parse:
#  1       Bir     bir     _       det     ind     2       det     _       _
#  2       kün     kün     _       n       nom     8       nmod    _       _
#  3       Nasreddin       Nasreddin       _       np      ant|m|nom       8       subj    _       _
#  4       oca     oca     _       n       nom     3       appos   _       _
#  5       eşegine eşek    _       n       px3sp|dat       6       nmod    _       _
#  6       minip   min     _       v       iv|gna_perf     8       advcl   _       _
#  7       yolğa   yol     _       n       dat     8       nmod    _       _
#  8       çıqa    çıq     _       v       iv|aor|p3|sg    0       root    _       _
#  9       .       .       _       sent    _       8       punct   _       _
#  
#  # LAS: 0.3333333333333333

state = 0;
topres = [];

for line in sys.stdin.readlines(): #{
	line = line.strip();

	if line.count('Gold parse:'): #{
		state = 1;
	elif line.count('LAS:') and state == 1: #{
		state = 2;
	#}

	if state == 2: #{
		state = 0;
		topres.append(line.split(' ')[2]);
	#}
#}

n = float(len(topres));
tot = 0.0
for top in topres: #{
	tot = tot + float(top);
#}
print('Oracle: %.4f' % (tot / n));
