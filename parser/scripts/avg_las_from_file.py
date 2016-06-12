import re

las_max = []
las_min = []
las_min_nozeros = []

with open('/Users/Sereni/PycharmProjects/Joint Parsing/parser/data/results/DT_experiments/tuvan_xbest_DT') as f:
	chunks = f.read().split('# Gold parse:')
	for chunk in chunks:
		lasses = [float(las) for las in re.findall('# LAS: ([0-9.]+)\n', chunk)]
		# print(lasses)
		if lasses:
			las_max.append(max(lasses))
			las_min.append(min(lasses))
		lasses = [las for las in lasses if las]
		if lasses:
			las_min_nozeros.append(min(lasses))
		
print('LAS max: ', float(sum(las_max))/len(las_max))
print('LAS min: ', float(sum(las_min))/len(las_min))
print('LAS min no zeros: ', float(sum(las_min_nozeros))/len(las_min_nozeros))