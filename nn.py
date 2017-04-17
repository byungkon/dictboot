from nltk.corpus import wordnet as wn
import numpy as np
import numpy.random as nr
import math
import cPickle as pk
import sys
import gzip

dw2id = {}

def sim(v1, v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def dist(v1, v2):
	return math.sqrt(((v1 - v2) ** 2).sum())

if __name__ == "__main__":
	test_mode = 'none'
	if len(sys.argv) < 2:
		exit()
	with open(sys.argv[1], 'rb') as f:
		model = pk.load(f)
	with open(sys.argv[2], 'rb') as f:
		supp = pk.load(f)
	if test_mode == 'ndw':
		print 'Testing init_dwe2'
		dic = supp['init_dwe2']
	elif test_mode == 'aw':
		with gzip.open('glove.pklz', 'rb') as f:
			dic = pk.load(f)
	else:
		dic = model['params']['dwe'].get_value()
		
	for (i, w) in enumerate(supp['id2dw']):
		dw2id[w] = i
	ind = [dw2id['hood.n.01'], dw2id['bank.n.01'], dw2id['paper.n.01'], dw2id['paper.n.05'], \
		dw2id['bank.n.05'], dw2id['apple.n.01'], dw2id['hood.n.02']]

	with open('nn_out.txt', 'w') as f:
		for w1 in ind:
			print 'Processing {}'.format(w1)
			lst = []
			for (w2, _) in enumerate(dic):
				if w1 == w2: continue
				s = sim(dic[w1], dic[w2])
				lst.append((s, supp['id2dw'][w2]))
			lst.sort(key = lambda e: e[0], reverse=True)
			nn_w = lst[0:20]
			f.write(str(supp['id2dw'][w1]) + '\t' + str(nn_w) + '\n')
