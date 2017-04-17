import theano
import theano.tensor as T
import lasagne
import numpy as np
import numpy.random as nr
import sys
import pdb
import os
import cPickle as pk
import time
import re
import string
import xmltodict as xd
from nltk.corpus import wordnet as wn
from lasagne.layers import GRULayer, get_all_params, Gate, InputLayer, LSTMLayer
import lasagne.nonlinearities as nonlinearities

class Sent2Vec:
	def __init__(self, in_path, concat=False, wsi_path=None, dat_path='data/dat.pkl', supp_path='data/supp.pkl'):
		self.in_path = in_path
		self.concat = concat
		self.wsi_path = wsi_path
		self.lm_mode = 'default'
		with open(self.in_path, 'rb') as f:
			p = pk.load(f)

		self.do_brnn = False
		if 'do_brnn' in p:
			self.do_brnn = p['do_brnn']
		self.is_lstm = 'Wxo' in p['params']
		self.is_gru = 'Whr' in p['params']
		if 'Wt' not in p: self.lm_mode = 'iden'
		elif p['Wt'].get_value().ndim == 1: self.lm_mode = 'diag'
		self.params = p['params']
		self.dwe = self.params['dwe'] # disambiguated word embeddings
		self.td = self.dwe.get_value().shape[1]
		self.hd = self.params['L'].get_value().shape[1]
		self.gc = 2
		self.l_mask = InputLayer((None, None), trainable=False)

		if self.is_lstm:
			self.l_gru_emb = LSTMLayer((None, None, self.td), self.hd, grad_clipping=self.gc, \
				ingate=Gate(W_in=self.params['Wxr'], W_hid=self.params['Whr'], b=self.params['br'], W_cell=None), \
				forgetgate=Gate(W_in=self.params['Wxu'], W_hid=self.params['Whu'], b=self.params['bu'], W_cell=None), \
				outgate=Gate(W_in=self.params['Wxo'], W_hid=self.params['Who'], b=self.params['bo'], W_cell=None), \
				cell=Gate(W_in=self.params['Wxc'], W_hid=self.params['Whc'], b=self.params['bc'], W_cell=None,\
					nonlinearity=nonlinearities.tanh),mask_input=self.l_mask, peepholes=False)
			if self.do_brnn:
				self.l_bgru_emb = LSTMLayer((None, None, self.td), self.hd, grad_clipping=self.gc, \
					ingate=Gate(W_in=self.params['bWxr'], W_hid=self.params['bWhr'], b=self.params['bbr'], W_cell=None), \
					forgetgate=Gate(W_in=self.params['bWxu'], W_hid=self.params['bWhu'], b=self.params['bbu'], W_cell=None), \
					outgate=Gate(W_in=self.params['bWxo'], W_hid=self.params['bWho'], b=self.params['bbo'], W_cell=None), \
					cell=Gate(W_in=self.params['bWxc'], W_hid=self.params['bWhc'], b=self.params['bbc'], W_cell=None,\
						nonlinearity=nonlinearities.tanh),mask_input=self.l_mask, peepholes=False, backwards=True)
		elif self.is_gru:
			self.l_gru_emb = GRULayer((None, None, self.td), self.hd, grad_clipping=self.gc, \
				resetgate=Gate(W_in=self.params['Wxr'], W_hid=self.params['Whr'], b=self.params['br'], W_cell=None), \
				updategate=Gate(W_in=self.params['Wxu'], W_hid=self.params['Whu'], b=self.params['bu'], W_cell=None), \
				hidden_update=Gate(W_in=self.params['Wxc'], W_hid=self.params['Whc'], b=self.params['bc'], W_cell=None,\
					nonlinearity=nonlinearities.tanh),mask_input=self.l_mask)
			if self.do_brnn:
				self.l_bgru_emb = GRULayer((None, None, self.td), self.hd, grad_clipping=self.gc, \
					resetgate=Gate(W_in=self.params['bWxr'], W_hid=self.params['bWhr'], b=self.params['bbr'], W_cell=None), \
					updategate=Gate(W_in=self.params['bWxu'], W_hid=self.params['bWhu'], b=self.params['bbu'], W_cell=None), \
					hidden_update=Gate(W_in=self.params['bWxc'], W_hid=self.params['bWhc'], b=self.params['bbc'], W_cell=None,\
						nonlinearity=nonlinearities.tanh),mask_input=self.l_mask, backwards=True)
		else:
			self.is_nlm = True
	
		with open(dat_path, 'rb') as f:
			d = pk.load(f)
		self.nw, self.mw, self.ms = d['def'].shape # num words, max num of words, max num of senses
		self.dw = d['dw'] # dw to index
		self.aw = d['aw']
		self.no = len(d['aw'])
		if 'spriors' in d:
			self.sense_priors = d['spriors']
		else:
			self.sense_priors = np.ones((self.no, self.ms))

		with open(supp_path, 'rb') as f:
			s = pk.load(f)
		self.id2aw = s['id2aw']
		self.id2dw = s['id2dw']
		self.aw2dw = s['aw2dw']

		self.build_encoder()

    # assume xml-style input
	# output: 'lemma.pos instance-id sense-name/rating'
	def perform_wsi(self):
		expr = '[' + string.punctuation + ']'
		jaccard = False
		for d in os.listdir(self.wsi_path):
			f = os.path.join(self.wsi_path, d)
			if not os.path.isfile(f): continue
			with open(f) as fin:
				wsi = xd.parse(fin.read())
			for inst in wsi['instances']['instance']:
				tok = inst['@token']
				txt = re.sub(expr, ' ', inst['#text'])
				lemma = inst['@lemma']
				pos = inst['@partOfSpeech']
				inst_id = inst['@id']
				ind = txt.split().index(tok)
				s, m, ptmp = self.to_indexes(txt, token=tok, pos=pos, lem=lemma)
				'''s = s.reshape(1, *s.shape)
				m = m.reshape(1, *m.shape)
				fu = np.asarray([ptmp]).astype(np.int32)
				weights = self.get_weights(s, m, fu, np.ones_like(s).astype(np.float32)) # mw x ms'''
				weights = self.get_vector([txt], mode='w', token=tok, pos=pos, lem=lemma)
				senses = s[ind, :]
				sweight = weights[0][ind, :]
				ratings = [(self.id2dw[senses[i]], sweight[i]) \
					for i in range(len(sweight)) \
						if self.id2dw[senses[i]].split('.')[0] == lemma and sweight[i] > 0.02]
				ratings.sort(key=lambda k: k[1], reverse=True)
				if len(ratings) == 0: pdb.set_trace()
				l = min(3, len(ratings))
				if jaccard:
					r = [k[0] for k in ratings[0:2]]
				else:
					r = [k[0] + '/' + str(k[1]) for k in ratings[0:l]]
				print '{}.{} {} {}'.format(lemma, pos, inst_id, ' '.join(r))

	def build_encoder(self):
		def to_vect(d, m, p):
			L0 = self.params['L0']
			hid_inp = self.dwe[d, :] # mw x ms x hd
			logit = T.exp(T.dot(hid_inp, L0)[:,:,p])# (mw x ms) x mw
			mk = T.switch(T.lt(p, 0), 0, 1) # mw: word-level mask (different mask from m)
			mask = mk.dimshuffle(0, 'x', 'x')
			l2 = logit * mask # mw x ms x mw
			l2 = T.sum(l2 * mk.dimshuffle('x', 'x', 0), axis=2) * m # mw x ms 
			w0 = l2 / T.sum(l2, axis=1).dimshuffle(0, 'x')
			w1 = T.switch(T.isnan(w0), 0, w0)
			w = w1.dimshuffle(0, 1, 'x') # mw x ms x 1
			res = T.sum(w * hid_inp, axis=1) # mw x hd
			return res #, logit, weights

		def to_weights(d, m, p, prior):
			hid_inp = self.dwe[d, :] # mw x ms x hd
			if self.is_lstm or self.is_gru:
				logit = T.exp(T.dot(hid_inp, L0)[:,:,p])# (mw x ms) x mw
				mk = T.switch(T.lt(p, 0), 0, 1) # mw: word-level mask (different mask from m)
				mask = mk.dimshuffle(0, 'x', 'x')
				l2 = logit * mask # mw x ms x mw
				l2 = T.sum(l2 * mk.dimshuffle('x', 'x', 0), axis=2) * m # mw x ms 
				w0 = l2 / T.sum(l2, axis=1).dimshuffle(0, 'x')
				w1 = T.switch(T.isnan(w0), 0, w0)
			else:
				if self.lm_mode == 'diag':
					B = hid_inp * Wt.dimshuffle('x', 'x', 0)
					tmp = T.tensordot(B, B.T, axes = 1)
				elif self.lm_mode == 'iden':
					logit = T.tensordot(self.dwe[d, :], self.dwe.T, axes=1)[:,:,d] # mw x ms x mw x ms
					cnt = T.sum(m, axis=1).dimshuffle('x', 'x', 0) # 1 x 1 x mw
					logit = T.sum(logit * m.dimshuffle('x', 'x', 0, 1), axis=3) / cnt # mw x ms x mw
					logit = T.exp(10*T.switch(T.isnan(logit), 0, logit)) # mw x ms x mw
					logit = T.prod(logit, axis=2) * prior # mw x ms
					sm = T.sum(logit * m, axis=1, keepdims=True) # mw x 1
					logit = (logit * m) / sm # mw x ms
					return T.switch(T.or_(T.isnan(logit), T.isinf(logit)), 0, logit)
				else:
					tmp = T.tensordot(T.dot(hid_inp, self.params['Wt']), hid_inp.T, axes=1) # mw x ms x ms x mw
				tmp = T.exp(tmp.dimshuffle(0, 1, 3, 2)) # mw x ms x mw x ms
				tmp = tmp * m.dimshuffle('x', 'x', 0, 1)
				nrm = T.sum(tmp, axis=3)
				tmp = tmp / nrm.dimshuffle(0, 1, 2, 'x')
				tmp = T.switch(T.isnan(tmp), 0, tmp)
				mk = T.switch(T.lt(p, 0), 0, 1) # mw: word-level mask (different mask from m)
				tmp = T.max(tmp, axis=3) * mk.dimshuffle('x', 'x', 0) # mw x ms x mw
				tmp = T.exp(T.sum(T.log(T.switch(T.eq(tmp, 0), 1, tmp)), axis=2)) * m # mw x ms
				tmp = tmp * prior
				tmp = tmp / T.sum(tmp, axis=1).dimshuffle(0, 'x')
				w1 = T.switch(T.isnan(tmp), 0, tmp)
			return w1

		st = T.itensor3('st') # bs x len x ms
		pd = T.imatrix('wi') # bs x len
		mk = T.itensor3('mk') # bs x len x ms
		wv = T.dmatrix('wv') # bs x hd
		pe = T.imatrix('pe') # bs x mew
		pr = T.tensor3('pr') # bs x len x ms
		weights, _ = theano.scan(fn = to_weights, sequences = [st, mk, pd, pr]) # bs x mw x ms
		mask = T.ones_like(pd).astype(theano.config.floatX) # bs x len

		if self.is_lstm or self.is_gru:
			enc, _ = theano.scan(fn = to_vect, sequences = [st, mk, pd]) # bs x mw x hd
			enc = enc.astype(theano.config.floatX)
			fdef_emb = self.l_gru_emb.get_output_for([enc, mask]) # bs x hd
			if self.do_brnn:
				bdef_emb = self.l_bgru_emb.get_output_for([enc, mask])
				if self.concat:
					def_emb = T.concatenate([fdef_emb[:,-1,:], bdef_emb[:,0,:]], axis=1)
				else:
					def_emb = T.dot(fdef_emb[:, -1, :], self.params['Wf']) + \
						T.dot(bdef_emb[:, 0, :], self.params['Wb']) + \
						self.params['by'].dimshuffle('x', 0) # bs x hd
			else:
				def_emb = fdef_emb[:, -1, :]
		else:
			hid_inp = self.dwe[st, :]
			dat = T.sum(weights.dimshuffle(0, 1, 2, 'x') * hid_inp, axis=2)
			def_emb = T.sum(T.dot(dat, self.params['L']), axis = 1)

		self.encode = theano.function([st, mk, pd, pr], def_emb)
		self.get_weights = theano.function([st, mk, pd, pr], weights)

	def preproc_word(self, w, pos=None):
		if pos == 'j': pos = 'a'
		w = re.sub(r'[\$,\{\}\[\]\(\)`\'\":;!\?\.]', '', w).lower()
		w = re.sub(r'\-', '_', w) # hyphen -> underscore
		if w == 'an': w = 'a' # dirty hack....
		if w == 'oclock': w = 'o\'clock'
		if w.isdigit(): w = '<NUM>'
		wp = wn.morphy(w, pos=pos)
		if wp is None: wp = w
		return wp

	# 'sents' is a list of sentences
	def get_vector(self, sents, mode='v', token=None, pos='any', lem=None):
		mw = max([len(s.split()) for s in sents])
		s = np.ones((len(sents), mw, self.ms), dtype=np.int32) * -1
		m = np.zeros(s.shape, dtype=np.int32)
		p = np.ones((len(sents), mw), dtype=np.int32) * -1
		pr = np.ones((len(sents), mw, self.ms), dtype=np.float32)
		sp = self.sense_priors # no x ms
		for (si, sn) in enumerate(sents):
			s[si], m[si], p_tmp = self.to_indexes(sn, mw, token=token, pos=pos, lem=lem)
			p[si][0:len(p_tmp)] = p_tmp
			for i in range(mw):
				if i >= len(sn): break
				pwid = p[si][i]
				pr[si][i] = sp[pwid, :]
		if mode == 'v':
			return self.encode(s, m, p, pr)
		else:
			return self.get_weights(s, m, p, pr)

	# 'sent' is a single string
	# mw is the maximum number of words (if called from get_vector())
	# Setting token = w and pos = p will restrict the processing of 'w' to ones having POS tag 'p'
	def to_indexes(self, sent, mw = None, token = None, pos = None, lem = None):
		def same_pos(a, b):
			if a is None or b is None or a == b: return True
			if (a == 'a' or a == 's') and b == 'j': return True
			return False

		p_tmp = []
		sn = sent.split()
		if mw is None:
			mw = len(sn)
		s = np.ones((mw, self.ms), dtype=np.int32) * -1
		m = np.zeros(s.shape, dtype=np.int32)
		for (ind, w) in enumerate(sn):
			filt = (token is not None) and (w == token) #filter the token using pos
			if filt: _pos = pos
			else: _pos = None
			w = self.preproc_word(w, pos=_pos)
			if w not in self.aw2dw or len(self.aw2dw[w]) == 0:
				s[ind, 0] = self.dw['<UNK>']
				m[ind, 0] = 1.0
			else:
				l = min(10, len(self.aw2dw[w]))
				if filt:
					cands = []
					if lem is not None: w = lem
					for wp in self.aw2dw[w]:
						try:
							if same_pos(wn.synset(wp).pos(), pos) and wp.split('.')[0] == w:
								cands.append(wp)
						except:
							continue
					#cands = [wp for wp in self.aw2dw[w] if same_pos(wn.synset(wp).pos(), pos)]
					l = min(25, len(cands))
					s[ind][0:l] = [self.dw[wp] for wp in cands][0:25]
				else:
					s[ind][0:l] = [self.dw[wp] for wp in self.aw2dw[w][0:l]]
				m[ind][0:l] = np.ones((l,))
				if l == 0: pdb.set_trace()
			if w in self.aw:
				p_tmp.append(self.aw[w])
			else:
				p_tmp.append(0)
		return s, m, p_tmp

def sim(x, y):
	return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def main(args):
	if len(args) < 2:
		print 'Usage: python sent2vec.py <model_file_path> <wsi_input_file>'
		return
	foo = Sent2Vec(args[0], concat=False, wsi_path=args[1])
	foo.perform_wsi()
	'''sents = ['I went to bank yesterday', 'my sister is not a student', 'brother goes to school', \
		'my father went to bank', 'dog jumped over fox', 'yesterday was my bore birthday']
	v = foo.get_vector(sents)
	w = foo.get_vector(sents, mode = 'w')
	pdb.set_trace()
	for i in range(len(sents)):
		for j in range(i, len(sents)):
			if i == j: continue
			print '{} vs {}: {}'.format(i, j, sim(v[i], v[j]))'''

if __name__ == "__main__":
	main(sys.argv[1:])
