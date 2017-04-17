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
from lasagne.layers import  InputLayer, GRULayer, Gate, get_all_params, LSTMLayer
from lasagne.updates import sgd, adam, adadelta, nesterov_momentum
from theano.ifelse import ifelse
from theano.compile.debugmode import DebugMode
from collections import OrderedDict
import lasagne.nonlinearities as nonlinearities
#from theano import ProfileMode

class Trainer:
	def __init__(self, in_path, out_path, resume_tag=None, tr_type='chain', sg=True, sgd=False, pretrain_dwe=True):
		self.hd = 600 # hidden vector dimension (for the GRUs)
		self.td = 300 # target dimension of the word vectors
		self.bs = 64 # minibatch size
		self.lr = 5e-2# learning rate
		self.gc = 10  # gradient clipping
		self.lam = 0.8 # trade-off coeff.
		self.neg_per_word = 10
		self.no_alt = False # No alternating training
		self.do_fixedpoint = True
		self.hinge_cost = True #False 
		self.do_brnn = True
		self.do_ortho = False
		self.do_rw = True
		self.init_dwe = True #False 
		self.reg_alpha = False #True
		self.use_prior = False #True
		self.recompute_prior = False
		self.num_epoch = 500
		self.tic = 100
		self.neg_weight = 1
		self.save_path = out_path
		self.params = {}
		self.use_skipgram = sg
		self.tr_type = tr_type
		self.in_path = in_path
		self.do_sgd = sgd
		self.tag = time.strftime('%Y%m%d%H%M')# + '-' + self.tr_type
		# a short description of the current setting (to be displayed during training
		#self.prog_id = 'probabilistic_cost_sgd_no_ccost' 
		self.prog_id = 'lr=0.05,nlm,rw0.8,1000x,hnge,noprior'
		self.pretrain_dwe = pretrain_dwe
		self.read_data(resume_tag)

		print 'Compiling trainer...'
		eval('self.build_{}_trainer()'.format(tr_type))
		#pdb.set_trace()
				
	def read_data(self, resume_tag):
		print 'Loading training data into memory...'
		with open(os.path.join(self.in_path, 'dat.pkl'), 'rb') as f:
			d = pk.load(f)
		with open(os.path.join(self.in_path, 'supp.pkl'), 'rb') as f:
			supp = pk.load(f)
		self.id2dw = supp['id2dw']
		self.dw2id = {}
		for (i, w) in enumerate(self.id2dw):
			self.dw2id[w] = i

		self.dat = self.shared_var('def', None, d['def']) # indexes... nw x mw x ms
		self.pd = self.shared_var('pd', None, d['def_plain']) # indexes... nw x mw
		self.ex = self.shared_var('ex', None, d['ex']) # plain example indexs... nw x mew
		self.dmask = self.shared_var('dmask', None, d['dmask']) # nw x mdw x ms
		self.nw, self.mw, self.ms = d['def'].shape
		self.no = len(d['aw']) # number of ambiguous words (i.e., observations)
		maxid = np.max(d['def'])
		self.true_nw = self.nw
		if maxid >= self.nw: 
			self.nw = maxid + 1
		hd = self.hd
		td = self.td
		nw = self.nw
		no = self.no		
		dtype = theano.config.floatX
	
		def dirichlet(a):
			g = [nr.gamma(a1, 1) for a1 in a]
			return [x / sum(g) for x in g]
		sprior = np.zeros(d['def'].shape) #nw x mw x ms
		if not self.use_prior: sprior = np.ones(d['def'].shape)
		conc_par = [1] * self.ms
		conc_par[0:2] = [4,2] # Skewed concentration parameters
		wmask = np.zeros(d['dmask'].shape[0:2]) # only the first two dimensions (word-level mask)
		for i in range(len(d['dmask'])): # for each of nw words
			for j in range(len(d['dmask'][i])): # for each of mw words	
				wd = d['dmask'][i, j] # ms
				wmask[i, j] = max(0, wd[0])
				if not self.use_prior or wd[0] <= 0: continue
				if 'spriors' in d and not self.recompute_prior:
					pw = d['def_plain'][i, j]
					sprior[i, j] = d['spriors'][pw, :]
				else:
					numsense = np.where(wd > 0)[0].shape[0]
					sprior[i, j, 0:numsense] = dirichlet(conc_par[0:numsense])
		self.wmask = self.shared_var('wmask', None, wmask, dtype)
		self.sense_priors = self.shared_var('sprior', init_dat = sprior, dtype = dtype)
		self.l_mask = InputLayer((None, None), trainable=False)
		self.idf = theano.shared(supp['idf'].astype(dtype), borrow=True) #self.shared_var('idf', init_dat = supp['idf'], dtype=dtype)

		if resume_tag: # continue pre-trained model...
			self.prog_id += '_resume'
			self.load_model(resume_tag) # actually, this should probably wait...
		else:
			self.params['dwe'] = self.shared_var('dwe', (nw, td), dtype=dtype) # disambiguated word embeddings
			#self.params['L0'] = self.shared_var('L0', (td, no), dtype=dtype) # 
			self.params['L1'] = self.shared_var('L1', shape=(hd, td), dtype=dtype) # 
			self.params['L'] = self.shared_var('L', shape=(td, hd), dtype=dtype)
			#self.params['Wt'] = self.shared_var('Wt', init_dat=nr.uniform(-500, 500, (td, )), dtype=dtype)
			if not self.hinge_cost:
				self.params['L2'] = self.shared_var('L2', shape=(td, td), dtype=dtype) # 
				self.params['B'] = self.shared_var('B', shape=(td,), dtype=dtype)
				self.params['B2'] = self.shared_var('B2', shape=(td,), dtype=dtype)
				
		dwe_key = 'no_init_dwe' # change this init_dwe2 for wordvec initialization
		if self.init_dwe and dwe_key in supp:
			#self.params['dwe'] = self.do_init_dwe(supp[dwe_key].astype(dtype))
			self.base = self.do_init_dwe(supp[dwe_key].astype(dtype))
			if resume_tag is None:
				self.params['dwe'] = self.shared_var('dwe', (nw, td), dtype=dtype)
		elif resume_tag is None:
			self.params['dwe'] = self.shared_var('dwe', (nw, td), dtype=dtype)
			self.base = theano.shared(self.params['dwe'].get_value(), borrow = True)
	
	def init_L(self, L):
		where0 = np.unique(np.where(L == 0)[0])
		for i in where0:
			L[i] = nr.randn(L.shape[1],)
		return self.shared_var('L0', init_dat=L.T, dtype=theano.config.floatX)
	
	def do_init_dwe(self, dat): # l_gru_emb and L0 must be initialized by now
		print 'Initializing dwe...'
		dtype = theano.config.floatX
		dwe = np.zeros((self.nw, self.td), dtype=dtype)
		dwe[0:dat.shape[0], :] = dat
		if self.nw > dat.shape[0]:
			for ind in range(dat.shape[0], self.nw):
				dwe[ind] = nr.uniform(-0.1, 0.1, (self.td,))
			
		return self.shared_var('dwe', init_dat=dwe, dtype=dtype)

	def load_model(self, tag):
		#with open(os.path.join(self.save_path, tag), 'rb') as f:
		with open(tag, 'rb') as f:
			par = pk.load(f)

		print 'Loading pre-trained model...'
		self.params = par['params']
		self.lr = par['lr']
		self.tr_type = par['tr_type']
		self.do_fixedpoint = par['do_fixedpoint']
		self.tag += '-resume'

	def save_model(self):
		model_save_name = self.tag + '-model.pkl'

		save = {'params': self.params, 'prog_id': self.prog_id, 'lr': self.lr, 'tr_type': self.tr_type, \
			'data_path': self.in_path, 'do_brnn': self.do_brnn, 'do_fixedpoint': self.do_fixedpoint} #, \
			#'spriors': self.sense_priors} #, 'gru': self.l_gru_emb}
		if not os.path.exists(self.save_path):
			os.mkdir(self.save_path)
		with open(os.path.join(self.save_path, model_save_name), 'wb') as f:
			pk.dump(save, f, protocol=pk.HIGHEST_PROTOCOL)

		print 'Saved to ' + os.path.join(self.save_path, model_save_name)

	def shared_var(self, n, shape=None, init_dat=None, dtype='int32', ortho_init=False):
		if ortho_init:
			if shape is None: return None
			tmp = nr.randn(*shape)
			value, _, _2 = np.linalg.svd(tmp)
			#value = value * 0.1
		elif init_dat is not None:
			value = init_dat
			#shape = init_dat.shape
		elif shape is not None:
			value = nr.uniform(-0.1, 0.1, shape)
		else:
			return None
		return theano.shared(value.astype(dtype), name=n, borrow=True)
		
	def build_chain_trainer(self):
		bs = self.bs
		td = self.td

		wi = T.ivector('wi') # bs (disamb. word indices)
		nwi = T.ivector('nwi') # negative samples
		lr = T.dscalar('lr').astype(theano.config.floatX) # learning rate
		lam = T.dscalar('lam').astype(theano.config.floatX)
		L = self.params['L']
		L1 = self.params['L1'] # hd x td
		#Wt = self.params['Wt']
		if not self.hinge_cost:
			L2 = self.params['L2']
			B = self.params['B'] # td
			B2 = self.params['B2']

		dwe = self.params['dwe']
		df = self.dat[wi, :] #T.itensor3('df')# bs x mw x ms
		pr = self.sense_priors[wi, :] # bs x mw x ms
		mk = self.dmask[wi, :] #T.itensor3('mk')# bs x mw x ms
		pd = self.pd[wi, :] #T.imatrix('pd') # bs x mdw (plain definition sentence)
		pe = self.ex[wi, :] # plain example sentences bs x mew
		dw = dwe[wi, :] # bs x td
		msk = self.wmask[wi, :].dimshuffle(0, 1, 'x') # bs x mw x 1
		ndw = dwe[nwi, :] # negative words
		
		def to_vect(d, m, p):
			hid_inp = dwe[d, :] # mw x ms x hd
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

		def to_weight(d, m, p, prior):
			logit = T.tensordot(dwe[d, :], dwe.T, axes=1)[:,:,d] # mw x ms x mw x ms
			cnt = T.sum(m, axis=1).dimshuffle('x', 'x', 0) # 1 x 1 x mw
			logit = T.sum(logit * m.dimshuffle('x', 'x', 0, 1), axis=3) / cnt # mw x ms x mw
			logit = T.exp(10*T.switch(T.isnan(logit), 0, logit)) # mw x ms x mw
			logit = T.prod(logit, axis=2) * prior # mw x ms
			sm = T.sum(logit * m, axis=1, keepdims=True) # mw x 1
			#mask = T.switch(T.lt(p, 0), 0, 1).dimshuffle(0, 'x') # 
			logit = (logit * m) / sm # mw x ms
			return T.switch(T.or_(T.isnan(logit), T.isinf(logit)), 0, logit)

		'''def to_weight(d, m, p, prior):
			A = dwe[d, :] # mw x ms x td
			#tmp = T.tensordot(T.dot(A, Wt), A.T, axes=1) # mw x ms x ms x mw
			#B = A * Wt.dimshuffle('x', 'x', 0) # 'diag' setting
			#tmp = T.tensordot(B, B.T, axes = 1)
			tmp = T.tensordot(A, A.T, axes = 1) # 'iden' setting
			tmp = T.exp(1000 * tmp.dimshuffle(0, 1, 3, 2)) # mw x ms x mw x ms
			tmp = tmp * m.dimshuffle('x', 'x', 0, 1)
			nrm = T.sum(tmp, axis=3)
			tmp = tmp / nrm.dimshuffle(0, 1, 2, 'x')
			tmp = T.switch(T.isnan(tmp), 0, tmp)
			mk = T.switch(T.lt(p, 0), 0, 1) # mw: word-level mask (different mask from m)
			tmp = T.max(tmp, axis=3) * mk.dimshuffle('x', 'x', 0) # mw x ms x mw
			tmp = T.exp(T.sum(T.log(T.switch(T.eq(tmp, 0), 1, tmp)), axis=2)) * m # mw x ms
			tmp = tmp * prior
			tmp = tmp / T.sum(tmp, axis=1).dimshuffle(0, 'x')
			return T.switch(T.isnan(tmp), 0, tmp)'''

		def cosim(x, y):
			return T.mean(T.sum(x * y, axis=1) / (x.norm(2, axis=1) * y.norm(2, axis=1)))
			
		#dat, _ = theano.scan(fn=to_vect, sequences=[df, mk, pd]) # bs x mw x td
		#ndat, _ = theano.scan(fn=to_vect_tmp, sequences=[ndf, nmk, npd]) # bs x mw x td
		weights, _ = theano.scan(fn=to_weight, sequences=[df, mk, pd, pr]) # bs x mw x ms
		hid_inp = dwe[df, :] # bs x mw x ms x td
		dat = T.sum(weights.dimshuffle(0, 1, 2, 'x') * hid_inp, axis=2) # bs x mw x td '''
		inp = dat.astype(theano.config.floatX)
		def_emb = T.sum(T.dot(inp, L) * msk, axis=1) # bs x hd
		#neg_inp = ndat.astype(theano.config.floatX)
		#def_emb = get_sentence(inp, msk) # bs x hd
	
		#neg_def_emb = get_sentence(neg_inp, neg_msk)

		#w_cost = T.sum((def_emb - dw) ** 2)
		#w_neg_cost = T.sum((def_emb - ndw) ** 2) 
		if self.hinge_cost:
			def_emb = T.dot(def_emb, L1)
			w_cost = -cosim(def_emb, dw)
			rep = nwi.shape[0] / wi.shape[0] # b/c there are more negative samples than pos.
			de = T.extra_ops.repeat(def_emb, rep, axis=0)
			w_neg_cost = -cosim(de, ndw)
			cost = T.mean(T.maximum(0, 0.01 + w_cost - w_neg_cost)) # hingeloss
		else:
			regress = T.dot(T.nnet.sigmoid(T.dot(def_emb, L1) + B), L2) + B2 # bs x td
			cost = T.mean((regress - dw) ** 2) + 0.01 * T.sum(abs(L2)) # only regularize the last

		if self.reg_alpha:
			cost += 0.1 * T.sum(abs(weights))
		#w_cost = get_word_probs(def_emb, wi, L1) #dwe.T) # dwe instead of L1
		#w_neg_cost = get_word_probs(def_emb, nwi, L1) #dwe.T) # dwe instead of L1

		#c_cost = -get_context_probs(def_emb, pe, L0) # negative of the likelihood
		#c_neg_cost = -get_context_probs(def_emb, npe, L0)

		#all_params = [self.params[k] for k in self.params if k != 'dwe' and not k.startswith('L')]
		all_params = [self.params[k] for k in self.params if k != 'dwe']
		#L_params = [L0]

		'''Copy of the same function in Lasagne (with minor changes)'''
		def apply_nesterov_momentum(ups, mom, shape=None):
			params = ups.keys()
			ups = OrderedDict(ups)
			if shape is None: shape = [p.get_value(borrow=True).shape for p in params]

			for (param, shp) in zip(params, shape):
				velocity = theano.shared(np.zeros(shp, dtype=theano.config.floatX),
										 broadcastable=param.broadcastable)
				x = mom * velocity + ups[param] - param
				ups[velocity] = x
				ups[param] = mom * x + ups[param]
			return ups

		dwe_params = [dw, ndw]
		if self.do_sgd:
			grads = T.grad(cost, all_params)
			updates = OrderedDict()
			for (p, g) in zip(all_params, grads):
				updates[p] = p - lr * g
			apply_nesterov_momentum(updates, mom=0.9)
			if self.no_alt or not self.do_fixedpoint: 
				dgrads = T.grad(cost, dwe_params)
				dwe_update = OrderedDict()
				for (p, g) in zip(dwe_params, dgrads):				
					dwe_update[p] = p - lr * g
					foo = lr * g
				apply_nesterov_momentum(dwe_update, mom=0.9, shape=[(bs, td), (bs, td)])
		else:
			updates = adadelta(cost, all_params, learning_rate = lr)
			#L_update = adadelta(cost, L_params, learning_rate = lr)
			if self.no_alt or not self.do_fixedpoint: 
				dwe_update = adadelta(cost, dwe_params, learning_rate = lr)
		
		if not self.no_alt and self.do_fixedpoint:# because no alternating training means optimization
			if self.do_rw:
				#posword = self.base[wi] + 0.3 * def_emb #0.3 * ((1 - self.lam) * def_emb + self.lam * dw)
				idf = self.idf[wi].dimshuffle(0, 1, 'x') # bs x mw x 1  (dat is bs x mw x hd)
				rw_term = T.sum(dat * idf , axis=1) # bs x hd
				disc_fact = 0.9
				if self.init_dwe:
					#posword = disc_fact * rw_term # + self.base[wi] # truerw
					posword = (1 - lam) * dw + lam * disc_fact * rw_term # + self.base[wi] # truerw
				else:
					base = self.lam * def_emb + (1 - self.lam) * dw
					posword = base + disc_fact * rw_term
				word_update = T.set_subtensor(dw, posword.astype(theano.config.floatX))
				dwe_update = {dwe: word_update}
				dwe_ret = T.max(T.abs_(posword - dw)) # max-norm of the increment 
			else:
				posword = (1 - self.lam) * def_emb + self.lam * dw
				word_update = T.set_subtensor(dw, posword - self.lam * ndw)
				dwe_update = {dwe: word_update}
				dwe_ret = word_update
		else: #elif not self.do_fixedpoint or self.no_alt:
			word_update = dwe_update[dw]
			word_update = T.set_subtensor(dw, word_update)
			nword_update = dwe_update[ndw]
			word_update = T.set_subtensor(word_update[nwi, :], nword_update)
			dwe_update = {dwe: word_update} #T.set_subtensor(dw, word_update)
			if self.no_alt:
				updates.update({dwe: word_update})
			dwe_ret = word_update
			#updates.update({dwe: dwe_update[dwe]}) #word_update})
		#updates.update({dwe: word_update})

		self.train_step = theano.function([wi, nwi, lr], [cost, weights], updates=updates) 
		if not self.no_alt:
			self.dwe_train_step = theano.function([wi, nwi, lam], [cost, dwe_ret, weights], updates=dwe_update)
		#self.L_train_step = theano.function([wi, nwi, lr], [cost], updates = L_update)

	def train(self, num_epoch=1000):
		print 'Training...'
		indexes = range(self.true_nw) 
		cur_ep = 0
		cur_lr = self.lr
		num_consec_train = 5 # number of consecutive epochs per parameter (W or L0)
		num_neg_samples = self.neg_per_word # number of negative samples per word
		mode = ['alph', 'rw']
		tol = 1e-3
		next_schedule = 5
		cur_mode = 1
		dwe_up_cnt = 0

		while cur_ep < num_epoch:
			nr.shuffle(indexes)
			neg_ind = indexes[self.bs * num_neg_samples:] + indexes[:self.bs * num_neg_samples]
			cost = 0
			totTime = 0
			max_diff = -np.inf
			cur_lam = self.lam ** (dwe_up_cnt + 1)
			#cur_mode = (cur_ep // num_consec_train) % len(mode)
			for cur_iter in np.arange(0, self.true_nw, self.bs):
				cur_ind = indexes[cur_iter : min(cur_iter + self.bs, self.true_nw - 1)]
				cur_neg_ind = np.tile(neg_ind[cur_iter : min(cur_iter + self.bs, self.true_nw - 1)], \
					num_neg_samples).astype(np.int32)
				#neg_ind[cur_iter : min(cur_iter + self.bs * num_neg_samples, self.true_nw - 1)]
				st = time.time()
				#test_ind = [5632, 3211, 432, 564, 9983, 10002]
				#test_neg_ind = [562, 211, 42, 64, 993, 102]
				if self.no_alt or cur_mode == 0:
					ret = self.train_step(cur_ind, cur_neg_ind, cur_lr)
					tc, w = ret[0], ret[1]
					if np.isnan(tc) or np.isnan(ret[-1]).any(): #or np.isnan(ret[0]):
						pdb.set_trace()
				elif not self.no_alt and cur_mode == 1:
					ret = self.dwe_train_step(cur_ind, cur_neg_ind, cur_lam)
					tc, diff = ret[0], ret[1]
					max_diff = max(max_diff, float(diff))
					dwe_up_cnt += 1
					if np.isnan(tc):
						pdb.set_trace()						
				et = time.time()
				totTime += (et - st)
				cost += tc
				if (cur_iter / self.bs) % self.tic == 0:
					print '[{0}: {1}] {2}, cost = {3:.3f}, max_diff = {5:.3f} ({4:.1f} sec.)'.format(\
						mode[cur_mode], self.prog_id, cur_iter / self.bs, cost, totTime, max_diff)
					totTime = 0
			if cur_mode == 1 and max_diff < tol:
				cur_mode = 0
				max_diff = -np.inf
				next_schedule = cur_ep + num_consec_train
			elif cur_mode == 0 and next_schedule == cur_ep:
				cur_mode = 1 # now perform the RW updates on dwe's
				dwe_up_cnt = 0
			#elif cur_mode == 1 and cur_ep == 0:
			#	cur_mode = 0 # run dwe update only once at the beginning and proceed normally

			print '\t*** Epoch {}, cost = {} ***'.format(cur_ep, cost)
			self.test()
			self.test(w = 'instrument.n.01')
			self.save_model()
			cur_ep += 1

	def test(self, w = 'apple.n.01'):
		if w not in self.dw2id:
			w = self.id2dw[1233]

		def sim(v1, v2):
			return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

		dwe = self.params['dwe'].get_value()
		v = dwe[self.dw2id[w]]
		nn = [(self.id2dw[ind], sim(v, wd)) for (ind, wd) in enumerate(dwe)]
		nn.sort(key = lambda e: e[1], reverse=True)
		print '10-NN of {}: {}'.format(w, str(nn[0:10]))

def main(args):
	if len(args) < 2:
		print 'Usage: python train.py <input_dir_path> <output_dir> [trained_model_file_name]'
		return
	if len(args) >= 3:
		tr = Trainer(args[0], args[1], resume_tag = args[2])
	else: tr = Trainer(args[0], args[1], sgd=True, pretrain_dwe=True)
	tr.train()

if __name__ == "__main__":
	#profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
	main(sys.argv[1:])
	#profmode.print_summary()
