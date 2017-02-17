import math
import numpy as np
import chainer
from chainer import cuda, optimizers, serializers, serializers, utils
from chainer import Link, Chain, ChainList, Function, gradient_check, Variable
import chainer.functions as F
import chainer.links as L
from chainer.utils import walker_alias
import collections

from TextProcessor import BCCWJProcessor, CharProcessor

class Char2Vec(chainer.Chain):

	def __init__(self, char_num, char_dim):
		super(Char2Vec, self).__init__(
			embed = L.EmbedID(char_num, char_dim),
		)

	def __call__(self, focus_chars, context_chars, sampler, neg_size):
		loss_sum = None
		for i in range(len(focus_chars)):
			focus_char = Variable(np.array([focus_chars[i]], dtype=np.int32))
			context_char = context_chars[i]
			loss = F.negative_sampling(context_char, focus_char, self.embed.W, sampler, neg_size)
			loss_sum = loss if loss_sum is None else loss + loss
		return loss_sum


class Char2VecManager:

	def set_IDDict(self, toID_dic:dict, fromID_dic:dict):
		self.toID_dic = toID_dic
		self.fromID_dic = fromID_dic
		self.char_num = len(toID_dic)
	def set_countID(self, countID:dict):
		self.countID = countID
	def set_sentIDData(self, sentIDData:list):
		self.document = sentIDData
		self.doc_size = len(sentIDData)
	def set_Char2VecModel(self):
		char_dim = 50
		self.model = Char2Vec(self.char_num, char_dim)
		# serializers.save_npz("c2v-"+str(epoch)+".npz", self.model)

	def makeBatchSet(self, ids):
		window_size = 3
		focus_chars, context_chars = [], []
		for pos in ids:
			focus_char = self.document[pos]
			for i in range(1, window_size):
				p = pos - i
				if p >= 0:
					focus_chars.append(focus_char)
					context_char = self.document[p]
					cc_vec = Variable(np.array([context_char], dtype=np.int32))
					cc_vec = self.model.embed(cc_vec)
					context_chars.append(cc_vec)
				p = pos + i
				if p < self.doc_size:
					focus_chars.append(focus_char)
					context_char = self.document[p]
					cc_vec = Variable(np.array([context_char], dtype=np.int32))
					cc_vec = self.model.embed(cc_vec)
					context_chars.append(cc_vec)
		return [focus_chars, context_chars]

	def train(self):
		batch_size = 50
		neg_size = 5

		cs = [self.countID[char] for char in range(len(self.countID))]
		power = np.float32(0.75)
		p = np.array(cs, power.dtype)
		sampler = walker_alias.WalkerAlias(p)

		optimizer = optimizers.Adam()
		optimizer.setup(self.model)

		for epoch in range(30):
			print('epoch: {0}'.format(epoch))
			indexes = np.random.permutation(self.doc_size)
			for pos in range(0, self.doc_size, batch_size):
				ids = indexes[pos : (pos+batch_size) if (pos+batch_size) < self.doc_size else self.doc_size]
				focus_chars, context_chars = self.makeBatchSet(ids)
				self.model.zerograds()
				loss = self.model(focus_chars, context_chars, sampler.sample, neg_size)
				loss.backward()
				optimizer.update()
			print(math.floor(loss.data))
			serializers.save_npz("c2v-"+str(epoch)+".npz", self.model)

		with open('w2v.model', 'w') as f:
			w = self.model.embed.W.data
			for i in range(w.shape[0]):
				v = ' '.join(['%f' % v for v in w[i]])
				f.write('%s %s\n' % (self.fromID_dic[i], v))
































