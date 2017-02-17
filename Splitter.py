import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from Char2Vec import Char2Vec

class Splitter(chainer.Chain):

	def __init__(self, char_num, char_dim):
		super(Splitter, self).__init__(
			H = L.Linear(char_dim, char_dim),
			W = L.Linear(char_dim, 1),
		)

	## train
	def __call__(self, s):
		accum_loss = None
		char_num, char_dim = self.embed.W.data.shape
		h = Variable(np.zeros((1, char_dim), dtype=np.float32))

		for char in s:
			char_id = np.array([char[0]], dtype=np.int32)
			EOS = Variable(np.array([char[1]], dtype=np.float32))
			x = self.embed(Variable(char_id, volatile="auto"))
			h = F.tanh(x + self.H(h))
			loss = F.mean_squared_error(self.W(h)[0], EOS)
			accum_loss = loss if accum_loss is None else accum_loss + loss
		return accum_loss


class SplitterManager:

	def __init__(self, char_num, char_dim):
		self.char_num = char_num
		self.char_dim = char_dim

	def set_sentIDData(self, sentIDData:list):
		self.document = sentIDData
	def set_EOSData(self, EOSData:list):
		self.EOSData = EOSData

	def train(self):
		C2Vmodel = Char2Vec(self.char_num, self.char_dim)
		serializers.load_npz("c2v-9.npz", C2Vmodel)
		trained_C2V = Link.copy(C2Vmodel.embed)

		self.model = Splitter(self.char_num, self.char_dim)
		self.model.add_link("embed", trained_C2V)

		optimizer = optimizers.Adam()
		optimizer.setup(self.model)

		for epoch in range(5):
			s = []
			for pos in range(len(self.document)):
				char_id = self.document[pos]
				EOS = self.EOSData[pos]
				s.append([char_id, EOS])
				if (EOS == 1):
					self.model.zerograds()
					loss = self.model(s)
					print(loss.data)
					loss.backward()
					optimizer.update()
					s = []
				if (pos % 100 == 0):
					print(pos, "/", len(self.document)," finished")
			outfile = "splitter-" + str(epoch) + ".model"
			serializers.save_npz(outfile, self.model)










