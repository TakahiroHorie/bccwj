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
			H = L.Linear(char_dim*2, char_dim*2),
			W = L.Linear(char_dim*2, 1),
		)

	def __call__(self, s):
		accum_loss = 0
		char_num, char_dim = self.embed.W.data.shape
		h = Variable(np.zeros((1, char_dim*2), dtype=np.float32))
		for char in s:
			if(char[0][0]=="PAD"): continue

			char2gram_vec = self.charIntoVector(char[0])
			h = F.tanh(char2gram_vec + self.H(h))
			EOS = Variable(np.array([char[1]], dtype=np.float32))
			loss = F.mean_squared_error(self.W(h)[0], EOS)
			accum_loss += loss
		return accum_loss

	def charIntoVector(self, char_2gram):
		char_id_prev = np.array([char_2gram[0]], dtype=np.int32)
		char_id      = np.array([char_2gram[1]], dtype=np.int32)
		x_prev = self.embed(Variable(char_id_prev, volatile="auto"))
		x      = self.embed(Variable(char_id, volatile="auto"))
		return F.concat((x_prev, x), axis=1)

class SplitterManager:

	def __init__(self, char_num:int, char_dim:int):
		self.char_num = char_num
		self.char_dim = char_dim

	def set_sentIDData_bi(self, sentIDData_bi:list):
		self.document = sentIDData_bi
	def set_EOSData(self, EOSData:list):
		self.EOSData = EOSData

	def train(self, epochs:int, trained_c2v:str):
		C2Vmodel = Char2Vec(self.char_num, self.char_dim)
		serializers.load_npz(trained_c2v, C2Vmodel)
		trained_C2V = Link.copy(C2Vmodel.embed)

		self.model = Splitter(self.char_num, self.char_dim)
		self.model.add_link("embed", trained_C2V)

		optimizer = optimizers.Adam()
		optimizer.setup(self.model)

		for epoch in range(epochs):
			s = []
			total_loss = 0
			## batch: 0~50, 25~75, 50~100... 文字目
			for i in range(0, len(self.document), 25):
				for j in range(50):
					pos = i+j
					if(pos < len(self.document)): s.append([self.document[pos], self.EOSData[pos]])
				self.model.zerograds()
				loss = self.model(s)
				loss.backward()
				total_loss += loss.data
				if(len(s) > 30): loss.unchain_backward()
				optimizer.update()
				s = []
				print(i, "/", len(self.document)," finished")
			print(epoch, total_loss)

			outfile = "splitter-" + str(epoch) + ".model"
			serializers.save_npz(outfile, self.model)










