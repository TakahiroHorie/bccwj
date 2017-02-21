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
			W = L.Linear(char_dim*2, 2),
		)
		self.char_dim = char_dim

	def __call__(self, s):
		accum_loss, accum_acc = 0, 0
		char_num, char_dim = self.embed.W.data.shape
		h = Variable(np.zeros((1, char_dim*2), dtype=np.float32))
		for char in s:
			if(char[0][0]=="PAD"): continue

			char2gram_vec = self.charIntoVector(char[0])
			h = F.tanh(char2gram_vec + self.H(h))
			EOS = Variable(np.array([char[1]], dtype=np.int32))
			loss = F.softmax_cross_entropy(self.W(h), EOS)
			accum_loss += loss
			acc = F.accuracy(self.W(h), EOS)
			accum_acc += acc
		return accum_loss, accum_acc

	def charIntoVector(self, char_2gram):
		## testの未知語対応
		if isinstance(char_2gram[0], str):
			x_prev = Variable(np.array([[0]*self.char_dim], dtype=np.float32), volatile="auto")
		if isinstance(char_2gram[1], str):
			x = Variable(np.array([[0]*self.char_dim], dtype=np.float32), volatile="auto")

		if not isinstance(char_2gram[0], str):
			char_id_prev = np.array([char_2gram[0]], dtype=np.int32)
			x_prev = self.embed(Variable(char_id_prev, volatile="auto"))
		if not isinstance(char_2gram[1], str):
			char_id = np.array([char_2gram[1]], dtype=np.int32)
			x = self.embed(Variable(char_id, volatile="auto"))
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
			## batch: 0~50, 25~75, 50~100... 文字目
			for i in range(0, len(self.document), 25):
				for j in range(50):
					pos = i+j
					if(pos < len(self.document)): s.append([self.document[pos], self.EOSData[pos]])
				self.model.zerograds()
				loss, acc = self.model(s)
				print(loss.data / 50, acc.data / 50)
				loss.backward()
				if(len(s) > 30): loss.unchain_backward()
				optimizer.update()
				s = []
			print(epoch)
			outfile = "model/sce2-splitter-" + str(epoch) + ".npz"
			serializers.save_npz(outfile, self.model)
























