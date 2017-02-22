import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class Splitter(chainer.Chain):

	def __init__(self, char_num, char_dim, window_size:int=7):
		super(Splitter, self).__init__(
			embed = L.EmbedID(char_num, char_dim),
			W = L.Linear(char_dim * window_size, 2),
		)
		self.char_dim = char_dim

	def __call__(self, chars_EOS):
		loss = 0
		char_num, char_dim = self.embed.W.data.shape

		charsVec = self.charsIntoVector(chars_EOS[0])
		EOS = Variable(np.array([chars_EOS[1]], dtype=np.int32))
		loss = F.softmax_cross_entropy(self.W(charsVec), EOS)
		return loss

	def charsIntoVector(self, chars):
		x = []
		for char in chars:
			## testの未知語対応
			if isinstance(char, str):
				x.append(Variable(np.array([[0]*self.char_dim], dtype=np.float32)))

			if not isinstance(char, str):
				char_id = np.array([char], dtype=np.int32)
				x.append(self.embed(Variable(char_id)))
		return F.concat((x[0], x[1], x[2], x[3], x[4], x[5], x[6]), axis=1)

class SplitterManager:

	def __init__(self, char_num:int, char_dim:int, window_size:int=7):
		self.char_num = char_num + 1 ## PAD分
		self.char_dim = char_dim
		self.window_size = window_size

	def set_sentIDData(self, sentIDData:list):
		self.document = sentIDData
		self.doc_size = len(sentIDData)
	def set_EOSData(self, EOSData:list):
		self.EOSData = EOSData

	def train(self, epochs:int, resumed_model:str=None):
		add_tag = ["PAD"]

		self.model = Splitter(self.char_num, self.char_dim, self.window_size)
		if resumed_model != None: serializers.load_npz(resumed_model, self.model)

		optimizer = optimizers.Adam()
		optimizer.setup(self.model)

		for epoch in range(epochs):
			print("epoch:", epoch)
			chars = []
			for i in range(0, len(self.document)):
				for j in range(self.window_size):
					pos = i+(j-3)
					if(pos < 0 or len(self.document) <= pos):
						chars.append(self.char_num-1)
					else:
						chars.append(self.document[pos])
				chars_EOS = [chars, self.EOSData[i]]
				chars = []
				self.model.zerograds()
				loss = self.model(chars_EOS)
				loss.backward()
				optimizer.update()
				if i % 1000 == 0:
					print(str(i) + " / " + str(self.doc_size), "finished")
			outfile = "model/splitter-" + str(epoch+1) + ".npz"
			serializers.save_npz(outfile, self.model)
























