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
		return F.concat(x, axis=1)

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
	def set_model(self, trained_model:str=None):
		self.model = Splitter(self.char_num, self.char_dim, self.window_size)
		if trained_model != None: serializers.load_npz(trained_model, self.model)

	def train(self, epochs:int):
		optimizer = optimizers.Adam()
		optimizer.setup(self.model)

		for epoch in range(epochs):
			print("epoch:", epoch)
			for i in range(0, len(self.document)):
				chars_EOS = [self.sliceByWindow(i), self.EOSData[i]]

				self.model.zerograds()
				loss = self.model(chars_EOS)
				loss.backward()
				optimizer.update()
				if i % 1000 == 0:
					print(str(i) + " / " + str(self.doc_size), "finished")
			outfile = "model/splitter-" + str(epoch+1) + ".npz"
			serializers.save_npz(outfile, self.model)

	def sliceByWindow(self, i:int):
		chars = []
		for j in range(self.window_size):
			pos = i+(j-3)
			charID = self.char_num-1 if(pos<0 or self.doc_size<=pos) else self.document[pos]
			chars.append(charID)
		return chars
























