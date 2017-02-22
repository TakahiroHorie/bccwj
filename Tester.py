import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from Splitter import Splitter

class Tester:

	def __init__(self, test_data:list , ans_data:list, toIDDict:dict, fromIDDict:dict):
		self.test_data = test_data
		self.ans_data = ans_data
		self.fromIDDict = fromIDDict
		self.char_num = len(toIDDict)
		self.char_dim = 50

	def set_model(self, trained_splitter:str):
		self.model = Splitter(self.char_num+1, self.char_dim)
		serializers.load_npz(trained_splitter, self.model)

	def splitTest(self):
		total = len(self.test_data)
		correct = 0
		TP, FP, FN = 0, 0, 0
		chars = []
		s = ""
		for i in range(len(self.test_data)):
			for j in range(7):
				pos = i+(j-3)
				if(pos < 0 or len(self.test_data) <= pos):
					chars.append(self.char_num-1)
				else:
					chars.append(self.test_data[pos])
			chars_EOS = [chars, self.ans_data[i]]
			chars = []

			charsVec = self.model.charsIntoVector(chars_EOS[0])
			y = self.model.W(charsVec)
			ans = chars_EOS[1]
			pred = np.argmax(y.data)

			if(ans == pred): correct += 1
			if(ans == 1 and pred == 1): TP += 1
			if(ans == 1 and pred == 0): FN += 1
			if(ans == 0 and pred == 1): FP += 1

			# s += self.fromIDDict[self.test_data[i]] if not isinstance(self.test_data[i], str) else str(self.test_data[i])
			# if(pred == 1):
			# 	print(s)
			# 	s = ""

		accuracy = correct / total
		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		F_score = (2*recall*precision) / (recall + precision)
		print("F-score:", F_score, "Accuracy-rate:", accuracy)






		