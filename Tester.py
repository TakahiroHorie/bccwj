import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from Splitter import SplitterManager

class Tester:

	def __init__(self, test_data:list , ans_data:list, toIDDict:dict, fromIDDict:dict):
		self.test_data = test_data
		self.ans_data = ans_data
		self.fromIDDict = fromIDDict
		self.char_num = len(toIDDict)

	def set_model(self, trained_splitter:str):
		char_dim = 50
		self.SplManager = SplitterManager(self.char_num, char_dim)
		self.SplManager.set_sentIDData(self.test_data)
		self.SplManager.set_EOSData(self.ans_data)
		self.SplManager.set_model(trained_splitter)

	def splitTest(self):
		total, correct = len(self.test_data), 0
		TP, FP, FN = 0, 0, 0
		s = ""
		for i in range(len(self.test_data)):
			chars_EOS = [self.SplManager.sliceByWindow(i), self.ans_data[i]]

			charsVec = self.SplManager.model.charsIntoVector(chars_EOS[0])
			y = self.SplManager.model.W(charsVec)
			ans = chars_EOS[1]
			pred = np.argmax(y.data)

			if(ans == pred): correct += 1
			if(ans == 1 and pred == 1): TP += 1
			if(ans == 1 and pred == 0): FN += 1
			if(ans == 0 and pred == 1): FP += 1

			s += self.fromIDDict[self.test_data[i]] if not isinstance(self.test_data[i], str) else str(self.test_data[i])
			if(pred == 1):
				print(s)
				s = ""

		accuracy = correct / total
		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		F_score = (2*recall*precision) / (recall+precision)
		print("F-score:", F_score, "Accuracy-rate:", accuracy)











		