import re, jaconv
from TextProcessor import CharProcessor

class Tagger:

	def init(self):
		self.sentData = []
		self.toID_dic = {}
		self.fromID_dic = {}
		self.EOSData = []

	def set_SentData(self, sentData:list):
		self.sentData = sentData
	def set_IDDict(self, toID_dic:dict, fromID_dic:dict):
		self.toID_dic = toID_dic
		self.fromID_dic = fromID_dic

	def makeEOSData(self, byID:bool):
		data = []
		for sent in self.sentData:
			for (i, char) in enumerate(sent):
				EOS = 1 if (i == len(sent)-1) else 0

				if (byID): char = self.toID_dic[char]
				data.append([char, EOS])
		self.EOSData = data

	def get_EOSData(self):
		return self.EOSData

	def printData(self, dataType:str):
		if dataType == "plain":
			for char_EOS in self.EOSData:
				print(char_EOS[0])
		if dataType == "withEOS":
			for char_EOS in self.EOSData:
				print(char_EOS[0], char_EOS[1])