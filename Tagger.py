import re, jaconv
from TextProcessor import CharProcessor

class Tagger:

	def __init__(self):
		self.sentData = []
		self.EOSData = []

	def setSentData(self, sentData:list):
		self.sentData = sentData

	def makeEOSData(self, byID:bool):
		data = []
		for sent in self.sentData:
			for (i, char) in enumerate(sent):
				EOS = 1 if (i == len(sent)-1) else 0

				if (byID): char = CharProcessor.convertChar2ID(char)
				data.append([char, EOS])
		self.EOSData = data

	def getEOSData(self):
		return self.EOSData

	def printData(self, dataType:str):
		if dataType == "plain":
			for char_EOS in self.EOSData:
				print(char_EOS[0])
		if dataType == "withEOS":
			for char_EOS in self.EOSData:
				print(char_EOS[0], char_EOS[1])