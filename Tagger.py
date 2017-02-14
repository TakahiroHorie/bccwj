import re, jaconv

class Tagger:

	def __init__(self):
		self.__doc = []
		self.__EOSData = []

	def setDoc(self, docData:list):
		self.__doc = docData

	def makeEOSData(self):
		data = []
		for sent in self.__doc:
			for (i, char) in enumerate(sent):
				EOS = 1 if (i == len(sent)-1) else 0
				data.append([char, EOS])
		self.__EOSData = data

	def getEOSData(self):
		return self.__EOSData

	def printData(self, dataType:str):
		if dataType == "plain":
			for string in self.__doc:
				for char in string:
					print(char)
		if dataType == "withEOS":
			for char_EOS in self.__EOSData:
				print(char_EOS[0], char_EOS[1])