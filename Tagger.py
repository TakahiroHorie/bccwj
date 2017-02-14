import re, jaconv

class Tagger:

	def __init__(self):
		self.__doc = None
		self.__EOSData = None

	def setDoc(self, docData:list):
		self.__doc = docData

	def makeEOSData(self):
		data = []
		for sent in self.__doc:
			for char in sent:
				data.append(char)
			data.append("EOS")
		self.__EOSData = data

	def getEOSData(self): return self.__EOSData

	def printData(self):
		for string in self.__EOSData:
			print(string)