import re
import lxml.html

class BCCWJProcessor:
	def __init__(self, corpus:str):
		self.corpus = corpus
		self.__sentData = []

	def makeSentData(self):
		inline_tag = ["ruby", "quote"]

		xml = open(self.corpus, "rb").read()
		root = lxml.html.fromstring(xml)
		sentences = root.xpath("//sentence")
		for sent in sentences:
			## quasiのsentenceは除外
			if len(sent.attrib) > 0:
				if sent.attrib["type"] == "quasi": continue

			self.__sentData.append(cleanText(sent.text_content()))

	def getSentData(self):
		return self.__sentData

class CharProcessor:
	def __init__(self):
		self.sentData = []
		self.toID_dic = {}
		self.fromID_dic = {}

	def setSentData(self, sentData:list):
		self.sentData = sentData

	def makeIDDict(self):
		for sentence in self.sentData:
			for char in sentence:
				if char not in self.toID_dic:
					index = len(self.toID_dic)
					self.toID_dic[char] = index
					self.fromID_dic[index] = char

	def get_toIDDict(self):
		return self.toID_dic
	def get_fromIDDict(self):
		return self.fromID_dic

	def convertChar2ID(char:str):
		return self.toID_dic[char]
	def convertID2Char(id:int):
		return self.fromID_dic[id]



def cleanText(text:str):
	text = text.replace('\n','')
	text = text.replace('\r','')
	text = text.replace('　','')
	text = text.replace('。','')
	return text







