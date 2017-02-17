import re
import lxml.html
import collections

class BCCWJProcessor:

	def __init__(self, corpus:str):
		self.corpus = corpus
		self.sentData = []

	def makeSentData(self):
		inline_tag = ["ruby", "quote"]

		xml = open(self.corpus, "rb").read()
		root = lxml.html.fromstring(xml)
		sentences = root.xpath("//sentence")
		for sent in sentences:
			## quasiのsentenceは除外
			if len(sent.attrib) > 0:
				if sent.attrib["type"] == "quasi": continue

			self.sentData.append(cleanText(sent.text_content()))

	def get_sentData(self):
		return self.sentData

class CharProcessor:
	
	def __init__(self):
		self.toID_dic = {}
		self.fromID_dic = {}
		self.sentIDData = []
		self.countID = collections.Counter()

	def set_sentData(self, sentData:list):
		self.sentData = sentData

	def make_IDDict(self):
		for sentence in self.sentData:
			for char in sentence:
				if char not in self.toID_dic:
					index = len(self.toID_dic)
					self.toID_dic[char] = index
					self.fromID_dic[index] = char
				self.countID[self.toID_dic[char]] += 1

	def make_sentIDData(self):
		for sent in self.sentData:
			for (i, char) in enumerate(sent):
				char = self.toID_dic[char]
				self.sentIDData.append(char)

	def get_sentIDData(self):
		return self.sentIDData
	def get_countID(self):
		return self.countID
	def get_toIDDict(self):
		return self.toID_dic
	def get_fromIDDict(self):
		return self.fromID_dic



def cleanText(text:str):
	text = text.replace('\n','')
	text = text.replace('\r','')
	text = text.replace('　','')
	text = text.replace('。','')
	return text







