import re
import lxml.html
import collections

class BCCWJProcessor:

	def __init__(self, corpus:str):
		self.corpus = corpus
		self.sentData = []

	def makeSentData(self):
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
		self.sentIDData_bi = []
		self.countID = collections.Counter()

	def set_toIDDict(self, toID_dic:list):
		self.toID_dic = toID_dic
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
			for char in sent:
				char = self.toID_dic[char]
				self.sentIDData.append(char)

	def make_sentIDData_bi(self):
		PAD_flg = 1
		for sent in self.sentData:
			for i in range(len(sent)):
				if (PAD_flg):
					char_prev = "PAD"
					char = self.toID_dic[sent[i]]
					PAD_flg = 0
				else:
					if sent[i-1] in self.toID_dic:
						char_prev = self.toID_dic[sent[i-1]]
					else:
						char_prev = sent[i-1]

					if sent[i] in self.toID_dic:
						char = self.toID_dic[sent[i]]
					else:
						char = sent[i]
				bigram = [char_prev, char]
				self.sentIDData_bi.append(bigram)

	def get_sentIDData(self):
		return self.sentIDData
	def get_sentIDData_bi(self):
		return self.sentIDData_bi
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







