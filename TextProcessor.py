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

	def getSentData(self): return self.__sentData

def cleanText(text:str):
	text = text.replace('\n','')
	text = text.replace('\r','')
	text = text.replace('　','')
	text = text.replace('。','')
	return text