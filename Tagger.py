class Tagger:

	def __init__(self):
		self.EOSData = []

	def set_sentData(self, sentData:list):
		self.sentData = sentData
		
	def make_EOSData(self):
		for sent in self.sentData:
			for (i, char) in enumerate(sent):
				EOS = 1 if (i == len(sent)-1) else 0
				self.EOSData.append(EOS)

	def get_EOSData(self):
		return self.EOSData