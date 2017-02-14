import glob
from Tagger import Tagger
from TextProcessor import BCCWJProcessor, CharProcessor

if __name__ == '__main__':

	## CharID管理
	CharProc = CharProcessor()

	corpora = glob.glob("corpora/bccwj/cxml/variable/pn/*.xml")
	for corpus in corpora:
		print("-----------" + corpus + "-----------")
		BCCWJProc = BCCWJProcessor(corpus)
		BCCWJProc.makeSentData()

		CharProc.set_SentData(BCCWJProc.get_SentData())
		CharProc.makeIDDict()

		TextTagger = Tagger()
		TextTagger.set_SentData(BCCWJProc.get_SentData())
		TextTagger.set_IDDict(CharProc.get_toIDDict(), CharProc.get_fromIDDict())
		TextTagger.makeEOSData(byID=True)
		TextTagger.printData("withEOS")