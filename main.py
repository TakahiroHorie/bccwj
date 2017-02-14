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

		CharProc.setSentData(BCCWJProc.getSentData())
		CharProc.makeIDDict()

		TextTagger = Tagger()
		TextTagger.setSentData(BCCWJProc.getSentData())
		TextTagger.makeEOSData(byID=True)
		TextTagger.printData("plain")