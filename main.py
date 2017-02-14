import glob
from Tagger import Tagger
from TextProcessor import BCCWJProcessor

if __name__ == '__main__':

	corpora = glob.glob("corpora/bccwj/cxml/variable/pn/*.xml")
	for corpus in corpora:
		# print("-----------" + corpus + "-----------")
		BCCWJProc = BCCWJProcessor(corpus)
		BCCWJProc.makeSentData()

		BCCWJTagger = Tagger()
		BCCWJTagger.setDoc(BCCWJProc.getSentData())
		BCCWJTagger.makeEOSData()
		BCCWJTagger.printData("plain")
