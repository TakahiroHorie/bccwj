import glob
from Tagger import Tagger
from TextProcessor import BCCWJProcessor, CharProcessor
from Char2Vec import Char2VecManager
from Splitter import SplitterManager
from Tester import Tester

CharProc = CharProcessor() ## CharID管理, ID化した文章を管理
TextTagger = Tagger() ## EOSデータ

if(True):
	corpora = glob.glob("corpora/bccwj/cxml/variable/pn_train/*.xml")
	for corpus in corpora:
		# print("-----------" + corpus + "-----------")
		BCCWJProc = BCCWJProcessor(corpus)
		BCCWJProc.makeSentData()

		CharProc.set_sentData(BCCWJProc.get_sentData())
		CharProc.make_IDDict()
		CharProc.make_sentIDData()

		TextTagger.set_sentData(BCCWJProc.get_sentData())
		TextTagger.make_EOSData()

	if(False):
		C2VManager = Char2VecManager()
		C2VManager.set_IDDict(CharProc.get_toIDDict(), CharProc.get_fromIDDict())
		C2VManager.set_countID(CharProc.get_countID())
		C2VManager.set_sentIDData(CharProc.get_sentIDData())
		C2VManager.set_Char2VecModel()
		C2VManager.train()

	if(False):
		char_num = len(CharProc.get_toIDDict())
		char_dim = 50
		SplManager = SplitterManager(char_num, char_dim)
		SplManager.set_sentIDData(CharProc.get_sentIDData())
		SplManager.set_EOSData(TextTagger.get_EOSData())
		SplManager.train()

SplTester = Tester(CharProc.get_sentIDData(), TextTagger.get_EOSData(), CharProc.get_toIDDict(), CharProc.get_fromIDDict())
if(True):
	corpora = glob.glob("corpora/bccwj/cxml/variable/pn_test/*.xml")
	for corpus in corpora:
		BCCWJProc = BCCWJProcessor(corpus)
		BCCWJProc.makeSentData()

		CharProc.set_sentData(BCCWJProc.get_sentData())
		CharProc.make_sentIDData()
		TextTagger.set_sentData(BCCWJProc.get_sentData())
		TextTagger.make_EOSData()

		SplTester.split_test()



























