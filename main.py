import glob
from Tagger import Tagger
from TextProcessor import BCCWJProcessor, CharProcessor
from Char2Vec import Char2VecManager

CharProc = CharProcessor() ## CharID管理, ID化した文章を管理
TextTagger = Tagger() ## EOSデータ

corpora = glob.glob("corpora/bccwj/cxml/variable/pn_short/*.xml")
for corpus in corpora:
	# print("-----------" + corpus + "-----------")
	BCCWJProc = BCCWJProcessor(corpus)
	BCCWJProc.makeSentData()

	CharProc.set_sentData(BCCWJProc.get_sentData())
	CharProc.make_IDDict()
	CharProc.make_sentIDData()

	TextTagger.set_sentData(BCCWJProc.get_sentData())
	TextTagger.make_EOSData()

C2VManager = Char2VecManager()
C2VManager.set_IDDict(CharProc.get_toIDDict(), CharProc.get_fromIDDict())
C2VManager.set_countID(CharProc.get_countID())
C2VManager.set_sentIDData(CharProc.get_sentIDData())
C2VManager.set_Char2VecModel()
C2VManager.train()