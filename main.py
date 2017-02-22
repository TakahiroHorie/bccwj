import sys
args = sys.argv
import glob

from Tagger import Tagger
from TextProcessor import BCCWJProcessor, CharProcessor
from Splitter import SplitterManager
from Tester import Tester

TRAIN_DIR = "corpora/bccwj/cxml/variable/oy_train/"
TEST_DIR  = "corpora/bccwj/cxml/variable/oy_test/"

MODEL_DIR  = "model/"

CharProc = CharProcessor() ## CharID管理, ID化した文章を管理
TextTagger = Tagger() ## EOSデータ
corpora = glob.glob(TRAIN_DIR+"*.xml")
for corpus in corpora:
	BCCWJProc = BCCWJProcessor(corpus)
	BCCWJProc.makeSentData()

	CharProc.set_sentData(BCCWJProc.get_sentData())
	CharProc.make_IDDict()
	CharProc.make_sentIDData()

	TextTagger.set_sentData(BCCWJProc.get_sentData())
	TextTagger.make_EOSData()

if(args[1] == "spl_learn"):
	epochs = args[2]
	CharProc_spl = CharProcessor()
	TextTagger_spl = Tagger()
	char_num = len(CharProc.get_toIDDict())
	char_dim = 50
	corpora = glob.glob(TRAIN_DIR+"*.xml")
	for corpus in corpora:
		BCCWJProc = BCCWJProcessor(corpus)
		BCCWJProc.makeSentData()

		CharProc_spl.set_sentData(BCCWJProc.get_sentData())
		CharProc_spl.set_toIDDict(CharProc.get_toIDDict())
		CharProc_spl.make_sentIDData()

		TextTagger_spl.set_sentData(BCCWJProc.get_sentData())
		TextTagger_spl.make_EOSData()

	SplManager = SplitterManager(char_num, char_dim)
	SplManager.set_sentIDData(CharProc_spl.get_sentIDData())
	SplManager.set_EOSData(TextTagger_spl.get_EOSData())
	SplManager.train(int(epochs))

if(args[1] == "test"):
	model_file = args[2]
	CharProc_tst = CharProcessor()
	TextTagger_tst = Tagger()
	corpora = glob.glob(TEST_DIR+"*.xml")
	for corpus in corpora:
		BCCWJProc = BCCWJProcessor(corpus)
		BCCWJProc.makeSentData()

		CharProc_tst.set_sentData(BCCWJProc.get_sentData())
		CharProc_tst.set_toIDDict(CharProc.get_toIDDict())
		CharProc_tst.make_sentIDData()

		TextTagger_tst.set_sentData(BCCWJProc.get_sentData())
		TextTagger_tst.make_EOSData()

	SplTester = Tester(CharProc_tst.get_sentIDData(), TextTagger_tst.get_EOSData(),\
				CharProc.get_toIDDict(), CharProc.get_fromIDDict())
	SplTester.set_model(MODEL_DIR+model_file)
	SplTester.splitTest()



























