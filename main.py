import sys
args = sys.argv
import glob

from Tagger import Tagger
from TextProcessor import BCCWJProcessor, CharProcessor
from Char2Vec import Char2VecManager
from Splitter import SplitterManager
from Tester import Tester

TRAIN_DIR = "corpora/bccwj/cxml/variable/ot_train/"
TEST_DIR  = "corpora/bccwj/cxml/variable/ot_test/"

MODEL_DIR  = "model/"


CharProc = CharProcessor() ## CharID管理, ID化した文章を管理
TextTagger = Tagger() ## EOSデータ
corpora = glob.glob(TRAIN_DIR+"*.xml")
for corpus in corpora:
	# print("-----------" + corpus + "-----------")
	BCCWJProc = BCCWJProcessor(corpus)
	BCCWJProc.makeSentData()

	CharProc.set_sentData(BCCWJProc.get_sentData())
	CharProc.make_IDDict()
	CharProc.make_sentIDData()
	CharProc.make_sentIDData_bi()

	TextTagger.set_sentData(BCCWJProc.get_sentData())
	TextTagger.make_EOSData()

## c2v_learn
if(args[1] == "c2v_learn"):
	C2VManager = Char2VecManager(CharProc.get_toIDDict(), CharProc.get_fromIDDict(), CharProc.get_countID())
	C2VManager.set_sentIDData(CharProc.get_sentIDData())
	C2VManager.train(100)

## spl_learn
if(args[1] == "spl_learn"):
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
		CharProc_spl.make_sentIDData_bi()

		TextTagger_spl.set_sentData(BCCWJProc.get_sentData())
		TextTagger_spl.make_EOSData()

	SplManager = SplitterManager(char_num, char_dim)
	SplManager.set_sentIDData_bi(CharProc_spl.get_sentIDData_bi())
	SplManager.set_EOSData(TextTagger_spl.get_EOSData())
	SplManager.train(100, "c2v-99.npz")

## test
if(args[1] == "test"):
	CharProc_tst = CharProcessor()
	TextTagger_tst = Tagger()
	corpora = glob.glob(TEST_DIR+"*.xml")
	for corpus in corpora:
		BCCWJProc = BCCWJProcessor(corpus)
		BCCWJProc.makeSentData()

		CharProc_tst.set_sentData(BCCWJProc.get_sentData())
		CharProc_tst.set_toIDDict(CharProc.get_toIDDict())
		CharProc_tst.make_sentIDData_bi()

		TextTagger_tst.set_sentData(BCCWJProc.get_sentData())
		TextTagger_tst.make_EOSData()

	SplTester = Tester(CharProc_tst.get_sentIDData_bi(), TextTagger_tst.get_EOSData(),\
				CharProc.get_toIDDict(), CharProc.get_fromIDDict())
	SplTester.set_model("short-c2v-99.npz", "splitter-99.npz")
	SplTester.splitTest()



























