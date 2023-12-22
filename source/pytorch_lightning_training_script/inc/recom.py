from ranx import Qrels, Run, evaluate, compare

class PaperDataClass:
    def __init__(self):
        self.paperDict = {}
        self.paperList = [] # ランダムに論文を選択する都合上、数字のインデックスでアクセスできるようにリストも持っておく
        self.abstList = []
        self.abstEmbList = []
        self.labelList = { # TF-IDFの処理の都合上、観点毎にリストに切り出す
            'title': [],
            'bg': [],
            'obj': [],
            'method': [],
            'res': [],
            'other': [],
        }

class allPaperDataClass(PaperDataClass):
    def __init__(self):
        super().__init__()
        self.testDataIndex = []
        self.titleToIndex = {}

class testPaperDataClass(PaperDataClass):
    def __init__(self):
        super().__init__()
        self.allDataIndex = []

def extractTestPaperEmbeddings(allPaperEmbeddingList, testPaperAllDataIndex):
    """
    allPaperDataをBOWやTF-IDFなどのベクトルに変換したリストから、testフラグが
    立っているデータを抜き出してtestPaperDataに代入して返す
    """
    testPaperEmbeddingList = []

    if type(allPaperEmbeddingList) != list:
        allPaperEmbeddingList = allPaperEmbeddingList.toarray()

    for i, paper_embedding in enumerate(allPaperEmbeddingList):
        if i in testPaperAllDataIndex:
            testPaperEmbeddingList.append(paper_embedding)
            
    return testPaperEmbeddingList

def genQrels(testPaperData):
    # Qrel（教師データ）を作成
    qrels_dict = {}
    for paper in testPaperData.paperList:
        citeDict = {}
        for cite in paper['cite']:
            citeDict[cite] = 1
        # ローカルではpaperにidを振っていたが、こっちではタイトルをidのように扱っているから
        qrels_dict[paper['title']] = citeDict
    
    # お決まりのやつ
    qrels = Qrels(qrels_dict)
    
    return qrels

def genRun(allPaperData, testPaperData, simMatrix):
    # Run（予測結果）を作成
    run_dict = {} 
    for i, oneTestPaperSim in enumerate(simMatrix):
        simDict = {}
        for j, score in enumerate(oneTestPaperSim):
            # testデータと同じ文書を含まないようにするため
            if allPaperData.paperList[testPaperData.allDataIndex[i]]['title'] == allPaperData.paperList[j]['title']:
                continue
            simDict[allPaperData.paperList[j]['title']] = score
        run_dict[testPaperData.paperList[i]['title']] = simDict 

    # お決まりのやつ
    run = Run(run_dict)
    
    return run