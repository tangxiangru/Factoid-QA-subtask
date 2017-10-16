#coding:utf-8
import numpy as np


def stat_docFreqs(docs):
    '''统计单词的文档频率
    '''
    docFreqs={}
    for doc in docs:
        for w in set(doc):
            if not w in docFreqs:
                docFreqs[w]=1
            else:
                docFreqs[w]+=1
    return docFreqs
def stat_idf(docs):
    '''统计idf
    '''
    doc_num=len(docs)
    docFreqs=stat_docFreqs(docs)
    idfs={}
    for w,docFreq in docFreqs.items():
        idfs[w]=np.log(1+(doc_num-docFreq+0.5)/(docFreq+0.5))
    return idfs
def stat_tf(doc):
    '''统计词频
    '''
    wf={}
    for w in doc:    
        if not w in wf:
            wf[w]=1
        else:
            wf[w]+=1
    return wf
    
#def get_data():
#    questions=[]
#    passages=[]
#    data=tokenizeTrainData()
#    for query_id,query_words,pass_id,pass_words,answer_words,start,end,score,qscore in data:
#        questions.append(query_words)
#        passages.append(pass_words)
#    return questions,passages
'''
参数设置：
1. bm25中的b设置大一些，效果比较好，可以充分考虑文档长度的影响。实验效果：b: 0.6>0.8>0.5>0.25
'''
class BM25Sim(object):
    def __init__(self,docs,k1=1.25,b=0.6):
        self.k1=k1
        self.b=b
        self.ds=None
        self.path=None
        self.idfs=stat_idf(docs)  #idf
        docNum=len(docs)
        self.avgl=np.sum([len(doc) for doc in docs])/docNum  #平均文档长度
        
    def predict_sim(self,questions,answers):
        '''计算每个问题与对应文档的BM25分数'''
        #去停用词
        #questions=[nlp.drop_stopwords(question,stopwords) for question in questions]
        #print(questions[0])
        #answers=[nlp.drop_stopwords(answer,stopwords) for answer in answers]
        scores=[]
        for question,answer in zip(questions,answers):
            scores.append(self.compute_score(question,answer))
        return scores

    def compute_score(self,question,answer):
        '''计算BM25得分
        question: 问题, 单词列表
        answer: 答案，单词列表
        docFreq: 每个单词出现的文档个数
        '''
        score=0.0
        doc_len=len(answer)
        word_freq=stat_tf(answer)
        position=1
        q_len=len(question)
        for word in question:
            idf=self.idfs.get(word,0)
            k=self.k1*(1-self.b+self.b*(doc_len/self.avgl))
            fi=word_freq.get(word,0)  #词频数比频率效果好点
            r=fi*(1+self.k1)/(fi+k)+1.0  #加1.0变成BM25+
            score+=idf*r #*(position/q_len)  #问题中位置靠后的单词比较重要，这里乘以位置权重,效果要好很多
            position+=1
        return score
            
if __name__=='__main__':
#    questions,passages=get_data()
#    model=BM25Sim(passages)
#    score=model.compute_score(questions[0],passages[0])
    pass