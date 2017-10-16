#coding:utf-8
import os
import sys
sys.path.append("..")
import config
import nltk
from nltk.text import Text
from nltk.corpus import WordListCorpusReader
import numpy as np

def drop_stopwords(words,stop_words):
    '''去停用词
    '''
    droped=[]
    for word in words:
        if not word in stop_words:
            droped.append(word)
    return droped
    
def load_stopwords(path,sep=None,encoding="utf-8"):
    '''加载停用词表
    '''
    f=open(path,encoding=encoding)
    stopwords=set()
    for line in f:
        line=line.strip()
        if sep is None:
            if not line in stopwords:
                stopwords.add(line)
        else:
            words=line.split(sep)
            for word in words:
                if not word in stopwords:
                    stopwords.add(word)
    return stopwords
    
def read_stopwords(path):
    '''使用nltk读停用词表
    '''
    root,fileid=os.path.split(path)
    stopwords=WordListCorpusReader(root,[fileid])
    return stopwords.words(fileid)
    
def stat_idf_nltk(docs):
    '''统计idf,使用nltk的ConditionalFreqDist
    '''
    doc_num=len(docs)
    pairs=[(w,i) for i in range(doc_num) for w in docs[i]]
    cfd=nltk.ConditionalFreqDist(pairs)
    idfs={}
    for c in cfd:
        docFreq=len(c)
        idfs[c]=np.log(1+(doc_num-docFreq+0.5)/(docFreq+0.5))
    return idfs
    
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
    
def stat_positions(doc):
    '''词的位置信息
    '''
    positions={}
    position=1
    for w in doc:    
        if not w in positions:
            positions[w]=[position]
        else:
            positions[w].append(position)
        position+=1
    return positions

if __name__=="__main__":
    stopwords=load_stopwords(config.stopwords_path)
    