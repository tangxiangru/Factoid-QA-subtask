#coding:utf-8
import json
import jieba
import nltk
import itertools
import numpy as np
import os
import pickle
import random
from BM25Sim import BM25Sim
import pandas as pd

data_path='./datasets/train_factoid_1.json'
tokenized_path="./datasets/temp/tokenized_data.pkl"
vocab_path="./datasets/temp/vocab.pkl"
passage_len=301
query_len=30
answer_len=5
vocab_size=50000

def computeDocAnswerScore(doc,answer,passage_len=passage_len):
    '''
    参数：
        doc: 文档passage单词
        answer：答案单词
    return:
        docAnswerScore： passage中是否包含答案
        start：开始位置
        end: 结束位置
    '''
    score=0
    start=passage_len-1
    end=passage_len-1
    if "".join(answer) in "".join(doc):
        start=0
        end=0
        score=1
    #计算start,end
    for i in range(len(doc)):
        if doc[i]==answer[0]:
            if len(doc)>=i+len(answer):
                start=i
                for j in range(len(answer)):
                    if doc[i+j]!=answer[j]:
                        start=0
                        break
            if start!=passage_len-1:
                end=i+j+1
            break  #如果包含多个答案怎么解决
    return score,min(start,passage_len-1),min(end,passage_len-1)
def computDocQuestionScore(question,doc,model):
    '''
    param:
        doc: 文档单词
        question: 问题单词
    return:
        docQuetionScore
    '''
    score=model.compute_score(question,doc)
    #计算score

    return score

def tokenize(text):
    '''分词'''
    return jieba.lcut(text)
    
def readTrainData(path=data_path):
    '''读取数据
    '''
    datas=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            data=json.loads(line.strip())
            datas.append(data)
    return datas


def tokenizeTrainData(path=data_path):
    '''
    输入:
        json格式的原始数据路径
    输出:
        训练数据：
    '''
    dump_path=tokenized_path
    if os.path.exists(dump_path):
        return pickle.load(open(dump_path,"rb"))
    else:
        trainData=readTrainData(path)
        res=[]
        for data in trainData:
            query_id=data['query_id']
            query=data['query']
            query_words=tokenize(query)
            passages=data['passages']
            answer=data['answer']
            answer_words=tokenize(answer)
            typ=data['type']
            for passage in passages:
                pass_id=passage['passage_id']
                pass_text=passage['passage_text']
                pass_words=tokenize(pass_text)
                score,start,end=computeDocAnswerScore(pass_words,answer_words)
                qscore=0#computDocQuestionScore(pass_words,answer_words)
                res.append([query_id,query_words,pass_id,pass_words,answer_words,start,end,score,qscore])
        pickle.dump(res,open(dump_path,"wb"))
        return res

def get_question_passages():
    '''所有的query和passages
    '''
    questions=[]
    passages=[]
    data=tokenizeTrainData()
    for query_id,query_words,pass_id,pass_words,answer_words,start,end,score,qscore in data:
        questions.append(query_words)
        passages.append(pass_words)
    return questions,passages
    
questions,passages=get_question_passages()
bm25_model=BM25Sim(passages)
    
def getSentences(tokenizedTrainData):
    sentences=[]
    for query_id,query_words,pass_id,pass_words,answer_words,start,end,score,qscore in tokenizedTrainData:
        sentences.append(query_words)
        sentences.append(pass_words)
    return sentences
    
def buildVocab(tokenized_sentences=None):
    #vocab_path=vocab_path
    if os.path.exists(vocab_path):
        return pickle.load(open(vocab_path,"rb"))
    else:
        if tokenized_sentences is None:
                res=tokenizeTrainData()
                tokenized_sentences=getSentences(res)
        # get frequency distribution
        freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("all words: ",len(freq_dist.keys()))
        vocab = freq_dist.most_common(vocab_size)
        # index2word
        index2word = ['<end>'] + ['UNK'] + [x[0] for x in vocab]
        # word2index
        word2index = dict([(w, i) for i, w in enumerate(index2word)])
        pickle.dump([index2word, word2index, freq_dist],open(vocab_path,'wb'))
        return index2word, word2index, freq_dist

def padding(sequences,max_len,value=0):
    '''padding'''
    padded=[]
    for seq in sequences:
        leng=len(seq)
        temp=list(seq)+[value for i in range(max(0,max_len-leng))]
        padded.append(temp[:max_len])
    return np.array(padded)
    
def buildTrainDataIndex(tokenizedTrainData,word2idx,question_len=query_len,passage_len=passage_len,answer_len=answer_len,filter_negtive=True):
    '''训练数据单词-->index
    '''
    query_ids=[]
    query_index=[]
    pass_ids=[]
    pass_index=[]
    answer_index=[]
    starts=[]
    ends=[]
    scores=[]
    qscores=[]
    overlaps=[] #passage中的词是否在query中出现
    columns=['qid','qwords','qidx','pid','pwords','pidx','awords','aidx','start','end','score','qscore','overlap']
    #统计每个问题对应的最大相似度
    temp_datas=[]
    for query_id,query_words,pass_id,pass_words,answer_words,start,end,score,qscore in tokenizedTrainData:
        qscore=computDocQuestionScore(query_words,pass_words,bm25_model)
        qidx=[word2idx.get(word,1) for word in query_words]
        pidx=[word2idx.get(word,1) for word in pass_words]
        aidx=[word2idx.get(word,1) for word in answer_words]
        overlap=[1 if w in query_words else 2 for w in pass_words]
        temp_datas.append([query_id,query_words,qidx,pass_id,pass_words,pidx,answer_words,aidx,start,end,score,qscore,overlap])
    
    df=pd.DataFrame(data=temp_datas,index=None,columns=columns)
    print("data shape",df.shape)
    #排序、分组,每组选前n个
    grouped=df.sort_values(by=['score','qscore'],ascending=False).groupby('qid').head(n=5)
    query_ids=grouped['qid'].values
    query_index=padding(grouped['qidx'].values,question_len)
    pass_ids=grouped['pid']
    pass_index=padding(grouped['pidx'].values,passage_len)
    starts=np.array(grouped['start'].values)
    ends=np.array(grouped['end'].values)
    scores=np.array(grouped['score'].values)
    qscores=np.array(grouped['qscore'].values)
    answer_index=padding(grouped['aidx'].values,answer_len)
    overlaps=padding(grouped['overlap'].values,passage_len)
    
    return query_ids,\
           query_index,\
           pass_ids,\
           pass_index,\
           answer_index,\
           starts,\
           ends,\
           scores,\
           qscores,\
           overlaps
    
def getTrainData(split=0.8):
    '''切分训练集和验证集'''
    dump_path="./datasets/temp/train_valid_data.np"
    if os.path.exists(dump_path):
        return pickle.load(open(dump_path,"rb"))
    else:
        res=tokenizeTrainData()
        num=len(res)
        #res=random.sample(res,num)
        train_num=int(num*0.8)
        
        sentences=getSentences(res)
        id2w,w2id,fre=buildVocab(sentences)
        
        train_res=res[:train_num]
        valid_res=res[train_num:]

        query_ids,query,pass_ids,passage,answer,starts,ends,scores,qscores,overlaps=buildTrainDataIndex(train_res,w2id)
        train_data=[query_ids,query,pass_ids,passage,answer,starts,ends,scores,qscores,overlaps]
        
        query_ids,query,pass_ids,passage,answer,starts,ends,scores,qscores,overlaps=buildTrainDataIndex(valid_res,w2id)
        valid_data=[query_ids,query,pass_ids,passage,answer,starts,ends,scores,qscores,overlaps]
        
        pickle.dump([train_data,valid_data],open(dump_path,"wb"))
        return train_data,valid_data
id2w,w2id,fre=buildVocab()    
def transSent2Idx(sent,max_len,pad_value=0,word2idx=w2id):
    '''句子转换成index'''
    words=tokenize(sent)
    seq=[word2idx.get(word,1) for word in words]
    leng=len(seq)
    padded=list(seq)+[pad_value for i in range(max(0,max_len-leng))]
    return words,np.array([padded[:max_len]])
    
    
if __name__=="__main__":
    res=tokenizeTrainData()
    sentences=getSentences(res)
    id2w,w2id,fre=buildVocab(sentences)
    query_ids,query,pass_ids,passage,answer,starts,ends,scores,qscores=buildTrainDataIndex(res,w2id)
    
    train_data,valid_data=getTrainData()
    query_ids,querys,pass_ids,passages,answer,starts,ends,scores,qscores=train_data
    vquery_ids,vquerys,vpass_ids,vpassages,vanswer,vstarts,vends,vscores,vqscores=valid_data
    

