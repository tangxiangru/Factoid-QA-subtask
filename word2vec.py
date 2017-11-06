#coding:utf-8
import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import KeyedVectors,LineSentence
import itertools
from data import tokenizeTrainData,data_path,buildVocab
import sys
import os

tokenize_data="datasets/tokenized_data.txt"
emb_dim=128
model_path="weights/word2vec.bin"
embedding_path="datasets/temp/embedding.np"

def get_corpora():
    data=tokenizeTrainData(path=data_path)
    with open(tokenize_data,"w",encoding="utf-8") as f:
        for query_id,query_words,pass_id,pass_words,answer_words,start,end,score,qscore in data:
            f.write(" ".join(query_words)+"\n")
            f.write(" ".join(pass_words)+"\n")
                
def train_word2vec():
    '''训练词项向量
    '''
    model=Word2Vec(sentences=LineSentence(tokenize_data),size=emb_dim,window=5,min_count=5,iter=5)
    model.wv.save_word2vec_format(model_path,binary=True)
    return model
    
def load_model():
    model=KeyedVectors.load_word2vec_format(model_path,binary=True)
    return model
    
def buildEmbedding():
    if os.path.exists(embedding_path):
        return np.load(open(embedding_path,"rb"))
    else:
        model=load_model()
        index2word, word2index, freq_dist=buildVocab()
        embedding=[]
        for word in index2word:
            if word in model:
                embedding.append(model[word])
            else:
                embedding.append(np.zeros(shape=[emb_dim]))
        embedding=np.array(embedding)
        np.save(open(embedding_path,"wb"),embedding)
        return embedding
    
    
if __name__=="__main__":
   get_corpora()
   train_word2vec()
   model=load_model()
   embedding=buildEmbedding()

