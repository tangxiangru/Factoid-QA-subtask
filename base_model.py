#!/bin/env python3
#-*- text-encoding: utf-8 -*-
import tensorflow as tf
import random
from data import buildTrainDataIndex,getTrainData,transSent2Idx,vocab_size,readTrainData,bm25_model,tokenize
import numpy as np
from word2vec import buildEmbedding
from myutils.train_util import get_weight_path
from myutils.rlog import *

base_weight_path="weights/"

class BaseModel(object):
    def __init__(self,embed_dim=128,vocab_size=vocab_size,learning_rate=0.001):
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.learning_rate=learning_rate
        self.embedding=buildEmbedding().astype("float32")
        print(self.embedding.shape)
        self.query_len=30
        self.passage_len=301
        self.ckpt_path=get_weight_path(self,base_weight_path)
        tf.reset_default_graph()
        
        
        self.query_in=tf.placeholder(dtype=tf.int32,shape=[None,self.query_len])
        self.passage_in=tf.placeholder(dtype=tf.int32,shape=[None,self.passage_len])
        self.overlap_in=tf.placeholder(dtype=tf.int32,shape=[None,self.passage_len])
        self.starts_in=tf.placeholder(dtype=tf.int32,shape=[None])
        self.ends_in=tf.placeholder(dtype=tf.int32,shape=[None])
        self.score_in=tf.placeholder(dtype=tf.float32,shape=[None])
        
        
        self.get_model(self.query_in,self.passage_in,self.overlap_in,self.starts_in,self.ends_in,self.score_in)
        
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        
    def get_model(self,query_in,passage_in,overlap_in,starts,ends,scores):
        '''模型
        '''
        starts=tf.clip_by_value(starts,clip_value_min=0,clip_value_max=self.passage_len-1)
        ends=tf.clip_by_value(ends,clip_value_min=0,clip_value_max=self.passage_len-1)
        with tf.variable_scope("embedding"):
            print(self.embedding[10:,])
            W=tf.get_variable(name="embedding",initializer=self.embedding,trainable=False)
            #W=tf.get_variable(name="embedding",initializer=None,trainable=True,shape=self.embedding.shape)
            query_embeded=tf.nn.embedding_lookup(W,query_in)
            passage_embeded=tf.nn.embedding_lookup(W,passage_in)
            
            overlap_W=tf.get_variable(name='overlapEmbedding',initializer=np.array([[0,0],[0,1],[1,0]]).astype("float32"),dtype=tf.float32,trainable=False)
            overlap_embeded=tf.nn.embedding_lookup(overlap_W,overlap_in)
            
            query_lengths=tf.ones(shape=[tf.shape(query_embeded)[0]])*self.query_len
            passage_lengths=tf.ones(shape=[tf.shape(passage_embeded)[0]])*self.passage_len
            
        with tf.variable_scope("query_encoder"):
            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
            query_h,query_state=tf.nn.dynamic_rnn(cell,query_embeded,sequence_length=None,dtype=tf.float32,swap_memory=False,time_major=False)
            
        with tf.variable_scope("passage_encoder"):
            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
            passage_h,passage_sate=tf.nn.dynamic_rnn(cell,passage_embeded,sequence_length=None,dtype=tf.float32,swap_memory=False,time_major=False)
            
            
        with tf.variable_scope("coattention"):
            att=tf.matmul(query_h,passage_h,transpose_b=True)
            
            query_ctx=tf.matmul(att,passage_h)
            query_ctx=tf.concat([query_h,query_ctx],axis=-1)
            
            passage_ctx=tf.matmul(att,query_ctx,transpose_a=True)
            passage_ctx=tf.concat([passage_h,passage_ctx],axis=-1)
        with tf.variable_scope("encoder_output"):
            cell_fw=tf.nn.rnn_cell.BasicLSTMCell(num_units=128,state_is_tuple=True)
            cell_bw=tf.nn.rnn_cell.BasicLSTMCell(num_units=128,state_is_tuple=True)
            
            encoder_outputs,encoder_states=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                           cell_bw=cell_bw,
                                                                           inputs=passage_ctx,
                                                                           sequence_length=None,
                                                                           dtype=tf.float32,
                                                                           time_major=False,
                                                                           swap_memory=False,
                                                                           parallel_iterations=1)
            encoder_outputs_fw,encoder_outputs_bw=encoder_outputs
            
            encoder_outputs=tf.concat(encoder_outputs,axis=-1)
            encoder_outputs=tf.concat([encoder_outputs,overlap_embeded],axis=-1)
            encoder_state_fw,encoder_state_bw=encoder_states
            h=tf.concat([encoder_state_fw.h,encoder_state_bw.h],axis=-1)
            c=tf.concat([encoder_state_fw.c,encoder_state_bw.c],axis=-1)
            
        with tf.variable_scope("output_label"):
            '''output_label: 是否是正确的样本'''
            ip=tf.reduce_max(encoder_outputs,axis=1,keep_dims=False)
            W=tf.get_variable(name="weights",shape=[self.embed_dim*2+2,128])
            b=tf.get_variable(name="b",shape=[128])
            hid=tf.nn.tanh(tf.matmul(ip,W)+b)
            
            W2=tf.get_variable(name='w2',shape=[128,1])
            b2=tf.get_variable(name="b2",shape=[1])
            out_label=tf.matmul(hid,W2)+b2
            out_label=tf.nn.sigmoid(out_label)
            self.out_label=out_label
#            self.out_label=tf.clip_by_value(out_label,clip_value_min=1e-6,clip_value_max=1-1e-6)
#            self.label_loss=-tf.reduce_mean(scores*tf.log(self.out_label)+(1-scores)*tf.log(1-self.out_label))
            self.label_loss=tf.reduce_mean(tf.square(scores-self.out_label))

            
        with tf.variable_scope("decoder_start") as scope:
            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.embed_dim,state_is_tuple=True)
            state=cell.zero_state(batch_size=tf.shape(query_embeded)[0],dtype=tf.float32)

            pass_embedding=tf.unstack(tf.transpose(encoder_outputs,perm=[1,0,2]),axis=0)
            start_inputs=pass_embedding[0]
            end_inputs=pass_embedding[-1]
            

            start_outputs=[]
            end_outputs=[]
            for time_step in range(4):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                inputs=tf.concat([start_inputs,end_inputs],axis=-1)
                h,state=cell(inputs,state=state)
                
                hmn_start=self.HMN(encoder_outputs,h,start_inputs,end_inputs,scope=scope,name="start")
                hmn_end=self.HMN(encoder_outputs,h,start_inputs,end_inputs,scope=scope,name="end")
                
                #当前的开始和结束位置
                next_start=tf.argmax(hmn_start,axis=-1)
                next_end=tf.argmax(hmn_end,axis=-1)
                start_onehot=tf.one_hot(next_start,depth=self.passage_len)
                end_onehot=tf.one_hot(next_end,depth=self.passage_len)
                
                #计算下一个开始和结束的位置对应的输入
                start_inputs=tf.squeeze(tf.matmul(tf.expand_dims(start_onehot,axis=1),encoder_outputs),axis=1)
                end_inputs=tf.squeeze(tf.matmul(tf.expand_dims(end_onehot,axis=1),encoder_outputs),axis=1)

                start_outputs.append(hmn_start)
                end_outputs.append(hmn_end)


            self.start_outputs=tf.reduce_mean(start_outputs,axis=0,keep_dims=False)
            self.end_outputs=tf.reduce_mean(end_outputs,axis=0,keep_dims=False)
            
            output_matrix=tf.reshape(tf.matmul(tf.expand_dims(self.start_outputs,1),
                                               tf.expand_dims(self.end_outputs,1),
                                               transpose_a=True),[-1,self.passage_len*self.passage_len])
            labels=starts*self.passage_len+ends
            self.outputs=output_matrix
#            self.loss=tf.losses.sparse_softmax_cross_entropy(logits=output_matrix,labels=labels)+self.label_loss
            
        start_loss=tf.losses.sparse_softmax_cross_entropy(logits=self.start_outputs,labels=starts) 
        start_loss=tf.reduce_mean(start_loss)
        end_loss=tf.losses.sparse_softmax_cross_entropy(logits=self.end_outputs,labels=ends)
        end_loss=tf.reduce_mean(end_loss)
        
        self.loss=start_loss+end_loss
        

        
        self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        
        #return start_outputs,end_outputs,loss,opt

    def HMN(self,encoder_outputs,h_i,start_inputs,end_inputs,scope,name):
        '''
        Highway Maxout Network
        '''
        with tf.variable_scope(scope):
            p = 10
            W1 = tf.get_variable(name=name+"W1", shape=[self.embed_dim * 3+2, self.embed_dim, p])
            b1 = tf.get_variable(name=name+'b1', shape=[self.embed_dim])
            W2 = tf.get_variable(name=name+"W2", shape=[self.embed_dim, self.embed_dim, p])
            b2 = tf.get_variable(name=name+'b2', shape=[self.embed_dim])
            W3 = tf.get_variable(name=name+'W3', shape=[self.embed_dim * 2, p])
            W_d = tf.get_variable(name=name+"W_d", shape=[self.embed_dim * 5+4, self.embed_dim])

            start_r = tf.nn.tanh(tf.matmul(tf.concat([h_i,start_inputs, end_inputs], axis=-1), W_d))
            ones = tf.ones(shape=[tf.shape(start_r)[0], self.passage_len, 1])
            start_r_repeat = tf.matmul(ones, tf.expand_dims(start_r, axis=1))
            start_u = tf.concat([encoder_outputs, start_r_repeat], axis=-1)
            start_m1 = tf.matmul(tf.reshape(start_u, shape=[-1, self.embed_dim * 3+2]),
                                 tf.reshape(W1, [self.embed_dim * 3+2, p * self.embed_dim]))
            start_m1 = tf.reduce_max(tf.reshape(start_m1, [-1, self.passage_len, p, self.embed_dim]) + b1,
                                     axis=2, keep_dims=False)
            start_m2 = tf.matmul(tf.reshape(start_m1, [-1, self.embed_dim]),
                                 tf.reshape(W2, [self.embed_dim, self.embed_dim * p]))
            start_m2 = tf.reduce_max(tf.reshape(start_m2, [-1, self.passage_len, p, self.embed_dim]) + b2,
                                     axis=2, keep_dims=False)
            m3 = tf.matmul(tf.reshape(tf.concat([start_m1, start_m2], -1), [-1, self.embed_dim * 2]),
                           W3)
            m3 = tf.reduce_max(tf.reshape(m3, [-1, self.passage_len, p]),
                               axis=-1, keep_dims=False)
            return m3

    def batch_generator(self,querys,passages,overlaps,starts=None,ends=None,scores=None,batch_size=128,shuffle=False):
        num=len(querys)
        if shuffle:
            ids=random.sample(list(range(num)),num)
            querys=querys[ids]
            passages=passages[ids]
            overlaps=overlaps[ids]
            if starts is not None and ends is not None and scores is not None:
                starts=starts[ids]
                ends=ends[ids]
                scores=scores[ids]
                
        for i in range((num+batch_size-1)//batch_size):
            s=i*batch_size
            e=(i+1)*batch_size
            if starts is not None and ends is not None:
                yield querys[s:e],passages[s:e],starts[s:e],ends[s:e],scores[s:e],overlaps[s:e]
            else:
                yield querys[s:e],passages[s:e],overlaps[s:e]
    def train(self,querys,passages,starts,ends,scores,overlaps,
              valid_querys=None,valid_passages=None,valid_starts=None,valid_ends=None,voverlaps=None,
              iter_num=10,batch_size=256):
        '''训练'''
        for i in range(iter_num):
            total_loss=0
            gen=self.batch_generator(querys,passages,overlaps,starts,ends,scores,shuffle=True,batch_size=batch_size)
            for q,p,s,e,score,overlap in gen:
                feed_dict={self.query_in:q,
                           self.passage_in:p,
                           self.starts_in:s,
                           self.ends_in:e,
                           self.score_in:score,
                           self.overlap_in:overlap}
                _,err=self.sess.run([self.opt,self.loss],feed_dict=feed_dict)
                total_loss+=err
                #print("iter num: %s,error:%s"%(i,err))
            avg_loss=total_loss/(len(querys)+batch_size-1)*batch_size
            log("[{}/{}] total loss={}, average loss={}".format(i, iter_num, total_loss, avg_loss))
            self.save_weights()
            #预测
            pre_starts,pre_ends,pre_scores=self.predict(vquerys,vpassages,voverlaps)
            acc_starts=np.sum((pre_starts==vstarts)&(pre_ends==vends))/len(vstarts)
            acc_ends=np.sum(pre_ends==vends)/len(vends)
            log("start accuracy: %s, end accuracy: %s"%(acc_starts,acc_ends))
                
        
    def predict(self,querys,passages,overlaps):
        '''预测'''
        gen=self.batch_generator(querys,passages,overlaps)
        pre_starts=[]
        pre_ends=[]
        pre_scores=[]
        for batch_query,batch_passage,batch_overlap in gen:
            feed_dict={self.query_in:batch_query,self.passage_in:batch_passage,self.overlap_in:batch_overlap}
            starts,ends=self.sess.run([self.start_outputs,self.end_outputs],feed_dict=feed_dict)
            starts=np.argmax(starts,axis=-1)
            ends=np.argmax(ends,axis=-1)

#            outputs,scores=self.sess.run([self.outputs,self.out_label],feed_dict=feed_dict)
#            outputs=np.argmax(outputs,axis=1)
#            #print(starts,ends.shape)
#            starts=outputs//self.passage_len
#            ends=outputs%self.passage_len
        
            pre_starts.extend(starts)
            pre_ends.extend(ends)
            pre_scores.extend(np.reshape(scores,[scores.shape[0],]))
            
        return np.array(pre_starts),np.array(pre_ends),np.array(pre_scores)
    def save_weights(self):
        saver=tf.train.Saver()
        saver.save(self.sess,self.ckpt_path)
    def restore_last_session(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, self.ckpt_path)
            
            
    def extractAnswer(self,query,passage):
        '''抽取答案
        param: 
            query: 问题文本
            passage: 段落文本
        return:
            返回答案文本
        '''
        qwords,qidx=transSent2Idx(query,max_len=self.query_len)
        pwords,pidx=transSent2Idx(passage,max_len=self.passage_len)
        overlaps=[1 if w in qwords else 2 for w in pwords]+[0 for i in range(self.passage_len-len(pwords))][:self.passage_len]
        overlaps=np.array([overlaps])
        starts,ends,score=self.predict(qidx,pidx,overlaps)
        
        start=min(starts[0],ends[0]-1)
        end=min(ends[0],start+20)
        answer=pwords[start:end]
#        print(starts,ends)
        return "".join(answer),score
        
def predictAnswer(datas,model,output="results.txt"):
    log('predicting answer.')
    with open(output,"w",encoding='utf-8') as f:
        right=0
        for line in datas:
            query=line['query']
            passages=line['passages']
            true_answer=line['answer']
            log("predict answer: question ~ " + query)
            log("answer ~ " + true_answer)
        
            my_answers=[]
            scores=[]
            for passage in passages:
                my_answer,score=model.extractAnswer(query,passage['passage_text'])
#                print(a,score,answer in ps[i]['passage_text'])
                bm25_score=bm25_model.compute_score(tokenize(query),tokenize(passage['passage_text']))
                scores.append(bm25_score)
                if my_answer is '':
                    continue
                answers.append(my_answer)
                print(my_answer,bm25_score,true_answer in passage['passage_text'])
            if true_answer in my_answers:
                right+=1
            f.write(query+"\tTrue Answer:"+true_answer+"\tPredict Answer:"+"\t".join(list(set(my_answers)))+"\n")
            log("right:%s/%s, acc:%s"%(right,len(datas),right/len(datas)))
        
if __name__=="__main__":
    train_data,valid_data=getTrainData()
    query_ids,querys,pass_ids,passages,answer,starts,ends,scores,qscores,overlaps=train_data
    vquery_ids,vquerys,vpass_ids,vpassages,vanswer,vstarts,vends,vscores,vqscores,voverlaps=valid_data
    
    model=BaseModel()
    #model.restore_last_session()
    model.train(querys,passages,starts,ends,scores,overlaps,
                vquerys,vpassages,vstarts,vends,voverlaps,batch_size=64,iter_num=20)
    model.save_weights()
            
    pre_starts,pre_ends,pre_scores=model.predict(vquerys,vpassages,voverlaps)
    datas=readTrainData()
    predictAnswer(datas[-10:],model)
    
        
