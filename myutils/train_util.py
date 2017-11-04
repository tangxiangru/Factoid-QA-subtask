#coding:utf-8
import random
import tensorflow as tf
import os


def shuffle_ids(sample_num):
    ids=random.sample(list(range(sample_num)),sample_num)
    return ids
    

def save_weights(sess,path):
    '''
    保存权重
    '''
    saver=tf.train.Saver()
    saver.save(sess,path)
def load_weights(sess,path):
    '''加载网络权重
    '''
    saver=tf.train.Saver()
    saver.restore(sess,path)
    return sess
    
def get_weight_path(model,base_weight_path):
    '''
    权重保存路径
    '''
    name=model.__class__.__name__
    weight_dir=os.path.join(base_weight_path,name)
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    weight_path=os.path.join(weight_dir,name+".h5")
    return weight_path
