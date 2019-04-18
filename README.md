## Dependenices
* python 3.5
* tensorflow 1.3
* Chinese Words Segementation：jieba

## Directory

1. datasets：data
2. data: preprocessing
3. models: model
4. weights: checkpoint
5. myutils: API
6. docs: document


## Data format
1. for each data：

(query_id, question, passage_id, passage, answer, start, end, docAnswerScore,docQuestionScore,overlaps)

 
 * question: word id
 * passage: word id
 * answer: word id
 * start: start position
 * end: end position
 * docAnswerScore: if answer is in document:1, if not0）
 * docQuestionScore: matching score
 * overlaps: if document word is in question:1，if not:0
 
 2. Label
 * predict： start, end 
 * sequence labeling： BIO
 


# Evaluation
q | a
---- | ---
task | Factoid Q&A subtask
metric |  Accuracy, F1
data num |  3w
candidate |  10documents each question
answer num |  1
answer length |  <20



# Sample
```
{
“query_id”: 10000,
“query”: “中国最大的内陆盆地是哪个”,
“passages”: [
{“passage_id”: 1, “url”: “https://zhidao.baidu.com/
question/713780769091877645”, “passage_text”: “中国新疆的塔里木盆地，是世界上最大的内陆盆地，东西长约1500公里，南北最宽处约600公里。盆地底部海拔1000米左右，面积53万平方公里。”},
{“passage_id”: 2, “url”: “http://www.doc88.com/p-093375971649.html”, “passage_text”: “中国最大的固定、半固定沙漠天山与昆仑山之间又有塔里木盆地，面积　５３　万平方公里，是世界最大的内陆盆地。　盆地中部是塔克拉玛干大沙漠，面积　３３．７　万平方公里，为世界第二大流动性沙漠。”},
……]
“answer”: “塔里木盆地”,
“type”: “factoid”
}
```



# Timeline

time | todo
---- | ---
2017/9/1 | Start
2017/9/22 | Implement
2017/10/10 | Data release
2017/10/16 | 	Submit
2017/10/23 | Optimize
2017/11/1 | Finish



