## 原始数据格式

* json格式
* 字段：{query_id, query, passages[passage_id,url,passage_text], answer,type}

## 提交结果的格式



## 数据处理

1. 训练数据格式

每条数据：

(query_id, question, passage_id, passage, answer, start, end, docAnswerScore,docQuestionScore,overlaps)

 (问题，文档，答案，开始位置，结束位置，文档类别, 文档与答案的匹配得分，文档与问题的匹配得分)
 
 * question: 问题中单词的id
 * passage: 文档中单词的id
 * answer: 答案中单词的id
 * start: 答案在文档里的开始位置
 * end: 答案在文档里的结束位置
 * docAnswerScore: 文档与答案的匹配得分（包含答案为1，不包含答案为0）
 * docQuestionScore:
 文档与问题的匹配得分
 * overlaps: 文档中词是否在问题中出现：出现为1，没有出现为0
 
 2. 对数据进行分词
 
 * 输入文件路径: train_data_path
 * 输出：分词后的数据
 * 格式为： [query_id, question, passage_id,passage, start,end, docAnswerScore]
 
 3. 建立词库
 
 4. 训练数据转换成词库id
 
 5. 计算docAnswerScore:
 * 输入: passage和answer
 * 输出：answer是否在passage中，1或0
 
 6. 计算docQuestionScore:
 * 输入：passage 和 question
 * 输出： docQuestionScore
 
 