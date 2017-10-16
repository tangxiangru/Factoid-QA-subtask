## 开发环境说明

* python 3.5
* tensorflow 1.3
* 分词工具：jieba

## 文件目录说明
1. datasets 存放数据
2. data: 处理数据
3. models: 模型
4. weights: 保存参数
5. myutils: 通用的API
6. docs: 文档 


## 训练数据格式
1. 每条数据：

(query_id, question, passage_id, passage, answer, start, end, docAnswerScore,docQuestionScore,overlaps)

 (问题，文档，答案，开始位置，结束位置，文档类别, 文档与答案的匹配得分，文档与问题的匹配得分,)
 
 * question: 问题中单词的id
 * passage: 文档中单词的id
 * answer: 答案中单词的id
 * start: 答案在文档里的开始位置
 * end: 答案在文档里的结束位置
 * docAnswerScore: 文档与答案的匹配得分（包含答案为1，不包含答案为0）
 * docQuestionScore:
 文档与问题的匹配得分
 * overlaps: 文档中词是否在问题中出现：出现为1，没有出现为0
 
 2. 特征抽取
 
 *	词向量
 *	字向量
 *	字->词 (FOFE, RNN，...)
 *	词性标注
 *	命名实体识别
 *	文档中词是否在问题中
 *	IF-IDF
 *	主题词，关键词抽取
 *	词典?
 *	...

 4. 数据标签
 * 预测开始和结束位置： start, end 标注
 * 序列标注的方式： BIO标注
 
 ## 测试
 
 1. 构建测试集
 
 2. 计算F1, Accuarcy
 
 
