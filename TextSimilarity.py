#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import time
from pprint import pprint
from gensim import corpora, models, matutils

# 基于向量空间模型实现的简单中文文本相似度计算，对于平均 200 词的 3200 篇文章，耗时约 45min

# 打印调试信息
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 读取文件并分词，同时过滤标点符号
count = 0
start_time = time.time()
with open('199801_clear.txt', 'r', encoding='GBK') as f_in:
    indexs = []   # 编号列表
    documents = []   # 文档分词列表
    index = ''   # 文档编号
    doc = []   # 文档分词
    for line in f_in:
        if line.strip():   # 跳过空行

            # 处理文章编号并以此为依据合并文章段落
            words = line.split()
            temp = words[0].rsplit('-', 1)[0]
            if index != temp:
                index = temp
                indexs.append(index)
                if doc:
                    documents.append(doc)
                doc = []
            for index1 in range(1, len(words)):
                if '/w' not in words[index1]:
                    doc.append(words[index1])
    documents.append(doc)

# 提取出现频数大于 1 的关键词作为词袋，以此为基础将文档向量化并使用 TF-IDF 作为词的权重
fre = {}
for doc in documents:
    for word in doc:
        if word in fre:
            fre[word] += 1
        else:
            fre[word] = 1
documents = [[word for word in doc if fre[word] > 1] for doc in documents]
bag = corpora.Dictionary(documents)   # 词袋（bag of words）
corpus = [bag.doc2bow(doc) for doc in documents]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# 计算两两文本之间的相似度并写入文件
similarity = []
num = len(corpus_tfidf)
for index1 in range(num):
    sim = []
    for index2 in range(index1 + 1, num):
        sim.append(matutils.cossim(corpus_tfidf[index1], corpus_tfidf[index2]))
    similarity.append(sim)
with open('result.txt', 'w') as f_out:
    for sim in similarity:
        f_out.write(','.join(map(str, sim)) + '\n')

print(time.time() - start_time)
