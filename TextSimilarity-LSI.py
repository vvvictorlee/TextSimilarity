#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import time
import math
from collections import defaultdict
from gensim import corpora, models, similarities

# 基于潜语义索引（LSI）实现的简单中文文本相似度计算，对于平均 170 词的 3148 篇文章，耗时约 24s
# 为了符合老师对作业的要求，LSI 前半部分构建文档-标引项矩阵要自己手动构建
# 后面 SVD 部分使用的是 Gensim：https://radimrehurek.com/gensim

# 打印调试信息
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 读取文件并分词，同时过滤无意义项，形成文档词频向量列表
start_time = time.time()
indexs = []  # 编号列表
documents = []  # 词频向量列表
with open('199801_clear.txt', 'r', encoding='GBK') as f_in:
    index = ''   # 文档编号
    doc = defaultdict(int)   # 文档词频统计
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
                doc = defaultdict(int)
            for i in range(1, len(words)):

                # 学习其他同学过滤掉无意义的标点符号和助词等
                word = words[i]
                if '/w' not in word and '/y' not in word and '/u' not in word \
                        and '/c' not in word:
                    doc[word] += 1
    documents.append(doc)

# 提取至少在两个文档中出现过的关键词作为词袋
fre = defaultdict(int)   # 出现过的文档数
for doc in documents:
    for word in doc:
        fre[word] += 1
bag = defaultdict(int)
count = 0
for word in fre:
    if fre[word] > 1:
        bag[word] = count
        count += 1

# 计算文档的 TF-IDF 向量，并形成文档-标引项矩阵（这里用的是稀疏矩阵表达）
docs_total = len(documents)   # 文档总数
corpus_tfidf = []   # 稀疏的文档-标引项矩阵
for doc in documents:
    words_total = 0
    vector = []
    for word in doc:
        word_total += doc[word]
    for word in doc:
        if word in bag:
            tf_idf = (doc[word] / words_total) * math.log(docs_total / (fre[word] + 1))
            vector.append((bag[word], tf_idf))
    corpus_tfidf.append(vector)

# 后面 SVD 部分就是直接使用 Gensim 来处理文档-标引项矩阵得到结果了
lsi = models.LsiModel(corpus_tfidf, num_topics=300)
corpus_lsi = lsi[corpus_tfidf]   # 转换成潜语义文档向量列表并持久化
corpora.MmCorpus.serialize('tmp/corpus_lsi.mm', corpus_lsi)
corpus_lsi = corpora.MmCorpus('tmp/corpus_lsi.mm')

# 构建文档相似度矩阵索引用于查询，再使用文档列表本身进行相似度查询（默认使用 Cosine）
index = similarities.MatrixSimilarity(corpus_lsi)
with open('result.csv', 'w') as f_out:
    for sims in index[corpus_lsi]:
        f_out.write(','.join(map(str, sims)) + '\n')

# 输出前 100 个文档最匹配的前 30 个文档编号
with open('100.csv', 'w') as f_out:
    for i in range(0, 100):
        sims = sorted(enumerate(index[corpus_lsi[i]]), key=lambda x: x[1], reverse=True)
        for j in range(0, 30):
            f_out.write(indexs[sims[j][0]] + ' - ' + str(sims[j][1]) + ',')
        f_out.write('\n')

print("总共耗时（秒）：" + str(time.time() - start_time))
