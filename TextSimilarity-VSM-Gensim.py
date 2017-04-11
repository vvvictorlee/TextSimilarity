#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import time
from gensim import corpora, models, similarities

# 基于向量空间模型实现的简单中文文本相似度计算，对于平均 170 词的 3148 篇文章，耗时约 20s
# 使用的是 Gensim：https://radimrehurek.com/gensim

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
corpus = [bag.doc2bow(doc) for doc in documents]   # 基于频数的文档向量列表
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]   # 转换成 TF-IDF 文档向量列表并持久化
corpora.MmCorpus.serialize('tmp/corpus_tfidf.mm', corpus_tfidf)
corpus_tfidf = corpora.MmCorpus('tmp/corpus_tfidf.mm')

# 构建文档相似度矩阵索引用于查询，再使用文档列表本身进行相似度查询（默认使用 Cosine）
index = similarities.SparseMatrixSimilarity(corpus_tfidf)
index.save('tmp/corpus_tfidf.index')
index = similarities.MatrixSimilarity.load('tmp/corpus_tfidf.index')
with open('result.txt', 'w') as f_out:
    for sim in index[corpus_tfidf]:
        f_out.write(','.join(map(str, sim)) + '\n')

print("总共耗时（秒）：" + str(time.time() - start_time))
