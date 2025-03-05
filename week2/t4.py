import jieba
import jieba.posseg as pseg
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# -------------------------------
# 1. 读取文件，打印前10行观察文本
# -------------------------------
with open('/Users/jm/Code/JM_Python_Weekly_Homework/week2/week2.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
print("文本总行数：", len(lines))

# 清除每行的首尾空白字符
documents = [line.strip() for line in lines if line.strip()]

print("前10行文本：")
for line in documents[:10]:
    print(line)
    
# -------------------------------
# 2. 使用jieba分词，对所有文档进行分词，并统计词频
# -------------------------------
all_words = []
for doc in documents:
    tokens = jieba.lcut(doc)
    all_words.extend(tokens)

# 使用Counter统计词频
word_counts = Counter(all_words)
print("\n【包含停用词】词频最高的10个词：")
print(word_counts.most_common(10))

# -------------------------------
# 3. 按词频排序（已用most_common排序），输出前10个词
# （上一步已实现）
# -------------------------------

# -------------------------------
# 4. 引入停用词过滤，重新观察词频排序结果
# 可自定义停用词表（以下为示例停用词）
# -------------------------------
stopwords = set(['的', '了', '在', '和', '是', '我', '有'])

filtered_words = [word for word in all_words if word not in stopwords and word.strip() != '']
filtered_word_counts = Counter(filtered_words)
print("\n【过滤停用词后】词频最高的10个词：")
print(filtered_word_counts.most_common(10))

# -------------------------------
# 5. 利用wordcloud对高频词进行可视化（词云）
# -------------------------------
# 请确保系统中有中文字体（这里使用 simhei.ttf），否则中文可能显示不全
wc = WordCloud(font_path='STHeiti Medium.ttc', width=800, height=400, background_color='white')
wc.generate_from_frequencies(filtered_word_counts)

plt.rcParams['font.sans-serif'] = ['STHeiti Light']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("词云 - 高频词")
plt.show()

# -------------------------------
# 6. （附加）对词性进行分析，统计不同词性的频率，并对名词生成词云
# -------------------------------
pos_counts = Counter()
pos_words = {}
for doc in documents:
    words = pseg.lcut(doc)
    for word, flag in words:
        pos_counts[flag] += 1
        pos_words.setdefault(flag, []).append(word)

print("\n词性频率统计：")
print(pos_counts.most_common())

# 针对名词（词性标记 'n'）生成词云
noun_list = pos_words.get('n', [])
# 过滤停用词和过短的词
noun_filtered = [word for word in noun_list if word not in stopwords and len(word.strip()) > 1]
noun_counts = Counter(noun_filtered)

wc_noun = WordCloud(font_path='STHeiti Medium.ttc', width=800, height=400, background_color='white')
wc_noun.generate_from_frequencies(noun_counts)

plt.figure(figsize=(10, 5))
plt.imshow(wc_noun, interpolation='bilinear')
plt.axis('off')
plt.title("词云 - 名词")
plt.show()

# -------------------------------
# 7. （附加）利用tuple表示bigram，统计所有bigram的频率，并可视化高频bigram
# -------------------------------
bigrams = []
for doc in documents:
    tokens = jieba.lcut(doc)
    if len(tokens) >= 2:
        # 生成bigram: (token[i], token[i+1])
        doc_bigrams = list(zip(tokens, tokens[1:]))
        bigrams.extend(doc_bigrams)

bigram_counts = Counter(bigrams)
print("\n最常出现的10个bigram：")
print(bigram_counts.most_common(10))

# 为词云将bigram转为字符串（使用下划线连接），可过滤低频bigram
bigram_str_counts = {'_'.join(bigram): count for bigram, count in bigram_counts.items() if count > 1}

wc_bigram = WordCloud(font_path='STHeiti Medium.ttc', width=800, height=400, background_color='white')
wc_bigram.generate_from_frequencies(bigram_str_counts)

plt.figure(figsize=(10, 5))
plt.imshow(wc_bigram, interpolation='bilinear')
plt.axis('off')
plt.title("词云 - 高频 Bigram")
plt.show()

# -------------------------------
# 8. （附加）利用词频筛选特征词，构建文本向量表示，并计算句子之间相似性
# -------------------------------
# 假设用过滤停用词后的结果中，选择前20个词作为特征词
top_features = [word for word, _ in filtered_word_counts.most_common(20)]
print("\nTop 20 特征词：", top_features)

# 为每个文档构建向量（统计每个特征词在文档中出现次数）
doc_vectors = []
for doc in documents:
    tokens = jieba.lcut(doc)
    token_counts = Counter(tokens)
    vector = [token_counts.get(word, 0) for word in top_features]
    doc_vectors.append(vector)

doc_vectors = np.array(doc_vectors)
# 计算文档间的余弦相似度矩阵
similarity_matrix = cosine_similarity(doc_vectors)

print("\n文档相似度矩阵：")
print(similarity_matrix)