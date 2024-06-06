#!/usr/bin/env python
# coding: utf-8

# 記述統計へのDP適用

# In[4]:


get_ipython().system('pip install ja-ginza')


# ## 1. 必要なライブラリのインストール

# In[5]:


get_ipython().system('pip install spacy')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install python-dp')


# ## 2. データの読み込みと形態素解析
# 

# In[6]:


import pandas as pd

data_path = './data/reviews_with_sentiment.csv'

df = pd.read_csv(data_path)
df


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
import spacy
import itertools
from typing import List, Tuple
# spaCyの日本語モデルのロード
nlp = spacy.load('ja_ginza')

# 抽出する品詞の指定
POS = ['ADJ', 'ADV', 'INTJ', 'PROPN', 'NOUN', 'VERB']
MAX_TERMS_IN_DOC = 5
NGRAM = 1
MAX_DF = 1.0
MIN_DF = 0.0
NUM_VOCAB = 10000
TOP_K = 20

def flatten(*lists) -> list:
    res = []
    for l in list(itertools.chain.from_iterable(lists)):
        for e in l:
            res.append(e)
    return res

def remove_duplicates(l: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    d = {}
    for e in l:
        d[e[0]] = e[1]
    return list(d.items())

# 形態素解析
df["doc"] = [nlp(review) for review in df["review"]]


# ## 3. Bag-of-Wordsの生成

# In[8]:


bows = {}
cvs = {}

for sentiment in df["sentiment"].unique():
    tokens = []
    for doc in df[df["sentiment"] == sentiment]["doc"]:
        similarities = [(token.similarity(doc), token.lemma_) for token in doc if token.pos_ in POS]
        similarities = remove_duplicates(similarities)
        similarities = sorted(similarities, key=lambda sim: sim[1], reverse=True)[:MAX_TERMS_IN_DOC]
        tokens.append([similarity[1] for similarity in similarities])
    
    cv = CountVectorizer(ngram_range=(1, NGRAM), max_df=MAX_DF, min_df=MIN_DF, max_features=NUM_VOCAB)
    bows[sentiment] = cv.fit_transform(flatten(tokens)).toarray()
    cvs[sentiment] = cv


# In[ ]:


get_ipython().system('pip install numpy')


# ## 4. 上位単語の頻度を計算

# In[10]:


from pydp.algorithms.laplacian import Count
import numpy as np# 上位単語の頻度を計算
vocabs = {}
term_frequencies = {}

for sentiment in df["sentiment"].unique():
    bow = bows[sentiment]
    cv = cvs[sentiment]
    
    vocab = cv.vocabulary_
    term_frequency = np.sum(bow, axis=0)
    vocabs[sentiment] = vocab
    term_frequencies[sentiment] = term_frequency
    
    indices_topk = np.argsort(term_frequency)[::-1][:TOP_K]
    bow_topk = np.take(bow, indices_topk, axis=1)
    reverse_vocab = {vocab[k]: k for k in vocab.keys()}
    words = [reverse_vocab[i] for i in indices_topk]
    
    print(sentiment, ":")
    for w, c in zip(words, term_frequency[indices_topk]):
        print(w, ":", c)


# ## 5. 差分プライバシーの適用
# 

# In[12]:


from pydp.algorithms.laplacian import Count

def preprocess_for_private_counts(tf: np.ndarray) -> List[np.ndarray]:
    repeated_words = []
    for i, term in enumerate(tf):
        repeated_words.append(np.repeat(i, term))
    return repeated_words

def cal_private_count(epsilon: float, max_partition_contributed: float, max_contributions_per_partition: float, repeated_words: List[np.ndarray]) -> List[int]:
    private_counts = []
    for repeated_word in repeated_words:
        counter = Count(epsilon, max_partition_contributed, max_contributions_per_partition)
        count = counter.quick_result(repeated_word)
        private_counts.append(count)
    return private_counts

def top_k_words_and_counts(k: int, tf: np.ndarray, vocab: dict) -> List[Tuple[str, int]]:
    indices_topk = np.argsort(tf)[::-1][:k]
    reverse_vocab = {vocab[key]: key for key in vocab.keys()}
    words = [reverse_vocab[i] for i in indices_topk]
    counts = [tf[i] for i in indices_topk]
    return list(zip(words, counts))

epsilons = [0.01, 0.05, 0.1, 0.3, 0.7, 1.0, 2.0, 3.0, 7.0, 10.0]
MAX_DUPLICATED_TERMS = 1

# 結果を保存するための辞書を初期化
results = {sentiment: {"no DP": None, **{eps: None for eps in epsilons}} for sentiment in df["sentiment"].unique()}

for eps in epsilons:
    print("ε: ", eps)
    for sentiment in df["sentiment"].unique():
        repeated_words = preprocess_for_private_counts(term_frequencies[sentiment])
        private_counts = cal_private_count(eps, MAX_TERMS_IN_DOC, MAX_DUPLICATED_TERMS, repeated_words)
        words_and_counts = top_k_words_and_counts(TOP_K, private_counts, vocabs[sentiment])
        results[sentiment][eps] = words_and_counts
        print(sentiment, ":")
        print(words_and_counts)

# 差分プライバシーなしの頻度計算を保存
for sentiment in df["sentiment"].unique():
    words_and_counts = top_k_words_and_counts(TOP_K, term_frequencies[sentiment], vocabs[sentiment])
    results[sentiment]["no DP"] = words_and_counts


# ## 6. 結果を表形式で表示
# 

# In[14]:


def display_results(results):
    for sentiment, data in results.items():
        print(f"\nSentiment: {sentiment}\n")
        df = pd.DataFrame(columns=["rank", "word", "count"] + [f"ε={eps}" for eps in epsilons])
        for rank, (word, count) in enumerate(data["no DP"], start=1):
            row = {"rank": rank, "word": word, "count": count}
            for eps in epsilons:
                if eps in data and rank <= len(data[eps]):
                    row[f"ε={eps}"] = data[eps][rank-1][1] if rank-1 < len(data[eps]) else None
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        print(df)

display_results(results)


# ## 7.グラフの生成

# In[20]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')


# In[29]:


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 日本語フォントの設定
font_path = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'  # フォントパスを指定
font_prop = fm.FontProperties(fname=font_path)

def calculate_match_rate(original_top_k, dp_top_k):
    match_count = len(set(original_top_k) & set(dp_top_k))
    return match_count / len(original_top_k) if original_top_k else 0

def plot_match_rate(results, epsilons, sentiment_label):
    top_k_values = [3, 5, 10, 20]
    match_rates = {k: [] for k in top_k_values}

    for eps in epsilons:
        for k in top_k_values:
            original_top_k = [word for word, count in results[sentiment_label]["no DP"][:k]]
            dp_top_k = [word for word, count in results[sentiment_label][eps][:k]]
            match_rate = calculate_match_rate(original_top_k, dp_top_k)
            match_rates[k].append(match_rate)

    plt.figure(figsize=(10, 6))
    for k, rates in match_rates.items():
        plt.plot(epsilons, rates, label=f'top{k}')

    plt.xlabel('ε', fontproperties=font_prop)
    plt.ylabel('一致率', fontproperties=font_prop)
    plt.title(f'元上位単語との一致率 ({sentiment_label})', fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.grid(True)
    plt.show()

epsilons = [0.01, 0.05, 0.1, 0.3, 0.7, 1.0, 2.0, 3.0, 7.0, 10.0]

plot_match_rate(results, epsilons, 'negative')
plot_match_rate(results, epsilons, 'neutral')
plot_match_rate(results, epsilons, 'positive')


# In[25]:


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 日本語フォントの設定
plt.rcParams['font.family'] = 'IPAGothic'

def calculate_match_rate(original_top_k, dp_top_k):
    match_count = len(set(original_top_k) & set(dp_top_k))
    return match_count / len(original_top_k) if original_top_k else 0

def plot_match_rate(results, epsilons, sentiment_label):
    top_k_values = [3, 5, 10, 20]
    match_rates = {k: [] for k in top_k_values}

    for eps in epsilons:
        for k in top_k_values:
            original_top_k = [word for word, count in results[sentiment_label]["no DP"][:k]]
            dp_top_k = [word for word, count in results[sentiment_label][eps][:k]]
            match_rate = calculate_match_rate(original_top_k, dp_top_k)
            match_rates[k].append(match_rate)

    plt.figure(figsize=(10, 6))
    for k, rates in match_rates.items():
        plt.plot(epsilons, rates, label=f'top{k}')

    plt.xlabel('ε')
    plt.ylabel('一致率')
    plt.title(f'元上位単語との一致率 ({sentiment_label})')
    plt.legend()
    plt.grid(True)
    plt.show()

epsilons = [0.01, 0.05, 0.1, 0.3, 0.7, 1.0, 2.0, 3.0, 7.0, 10.0]

plot_match_rate(results, epsilons, 'negative')
plot_match_rate(results, epsilons, 'neutral')
plot_match_rate(results, epsilons, 'positive')


# ## 分析ケース②: MLアルゴリズムへの差分プライバシー適用

# In[26]:


get_ipython().system('pip install diffprivlib')


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
import numpy as np
import spacy
from typing import List, Tuple
import itertools
import matplotlib.pyplot as plt
from diffprivlib.models import GaussianNB as DPGaussianNB

# データの読み込み
data_path = 'path/to/reviews_with_sentiment.csv'  # 実際のファイルパスに変更してください
df = pd.read_csv(data_path)

# spaCyの日本語モデルのロード
nlp = spacy.load('ja_ginza')

# 抽出する品詞の指定
POS = ['ADJ', 'ADV', 'INTJ', 'PROPN', 'NOUN', 'VERB']
MAX_TERMS_IN_DOC = 5
NGRAM = 1
MAX_DF = 1.0
MIN_DF = 0.01
NUM_VOCAB = 10000

def flatten(*lists) -> list:
    res = []
    for l in list(itertools.chain.from_iterable(lists)):
        for e in l:
            res.append(e)
    return res

def remove_duplicates(l: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    d = {}
    for e in l:
        d[e[0]] = e[1]
    return list(d.items())

# 形態素解析とBoWの生成
tokens = []
for doc in df["review"]:
    parsed_doc = nlp(doc)
    similarities = [(token.similarity(parsed_doc), token.lemma_) for token in parsed_doc if token.pos_ in POS]
    similarities = remove_duplicates(similarities)
    similarities = sorted(similarities, key=lambda sim: sim[1], reverse=True)[:MAX_TERMS_IN_DOC]
    tokens.append([similarity[1] for similarity in similarities])

cv = CountVectorizer(ngram_range=(1, NGRAM), max_df=MAX_DF, min_df=MIN_DF, max_features=NUM_VOCAB)
bow = cv.fit_transform([" ".join(ts) for ts in tokens]).toarray()

# ラベルの付与とデータセットの分割
m = {
    "positive": 1,
    "neutral": 0,
    "negative": 0,
}
df["sentiment"] = df["sentiment"].map(m)
df["bow"] = bow.tolist()

X_train, X_test, y_train, y_test = train_test_split(df["bow"], df["sentiment"], test_size=0.2)
X_train = [list(x) for x in X_train]
X_test = [list(x) for x in X_test]

# 差分プライバシーなしのナイーブベイズ
clf = SklearnGaussianNB()
clf.fit(X_train, y_train)
print("Non-DP accuracy: ", clf.score(X_test, y_test))

# 差分プライバシーありのナイーブベイズ
epsilons = np.logspace(-2, 2, 50)
dim = np.array(X_train).shape[1]
lowers = np.zeros(dim)
uppers = np.ones(dim)
accuracies = {}

for epsilon in epsilons:
    accuracy = []
    for _ in range(20):
        dp_clf = DPGaussianNB(bounds=(lowers, uppers), epsilon=epsilon)
        dp_clf.fit(X_train, y_train)
        accuracy.append(dp_clf.score(X_test, y_test))
    accuracies[epsilon] = accuracy

# 結果をグラフに描画
x = epsilons
y = [np.mean(accuracies[eps]) for eps in epsilons]
e = [np.std(accuracies[eps]) for eps in epsilons]

plt.figure(figsize=(10, 6))
plt.semilogx(x, y)
plt.errorbar(x, y, yerr=e, marker='o', capthick=1, capsize=10, lw=1)
plt.xlabel('ε')
plt.ylabel('accuracy')
plt.ylim(0, 1)
plt.title('ナイーブベイズにおけるεとaccuracyの関係')
plt.grid(True)
plt.show()

