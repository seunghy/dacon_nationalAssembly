import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hanja
# import folium
# import googlemaps
from wordcloud import WordCloud
import random
import matplotlib.font_manager as fm
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from konlpy.tag import Kkma
from gensim.models import Word2Vec
import networkx as nx

fm.get_fontconfig_fonts()
font_location = 'C:/Windows/Fonts/H2GTRE.ttf' # For Windows
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

path = "C:/Users/seunghyeon/Desktop/dacon/emotion/open/"
people = pd.read_csv(path+"new_people.csv", encoding='cp949')
suggest = pd.read_csv(path+"suggest.csv", encoding='utf-8')
process = pd.read_csv(path+"process.csv", encoding='utf-8')

people.head()
people.columns

#######################
#1. 출생지별 인원수 구하기
#######################
sum(people['BON'].isna()) #전체 인원 중 약 50%는 본관 확인 불가
people_sub = people[-people['BON'].isna()]
people_sub = people_sub.reset_index(drop=True)
people_sub['BON'] = [people_sub['BON'][x][:2] for x in range(len(people_sub))]
people_sub['BON'] = [hanja.translate(people_sub['BON'][x], 'substitution') for x in range(len(people_sub))]
people_sub['BON'] = [people_sub['BON'][x].strip() for x in range(len(people_sub))]

random.seed(1)
sub_data = people_sub['BON'].value_counts()
wc=WordCloud(font_path='C:/Windows/Fonts/H2GTRE.ttf',max_font_size=150, width=600, height=350).generate_from_frequencies(sub_data)
plt.imshow(wc,interpolation='bilinear')
plt.show()

#######################
#2. 법안별 특성확인
#######################
process_sub = process[-process['AGE'].isna()]
process_sub = process_sub[-process_sub['BILL_KIND'].isna()]
process_sub = process_sub[-process_sub['PROC_RESULT_CD'].isna()]

df = pd.DataFrame(process.groupby(by=['AGE','BILL_KIND'], as_index=False).size())
df = df.reset_index()
df.columns = ["age","bill_kind","count"]

df = df.pivot_table(values='count',index=df.age,columns=df.bill_kind)
df = df.fillna(0)
df['sum'] = df.sum(axis=1)


#비율기준
df['결산'] = df['결산'] / df['sum']
df['기타'] = df['기타'] / df['sum']
df['법률안'] = df['법률안'] / df['sum']
df['예산안'] = df['예산안'] / df['sum']
del df['sum']

df.plot.area()
plt.show() #-----21대 국회에 와서 예산안 비율이 높아짐을 확인


#######################
# 3. 법안명으로 clustering 
#######################
### 4-1. kmeans clustering

docum = suggest['BILL_NAME'] #--------법안명만 docum (list형태)으로 저장
docum = list(docum)

kkma = Kkma() ##----------konlpy의 kkma 이용하여 형태소 분석함

sample_size = 2000
split_list = []
for x in range(sample_size):
    temp = docum[x].replace("일부개정법률안","")   #------"일부개정법률안"과 "법률"의 경우 대다수 문장에 포함되어 제외시킴
    temp = temp.replace("법률","")
    nouns = ' '.join(kkma.morphs(temp))
    split_list.append(nouns)

    tfidf = TfidfVectorizer() #------------vector화
    tfidf_fit = tfidf.fit_transform(split_list)

split_list[:3]

voca_list = [] #-------vector화한 단어를 나누기위한 리스트
for item in split_list:
    voca_list.append(item.split(' '))

vocab = [] #------각 문장 내 단어가 한 단어인 것은 제외시킴
for i in range(len(voca_list)):
    ele = []
    for j in range(len(voca_list[i])):
        if len(voca_list[i][j])>1:
            ele.append(voca_list[i][j])
    vocab.append(ele)

df_dict = {"token":split_list, "sen":docum[:sample_size]}
df = pd.DataFrame(df_dict)
del df_dict
df['title'] = 0
df['num']= 0

for k in range(len(df)):
    df.iloc[k,3] = k
    df.iloc[k,2] = docum[k]
df.tail()


x = normalize(tfidf_fit)

def elbow(normalizedData, Clusters):  #Clustering size 가늠하기
    sse = []
    for i in range(1,Clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(normalizedData)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(7,6))
    plt.plot(range(1,Clusters), sse, marker='o')
    plt.xlabel('number of cluster')
    plt.xticks(np.arange(0,Clusters,1))
    plt.ylabel('SSE')
    plt.title('Elbow Method - number of cluster : '+str(Clusters))
    plt.show()

elbow(x, 50)  #----------cluster_num 확인용

tfidf = TfidfVectorizer()
feature_vec = tfidf.fit_transform(df['token'])

clusters_num = 12
km_cluster = KMeans(n_clusters=clusters_num, max_iter=10000, random_state=0) #---------kmeans clustering
km_cluster.fit(feature_vec)
cluster_label = km_cluster.labels_
cluster_centers = km_cluster.cluster_centers_


df['cluster_label'] = cluster_label
df.head()
# for i in range(clusters_num):
#   print('<<Clustering Label {0}>>'.format(i)+'\n')
#   print(df.loc[df['cluster_label']==i])
df.loc[df['cluster_label'] == 4] #-----눈으로 확인용


### 4-2. word cloud로 시각화
##형태소별 빈도수 확인
import collections
count_df = sum(vocab,[])
# collections.Counter(count_df)
count_df = pd.DataFrame(count_df)
count_df.columns = ["word"]
count_df = count_df["word"].value_counts()
temp = WordCloud(font_path='C:/Windows/Fonts/H2GTRE.ttf',max_font_size=120, width=600, height=350).generate_from_frequencies(count_df)
plt.imshow(temp)
plt.show()


##word2Vec 이용하여 유사도 확인
test = Word2Vec(vocab, size=sample_size/200, window=sample_size*0.05, min_count=sample_size/500, iter=200, sg=1) #window:중심단어 기준 좌우 n개 단어 학습, sg=1:skip_gram
# print(test.most_similar(positive=['운영'], topn=10))
# print(test.similarity("농업","농수산물"))
vocab[:10]

# sample = list(pd.DataFrame(sum(vocab,[]))[0].value_counts().index)
# test.most_similar(positive=['세법'], topn=1)[0][1]

word_list1 = [] #word2vec을 통해 정제된 단어리스트
word_list2 = [] #word_list1에 대응하는 단어리스트
weight_list = [] #word_list1의 단어와 word_list2의 각 단어 간 유사도

for x in test.wv.vocab.keys():
    temp = test.most_similar(positive=x, topn=3)
    for y in range(len(temp)):
        word_list1.append(x)
        word_list2.append(temp[y][0])
        weight_list.append(temp[y][1])

df = pd.DataFrame({"word1":word_list1,"word2":word_list2, "weight":weight_list})


#################################################################################
# networkx를 이용한 network 그리기
# 1안
font_name = fm.FontProperties(fname="C:/Windows/Fonts/H2GTRE.ttf").get_name()

G = nx.from_pandas_edgelist(df, source="word1",target="word2")
p = nx.spring_layout(G, k=0.4)
nx.draw_networkx_nodes(G,p)
nx.draw_networkx_edges(G,p)
nx.draw_networkx_labels(G,p,font_family=font_name,font_size=10)
plt.axis('off')
plt.show()

#2안
e = [(df.iloc[i,0],df.iloc[i,1]) for i in range(len(df))]
G = nx.Graph(e )
p = nx.spring_layout(G, k=0.4, random_state=123,fixed=None)
nx.draw_networkx_labels(G,p,font_family=font_name,font_size=10)
nx.draw_networkx(G,p, with_labels=False)
plt.axis('off')
plt.show()

#3안
node_list = []
node_list.extend(list(df.word1.unique()))
node_list.extend(list(df.word2.unique()))
node_list = set(node_list)  #from, to를 고려하여 unique한 node list생성

edge_list = [(df.iloc[i,0],df.iloc[i,1],df.iloc[i,2]) for i in range(len(df))] #df를 이용하여 edge list 생성
G = nx.Graph()

G.add_edges_from((e[0], e[1],{'weight':e[2]}) for e in edge_list)
central = dict(nx.degree(G)) #nx.degree(G)
G.add_nodes_from((n, {"size":central[n]/max(central.values())*25, "title":n})for n in node_list)

pos = nx.spring_layout(G, random_state=123)
nx.draw_networkx_nodes(G,pos,node_size=[size/max(central.values()) *300 for size in central.values()])

widths = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edges(G, pos, width=[i/max(widths.values())*2 for i in list(widths.values())],edge_color="grey")
nx.draw_networkx_labels(G,pos,font_family=font_name,font_size=15)
plt.axis('off')
plt.show()

#################################################################################
## interactive network - pyvis 이용
## 1안
from pyvis.network import Network

net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

for e in edge_list:
    source = e[0]
    target = e[1]
    weight = e[2]/max(edge_list)[2] * 100

    net.add_node(source,source, title=source)
    net.add_node(target, target, title=target)
    net.add_edge(source, target, title=weight)

neighbor = net.get_adj_list()
for n in net.nodes:
    n["title"] += "의 이웃노드:<br>" + "<br>".join(neighbor[n["id"]])

net.show("billNetwork.html")

##2안--기존의 networkx를 이용한 그래프 이용
net = Network(height="750px", width="120%", bgcolor="#222222", font_color="white")
net.from_nx(G)

neighbor = net.get_adj_list()
for n in net.nodes:
    n["title"] += "의 인접노드:<br>" + "<br>".join(neighbor[n["id"]])

net.show("billNetwork.html")