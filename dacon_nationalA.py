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
import seaborn as sns

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

# 제안되는 법안명에 집중하여 법안이 주로 어떤 내용인 것인지에 대한 탐구
# 주된 법안들은 어떤 법안들이 제안되고 통과되는지 확인해보자 한다.

#######################
# 추가 정보. 출생지별 인원수 구하기
#######################
sum(people['BON'].isna()) #전체 인원 중 약 50%는 본관 확인 불가

# 출생지별 인원수 파악을 위해 본관이 없는 경우를 제외하고 확인해본다.
people_sub = people[-people['BON'].isna()]
people_sub = people_sub.reset_index(drop=True)

# "경남 진주"와 같이 혼재되어 있어 앞의 두 글자만을 본관으로 딴다. 한자의 경우 해석.
people_sub['BON'] = [people_sub['BON'][x][:2] for x in range(len(people_sub))]
people_sub['BON'] = [hanja.translate(people_sub['BON'][x], 'substitution') for x in range(len(people_sub))]
people_sub['BON'] = [people_sub['BON'][x].strip() for x in range(len(people_sub))]

# 본관을 기준으로 워드클라우드로 빈도수를 확인해본다.
# 김해, 밀양, 전주, 경주, 진주 등 경상도와 전라도 지역이 돋보임
random.seed(1)
sub_data = people_sub['BON'].value_counts()
wc=WordCloud(font_path='C:/Windows/Fonts/H2GTRE.ttf',max_font_size=150, width=600, height=350).generate_from_frequencies(sub_data)
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()

#######################
#1. 법안별 특성확인
#######################
def drop_nas(df, variable): #dataframe에서 na있는 변수drop
    df = df[-df[variable].isna()]
    return df

####################### 본회의 처리 안건 기준으로 확인
process_sub = drop_nas(process, 'AGE')
process_sub = drop_nas(process_sub, 'BILL_KIND')   
process_sub = drop_nas(process_sub, 'PROC_RESULT_CD')  

df = pd.DataFrame(process.groupby(by=['AGE','BILL_KIND'], as_index=False).size())
df = df.reset_index()
df.columns = ["age","bill_kind","count"]

# 의안활동구분(bill_kind)는 국회(age)별로 각 비중에 차이가 있을까?
df = df.pivot_table(values='count',index=df.age,columns=df.bill_kind)
df = df.fillna(0)
df['sum'] = df.sum(axis=1)

#비율기준값으로 계산
df['결산_ratio'] = df['결산'] / df['sum']
df['기타_ratio'] = df['기타'] / df['sum']
df['법률안_ratio'] = df['법률안'] / df['sum']
df['예산안_ratio'] = df['예산안'] / df['sum']

df[['결산',"기타","법률안","예산안"]].plot.area()
plt.title("국회별 본회의에서 처리되는 처리 안건 건수 비교")
plt.xlabel("국회(AGE)")
plt.ylabel("건 수")
plt.show() #-----최근 국회에 들어 본회의에서 처리되는 법률안의 건이 급격히 상승함

df[['결산_ratio','기타_ratio','법률안_ratio','예산안_ratio']].plot.area() 
plt.title("국회별 본회의에서 처리되는 처리 안건 간 비율 비교")
plt.xlabel("국회(AGE)")
plt.ylabel("비율")
plt.show() #-----21대 국회에 와서 예산안 비율이 높아짐을 확인

# 21대 국회는 2020.05.30부터 시작된 국회로 현재 임기 중(만기 예정일: 2024.05.29)
# 21대 국회의 예산안 관련 법안 확인

# 국회별 예산안 건수 확인
# 확인 결과 지난번 국회(20대)보다 이미 전체 처리 안건보다 보다 많음
fig, ax1 = plt.subplots(figsize=(12,6))
ax1.set_xlabel("국회(age)")
ax1.set_ylabel("전체 건수(line)")
ax1 = sns.lineplot(data=df["sum"], marker='o',ax=ax1)
ax1.legend(["전체 건수"])
ax2 = ax1.twinx()
ax2.set_ylabel("예산안 건수(bar)")
ax2 = sns.barplot(x=df.index,y="예산안",data=df, ax=ax2, alpha=0.5)
plt.show()


# def show_conditional(df, var1, value1, var2,value2):
#     df = df[(df[var1]==value1) & (df[var2]==value2)]
#     return df

# bills = show_conditional(process_sub, 'AGE',18,'BILL_KIND',"예산안")['BILL_NAME']
# kkma = Kkma() 
# bill_vocab = []
# for x in bills:
#     temp = x.replace("(","")
#     temp = temp.replace(")","")
#     temp = temp.replace("년도","")
#     temp = temp.replace("안","")
#     temp = temp.replace("정부","")
#     temp = temp.replace("계획","")
#     bill_vocab.extend(kkma.morphs(temp))

# bill_vocab_freq = pd.DataFrame({"vocab":bill_vocab})
# bill_vocab_freq = bill_vocab_freq["vocab"].value_counts()
# wc = WordCloud(font_path='C:/Windows/Fonts/H2GTRE.ttf',max_font_size=110, width=600, height=350).generate_from_frequencies(bill_vocab_freq)
# plt.imshow(wc)
# plt.axis('off')
# plt.show()


#######################국회의원 발의법의안 기준으로 확인

# 변수별 결측값 확인
suggest.isnull().sum()

# 국회별 처리상태 확인-임기만료폐기 건이 압도적으로 다수
sns.countplot(y='PROC_RESULT',data=suggest)
plt.show()

# 임기만료폐기 건의 국회별 건수 확인 - 건수가 최근 압도적으로 증가 --> 다른 처리상태와 비교해보자
sns.countplot(x="AGE",data=suggest[suggest['PROC_RESULT']=="임기만료폐기"])
plt.show()

# 처리상태(proc result)별 건수/비율기준으로 그래프 확인
def get_ratio(df,lst):
    df['sum'] = df[lst].sum(axis=1)
    for x in lst:
        df[x] = df[x] / df['sum']
    return df

suggest_ratio = pd.DataFrame(suggest.groupby(by=['AGE','PROC_RESULT'], as_index=False).size())
suggest_ratio.reset_index(inplace=True)
suggest_ratio = suggest_ratio.fillna(0)
suggest_ratio.columns = ['AGE','PROC_RESULT','count']
suggest_ratio = suggest_ratio.pivot_table(values="count",index=suggest_ratio.AGE, columns=suggest_ratio.PROC_RESULT)
suggest_ratio = suggest_ratio.fillna(0)

# 건수 기준의 stacked area plot
# -> 대략 18대 국회 이후, 임기만료폐기 건과 대안반영폐기 건이 급격히 증가
suggest_ratio.plot.area()
plt.show()

proc_result = suggest_ratio.columns.values
suggest_ratio = get_ratio(suggest_ratio, proc_result)

# 전체 건수 중 비율 기준 stacked area plot
# -> 비율기준으로 보았을 대, 21대의 국회는 최근 임기시작했으므로 임기만료폐기 건은 없으나 대안반영폐기 건이 이전 국회보다 증가
suggest_ratio[proc_result].plot.area()
plt.show()



#######################
# 3. 법안명으로 clustering 
#######################
### 3-1. kmeans clustering

docum = suggest['BILL_NAME'] #--------법안명만 docum (list형태)으로 저장
docum = list(docum)

kkma = Kkma() ##----------konlpy의 kkma 이용하여 형태소 분석함

sample_size = 5000
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


### 3-2. word cloud로 시각화
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