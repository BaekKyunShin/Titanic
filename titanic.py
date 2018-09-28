import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # setting saeborn default for plots

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# train.head() # train의 앞 5개만 출력
# train.shape # (891, 12) 출력 / 891승객 정보, 12개의 feature정보
# train.isnull().sum() # feature별 null 갯수

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    '''
    train['Survived']==1 은 Survived가 1이면 True, 아니면 False로 모든 행 return
    train[train['Survived']==1]은 Survived가 1인 행만 return
    train[train['Survived']==1]['Sex']은 위 DF에서 Sex 칼럼만 return
    train[train['Survived']==1]['Sex'].value_counts()는 칼럼의 고유 요소의 갯수 return
    따라서 Survived는 (2,)인 DataFrame임, 즉 Series
    '''
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead]) # survived가 1행, dead가 2행
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    plt.show()


trainTestData = [train, test] # combining train and test dataset

for dataset in trainTestData: # 이름에서 Mr. Mrs. Miss. 등만 추출하여 title로 저장
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False) 

# title을 숫자로 mapping / Mr, Miss, Mrs를 제외하고는 모두 others로 간주
titleMapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in trainTestData: 
    dataset['Title'] = dataset['Title'].map(titleMapping) # dataset['Title']은 Series 객체이므로 map 함수 적용가능

# DataFrame에서 Name 열은 삭제 (Title로 뽑았기 때문에 더 이상 필요 없음)
train.drop('Name', axis=1, inplace=True) # inplace = False는 copy를 만듦, 원본 DF는 변화 없음, True는 원본 DF를 변화시키고, 현 라인에서는 None을 return 함
test.drop('Name', axis=1, inplace=True)


sexMapping = {"male": 0, "female": 1}
for dataset in trainTestData:
    dataset['Sex'] = dataset['Sex'].map(sexMapping)

# 4.4Age in[33]