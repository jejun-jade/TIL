Deep Learning Study
=================

# neural net(fully connected) 
- input layer
	+ input layer의 노드의 개수는 데이터의 개수(eg.28*28 이미지의 경우 28*28)
- hidden layer
	+
- output layer
	+ node의 개수는 labeling에 해당하는 경우를 찾는 것
	+ 개/고양이를 분류하는 모델이라고 했을 때 계산만 하면 predict -> back propagation(역전파)

- 역전파를 하기 위해서?
	+ loss function이 필요하다. loss를 가장 줄이는 방법으로 학습시킨다.
	+ 계산 -> activate fn. (eg. 젤루, 렐루함수 등)

- bert: 워드 임베딩 
- FCNN: Fully Connected Nueral Net(여기까지는 2시간이면 가능)
- overfitting: loss function을 너무 많이 주는 경우에 나타나게 됨
	+ 과적합을 막기 위해서 drop out, 앙상블,  L2 regulation

# CNN
- CNN: 2 * 2를 유지할 수 있기 때문에 (fully는 무조건 1차원으로 분석) clisification(분류)에 많이 활용
- 텍스트 분석도 CNN으로 하면 
- CNN의 특징 2가지:
	+ convolution layer: 
	+ pooling layer: 2 * 2중에 가장 큰 값을 가져와서 새로운 벡터를 만드는 것(max pooling) -> 특징을 뽑을 수 있음. 주변의 노이즈를 제거한다.	

# NLTK
- 동사 원형 처리, 필요없는 데이터 제거(a, an 등)
- word를 vector로 바꿔야

# RNN, LSTM (순환신경망)
- 대화문에서 대화의 시간차를 고려해야 했다면 순환신경망을 사용했어야 했다. 

# 대회 리뷰
- 머신러닝으로 진행
- 데이터가 많이 않아서 뉴럴넷을 활용하기 어려웠음
- 학습 데이터에 overfitting되었을 가능성. 
- 머신러닝 통계기법
	+ SVM(Support Vector Machine): x/y축을 두고 두 그룹을 나누는 평면(초평면)을 그어 주는 것. 보통 고차원의 벡터임
	+ Naive Bayse: 예를 들어 스팸메일이 왔다고 했을때 ..? 블로그 글 보내 주기로

# Word Imbeding[Sparse(분리되어 있음) - 그래프에 표시했을때 각 점이 떨어져있음]
- 단어를 분석할 수 있는 데이터로 변환하는 것(=숫자, 벡터)
- input layer의 형태를 유지하기 위해서 같은 크기의 벡터로 변환한다. eg. 강아지 = [1,0,0], 고양이 = [0,1,0] = one-hot incoding <- count based representation(숫자를 몇 번 세느냐?)
- 단점: 단어가 많아지면 벡터가 너무 커짐, 단어 사이의 관계를 알 수가 없다. 
- eg. I am a boy. boy is 라는 문장을 벡터로 표현하려면? 먼저 사전을 만들어야 한다. 
- 1:I, 2:an, 3:a, 4:boy, 5:is -> [1,1,1,2,1]
- bag of word: 단어들이 가방에 들어있음  
- count based > back of word, one-hot incoding, 등등 
- TF(Turn Frequncy)-IDF(Inverse Document Frequncy): 단어빈도수, 역문서빈도수 
- eg. 뉴스 기사가 5만개 있을 때 문서에 하나의 기사 안에 '나는'이 몇 번 나오는지: TF
- eg. 전체 기사에서 '나는'이 나오는 다큐먼트 개수 : DF,  : IDF 
- 모든 문서에서 너무 많이 언급이 되면 이걸 제거하기 위해서 DF-IDF를 사용하는 것. -> 사용하게 되면 [1,0,0,0] -> [0.7,0,0,0] 이런식으로 가중치를 줘서 쓸수 있음. 하지만 여전히 단어 사이의 관계는 알 수가 없음.
- 여기까지가 1세대

# Word Imbeding 2세대[dense]
- neural net representaton
- word2vec, fasttext, globe 
- w2v을 하게 되면 king-man+women-queen이 가능하게 된다.
- 쉘로우(단층) 뉴럴넷을 사용
- 특징: 0이 없어! eg.boy = [0.2,0.3 ... ,7.2, 3]
- 특징을 뽑아서(검정색, 빨간색, 파란색, 동그라미, 세모) 특징이 2가지로 줄여서 임베딩으로 표현한 것. 단! 하나의 숫자가 하나의 특징은 아님
- imbeding dimention(=하나의 단어가 얼마나 긴 벡터로 표현될 지)는 사용자가 정할 수 있다. 
- 3차원에서 man, woman, king, queen 이 제 위치에 들어가 있어서 실제로 king-man+woman=queen
- 그러면 위치는 어떻게 알 수 있냐?

# 위치를 아는 방법?
- CBOW:countinous bag of word(연결된 가방?) < word2vec 기법 중 하나(Skip-gram도 있음)
	+ 주변 단어로 중심단어를 예측하는 것
	+ 반경 범위를 2라고 할 때 I am a boy -> I를 타겟단어라고 했을 때 [key,value] -> [주변단어, 타겟단어]
	+ [am, i],[a,i] <- i를 타겟단어
	+ [i, am],[a,am],[boy,am] <- am이 타겟단어
	+ 이제 뉴럴넷에 넣어보자

## 뉴럴넷에 넣어보자
- 딕셔너리가 4개니까 input도 4개.
- I를 넣어서 am을 예측해보자.
- [1,0,0,0]을 넣어서 [0,1,0,0]이 나오면 정답. [0,0,1,0]이 나오면 오답.  
- 역전파를 하면서 weight가 업데이트 되면서 i, am이 가까워진다.
- input-hidden layer사이의 n * v (웨이트의 값들) -> 벡터로 표시.
- 나온 벡터값들을 딕셔너리를 사용. 없는 단어는 패딩으로 줌
- bert는 이미 단어간 사전을 만들어 둔 걸 사용
- word2vec은 쉘로우로 되어있는데 깊지 않음. 오버피팅되지 않음. 

- Skip gram: 중심 단어 하나로 주변 단어를 예측하는 것. () am () boy.
- I () a boy. 일때 [i,am], [am, a], [am, boy] 
- skip gram이 역전파를 더 많이 일으키기 때문에 정확도가 높아짐
- 단어간의 관계를 파악하기 위해서. 
- 지도/비지도/강화학습 중에 이건 비지도학습
- 2세대의 문제점은 문맥에 따라 단어의 진짜 뜻을 파악할 수 없다. 







