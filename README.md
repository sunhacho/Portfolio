# <머신러닝_ Chatgpt를 통한 감정분석>

### 문제 정의 (주제)

IT 시장을 바꾼 ChatGPT에 대한 주제를 선정.

- 기존 주제 :  ChatGPT 사용자에 대한 인식과 관련된 데이터 크롤링 및 감정분석을하여 만족도를 알아보기
- 바뀐 주제 : ChatGPT를 통해 감정 분석한 모델과 인간이 만든 모델의 성능을 비교

→ ChatGPT를 사용 여부가 아닌 ‘ChatGPT를 어떻게 활용할 수 있을까’에 대한 주제로 변경. 

WHY? : 이젠 ChatGPT를 사용해야 한다고 생각하기 때문. 

### 데이터 수집

- 기존의 데이터 수집 : 영화 ‘인어공주’에 대한 리뷰에 대해 감정 분석을 하기 위해 트위터 데이터 크롤링.
    - 바꾼 이유 → 트위터에 뉴스 등과 같은 글이 많아 중립 비율이 90%로, 감정 분석을 하기 힘든 데이터가 수집되었기 때문.
- 새로 정의한 데이터 : Kaggle의 ‘ChatGPT’와 관련된 감정 분석이 가능한 데이터

### 데이터 전처리 / 데이터 분석 / 모델링

- 데이터 전처리
    - 불필요한 열 제거, 영어 데이터만 사용, 리트윗/특수문자/URL 제거 등의 문자에 대한 전처리를 진행함.
    - 여전히 중립 비율이 50%에 가까워 데이터 비율을 맞춰주기 위해 중립을 부정에 포함하여 프로젝트를 진행함.
    - code
        
        ```python
        import pandas as pd
        import multiprocessing
        import numpy as np
        from collections import defaultdict
        from tqdm import trange
        import re
        
        df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/chatgpt1.csv',engine='python',encoding='utf-8')
        #불필요한 열 제거
        del df['Tweet Id']
        del df['Permalink']
        del df['User']
        del df['Outlinks']
        del df['CountLinks']
        del df['ReplyCount']
        del df['RetweetCount']
        del df['LikeCount']
        del df['Source']
        del df['Media']
        del df['QuotedTweet']
        del df['MentionedUsers']
        del df['Datetime']
        del df['Username']
        del df['QuoteCount']
        del df['ConversationId']
        del df['hashtag']
        del df['hastag_counts'] 
        
        #영어만 뽑아옴
        en = df[df['Language'] == 'en']
        
        #랜덤으로 800개의 행을 뽑아옴
        chatGPT_sentiment_english = en.sample(800, replace = False)
        
        #올바르게 뽑아왔는지에 대한 확인코드 
        chatGPT_sentiment_english.to_csv('data1.csv',encoding='utf-8')
        chat_data = pd.read_csv('/content/data1.csv',engine='python',encoding='utf-8')
        
        #데이터 전처리 (대문자/소문자 구분 x, url 제거 등)
        list_text = []
        
        for i in trange(len(chat_data)):
        
            pre_text = chat_data['Text'][i]
                # Retweets 제거
            text = re.sub('RT @[\w_]+: ', '', pre_text)
        
                # enticons 제거
            text = re.sub('@[\w_]+', '', text)
        
                # URL 제거
            text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ' ',
                              text)  # http로 시작되는 url
            text = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", ' ',
                              text)  # http로 시작되지 않는 url
                #     pattern = '(http|ftp|https)://(?:[-\w.]|(?:\da-fa-F]{2}))+'
                #     text = re.sub(pattern = pattern, repl = ' ',string=text)
        
                # Hashtag 제거
            text = re.sub('[#]+[0-9a-zA-Z_]+', ' ', text)
        
                # 쓰레기 단어 제거
            text = re.sub('[&]+[a-z]+', ' ', text)
        
                # 특수문자 제거
            text = re.sub('[^0-9a-zA-Zㄱ-ㅎ가-힣]', ' ', text)
        
                # 띄어쓰기 제거
            text = text.replace('\n', ' ')
        
            text = re.sub(r'can t', 'can not', text)
            text = re.sub(r'don t', 'do not', text)
            text = re.sub(r'couldn t', 'could not', text)
        
            #소문자 대문자
            text = text.lower()
        
                # 정리
            text = ' '.join(text.split())
            list_text.append(text)
        
        #문자 전처리 결과를 데이터 프레임으로 정리
        real_df=pd.DataFrame(list_text, columns = ['text'])
        ```
        

- 데이터 분석 - 데이터 라벨링 (ChatGPT)
    - 각 데이터에 긍정/부정/중립이라는 감정을 부여할 수 있게 질문하고 이에 따른 답을 ChatGPT에게 받아 엑셀로 수집
    - 아래와 같이 진행함.
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e4d6070c-479a-4355-8b1d-d450cf489330/a8e7844c-41b3-44d1-a0ff-60d4772d33e4/Untitled.png)
        

- 데이터 분석 - 데이터 라벨링 (인간)
    - 문장에 대해 각 단어로 나눠서(토큰화)하여 감정을 부여
        - TreebankWordTokenizer()을 사용하여 토큰화
        - nltk의 SentimentIntensityAnalyzer을 사용하여 감정 부여
        
- 모델링
    - TF-IDF를 사용하여 벡터화
    - linearSVC를 사용하여 학습하고 recall, precision,accuracy를 확인
    - code
        
        ```python
        # 문장과 라벨 분리
        sentences = data["문장"]
        labels = data["감정"]
        
        # train, test 데이터셋 분리
        train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer()
        train_features = vectorizer.fit_transform(train_sentences)
        test_features = vectorizer.transform(test_sentences)
        
        # 분류 모델 학습
        model = LinearSVC()
        model.fit(train_features, train_labels)
        
        # 테스트 데이터셋 예측
        predictions = model.predict(test_features)
        
        # 분류 결과 출력
        print(classification_report(test_labels, predictions))
        ```
        
    

### 결과

- ChatGPT 모델링 결과
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e4d6070c-479a-4355-8b1d-d450cf489330/b829cf6c-3522-49e1-968e-5533c873442e/Untitled.png)
    
    - **precision ( TP / TP + FP )**
        - 긍정이라고 예측한 데이터 중 실제로 긍정인 비율 : 0.43
        - 부정이라고 예측한 데이터 중 실제로 부정인 비율 : 0.78
    - **recall ( TP / TP + FN )**
        - 실제 긍정인 데이터( TP+FN )들 중 긍정이라고 예측한 비율 : 0.08
        - 실제 부정인 데이터 ( TP+FN )들 중 부정이라고 예측한 비율 : 0.97
    - **accuracy : 0.77 → 77%**
    
- 인간 모델링 결과
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e4d6070c-479a-4355-8b1d-d450cf489330/c1fff3db-f4b3-49c2-841a-c60632be2d33/Untitled.png)
    
    - **precision ( TP / TP + FP )**
        - 긍정이라고 예측한 데이터 중 실제로 긍정인 비율 : 0.79
        - 부정이라고 예측한 데이터 중 실제로 부정인 비율 : 0.74
    - **recall ( TP / TP + FN )**
        - 실제 긍정인 데이터( TP+FN )들 중 긍정이라고 예측한 비율 : 0.67
        - 실제 부정인 데이터 ( TP+FN )들 중 부정이라고 예측한 비율 : 0.85
    - **accuracy : 0.76 → 76%**
    
    ### 결과
    
    1. ChatGPT의 모델링 결과 중 긍정에 대한 recall 값이 0.08로 낮음. 
        
        → 중립을 부정에 포함 시켜 **부정 비율이 너무 높았기 때문에 recall값이 낮은 것**으로 예측. 
        
        - 이를 해결하기 위해 ChatGPT에게 긍정/부정으로 질문하였으나 ChatGPT가 명확히 판단할 수 없다는 이유로 분석하지 않음..
        
    2. 정확도는 ChatGPT가 77%, 인간이 76%로 거의 비슷함. → **ChatGPT를 감정 분석에 활용하는 것이 시간과 정확도 부분에서 좋음.** 
    
    1. 시간과 정확도 부분 외에서 ChatGPT는 감정 부여 시 중립으로 결론을 내는 경향이 있어 이를 유의해서 사용해야함.
        
         추천 방법 : 
        
         → 1. 중립을 제외할 수 있도록 ChatGPT를 유도하는 질문 만들기 
        
         → 2. 라벨링이 긍정/부정으로 명확하게 나누어질 수 있는 데이터를 사용하기
