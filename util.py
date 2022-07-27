# -*- coding: utf-8 -*-
from cProfile import label
from operator import index
import os
import sys
import random
import torch
import numpy as np
import pandas as pd
from itertools import product
import torch.nn.functional as F
from torch import nn

#from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#sample = '국민연금 월평균 수급액 1위 광역자치단체는 75만7천200원의 울산으로 나타났다. 반면 하위권에 속한 대구는 52만9천700원에 그쳐 노후 보장마저 지역 간 격차가 심각하다는 지적이다.김회재 더불어민주당 의원이 12일 국민연금공단으로부터 #제출받은 자료에 따르면, 올해 2월 기준 국민연금(노령연금) 월평균 수급액 #상위 5개 지자체는 울산에 이어 ▷세종 61만800원 ▷서울 60만4천700원 ▷경기 59만2천100원 ▷경남 58만3천700원으로 나타났다.하위 5개 지자체는 ▷전북 50만3천200원 ▷전남 51만9천400원 ▷충남 52만5천700원 ▷대구 52만9천700원 ▷제주 53만5천500원 순이었다.1위 울산과 꼴찌 전북의 격차가 약 25만 원에 이른 것이다. 경북은 17개 지자체 중 중위권인 55만6천700원이었다.김회재 의원은 "국토 불균형, 수도권 집중화 등으로 인해 지역 간 격차가 심화된 가운데, 노후 대비를 위한 1차 사회안전망인 국민연금에서조차 격차가 발생하고 있다"며 "지역에 질 좋은 일자리와 고부가가치산업이 부족해 발생하는 소득 격차가 노후보장 수준에까지 영향을 미치고 있다"고 밝혔다. 이어 "소외 지역에서 발생하는 노후보장 격차의 해결을 위한 추가적인 사회안전망 구축을 추진해야 한다"고 강조했다.'
#target = ['여러 사업에 도전했지만 실패한 아버지 김기택(송강호 扮), 해머던지기 선수 출신인 어머니 박충숙(장혜진 扮), 명문대 지망 4수생 첫째(장남) 김기우(최우식 扮), 미대 지망생 둘째(장녀) 김기정(박소담 扮)은 반지하 집에서 살아가는 백수 가족이다. 그들은 윗집이나 근처 카페에서 나오는 무료 와이파이에 매달리고, 피자박스 접기[1]로 생계를 유지한다. 집안은 꼽등이와 바퀴벌레가 득실거리고, 소독차가 다니는 날이면 공짜로 집안 소독이나 하자며 창문을 닫지 않으며, 주정뱅이가 노상방뇨하는 것을 반지하 창문 너머로 지켜보는 것이 일상인, 밑바닥 같은 나날을 보내고 있다.어렵사리 기우가 피자집 아르바이트 자리를 마련하고 조촐한 가족 파티[2]를 열고 있던 어느 날, 기우의 친구 민혁(박서준 扮)이 집으로 찾아온다.[3] 민혁은 명문대에 다니고 있고, 고등학생 과외 아르바이트를 하고 있다. 민혁에게서 과외를 받는 박다혜(정지소 扮)는 굉장한 부잣집 딸로, 다혜의 아버지 박동익(이선균 扮)은 글로벌 IT 기업의 CEO이다.기우네 가족들이 반지하 창문 너머로 지켜보는 가운데, 민혁은 집 앞에서 노상방뇨하던 주정뱅이에게 "정신 차려, 정신!"이라고 호통을 치며 쫓아내고, 가족들은 "역시 대학생은 다르다"며 감탄한다.[4] 집 안으로 들어온 민혁은 기택과 충숙 내외에게 안부 인사를 한 뒤 들고 온 고풍스러운 상자 안에서 값비싼 수석[5]을 꺼내어 선물한다. 민혁은 "저희 할아버지[6]가 가져다 주라고 하셨는데, 집안에 재물 운과 합격 운을 가져다 주는 물건이다"라고 설명한다. 기우는 수석을 유심히 바라보며 되게 상징적이라고 하고, 기택 역시 참으로 시의적절하다며[7] 고마워한다. 하지만 충숙은 "먹을 것이 아니네."라며 실망한다.','국민연금이 건강보험의 유탄을 맞았다. 정부가 지난달 29일 ‘소득 중심의 건강보험료 2차 개선안’을 내놓고 9월 시행을 예고하면서다. 이번에 건보 직장가입자의 피부양자 기준을 대폭 강화하면서 국민연금 수령자가 영향을 받게 됐다. 그동안 연금을 늘리기 위해 수령 시기 연장, 추후 보험료 납부(추납) 등의 노력을 해왔는데, 연금 증가로 인해 피부양자 탈락에다 재산 건보료라는 벽에 부딪혔다. 하지만 ‘무임승차 축소’라는 명분 앞에서 크게 불만을 토로하기도 힘든 상황이다. 국민연금공단은 며칠 새 건보료 문의가 쇄도하자 발 빠르게 움직이기 시작했다. 공단은 노령연금(일반적 형태의 국민연금) 수급개시(만 62세) 안내장에 유의사항을 명시해 9월부터 발송한다. “반납금 납부, 추후 납부로 노령연금 수령액이 증가할 경우 연금소득세 및 건강보험료에 영향을 줄 수 있다”는 게 골자다. 반납과 추납대상자는 4월부터 안내하고 있다. 공단은 이와 함께 연금 수령 개시자에게 건보료 상담서비스를 하기 위해 전산프로그램 개발에 착수했다. 66세 은퇴자 건보료 0원→15만원 경기도 안양시 정모(66)씨는 대기업에서 퇴직했다. 국민연금(120만원), 금융 이자소득 등으로 월 200만원가량의 소득이 있다. 그동안 직장가입자인 아들(36)의 건보증에 피부양자로 얹혔다. 최근 뉴스를 보고 걱정이 돼 건보공단 지사를 찾았더니 ‘피부양자 탈락’ 대상이었다. 정씨는 “부부가 빠듯하게 사는데 매달 15만원 넘게 건보료 나온다니 걱정”이라며 “4대 사회보험이 되는 단기 일자리라도 알아봐야겠다”고 말했다. 지금은 연금·이자·배당·임대·사업 등의 소득이 연간 3400만원 넘지 않으면 피부양자가 된다. 9월 이 기준이 2000만원(월 167만여원)으로 강화돼 27만3000명이 피부양자에서 탈락한다. 앞으로 국민연금 수령자와 연금액이 계속 늘어나게 돼 있어 탈락자가 더 늘어날 전망이다. 그동안 연금공단이나 복지부, 노후소득 전문가 등이 한목소리로 ‘연금 늘리기’를 권장해 왔다. 기자도 마찬가지다. 가령 국민연금 수령 시기를 최대 5년 연장하면 36%(연간 7.2%)의 연금을 더 받으니 여건이 되면 이를 활용하는 게 낫다는 식이다. 과거 일시금으로 받은 보험료를 반납하거나 과거에 못 낸 보험료를 나중에 내거나 60세 이후에도 보험료를 계속 내도록 권장했다. 이런 걸 활용해서 연금을 늘려 평생 받는 게 나은지, 건보료를 덜 내는 게 나은지 한마디로 정의하기 쉽지 않다. 그래서 연금공단이 조언자로 나서기로 한 것이다.피부양자는 무임승차 제도이다. 경제 능력이 없는 가족을 보호하는 장치이다. 직장가입자당 평균 0.95명이 얹혀있다. 독일(0.29명)의 경우 부모는 피부양자가 될 수 없고 미성년 자녀만 가능하다. 부모·자식 관계가 옅은 유럽의 특징을 반영했다. 일본은 0.68명이다. 한국은 초저출산 여파로 피부양자가 계속 줄지만 다른 나라보다 많은 편이다. 그래서 5년 전 국회에서 피부양자 축소에 합의했고, 이번에 시행한다.']

def diff(queries, keys):
    return queries.reshape((-1, 1)) - keys.reshape((1, -1))

def attention_pool(query_key_diffs, values):
    attention_weights = F.softmax(- query_key_diffs**2 / 2, dim=1)
    return torch.matmul(attention_weights, values), attention_weight

def preprocess(text):
    preprocess = []
    for word in text.split():
        if not word.startswith('#'):
            preprocess.append(word)
    text_preprocessed = ' '.join(preprocess)
    return text_preprocessed


#mixup word embedding
def mixup(sample_df, target_df, num_rows): #string, list
    print("---mixup function---")
    #target데이터셋에서 sample 문서와 비교해서 유사도가 가장 낮은 문서 찾기
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    
    sample = sample_df["contents"].values
    target = target_df["contents"].values
    
    sample_embedding = [model.encode(s) for s in sample] #sentence_transformer로 document를 대신해도 되는가?
    target_embedding = [model.encode(t) for t in target]
    
    result_contents = []
    result_labels = [1]*num_rows

    for idx, emb in enumerate(sample_embedding):
        sim_list = [emb] + target_embedding
        similarity = cosine_similarity(sim_list,sim_list)
    
        #mixup!!
        index_min = np.argmin(similarity[0][1:]) #[0][0] : sample
        target_ = target[index_min]
        
        sample_token = sample[idx].split() #[token1, token2, ...]
        target_token = target_.split() #[t1, t2, ..]

        if len(sample_token) < len(target_token):
            length = int(len(sample_token)/2) #the number of tokens
        else:
            length = int(len(target_token)/2)
        
        doc1, doc2 = [sample_token[:length],sample_token[length:]],[target_token[:length],target_token[length:]]
        comb_list = list(product(doc1, doc2))
        comb_list = [c[0]+c[1] for c in comb_list]

        for comb in comb_list:
            random.shuffle(comb)
            result_document = ' '.join(comb)
            result_contents.append(result_document)
        print("result contents count",len(result_contents))
        print(result_contents[idx])
    
    aug_data = pd.DataFrame({'contents':result_contents,'label':result_labels})
    print(len(aug_data))
    return aug_data


class FocalLoss(nn.Module):
    def __init__(self,alpha = 0.75, gamma = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha= alpha
        self.gamma = gamma
        #self.reduction = reduction
        self.eps = 1e-6
        #self.ignore_index = ignore_index

    def forward(self, input, target): #input : logits
        #target : one_hot vector
        target_one_hot = F.one_hot(target) #Tensor (32,2)
        
        alpha_tensor = (1-self.alpha) + target * (2*self.alpha -1) # alpha if target = 1 and 1 - alpha if target = 0
        
        #logit을 probability로 바꿔주고 계산
        prob = F.softmax(input,dim=1) + self.eps
        
        focal = -alpha_tensor.view([-1,1]).mul(torch.pow((1-prob+self.eps), self.gamma)) * torch.log(prob)
        focal_loss = torch.sum(target_one_hot*focal, dim = 1)
        loss = torch.mean(focal_loss)
        return loss
