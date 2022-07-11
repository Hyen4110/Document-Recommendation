# -*- coding: utf-8 -*-
import numpy as np 
import torch
from torch import nn
from sentence_transformers import SentenceTransformer, util


class KRSentenceBERT(nn.Module):
    def __init__(self, dropout_p=.3):
        super().__init__()
        
        self.sbert = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
           
           
    def forward(self, sentence1, sentence2):    
        sent1_embeddings = self.sbert(sentence1)['sentence_embedding']  
        sent2_embeddings = self.sbert(sentence2)['sentence_embedding']
        # print(f"sent1_embeddings shape : {sent1_embeddings.shape}") # torch.Size([64, 768])
        return sent1_embeddings, sent2_embeddings

if __name__ == "__main__":
    # sbert = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    sbert = KRSentenceBERT()
    sent1 = "문장 간 구문 유사도(syntactic similarity) 검사는 문장 생성 모델의 자동평가 도구로 이용되는 중요한 기술이다. 최근 딥러닝을 이용한 문장 임베딩(sentence embedding), 즉 문장 간 인코더-디코더에 관한 연구들이 진행 되고 있으며, 이를 이용한 기계 번역이 괄목할 만한 성과를 거두고 있다. 본 연구에서는 패러프레이징의 평가 도구로서 인코더-디코더 모델을 활용하고자 한다. 본 논문에서는 한국어 문장을 한국 어 위키피디아 말뭉치를 이용해 RNN(recurrent neural network)으로 학습한 인코더-디코더 모델을 이용한 문장 간 유사도 분석 실험을 실시하였다."
    sent2 = "최근 인코더-디코더 구조의 자연어 처리모델이 활발하게 연구가 이루어지고 있다. 인코더-디코더기반의 언어모델은 특히 본문의 내용을 새로운 문장으로 요약하는 추상(Abstractive) 요약 분야에서 널리 사용된다. 그러나 기존의 언어모델은 단일 문서 및 문장을 전제로 설계되었기 때문에 기존의 언어모델에 다중 문장을 요약을 적용하기 어렵고 주제가 다양한 여러 문장을 요약하면 요약의 성능이떨어지는 문제가 있다. 따라서 본 논문에서는 다중 문장으로 대표적이고 상품 리뷰를 워드 임베딩의 유사도를 기준으로 클러스터를 구성하여 관련성이 높은 문장 별로 인공 신경망 기반 언어모델을 통해 요약을 수행한다. 제안하는 모델의 성능을 평가하기 위해 전체 문장과 요약 문장의 유사도를 측정 하여 요약문이 원문의 정보를 얼마나 포함하는지 실험한다. 실험 결과 기존의 RNN 기반의 요약 모델 보다 뛰어난 성능의 요약을 수행했다."
    sent3 = ""

    sent1_embeddings, sent2_embeddings = sbert(sent1, sent2)
    # vec2 = sbert.encode(sent2, show_progress_bar=True, convert_to_tensor=True)
    
    # print(torch.matmul(vec1, vec2))
    # res = util.cos_sim(vec1, vec2).squeeze()



    print(sent1_embeddings.shape)
    print(sent2_embeddings.shape)
