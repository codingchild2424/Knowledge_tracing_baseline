import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

#2015데이터 경로 사전에 지정
DATASET_DIR = "data/2015_100_skill_builders_main_problems.csv"

#클래스를 호출할때는 Assist2015_loader()를 선언해야함, 만약 Assist2015_loader로 선언하면, 오류가 발생함

#Dataset을 상속받으면, __init__, __getitem__, __len__ 세 가지를 기본적으로 설정해야 함
class Assist2015_loader(Dataset):
    def __init__(self, data_path = DATASET_DIR):
        #Dataset 상속
        super().__init__()

        #경로 받아오기
        self.data_path = data_path

        #self.preprocess()를 통해 전처리된 데이터 받기
        self.user_list, self.que_list, self.user2idx, self.que2idx, \
            self.que_seqs, self.res_seqs = self.preprocess()

        #batches 정의
        #self.make_one_hot_batches()를 사용해서 batches 만들기
        self.batches = self.make_one_hot_batches()

        #batches 상태 확인, 필요시 사용
        #print('batches: ', self.batches)

        #__len__ 정의를 위해 사용함
        self.len = len(self.batches)

    #__getitem__은 DataLoader로 Dataset 클래스를 로드했을때, 반환하는 값에 대한 메쏘드임
    def __getitem__(self, index):
        #batches는 학생별 문항풀이 원핫벡터 데이터가 들어있음
        #index별로 내보낸다는 것은 각 학생 데이터를 하나씩 내보낸다는 것
        return self.batches[index]
        #return self.batches

    #__len__은 길이를 반환
    def __len__(self):
        #len 정의하기
        return self.len

    #여기서 preprocess를 정의하고, 초기 데이터 전처리
    def preprocess(self):
        df = pd.read_csv(self.data_path, encoding = "ISO-8859-1")

        #correct 열에 있는 값 중 0과 1 값만 df로 저장함(다른 값 제외)
        df = df[ (df["correct"] == 0).values + (df["correct"] == 1).values ]

        #유일한 user와 que의 리스트
        user_list = np.unique( df["user_id"].values )
        que_list = np.unique( df["sequence_id"].values )

        #user와 que별로 인덱스를 붙여준 딕셔너리
        user2idx = { u: idx for idx, u in enumerate(user_list) }
        que2idx = { q: idx for idx, q in enumerate(que_list) }

        #학생별 질문 리스트와 응답 리스트를 만듦
        que_seqs = []
        res_seqs = []

        for user in user_list:
            #한명의 user를 대상으로 한 데이터 프레임을 df_user로 만듦, 이때 log_id를 기준으로 순서대로 정렬함
            df_user = df[df["user_id"] == user].sort_values("log_id")

            que_seq = np.array([ que2idx[que] for que in df_user["sequence_id"].values] )
            res_seq = df_user["correct"].values

            que_seqs.append(que_seq)
            res_seqs.append(res_seq)

        return user_list, que_list, user2idx, que2idx, que_seqs, res_seqs

    #one_hot_vector
    def make_one_hot_batches(self):

        batches = []

        for que_seq, res_seq in zip(self.que_seqs, self.res_seqs):
            
            #user 한명이 푼 문제와 정답에 대한 정보가 담긴 벡터
            one_hot_vectors = []

            for i in range( len(que_seq) ):
                #전체 문항의 갯수의 2배 크기의 one_hot벡터를 만들 것임
                #전체 문항 갯수의 2배 크기인 이유는 오답일 경우 '처음~문항 갯수' 중에 문항번호에 해당되는 부분에 1을 기록하고,
                #정답일 경우에는 '문항 갯수+1~문항 갯수*2' 중 문항번호에 해당되는 부분에 1을 기록하도록 함
                zero_vector = np.zeros( len(self.que_list) * 2 )

                #one_hot_index를 정의함, que_seq[i]는 문항의 번호임
                #만약 정답이라면 res_seq[i]가 1이므로, 문항번호에 que_list만큼 더한 값만큼 인덱스가 정해짐
                #만약 오답이라면 res_seq[i]가 0이므로, 뒤에 더할 값이 없음. 문항번호 자체가 인덱스가 됨
                one_hot_idx = que_seq[i] + ( res_seq[i] * len(self.que_list) )

                #zero_vector에 인덱스에 1을 기록하여 원핫벡터로 만듦
                zero_vector[int(one_hot_idx)] += 1

                #one_hot_vectors에 zero_vector(원핫인 상태)를 더함
                one_hot_vectors.append(zero_vector)

            #one_hot_vectors 자체를 torch.Tensor로 만듦
            one_hot_vectors = torch.Tensor(one_hot_vectors)

            #torch.Tensor상태의 user 1명의 정보를 batches에 더함
            batches.append(one_hot_vectors)

        #batches = torch.Tensor(batches)

        return batches