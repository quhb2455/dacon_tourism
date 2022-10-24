from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from albumentations.pytorch.transforms import ToTensorV2
from sklearn import preprocessing
import albumentations as A

from transformers import AutoTokenizer

import torch
import cv2
import pandas as pd
import os
import numpy as np
import re
from textrankr import TextRank
from konlpy.tag import Okt
from typing import List

import options
arr = ['5일장', 'ATV', 'MTB', '강', '게스트하우스', '계곡', '고궁', '고택', '골프', '공연장',
   '공예,공방', '공원', '관광단지', '국립공원', '군립공원', '기념관', '기념탑/기념비/전망대',
   '기암괴석', '기타', '기타행사', '농.산.어촌 체험', '다리/대교', '대중콘서트', '대형서점',
   '도립공원', '도서관', '동굴', '동상', '등대', '래프팅', '면세점', '모텔', '문', '문화관광축제',
   '문화원', '문화전수시설', '뮤지컬', '미술관/화랑', '민물낚시', '민박', '민속마을', '바/까페',
   '바다낚시', '박람회', '박물관', '발전소', '백화점', '번지점프', '복합 레포츠', '분수', '빙벽등반',
   '사격장', '사찰', '산', '상설시장', '생가', '서비스드레지던스', '서양식', '섬', '성',
   '수련시설', '수목원', '수상레포츠', '수영', '스노쿨링/스킨스쿠버다이빙', '스카이다이빙', '스케이트',
   '스키(보드) 렌탈샵', '스키/스노보드', '승마', '식음료', '썰매장', '안보관광', '야영장,오토캠핑장',
   '약수터', '연극', '영화관', '온천/욕장/스파', '외국문화원', '요트', '윈드서핑/제트스키',
   '유람선/잠수함관광', '유명건물', '유스호스텔', '유원지', '유적지/사적지', '이색거리', '이색찜질방',
   '이색체험', '인라인(실내 인라인 포함)', '일반축제', '일식', '자동차경주', '자연생태관광지',
   '자연휴양림', '자전거하이킹', '전문상가', '전시관', '전통공연', '종교성지', '중식', '채식전문점',
   '카약/카누', '카지노', '카트', '컨벤션', '컨벤션센터', '콘도미니엄', '클래식음악회', '클럽',
   '터널', '테마공원', '트래킹', '특산물판매점', '패밀리레스토랑', '펜션', '폭포', '학교', '한식',
   '한옥스테이', '항구/포구', '해수욕장', '해안절경', '헬스투어', '헹글라이딩/패러글라이딩', '호수',
   '홈스테이', '희귀동.식물']
class OktTokenizer:
    okt: Okt = Okt()

    def __call__(self, text: str) -> List[str]:
        tokens: List[str] = self.okt.pos(text, norm=True, stem=True, join=True)
        return tokens

class CustomDataset(Dataset):
    def __init__(self, all_df, transforms, infer=False, isCleaned=False):
        self.okttokenizer = OktTokenizer()
        self.textrank = TextRank(self.okttokenizer)
        self.sentK = 20
        self.img_path_list = all_df['img_path'].values
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
        self.text_list = all_df['overview'].values
        # self.text_vectors = [self.tokenizer.encode(self.textPreprocessing(text),
        #                                         padding='max_length',
        #                                        max_length=256) for text in text_vectors]
        if not infer:
            self.cat1_list = all_df['cat1'].values
            self.cat2_list = all_df['cat2'].values
            self.cat3_list = all_df['cat3'].values

        self.transforms = transforms
        self.infer = infer
        self.isCleaned = isCleaned
        self.max_length = 512

    def textPreprocessing(self, text):
        hang = re.compile('[^ㄱ-ㅣ가-힣. ]')
        sentence = text.replace('\n', ' ')
        sentence = self.textrank.summarize(sentence, self.sentK)

        result = hang.sub('', sentence)

        result = result.replace('..', '.')

        res = result.replace('\n', '. ')
        res = res + '.'
        return res
    def textPosExtractor(self, text):
        okt_pos = Okt().pos(text, norm=True, stem=True)
        okt_filtering = [x for x, y in okt_pos if y in ['Noun', 'Adjective', 'Verb']]

        result = []
        stopwords = ['하다', '있다', '되다', '수', '이', '되어다', '않다', '없다', '이다', '로', '당신', '아니다', '등', '등등', '년','이상','리','것',\
                    '보다','약','분','내','곳','월','일','나','로부터','애','그','후','겸','호','옆','외','곳곳','위','몇','하나','당시','다시',\
                    '그동안','앞','옆면','둘','셋','넷','다섯','여섯','일곱','여덟','아홉','영','일','시','제','위해','모든','최선','다','로서','뿐',\
                    '때문','정도','및','모두','최상','점점','보고','대한','대다','들이다','지다','제일','높다','이루다','주변','건너','전혀','그대로',\
                    '내다','더욱','봄','여름','가을','겨울','우리','또한','널리','종종','크다','크게','직접','작다','가까이','한쪽','쪽','무엇','류',\
                    '별로','듯','차차','란','현재','잠시','중','층','아래','최근','사이','소위','때','올해','이번','전','과','앞쪽','빙','다','바로',\
                    '누구']
        for word in okt_filtering:
            if word in stopwords:
                continue
            result.append(word)
        return ' '.join(result)

    def __getitem__(self, index):
        # NLP
        text_vector = self.text_list[index]
        #print(f'[raw data] {self.text_list[index]}')
        if self.isCleaned == False:
            text_vector = self.textPreprocessing(text_vector)

        #print(f'[preprocessing] {text_vector}')
        text_vector = self.textPosExtractor(text_vector)

        #print(f'[pos extractor] {text_vector}')
        #print(text_vector)
        # Image
        img_path = os.path.join('./data', self.img_path_list[index])
        image = cv2.imread(img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        text_vector = self.tokenizer.encode_plus(text_vector, padding='max_length', max_length=self.max_length, truncation = True, return_token_type_ids=False, return_attention_mask=True,return_tensors='pt')
        #print(len(text_vector[0]))
        valid_length = len(text_vector)
        segment_ids = [0] * self.max_length
        # Label
        if self.infer:
            return image, text_vector['input_ids'].flatten(), text_vector['attention_mask'].flatten(), self.text_list[index]
        else:
            label = [self.cat1_list[index], self.cat2_list[index], self.cat3_list[index]]
            #print('[', arr[label[2]], ']', self.text_list[index])
            return image, text_vector['input_ids'].flatten(), text_vector['attention_mask'].flatten(),\
                    torch.tensor(self.cat1_list[index], dtype=torch.long),\
                    torch.tensor(self.cat2_list[index], dtype=torch.long),\
                    torch.tensor(self.cat3_list[index], dtype=torch.long),\
                    self.text_list[index]

    def __len__(self):
        return len(self.img_path_list)

def loadData(path):
    all_df = pd.read_csv(path)
    #all_df = all_df.loc[:100]
    train_df, val_df, _, _ = train_test_split(all_df, all_df['cat3'], test_size=0.01, random_state=options.SEED)

    le_classes = []
    category_name = ['cat1', 'cat2', 'cat3']
    for i in range(3):
        _le = preprocessing.LabelEncoder()
        _le.fit(train_df[category_name[i]].values)
        print(_le)
        print(_le.classes_)
        train_df[category_name[i]] = _le.transform(train_df[category_name[i]].values)
        val_df[category_name[i]] = _le.transform(val_df[category_name[i]].values)
        le_classes.append(_le.classes_)

    return train_df, val_df, le_classes

def createDataLoader(path, isCleaned):
    train_df, val_df, classes = loadData(path)

    train_transform = A.Compose([
                                A.Resize(options.IMG_SIZE,options.IMG_SIZE),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    test_transform = A.Compose([
                                A.Resize(options.IMG_SIZE,options.IMG_SIZE),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    train_dataset = CustomDataset(train_df, train_transform, isCleaned=isCleaned)
    train_loader = DataLoader(train_dataset, batch_size = options.BATCH_SIZE, shuffle=True, num_workers=1)

    val_dataset = CustomDataset(val_df, test_transform, isCleaned=isCleaned)
    val_loader = DataLoader(val_dataset, batch_size=options.BATCH_SIZE, shuffle=False, num_workers=1)

    return train_loader, val_loader, classes

def createTestLoader(path, isCleaned):
    all_df = pd.read_csv(path)

    test_transform = A.Compose([
                                A.Resize(options.IMG_SIZE,options.IMG_SIZE),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    test_dataset = CustomDataset(all_df, test_transform, infer=True, isCleaned=isCleaned)
    test_loader = DataLoader(test_dataset, batch_size = options.BATCH_SIZE, shuffle=False, num_workers=1)
    return test_loader
