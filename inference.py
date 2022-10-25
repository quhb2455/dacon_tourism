from data_utils import createTestLoader
from models import TourismModel
from transformers import BertModel, AutoModel
import utils
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import options
import os
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

def inference(model, data_loader, device):
    model = model.eval()
    preds_arr = []

    for img, text, attention_mask, raw in tqdm(iter(data_loader)):
        with torch.no_grad():
            img = img.float().to(device)
            text = text.to(device)
            attention_mask = attention_mask.to(device)

            model_pred = model(img, text, attention_mask)

            _, preds = torch.max(model_pred[-1], dim=1)
            # for p, r in zip(preds, raw):
            #     print('[', arr[int(p)], ']', r)
            preds_arr.extend(preds.cpu().numpy())
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    return preds_arr

def makeSubmission(results, save_name):
    submit = pd.read_csv('./data/sample_submission.csv')
    for i in range(len(results)):
        submit.loc[i,'cat3'] = arr[results[i]]
    save_name = os.path.join('./results/', save_name)
    submit.to_csv(save_name, index=False)

def infer_call(model, device, save_name):
    isCleaned = False
    test_loader = createTestLoader('./data/test.csv', isCleaned)
    model.eval()
    results = inference(model, test_loader, device)
    makeSubmission(results, save_name)

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    ckpt_path = './ckpt/checkpoint_10.pt'
    isCleaned = False
    test_loader = createTestLoader('./data/test.csv', isCleaned)

    # all_df = pd.read_csv('./data/train_cleaned.csv')
    # #all_df = all_df.loc[:100]
    # train_df, val_df, _, _ = train_test_split(all_df, all_df['cat3'], test_size=0.01, random_state=options.SEED)
    # _le = preprocessing.LabelEncoder()
    # _le.fit(train_df['cat3'].values)
    # print(_le.classes_)
    # classes = _le.classes_
    kobert = AutoModel.from_pretrained("klue/roberta-large")
    model = TourismModel(kobert, 1024)
    model, _ = utils.load_model(model, ckpt_path, device)
    model.eval()

    results = inference(model, test_loader, device)
    makeSubmission(results, 'test_roberta-s_E10_lr3e5_adamw_fullPreprocessing_PosExtractor_S20.csv')
