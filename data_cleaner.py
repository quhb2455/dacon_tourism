import pandas as pd
import re

from textrankr import TextRank
from konlpy.tag import Okt
from typing import List
from tqdm import tqdm
class OktTokenizer:
    okt: Okt = Okt()

    def __call__(self, text: str) -> List[str]:
        tokens: List[str] = self.okt.pos(text, norm=True, stem=True, join=True)
        return tokens
class CustomDataset():
    def __init__(self):
        self.tokenizer = OktTokenizer()
        self.textrank = TextRank(self.tokenizer)
        self.sentK = 5

    def textPreprocessing(self, text):
        hang = re.compile('[^ㄱ-ㅣ가-힣. ]')
        sentence = text.replace('\n', ' ')
        sentence = self.textrank.summarize(sentence, self.sentK)

        result = sentence.replace('\n', '. ')
        result = hang.sub('', result)
        result = result.strip()

        result = result.replace('..', '.')
        result = result.replace('. .', '.')
        if result[-1] != '.':
            result = result + '.'

        while True:
            if '  ' in result:
                result = result.replace('  ', ' ')
            else:
                break

        return result


def main():
    data_path = './data/train.csv'
    output_path = './data/train_cleaned.csv'
    all_df = pd.read_csv(data_path)
    new_df = pd.DataFrame(columns=all_df.columns)

    customData = CustomDataset()
    for idx, row in tqdm(all_df.iterrows()):
        sent = row.overview
        row.overview = customData.textPreprocessing(sent)

        new_df.loc[idx] = row

    new_df.to_csv(output_path, mode='w', index=False)
if __name__ == '__main__':
    main()
