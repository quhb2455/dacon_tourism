from konlpy.tag import Okt
from typing import List
from textrankr import TextRank
class OktTokenizer:
    okt: Okt = Okt()

    def __call__(self, text: str) -> List[str]:
        tokens: List[str] = self.okt.pos(text, norm=True, stem=True, join=True)
        return tokens
k = 3   
tok = OktTokenizer()
textrank = TextRank(tok)

stri = '''
소안항은 조용한 섬으로 인근해안이 청정해역으로 일찍이 김 양식을 해서 높은 소득을 올리고 있으며 바다낚시터로도 유명하다. 항 주변에 설치된 양식장들은 섬사람들의 부지런한 생활상을 고스 란히 담고 있으며 일몰 때 섬의 정경은 바다의 아름다움을 그대로 품고 있는 듯하다. 또한, 섬에는 각시여 전설, 도둑바위 등의 설화가 전해 내려오고 있으며, 매년 정월 풍어제 풍속이 이어지고 있다.<br>
'''

b = textrank.summarize(stri, k, verbose=False)

from konlpy.tag import Okt
voc = '형태소 란 문장을 구성하는 의미 요소 중 가장 작은 단위를 말한다.'
okt_pos = Okt().pos(voc, norm=True, stem=True)
okt_filtering = [x for x, y in okt_pos if y in ['Noun', 'Adjective', 'Verb']]
print(okt_filtering)


print(b)