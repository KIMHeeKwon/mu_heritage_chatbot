import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. 데이터 불러오기
df = pd.read_csv("full_mu_heritage_texts.csv")
texts = df["text_full"].tolist()

# 2. 임베딩 모델 로딩 (한국어 특화)
model = SentenceTransformer("jhgan/ko-sroberta-multitask")
embeddings = model.encode(texts, convert_to_numpy=True)

# 3. FAISS 인덱스 생성
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 4. 인덱스 저장
faiss.write_index(index, "full_mu_heritage_vector.index")

# 5. 메타데이터 저장 (소장품번호, 명칭, 링크)
df[["소장품번호", "명칭", "연결페이지"]].to_csv("full_mu_heritage_metadata.csv", index=False, encoding="utf-8-sig")

print("✅ 완료: 벡터 인덱스와 메타데이터가 생성되었습니다.")
