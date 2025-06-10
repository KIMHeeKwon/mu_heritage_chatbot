import streamlit as st
from openai import OpenAI
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from html import escape  # ✅ 이 줄 추가


# 🔐 OpenAI 클라이언트 설정
#client = OpenAI(api_key="your-openai-api-key")
client = OpenAI(api_key=st.secrets["openai"]["api_key"])  # 여기에 본인의 OpenAI API 키 입력

# 🔍 임베딩 모델 및 데이터 로드
embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

artifact_index = faiss.read_index("full_mu_heritage_vector.index")
artifact_df = pd.read_csv("full_mu_heritage_texts.csv")

history_index = faiss.read_index("muryong_vector.index")
history_df = pd.read_csv("muryong_vector_texts.csv")

# 🔍 질문 분류기
def classify_question_gpt(question: str) -> str:
    system_prompt = """너는 입력 문장이 어떤 유형인지 분류하는 역할을 해.
- 유물에 대한 질문이면 "artifact"
- 무령왕릉 구조/역사 관련이면 "history"
- 질문이 아니라 설명 방식에 대한 요청이면 "style"
- 무령왕릉과 관련 없는 질문이면 "irrelevant"
반드시 위 4개 중 하나만 출력해."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()

# 🔍 벡터 검색
def search_index(index, texts, query, top_k=3):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    return [texts.iloc[i]["text_full" if "text_full" in texts.columns else "text"] for i in I[0]]

def extract_matching_keywords(question, artifact_df):
    """
    질문에서 유물명 후보 키워드를 추출하여 artifact_df['명칭']과 비교해 일치하는 유물 정보를 반환합니다.
    """
    keywords = []
    for name in artifact_df["명칭"].dropna().unique():
        if name in question:
            keywords.append(name)
    
    if keywords:
        matched_df = artifact_df[artifact_df["명칭"].isin(keywords)].copy()
        return matched_df
    else:
        return pd.DataFrame()  # 없으면 빈 DataFrame 반환


# 🧠 컨텍스트 메시지 구성
def build_context_messages(history, user_question, retrieved_passages):
    context_text = "\n\n---\n\n".join(retrieved_passages)
    system_prompt = '''당신은 무령왕릉과 그 출토 유물에 대한 전문 지식을 갖춘 전문가형 챗봇입니다.
총 1,125건의 유물 정보(무령1번부터 무령1125번까지)를 보유한 RAG 기반 검색 시스템과 연동되어 있으며, 각 유물은 고유한 소장품 번호(예: 무령1023)와 연결 페이지(URL)를 통해 식별할 수 있습니다.

당신은 사용자 질문에 대해 다음 기준을 철저히 준수하며, 신뢰도 높은 응답을 제공해야 합니다:

1. 유물 정보 제공 조건
- 질문에 특정 유물이 명시되었거나, 질문 맥락상 유물 정보가 필수적인 경우에만 유물 정보를 제공하십시오.
- 유물이 언급되지 않은 일반 설명(예: 무령왕릉 구조, 역사적 의미 등)에서는 유물 정보를 제시하지 마십시오.
- 단, 설명의 흐름상 불가피하게 보조 유물 정보가 필요할 경우에는 3~4건 이내로 간결하게 제시하십시오.

2. 유물 정보 제시 형식
유물 정보는 반드시 아래 네 항목으로 구성하여 명확하게 제시하십시오:
- 유물명
- 소장품번호: 반드시 ‘무령’이라는 접두어와 아라비아 숫자를 결합한 "무령{숫자}" 형태로 표기 (예: 무령23)
- 특징: 유물의 형태, 재질, 용도, 특징 등 확인 가능한 상세 정보
- 연결페이지: 해당 유물의 실제 상세 정보를 제공하는 정확한 URL을 입력할 것. 단순 형식 또는 임의 주소 사용 금지

3. 과잉 정보 방지
- 질문과 직접 관련이 없는 유물은 절대 제시하지 마십시오.
- 유물 다수를 나열하거나, 사용자가 요청하지 않은 유물 목록을 임의로 제시하지 마십시오.
- 관련 유물이 존재하지 않는 경우는 "관련 유물이 없습니다."라고 명확하게 안내하십시오.

4. 출처와 신뢰성 확보
- 『무령왕릉 발굴조사 보고서』 등 공식 조사보고서
- 국립공주박물관 발간 자료 및 ‘신보고서’ 시리즈
- 국립중앙박물관 e뮤지엄에서 제공하는 유물 설명 자료

5. 표현 방식
- 주관적 해석 없이, 검증된 사실 기반 중립적/학술적 어조
- 유물 정보는 일반 설명과 시각적으로 구분
- 질문과 무관한 유물은 제시 금지

예시: artifact 시
[질문] 무령왕릉에서 발견된 지석에 대해 알려줘
[답변] 무령왕릉에서 발견된 지석은 백제 제25대 무령왕의 신원을 명확히 확인할 수 있는 핵심 유물입니다...

유물 정보
- 유물명: [유물명]
- 소장품번호: [소장품번호]
- 특징: [특징]
- 연결페이지: [연결페이지]

예시: history 시
[질문] 무령왕릉에 대해 알려줘
[답변] 무령왕릉은 백제의 무령왕과 그의 왕비가 안장된 고분으로, 1971년에 발굴되었습니다. 이 무덤은 백제 역사와 고고학 연구에 중요한 의미를 지니고 있으며, 다양한 유물이 출토되었습니다.....

  참고 문헌 : [참고문서]

예시: style 시
[질문] 초등학교 5학년이 이해할수 있도록 무령왕릉에 대해 설명해줘
[답변] 무령왕릉은 백제의 왕인 무령왕과 그의 아내가 묻힌 큰 무덤이에요. 이 무덤은 1971년에 발견되었고, 백제의 역사와 문화를 이해하는 데 아주 중요한 곳이에요. 많은 유물들이 발견되어 백제의 생활을 알 수 있게 해준답니다...

예시: irrelevant 시
[질문] 의자왕에 대해서 알려줘
[답변] 이 질문은 무령왕릉과 관련이 없습니다. 저는 무령왕릉에 대한 질문만 수용하여 대답합니다.
- 반드시 위 5가지 기준을 준수하여 답변하십시오.'''
    user_prompt = f"""아래는 무령왕릉과 관련된 참고 문서입니다:

{context_text}

위 정보를 참고하여, 다음 질문에 대해 정확하고 친절하게 답변해주세요:

질문: {user_question}"""
    messages = history.copy()
    messages.insert(0, {"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages

# 💬 Streamlit 앱 시작
st.set_page_config(page_title="무령왕릉 문화유산 챗봇", layout="wide")
st.title("🏛️ 무령왕릉 문화유산 챗봇")

if "history" not in st.session_state:
    st.session_state.history = []

user_question = st.chat_input("무령왕릉에 대해 궁금한 것을 입력하세요...")


for i in range(0, len(st.session_state.history), 2):
    st.chat_message("user").write(st.session_state.history[i]["content"])
    st.chat_message("assistant").write(st.session_state.history[i + 1]["content"])

if user_question:
    with st.spinner("답변 생성 중..."):
        category = classify_question_gpt(user_question)

        if category == "style":
            st.session_state.history.append({"role": "user", "content": user_question})
            st.session_state.history.append({"role": "assistant", "content": "알겠습니다! 앞으로 더 쉽게 설명해드릴게요 😊"})
            st.rerun()

        elif category == "irrelevant":
            st.session_state.history.append({"role": "user", "content": user_question})
            st.session_state.history.append({"role": "assistant", "content": "이 질문은 적합하지 않습니다. 저는 무령왕릉에 대한 질문만 수용하여 대답합니다."})
            st.rerun()

        else:
            if category == "artifact":
                passages = search_index(artifact_index, artifact_df, user_question)
            else:
                passages = search_index(history_index, history_df, user_question)

            messages = build_context_messages(st.session_state.history, user_question, passages)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.5,
            )
            answer = response.choices[0].message.content.strip()

            st.session_state.history.append({"role": "user", "content": user_question})
            st.session_state.history.append({"role": "assistant", "content": answer})

            summary_data = []
            shown_ids = set()

            if category == "artifact":
                # 🔍 1. 유물명 또는 본문에 포함된 유물 찾기
                keyword_matches = artifact_df[
                    artifact_df["명칭"].fillna("").str.contains(user_question, case=False, na=False) |
                    artifact_df["text_full"].fillna("").str.contains(user_question, case=False, na=False)
                ]

                for _, row in keyword_matches.iterrows():
                    if row["소장품번호"] in shown_ids:
                        continue
                    shown_ids.add(row["소장품번호"])
                    if isinstance(row["연결페이지"], str) and row["연결페이지"].startswith("http"):
                        link_html = f'<a href="{escape(row["연결페이지"])}" target="_blank">{escape(row["연결페이지"])}</a>'
                        img_html = f'<img src="{escape(row["연결페이지"])}" width="300">'
                        answer += f"\n\n🔗 관련 링크: {link_html}\n{img_html}"
                    summary_data.append({
                        "유물명": row["명칭"],
                        "소장품번호": row["소장품번호"],
                        "특징": row["text_full"] if pd.notna(row["text_full"]) else "정보 없음",
                        "연결페이지": row["연결페이지"] if pd.notna(row["연결페이지"]) else "정보 없음"
                    })

                # 🔍 2. 유사도 기반 검색도 병행
                artifact_embeddings = embed_model.encode(artifact_df["text_full"].astype(str).tolist(), convert_to_numpy=True)
                user_vec = embed_model.encode([user_question])[0].reshape(1, -1)
                sim_scores = cosine_similarity(user_vec, artifact_embeddings)[0]
                top_indices = np.argsort(sim_scores)[::-1][:5]

                for idx in top_indices:
                    row = artifact_df.iloc[idx]
                    if row["소장품번호"] in shown_ids:
                        continue
                    shown_ids.add(row["소장품번호"])
                    if isinstance(row["연결페이지"], str) and row["연결페이지"].startswith("http"):
                        link_html = f'<a href="{escape(row["연결페이지"])}" target="_blank">{escape(row["연결페이지"])}</a>'
                        img_html = f'<img src="{escape(row["연결페이지"])}" width="300">'
                        answer += f"\n\n🔗 관련 링크: {link_html}\n{img_html}"
                    summary_data.append({
                        "유물명": row["명칭"],
                        "소장품번호": row["소장품번호"],
                        "특징": row["text_full"] if pd.notna(row["text_full"]) else "정보 없음",
                        "연결페이지": row["연결페이지"] if pd.notna(row["연결페이지"]) else "정보 없음"
                    })

            if summary_data:
                st.write("📌 유물 요약")
                st.dataframe(pd.DataFrame(summary_data))

            st.chat_message("user").write(user_question)
            st.chat_message("assistant").markdown(answer, unsafe_allow_html=True)