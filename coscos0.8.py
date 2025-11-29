import csv
import os
import random
from json import loads
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from datasets import Dataset
import pdfplumber
import numpy as np

# ===== PDF → CSV変換 =====
def pdf_to_csv(pdf_file, csv_file="Book1.csv"):
    explanations = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                for line in text.split("\n\n"):
                    line = line.strip()
                    if line:
                        explanations.append([line])
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(explanations)

def load_explanations_from_csv(filename="Book1.csv"):
    explanations = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                explanations.append(row[0])
    return explanations


# ===== OpenAI API キーの読み込み =====
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# ===== 解説文の意味補正 =====
def refine_explanation(raw_text: str, client: OpenAI) -> str:
    """抽出文を自然で意味の通る説明文にリライト"""
    response = client.chat.completions.create(
        model="gpt-4.0",
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは教育教材の編集者です。"
                    "以下の文章を自然で意味の通る日本語に直してください。"
                    "文のつながりを補い、読んで理解できる形にしてください。"
                    "歴史や地名などの固有名詞は正確さを保ち、改変しすぎないように注意してください。"
                )
            },
            {"role": "user", "content": raw_text},
        ],
        temperature=0.8
    )
    refined_text = response.choices[0].message.content.strip()
    return refined_text


# ===== Streamlit UI =====
st.title("兵庫学検定試験対策ツール（意味補正＋5問同時出題＋コサイン類似度付き）")

uploaded_file = st.file_uploader("クイズに使うPDFファイルを選んでください", type=["pdf"])

if uploaded_file is not None:
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    pdf_to_csv(pdf_path, "Book1.csv")
    st.session_state.explanations = load_explanations_from_csv("Book1.csv")
    st.success(f"PDFを変換して {len(st.session_state.explanations)} 件の文章を読み込みました！")


# ===== クイズ出題処理 =====
if "explanations" in st.session_state and st.session_state.explanations:
    explanations = st.session_state.explanations

    if "generated_answers" not in st.session_state or st.session_state.get("next_question", False):
        # 1つ選択
        QuestionNum = random.randint(0, len(explanations) - 1)
        SelectedQuestion = explanations[QuestionNum]

        # ===== Step 1: 意味の通る解説文にリライト =====
        st.write("解説文を整えています")
        CleanedExplanation = refine_explanation(SelectedQuestion, client)

        # ===== Step 2: 5問同時生成 =====
        prompt = f"""
あなたは教育用クイズ作成AIです。
次の解説文をもとに四択問題を5問作成してください。
各問題には「問題文」「選択肢4つ」「正解番号」「解説」を含めてください。

--- 解説文 ---
{CleanedExplanation}

出力形式（JSON配列）:
[
  {{
    "Question": "～",
    "Choice1": "～",
    "Choice2": "～",
    "Choice3": "～",
    "Choice4": "～",
    "CorrectAnswer": 1,
    "Explanation": "～"
  }},
  ...
]
"""

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "あなたは正確で教育的なクイズ作成AIです。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        )

        try:
            output_json = response.choices[0].message.content
            generated_answers = loads(output_json)
        except Exception as e:
            st.error(f"JSON解析エラー: {e}")
            st.stop()

        st.session_state.generated_answers = generated_answers
        st.session_state.explanation = CleanedExplanation
        st.session_state.next_question = False

        # ===== コサイン類似度 =====
        correct_texts = [q[f"Choice{q['CorrectAnswer']}"] for q in generated_answers]

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        embeddings = [
            client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
            for text in correct_texts
        ]

        n = len(embeddings)
        similarities = [
            cosine_similarity(embeddings[i], embeddings[j])
            for i in range(n) for j in range(i + 1, n)
        ]
        avg_cosine_similarity = np.mean(similarities)
        st.session_state.avg_cosine_similarity = avg_cosine_similarity

        # ===== RAGAS評価 =====
        sample = generated_answers[0]
        data = Dataset.from_dict({
            "question": [sample["Question"]],
            "answer": [sample[f"Choice{sample['CorrectAnswer']}"]],
            "contexts": [[CleanedExplanation]],
        })

        result1 = evaluate(data, metrics=[faithfulness])
        result2 = evaluate(data, metrics=[answer_relevancy])

        st.session_state.faithfulness = result1["faithfulness"][0]
        st.session_state.answer_relevance = result2["answer_relevancy"][0]

    # ===== 出力 =====
    st.subheader("整えた解説文")
    st.info(st.session_state.explanation)

    st.subheader("生成された5問（問題＋解説付き）")
    for idx, q in enumerate(st.session_state.generated_answers, start=1):
        st.markdown(f"### 問題 {idx}: {q['Question']}")
        st.markdown(
            f"1. {q['Choice1']}  \n"
            f"2. {q['Choice2']}  \n"
            f"3. {q['Choice3']}  \n"
            f"4. {q['Choice4']}  \n"
            f"✅ 正解: {q[f'Choice{q['CorrectAnswer']}']}"
        )
        st.info(f"解説：{q['Explanation']}")
        st.write("---")

    # ===== スコア表示 =====
    st.subheader("スコアまとめ")
    st.write(f"Faithfulnessスコア: {st.session_state.faithfulness}")
    st.write(f"Answer Relevancyスコア: {st.session_state.answer_relevance}")
    st.write(f"平均コサイン類似度（小さいほど多様性が高い）: {st.session_state.avg_cosine_similarity:.4f}")

    if st.button("次の問題へ"):
        st.session_state.next_question = True
        st.session_state.pop("generated_answers", None)
        st.session_state.pop("avg_cosine_similarity", None)
        st.rerun()

else:
    st.info("まずはPDFファイルをアップロードしてください。")
