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

# ===== Streamlit UI =====
st.title("兵庫学検定試験対策ツール（5問同時出題＋平均コサイン類似度付き）")

# ファイルアップロード
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
        QuestionNum = random.randint(0, len(explanations) - 1)
        SelectedQuestion = explanations[QuestionNum]

        NUM_VARIANTS = 15
        generated_answers = []

        for i in range(NUM_VARIANTS):
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "あなたはクイズの出題者です。以下の文から四択問題を作成してください。"
                            "本文内容に基づいた問題にしてください。"
                            "出力はJSON形式で返してください。"
                        )
                    },
                    {"role": "user", "content": SelectedQuestion},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "QuestionData",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "Question": {"type": "string"},
                                "Choice1": {"type": "string"},
                                "Choice2": {"type": "string"},
                                "Choice3": {"type": "string"},
                                "Choice4": {"type": "string"},
                                "CorrectAnswer": {"type": "number"},
                            },
                            "required": ["Question", "Choice1", "Choice2", "Choice3", "Choice4", "CorrectAnswer"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
                temperature=1.4
            )

            output_text = response.choices[0].message.content
            try:
                data = loads(output_text)
                generated_answers.append(data)
            except Exception as e:
                st.error(f"JSON読み込み失敗: {e}")

        st.session_state.generated_answers = generated_answers
        st.session_state.explanation = SelectedQuestion
        st.session_state.selected_question = SelectedQuestion
        st.session_state.next_question = False

        # ===== コサイン類似度計算 =====
        all_correct_answers = [
            d[f"Choice{d['CorrectAnswer']}"] for d in generated_answers
        ]

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        embeddings = [
            client.embeddings.create(input=ans, model="text-embedding-3-small").data[0].embedding
            for ans in all_correct_answers
        ]

        n = len(embeddings)
        similarities = [
            cosine_similarity(embeddings[i], embeddings[j])
            for i in range(n) for j in range(i + 1, n)
        ]
        avg_cosine_similarity = np.mean(similarities)
        st.session_state.avg_cosine_similarity = avg_cosine_similarity

    # ===== RAGAS評価（最初の1問で代表評価） =====
    sample = st.session_state.generated_answers[0]
    data = Dataset.from_dict({
        "question": [sample["Question"]],
        "answer": [sample[f"Choice{sample['CorrectAnswer']}"]],
        "contexts": [[st.session_state.selected_question]],
    })

    # Faithfulness のみ評価
    result1 = evaluate(data, metrics=[faithfulness])
    st.session_state.faithfulness = result1["faithfulness"][0]

    # ===== UI表示（5問まとめて） =====
    st.subheader("生成された5問すべての問題")
    for idx, q in enumerate(st.session_state.generated_answers, start=1):
        st.markdown(f"### 問題 {idx}: {q['Question']}")
        st.markdown(
            f"1. {q['Choice1']}  \n"
            f"2. {q['Choice2']}  \n"
            f"3. {q['Choice3']}  \n"
            f"4. {q['Choice4']}  \n"
            f"✅ 正解: {q[f'Choice{q['CorrectAnswer']}']}"
        )
        st.info(f"解説：{st.session_state.explanation}")
        st.write("---")

    # ===== スコア表示 =====
    st.subheader("スコアまとめ")
    st.write(f"Faithfulnessスコア: {st.session_state.faithfulness}")
    #st.write(f"Answer Relevancyスコア: {st.session_state.answer_relevance}")
    st.write(f"平均コサイン類似度: {st.session_state.avg_cosine_similarity}")

    if st.button("次の問題へ"):
        st.session_state.next_question = True
        st.session_state.pop("generated_answers", None)
        st.session_state.pop("avg_cosine_similarity", None)
        st.rerun()
else:
    st.info("まずはPDFファイルをアップロードしてください。")
