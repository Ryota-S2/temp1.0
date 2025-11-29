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

# ===== PDF → CSV変換 =====
def pdf_to_csv(pdf_file, csv_file="Book1.csv"):
    """PDFを読み込み、各ページをCSVに保存"""
    explanations = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                explanations.append([text.strip()])
    # 保存
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(explanations)

# ===== CSV読み込み =====
def load_explanations_from_csv(filename="Book1.csv"):
    explanations = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # 空行スキップ
                explanations.append(row[0])
    return explanations

# ===== OpenAI API キーの読み込み =====
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ===== Streamlit UI =====
st.title("兵庫学検定試験対策ツール")

# ファイルアップロード
uploaded_file = st.file_uploader("クイズに使うPDFファイルを選んでください", type=["pdf"])

if uploaded_file is not None:
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # PDF → CSV 変換
    pdf_to_csv(pdf_path, "Book1.csv")

    # CSVを即ロードしてセッションに保持
    st.session_state.explanations = load_explanations_from_csv("Book1.csv")
    st.success(f"PDFを変換して {len(st.session_state.explanations)} 件の文章を読み込みました！")

# ===== クイズ出題処理 =====
if "explanations" in st.session_state and st.session_state.explanations:
    explanations = st.session_state.explanations

    if "question_data" not in st.session_state or st.session_state.get("next_question", False):
        QuestionNum = random.randint(0, len(explanations) - 1)
        SelectedQuestion = explanations[QuestionNum]

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたはクイズの出題者です。以下の文から四択問題を作成してください。"
                        "必ず本文の内容理解に基づいた問題にしてください。"
                        "ページ番号や位置情報（例: ○ページに書いてある、何行目など）に関する問題は出さないでください。"
                        "出力は必ずJSON形式で返してください。"
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
            temperature=1.0
        )

        # ===== GPTの応答からJSON抽出 =====
        output_text = response.choices[0].message.content
        try:
            data = loads(output_text)
            st.session_state.question_data = data
            st.session_state.explanation = SelectedQuestion
            st.session_state.next_question = False
        except Exception as e:
            st.error(f"JSON読み込み失敗: {e}")

    # ===== UI表示 =====
    st.write("以下の問題に答えてください：")
    st.write(st.session_state.question_data['Question'])

    choices = [
        f"1. {st.session_state.question_data['Choice1']}",
        f"2. {st.session_state.question_data['Choice2']}",
        f"3. {st.session_state.question_data['Choice3']}",
        f"4. {st.session_state.question_data['Choice4']}"
    ]

    selected = st.radio("選択肢：", choices)

    if st.button("解答"):
        selected_index = choices.index(selected) + 1
        correct_index = st.session_state.question_data["CorrectAnswer"]
        correct_answer = st.session_state.question_data[f"Choice{correct_index}"]

        if selected_index == correct_index:
            st.success("正解！")
        else:
            st.error(f"不正解… 正解は {correct_index} 番: {correct_answer}")

        st.info(f"解説：{st.session_state.explanation}")

        # === Faithfulness / Answer relevancy 評価 ===
        data = Dataset.from_dict({
            "question": [st.session_state.question_data["Question"]],
            "answer": [st.session_state.question_data[f"Choice{st.session_state.question_data['CorrectAnswer']}"]],
            "contexts": [[st.session_state.explanation]],
        })

        result1 = evaluate(data, metrics=[faithfulness])
        result2 = evaluate(data, metrics=[answer_relevancy])

        st.write("Faithfulnessスコア:", result1["faithfulness"][0])
        st.write("Answer Relevancyスコア:", result2["answer_relevancy"][0])

    if st.button("次の問題へ"):
        st.session_state.next_question = True
        st.rerun()
else:
    st.info("まずはPDFファイルをアップロードしてください。")
