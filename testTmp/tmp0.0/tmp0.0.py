import json
from json import loads
import random
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Book1.csv")


def load_csv(path):
    return pd.read_csv(path, encoding="utf-8", header=None)


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

st.title("📘 CSV教材 → 四択問題生成アプリ（Structured Outputs / Temperature=0.0）")

# ===== Book1.csv を読み込む =====
if not os.path.exists(CSV_PATH):
    st.error(f"Book1.csv が見つかりません: {CSV_PATH}")
    st.stop()

try:
    df = load_csv(CSV_PATH)
except UnicodeDecodeError:
    st.error("Book1.csv を UTF-8 に変換して保存し直してください。")
    st.stop()

# ===== 1列目を教材文章として使用 =====
explanations_list = df[0].dropna().astype(str).tolist()

# ===== 初期化 =====
if "question_data" not in st.session_state:
    st.session_state.next_question = True


if st.session_state.next_question:

    SelectedText = random.choice(explanations_list)

    response = client.chat.completions.create(
        model="gpt-4.1",   # ← 安定性重視
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたはクイズの出題者です。"
                    "与えられた文章を読んで、内容に基づく四択問題を作成してください。"
                    "出力は指定された JSON Schema に厳密に従ってください。"
                )
            },
            {"role": "user", "content": SelectedText},
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
                    "required": [
                        "Question",
                        "Choice1",
                        "Choice2",
                        "Choice3",
                        "Choice4",
                        "CorrectAnswer"
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        temperature=0.0
    )

    data = json.loads(response.choices[0].message.content)

    st.session_state.question_data = data
    st.session_state.explanation = SelectedText
    st.session_state.next_question = False

q = st.session_state.question_data

st.subheader("🔍 問題")
st.write(q["Question"])

choices = [
    f"1. {q['Choice1']}",
    f"2. {q['Choice2']}",
    f"3. {q['Choice3']}",
    f"4. {q['Choice4']}",
]

selected = st.radio("選択肢：", choices)

if st.button("解答"):
    selected_index = choices.index(selected) + 1
    if selected_index == q["CorrectAnswer"]:
        st.success("正解！")
    else:
        st.error("不正解")
    st.info(f"📘 元の文章：\n{st.session_state.explanation}")

if st.button("次の問題へ"):
    st.session_state.next_question = True
    st.rerun()
