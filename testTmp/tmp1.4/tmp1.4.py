import json
from json import loads
import re
import random
import os
from dotenv import load_dotenv
import streamlit as st
import openai
from openai import OpenAI
import csv

# ===== 解説CSV読み込み関数（nullバイト対応・ヘッダーなし対応） =====
def load_explanations_from_csv(filename="Book1.csv"):
    explanations = []
    with open(filename, "rb") as f:
        content = f.read().replace(b'\x00', b'')  # null バイト削除
    decoded = content.decode("utf-8")
    reader = csv.reader(decoded.splitlines())
    for row in reader:
        if row:  # 空行チェック
            explanations.append(row[0])
    return explanations

# ===== OpenAI API キーの読み込み =====
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ===== 説明文の読み込みとセッション保持 =====
if "explanations" not in st.session_state:
    st.session_state.explanations = load_explanations_from_csv("Book1.csv")

explanations = st.session_state.explanations

# ===== クイズの出題処理 =====
if "question_data" not in st.session_state or st.session_state.get("next_question", False):
    QuestionNum = random.randint(0, len(explanations) - 1)
    SelectedQuestion = explanations[QuestionNum]

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "あなたはクイズの出題者です。以下の文から4択問題を出題してください。"
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

    # ===== GPTの応答からJSON抽出 =====
    output_text = response.choices[0].message.content
    match = re.search(r"\{.*\}", output_text, re.DOTALL)

    if match:
        try:
            json_str = match.group()
            data = loads(json_str)
            st.session_state.question_data = data
            st.session_state.explanation = SelectedQuestion
            st.session_state.next_question = False
        except Exception as e:
            st.error(f"JSON読み込み失敗: {e}")
    else:
        st.error("JSON形式の出力が見つかりませんでした。")

# ===== UI表示 =====
st.title("兵庫学検定試験対策ツール Temperature=1.4")
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
    if selected_index == st.session_state.question_data["CorrectAnswer"]:
        st.success("正解！")
    else:
        st.error("不正解…")
        st.info(f"解説：{st.session_state.explanation}")

if st.button("次の問題へ"):
    st.session_state.next_question = True
    st.rerun()
