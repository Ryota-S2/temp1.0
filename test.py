import csv
import os
import random
import re
from json import loads
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI



def load_explanations_from_csv(filename="Book1.csv"):
    explanations = []
    with open(filename, "rb") as f:
        content = f.read().replace(b'\x00', b'')
    decoded = content.decode("utf-8")
    reader = csv.reader(decoded.splitlines())
    for row in reader:
        if row:
            explanations.append(row[0])
    return explanations



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)



if "explanations" not in st.session_state:
    st.session_state.explanations = load_explanations_from_csv("Book1.csv")

explanations = st.session_state.explanations



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
        temperature=1.0
    )

    output_text = response.choices[0].message.content
    try:
        data = loads(output_text)
        st.session_state.question_data = data
        st.session_state.explanation = SelectedQuestion
        st.session_state.next_question = False
    except Exception as e:
        st.error(f"JSON読み込み失敗: {e}")





st.title("兵庫学検定試験対策ツール")
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
        st.error(f"不正解… 正解は {correct_index} 番です。")

    st.info(f"解説：{st.session_state.explanation}")



if st.button("次の問題へ"):
    st.session_state.next_question = True
    st.rerun()
