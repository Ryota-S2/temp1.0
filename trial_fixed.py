import json
from json import loads
import re
import random
import os
from dotenv import load_dotenv
import streamlit as st
import openai
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=api_key)
explanations = ["神戸市には東灘区、灘区、中央区、兵庫区、長田区、須磨区、垂水区、北区、西区の9つの区がある。",
                "兵庫県の中央には中国山地があり、中国山地の北側と南側で気候が異なります。北部は山々が連なり、南部は平野が広がっています。また、山崎断層帯、六甲・淡路島断層帯、有馬ー高槻断層帯、中央構造線といった断層が分布している。",
                "五国の由来。 摂津：『古事記』 『日本書紀』 には、「津国」とある。津は港を意味する。国名は難波津、武庫水門などの良港があったことに由来する。 播磨：もとは針間国・針間鴨国・明石国の3か国に分かれていた 但馬：『古事記』 には 「多遅麻」 「多遅摩」 とある。 『日本書紀』 ではすべて但馬である。「谷間」に由来するといわれる。 丹波：赤米のが波のように見えたことから丹波と名付けられた.山岳が重なっている山国の底であるからされ、 それが丹波に変化した 淡路：阿波の国へ行く道 (路) に由来するといわれる。",
                "青森県から山口県まで本州を縦断するとき、 必す通らなければならない唯一の都道府県が私たちの住む兵庫県です。 北は日本海、南は瀬戸内海(せとないかい)、太平洋に面する広い兵庫県生には、大久保利通(としみち)が関わっています。また、兵庫県の初代知事は伊藤博文(いとうひろふみ)です。明治政府が重要視した場所ー兵庫県の変をみていきましよう。",
                "丹波地域は、 丹波篠山市と丹波市からなる、 兵庫県の中東部にる地域と京都府中部を合わせた地域てす 山々か重なり、 深い鬟土の蕓りに包まれた丹波地域は、 山林面積か約75 %を占めており盆地が多く、 昼夜の気温差か大きい独特の気候・風土か特色です",
                ]

if "question_data" not in st.session_state or st.session_state.get("next_question", False):
    QuestionNum = random.randint(0, len(explanations)-1)
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
    )

    output_text = response.choices[0].message.content
    match = re.search(r"\{.*\}", output_text, re.DOTALL)

    if match:
        try:
            json_str = match.group()
            data = loads(json_str)
            st.session_state.question_data = data
            st.session_state.explanation = SelectedQuestion
            st.session_state.next_question = False  # フラグをリセット！

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
    if selected_index == st.session_state.question_data["CorrectAnswer"]:
        st.success("正解！")
    else:
        st.error("不正解…")
        st.info(f"解説：{st.session_state.explanation}")

if st.button("次の問題へ"):
    # 次の問題フラグをセットし、再実行で新しい問題へ
    st.session_state.next_question = True
    st.rerun()