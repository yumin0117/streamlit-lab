# 📋 app.py · [실습 5-2] 심화 ① AI 퀴즈 앱 — JSON 으로 받은 문제를 위젯으로 풀기 (퀴즈 만들 때마다 요청 1번)
import json
import streamlit as st
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.5-flash-lite"

# 문제 하나 = 질문 · 보기 4개 · 정답 번호(0~3) · 해설
QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "choices": {"type": "array", "items": {"type": "string"}, "description": "보기 4개"},
                    "answer_index": {"type": "integer", "description": "정답 보기의 번호 (0부터 셈)"},
                    "explanation": {"type": "string", "description": "정답 해설 한 문장"},
                },
                "required": ["question", "choices", "answer_index", "explanation"],
            },
        }
    },
    "required": ["questions"],
}


@st.cache_resource
def get_client():
    """Gemini client 를 만들어 돌려준다 (키는 st.secrets 에서)"""
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            timeout=120_000,
            retry_options=types.HttpRetryOptions(http_status_codes=[500, 503]),
        ),
    )


def explain_error(error):
    """오류 번호를 보고 할 일을 한글로 알려 준다"""
    code = getattr(error, "status_code", None)
    if code == 429:
        return "⏳ 무료 한도를 넘음 — 1분 기다렸다 다시 눌러 보세요."
    if code in (400, 403):
        return "❌ 키 문제 — 키를 다시 복사했는지, AI Studio 에서 키 상태를 확인하세요."
    return f"❌ 오류 ({code}) : {error}"


def make_quiz(topic, count, level):
    """주제 · 문제 수 · 난이도로 퀴즈를 만들어 문제 리스트를 돌려준다"""
    reply = client.interactions.create(
        model=MODEL_NAME,
        store=False,
        system_instruction="너는 대학 1학년 수업용 객관식 퀴즈 출제자야. 보기는 반드시 4개, 정답은 하나만 둔다. 사실이 확실한 내용만 출제한다.",
        input=f"주제 '{topic}' 에 대해 {level} 난이도 문제 {count}개를 만들어 줘.",
        response_format={"type": "text", "mime_type": "application/json", "schema": QUIZ_SCHEMA},
    )
    questions = json.loads(reply.output_text)["questions"]
    # 모델이 규칙을 어긴 문제(보기가 4개가 아니거나 정답 번호가 이상함)는 빼고 씀
    return [q for q in questions if len(q["choices"]) == 4 and 0 <= q["answer_index"] <= 3][:count]


try:
    client = get_client()
except Exception:
    st.error("🔑 GEMINI_API_KEY 를 찾지 못했습니다 — Manage app → ⋮ → Settings → Secrets 칸을 확인하세요.")
    st.stop()

st.title("🧩 AI 퀴즈")

# ---------- ① 사이드바에서 퀴즈 만들기 ----------
with st.sidebar:
    st.header("퀴즈 설정")
    topic = st.text_input("주제", value="파이썬 리스트")
    count = st.slider("문제 수", min_value=3, max_value=5, value=3)
    level = st.radio("난이도", ["쉬움", "보통", "어려움"], horizontal=True)
    if st.button("✨ 퀴즈 만들기", disabled=(topic.strip() == "")):
        with st.spinner("문제를 만드는 중..."):
            try:
                st.session_state.quiz = make_quiz(topic, count, level)
                st.session_state.graded = False           # 새 퀴즈 → 아직 채점 전
            except Exception as error:
                st.error(explain_error(error))

if "quiz" not in st.session_state:
    st.info("👈 왼쪽에서 주제를 정하고 [✨ 퀴즈 만들기] 를 누르세요.")
    st.stop()

# ---------- ② 문제 풀기 — form 안의 위젯은 [채점하기] 를 누를 때 한꺼번에 전달됨 ----------
quiz = st.session_state.quiz
if not quiz:
    st.warning("규칙에 맞는 문제가 하나도 없었습니다. [✨ 퀴즈 만들기] 를 다시 눌러 보세요.")
    st.stop()
with st.form("quiz_form"):
    for number, question in enumerate(quiz):
        st.radio(f"**{number + 1}. {question['question']}**", question["choices"], index=None, key=f"answer_{number}")
    submitted = st.form_submit_button("채점하기")

if submitted:
    st.session_state.graded = True

# ---------- ③ 채점 ----------
if st.session_state.graded:
    score = 0
    for number, question in enumerate(quiz):
        picked = st.session_state.get(f"answer_{number}")            # 고른 보기의 글 (안 골랐으면 None)
        correct = question["choices"][question["answer_index"]]
        if picked == correct:
            score += 1
            st.success(f"{number + 1}번 ✅ 정답 — {question['explanation']}")
        else:
            st.error(f"{number + 1}번 ❌ 정답은 '{correct}' — {question['explanation']}")
    st.metric("점수", f"{score} / {len(quiz)}")
    if score == len(quiz):
        st.balloons()                       # 만점이면 풍선
