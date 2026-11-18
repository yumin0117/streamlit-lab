# 📋 app.py · [실습 7-1] 화면 — 사진 · 음성 도우미 (Community Cloud 에 올릴 app.py · 분석 한 번에 요청 1번)
# · 이 화면에는 Gemini 키가 없음 — 사진 · 음성을 내 API(Vercel)로 보내고 답만 받아 보여 줌
# · Secrets 두 줄 : API_URL (내 vercel.app 주소) · APP_ACCESS_CODE (Vercel 과 같은 값)
import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

MAX_SIDE = 1024                     # 사진의 긴 쪽을 1024px 이하로 줄여 보냄 — Vercel 요청 본문 4.5MB 안에 들도록

try:
    API_URL = st.secrets["API_URL"]                  # Secrets 의 이름으로 꺼냄 (대괄호 안에 주소를 쓰지 않음)
except Exception:
    API_URL = "http://localhost:8000"                # Secrets 가 없으면(Colab 점검) 같은 런타임의 API 서버
try:
    ACCESS_CODE = st.secrets["APP_ACCESS_CODE"]
except Exception:
    ACCESS_CODE = ""

st.set_page_config(page_title="사진 · 음성 도우미", page_icon="📷")
st.title("📷 사진 · 음성 도우미")
st.caption(f"화면은 Streamlit · 일은 내 API 가 합니다 · API 주소 : {API_URL} · 얼굴 · 신분증 · 남의 목소리는 올리지 마세요")


def photo_b64(raw):
    """사진을 긴 쪽 1024px 이하 JPEG 로 줄이고 base64 글자로 바꾼다"""
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_api(path, body):
    """내 API 에 JSON 을 보내고 답 또는 오류를 화면에 보여 준다"""
    try:
        response = requests.post(f"{API_URL}{path}", json=body, headers={"X-Access-Code": ACCESS_CODE}, timeout=120)
    except requests.RequestException:
        st.error("❌ API 서버에 닿지 못함 — 화면 위 'API 주소' 가 내 vercel.app 주소인지 확인하세요.")
        return
    if response.status_code == 200:
        data = response.json()
        st.markdown(data["answer"])
        st.caption(f"상태 번호 200 · 토큰 {data['tokens']}")
    else:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text[:100]
        st.error(f"❌ 상태 번호 {response.status_code} : {detail}")


photo_tab, voice_tab = st.tabs(["📷 사진", "🎙️ 음성"])

with photo_tab:
    question = st.text_input("사진에 대한 질문", value="이 사진에 무엇이 있는지 3문장 이내로 설명해 줘.")
    camera = st.camera_input("카메라로 찍기") if st.toggle("카메라 켜기") else None
    upload = st.file_uploader("또는 사진 파일 올리기", type=["png", "jpg", "jpeg", "webp"])
    photo = camera or upload
    if photo is not None:
        st.image(photo, width=240)
    if st.button("사진 분석하기", key="photo_go", disabled=photo is None):
        with st.spinner("내 API 에 묻는 중..."):
            call_api("/describe", {"image_b64": photo_b64(photo.getvalue()), "question": question})
    if os.path.exists("receipt.png") and st.button("예시 사진(영수증)으로 해 보기", key="photo_sample"):
        st.image("receipt.png", width=240)
        with st.spinner("내 API 에 묻는 중..."):
            call_api("/describe", {"image_b64": photo_b64(open("receipt.png", "rb").read()), "question": question})

with voice_tab:
    task = st.selectbox("할 일", ["받아쓰기", "한 줄 요약", "영어로 번역"])
    voice = st.audio_input("녹음하기 (마이크 버튼 → 말하기 → 정지 · 1분 이내)")
    if st.button("음성 분석하기", key="voice_go", disabled=voice is None):
        with st.spinner("내 API 에 묻는 중..."):
            call_api("/listen", {"audio_b64": base64.b64encode(voice.getvalue()).decode("utf-8"), "task": task})
    if os.path.exists("voice.wav") and st.button("예시 음성으로 해 보기", key="voice_sample"):
        st.audio("voice.wav")
        with st.spinner("내 API 에 묻는 중..."):
            call_api("/listen", {"audio_b64": base64.b64encode(open("voice.wav", "rb").read()).decode("utf-8"), "task": task})
