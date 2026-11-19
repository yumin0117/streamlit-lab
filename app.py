"""AI 냉장고 탐정 화면. 기존 API_URL / APP_ACCESS_CODE Secrets를 사용합니다."""
import base64
import csv
import hashlib
import html
import io
import json
import os
import re
import warnings
from collections import Counter

import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_SIDE = 1280
MAX_UPLOAD = 20 * 1024 * 1024
DEFAULT_PANTRY = ["설탕", "간장", "소금", "고추장", "고춧가루", "후추", "식용유", "참기름", "식초", "다진 마늘", "물"]
CATEGORIES = ["전체", "과일", "채소", "유제품·달걀", "육류·생선", "음료·기타"]
COLORS = {"과일": "#e29024", "채소": "#219b63", "유제품·달걀": "#437ac5",
          "육류·생선": "#d05a71", "음료·기타": "#8764b5"}


def setting(name, default=""):
    try:
        return str(st.secrets[name]).strip()
    except (KeyError, FileNotFoundError):
        return os.environ.get(name, default).strip()


st.set_page_config(page_title="AI 냉장고 탐정", page_icon="🧊", layout="wide")
API_URL = setting("API_URL", "https://my-ai-api-flyn.vercel.app").rstrip("/")
ACCESS_CODE = setting("APP_ACCESS_CODE")

st.markdown("""
<style>
.stApp { background: #f7faf7; }
.block-container { max-width: 1180px; padding-top: 2.4rem; }
h1, h2, h3 { color: #173f31; }
[data-testid="stMetric"] { background: #eaf4ee; border-radius: 14px; padding: 16px; }
div.stButton > button[kind="primary"] { background: #207a50; border-color: #207a50; }
</style>
""", unsafe_allow_html=True)
st.caption("FRIDGE DETECTIVE · 우리 집 식재료 찾기")
st.title("🧊 AI 냉장고 탐정")
st.write("사진으로 냉장고 속 재료를 찾고, 그 재료로 만들 음식과 조리 순서까지 알아보세요.")


def prepare_photo(raw):
    if len(raw) > MAX_UPLOAD:
        raise ValueError("사진은 20MB 이하로 올려 주세요.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source)
            image.thumbnail((MAX_SIDE, MAX_SIDE))
            image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    data = buffer.getvalue()
    if len(base64.b64encode(data)) > 4_000_000:
        raise ValueError("변환한 사진이 너무 큽니다. 사진 크기를 줄여 주세요.")
    return data, image.width, image.height


def call_api(body, path="/detect"):
    try:
        response = requests.post(f"{API_URL}{path}", json=body,
                                 headers={"X-Access-Code": ACCESS_CODE}, timeout=(10, 115))
    except requests.Timeout:
        st.error("요청 시간이 초과되었습니다. 잠시 뒤 다시 시도해 주세요.")
        return None
    except requests.RequestException:
        st.error("분석 서버에 연결할 수 없습니다. 잠시 뒤 다시 시도해 주세요.")
        return None
    try:
        data = response.json()
    except ValueError:
        st.error(f"분석 서버가 올바른 응답을 보내지 않았습니다. (상태 {response.status_code})")
        return None
    if not response.ok:
        detail = data.get("detail", "서버 오류가 발생했습니다.") if isinstance(data, dict) else ""
        if isinstance(detail, list):
            detail = "입력한 내용이나 설정이 올바르지 않습니다. 재료 이름과 입력값을 확인해 주세요."
        if response.status_code == 404:
            detail = "분석 서버를 업데이트하는 중입니다. 잠시 뒤 다시 시도해 주세요."
        st.error(f"{detail} (상태 {response.status_code})")
        return None
    if path == "/recipe-by-name":
        valid_result = isinstance(data, dict) and "recipe" in data and (data["recipe"] is None or isinstance(data["recipe"], dict))
    else:
        result_field = "recipes" if path == "/recipes" else "detections"
        valid_result = isinstance(data, dict) and isinstance(data.get(result_field), list)
    if not valid_result:
        st.error("분석 결과 형식이 올바르지 않습니다. 다시 분석해 주세요.")
        return None
    return data


def show_photo(jpeg, width, height, rows, show_boxes):
    # HTML 텍스트를 이스케이프하여 모델 응답이 코드로 해석되지 않게 합니다.
    boxes = []
    if show_boxes:
        for row in rows:
            ymin, xmin, ymax, xmax = row["box"]
            color = "#65746c" if row["uncertain"] else COLORS.get(row["category"], "#219b63")
            label = html.escape(f'{row["label"]}{"?" if row["uncertain"] else ""} {row["confidence"]:.0%}')
            align = "right:0" if xmin > 650 else "left:0"
            boxes.append(
                f'<div class="box" style="left:{xmin/10}%;top:{ymin/10}%;'
                f'width:{(xmax-xmin)/10}%;height:{(ymax-ymin)/10}%;border-color:{color}">'
                f'<span style="background:{color};{align}">{label}</span></div>')
    encoded = base64.b64encode(jpeg).decode("ascii")
    display_height = min(580, max(200, round(height / width * 680)))
    st.iframe(
        f'<style>body{{margin:0;font-family:system-ui,sans-serif}}'
        f'.stage{{position:relative;width:fit-content;max-width:100%;margin:auto}}'
        f'img{{display:block;max-width:100%;max-height:{display_height}px;border-radius:12px}}'
        f'.box{{position:absolute;border:2px solid;box-sizing:border-box}}'
        f'.box span{{position:absolute;top:0;color:white;font-size:12px;padding:2px 5px;'
        f'border-radius:0 0 4px 0;white-space:nowrap;max-width:240px;overflow:hidden}}'
        f'</style><div class="stage"><img src="data:image/jpeg;base64,{encoded}" '
        f'alt="냉장고 식품 분석 사진">{"".join(boxes)}</div>',
        height=display_height + 8, alt="식품 탐지 위치와 한국어 이름")


with st.sidebar:
    st.header("탐정의 분석 설정")
    image_kind = st.radio("이미지 유형", ["사진", "그림·일러스트"])
    category = st.selectbox("찾을 식품", CATEGORIES)
    st.divider()
    st.subheader("결과 표시")
    threshold = st.slider("최소 신뢰도", 10, 95, 40, 5, format="%d%%")
    show_boxes = st.toggle("탐지 박스 표시", value=True)
    hide_uncertain = st.toggle("불확실한 식품 숨기기", value=False)
    st.caption("신뢰도와 표시 옵션은 저장된 결과에 바로 적용됩니다. 추가 AI 요청은 발생하지 않습니다.")
    with st.expander("사용 안내"):
        st.write("① 밝은 곳에서 식품이 보이게 촬영하세요.\n\n② 사진을 올리고 분석을 눌러 주세요.\n\n③ 박스와 목록을 비교해 확인하세요.")
        st.caption("신뢰도는 AI의 추정치입니다. 가려진 식품과 묶음의 개수는 부정확할 수 있습니다.")
        st.caption("사진은 분석을 위해 서버와 Gemini로 전송됩니다.")
        st.markdown("[참고 프로젝트 · AI 냉장고 탐정](https://huggingface.co/spaces/yumin0601/ai_fridge_detective)")

left, right = st.columns([1.35, 1], gap="large")
with left:
    st.subheader("1. 사진 준비")
    upload_tab, camera_tab = st.tabs(["📁 사진 올리기", "📷 카메라로 찍기"])
    with upload_tab:
        upload = st.file_uploader("JPG, PNG, WEBP · 최대 20MB", type=["jpg", "jpeg", "png", "webp"], max_upload_size=20)
    with camera_tab:
        camera = st.camera_input("냉장고나 식재료를 촬영해 주세요") if st.toggle("카메라 켜기") else None
    if camera is not None and upload is not None:
        source = st.radio("분석할 사진", ["업로드 사진", "카메라 사진"], horizontal=True)
        photo = camera if source == "카메라 사진" else upload
    else:
        photo = camera or upload

    prepared = None
    if photo is not None:
        try:
            prepared = prepare_photo(photo.getvalue())
        except (UnidentifiedImageError, OSError, ValueError,
                Image.DecompressionBombError, Image.DecompressionBombWarning):
            st.error("사진을 열 수 없습니다. 20MB 이하의 JPG, PNG 또는 WEBP 사진으로 다시 올려 주세요.")

    key = None
    if prepared:
        jpeg, width, height = prepared
        key = (hashlib.sha256(jpeg).hexdigest(), image_kind, category, API_URL)
    if st.session_state.get("fridge_result_key") != key:
        st.session_state.pop("fridge_result", None)
        st.session_state.pop("fridge_result_key", None)

    analyze = st.button("🔎 냉장고 분석하기", type="primary", width="stretch",
                        disabled=prepared is None)
    if analyze and prepared:
        with st.spinner("식품을 찾고 있습니다. 잠시만 기다려 주세요…"):
            result = call_api({"image_b64": base64.b64encode(jpeg).decode("ascii"),
                               "mime_type": "image/jpeg", "image_kind": image_kind,
                               "category": category})
        if result is not None:
            st.session_state["fridge_result"] = result
            st.session_state["fridge_result_key"] = key
            st.session_state.pop("recipe_source", None)

result = st.session_state.get("fridge_result")
candidates = [r for r in result["detections"] if r["confidence"] >= threshold / 100] if result else []
visible = [r for r in candidates if not (hide_uncertain and r["uncertain"])]
confirmed = [r for r in candidates if not r["uncertain"]]
counts = Counter(r["label"] for r in confirmed)

with left:
    if prepared:
        show_photo(jpeg, width, height, visible, show_boxes)
        st.caption(f"분석 이미지 {width} × {height}px · 이름과 박스로 식품의 위치를 확인하세요.")
    else:
        st.info("사진을 올리면 이곳에 식품의 위치와 한국어 이름이 표시됩니다.")

with right:
    st.subheader("2. 식품 탐지 결과")
    if result is None:
        st.write("사진 속 식품을 찾아 종류별로 정리해 드려요.")
        st.markdown("🍎 **과일** · 사과, 바나나 등\n\n🥬 **채소** · 당근, 양배추 등\n\n🥚 **유제품·달걀** · 우유, 치즈 등\n\n🐟 **육류·생선** · 고기, 생선 등\n\n🧃 **음료·기타** · 음료, 소스 등")
        st.caption("사진이나 분석 범위를 바꾸면 이전 결과는 초기화됩니다.")
    else:
        a, b = st.columns(2)
        a.metric("발견한 종류", f"{len(counts)}종")
        b.metric("집계한 식품", f"{sum(counts.values())}개")
        st.write(result.get("summary", ""))
        if counts:
            st.dataframe([{"식품": name, "개수": count} for name, count in sorted(counts.items())],
                         hide_index=True, width="stretch")
        elif candidates:
            st.info("확실하게 구분된 식품이 없습니다. 신뢰도를 조절하거나 더 선명한 사진으로 다시 분석해 주세요.")
        else:
            st.info("조건에 맞는 식품을 찾지 못했습니다. 신뢰도를 낮추거나 식품이 크게 보이는 사진으로 다시 시도해 주세요.")
        uncertain = [r for r in candidates if r["uncertain"]]
        if uncertain:
            st.warning(f"종류가 불확실한 {len(uncertain)}개는 집계에서 제외했습니다.")
            with st.expander("불확실한 식품 확인"):
                for row in uncertain:
                    alternative = f' / 다른 후보: {row["alternative"]}' if row.get("alternative") else ""
                    st.write(f'• {row["label"]} ({row["confidence"]:.0%}){alternative}')
        if visible:
            with st.expander("탐지 상세 보기"):
                st.dataframe([{"식품": r["label"], "분류": r["category"],
                               "추정 신뢰도": f'{r["confidence"]:.0%}',
                               "확인 상태": "불확실" if r["uncertain"] else "집계 포함"} for r in visible],
                             hide_index=True, width="stretch")
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["식품", "개수"])
        writer.writerows(sorted(counts.items()))
        st.download_button("📥 식품 목록 CSV", csv_buffer.getvalue().encode("utf-8-sig"),
                           file_name="fridge_inventory.csv", mime="text/csv")
        export = {**result, "display_threshold": threshold / 100,
                  "filtered_inventory": [{"label": k, "count": v} for k, v in sorted(counts.items())]}
        st.download_button("📄 전체 분석 JSON", json.dumps(export, ensure_ascii=False, indent=2),
                           file_name="fridge_result.json", mime="application/json")
        st.caption(f'분석 완료 · 사용 토큰 {result.get("tokens") or "정보 없음"}')

st.divider()
st.subheader("3. 이 재료로 무엇을 만들까요?")
st.write("찾은 재료를 확인하고 요리를 추천받으세요. 잘못 인식한 이름을 고치거나 집에 있는 재료를 추가할 수 있어요.")

recipe_source = (st.session_state.get("fridge_result_key"), tuple(sorted(counts)))
if st.session_state.get("recipe_source") != recipe_source:
    st.session_state["recipe_ingredients"] = ", ".join(sorted(counts))
    st.session_state["recipe_source"] = recipe_source
    st.session_state.pop("recipe_result", None)
    st.session_state.pop("recipe_result_key", None)

ingredient_text = st.text_area("추천에 사용할 재료", key="recipe_ingredients",
                               placeholder="예: 달걀, 두부, 양파, 당근", height=90,
                               help="쉼표나 줄바꿈으로 나눠 주세요. 불확실하게 탐지된 식품은 자동으로 포함하지 않습니다.")
ingredients = list(dict.fromkeys(name.strip() for name in re.split(r"[,\n]+", ingredient_text) if name.strip()))
cuisine_col, servings_col = st.columns([2, 1])
with cuisine_col:
    cuisine = st.selectbox("요리 종류", ["상관없음", "한식", "중식", "양식", "일식"], key="recipe_cuisine",
                           help="선택한 종류와 보유 재료에 맞는 음식을 추천합니다.")
with servings_col:
    servings = st.number_input("몇 인분을 만들까요?", min_value=1, max_value=6, value=2, step=1)
with st.expander("집에 있다고 가정하는 기본 양념", expanded=True):
    pantry = st.multiselect("사용할 수 있는 양념", DEFAULT_PANTRY, default=DEFAULT_PANTRY,
                            help="없는 양념은 빼 주세요. 밥·면·밀가루 등은 위의 재료 칸에 직접 적어 주세요.")
    st.caption("설탕·간장·소금·고추장·고춧가루 등 선택한 양념은 이미 있다고 가정합니다.")

recipe_key = (tuple(ingredients), int(servings), tuple(pantry), cuisine, API_URL)
if st.session_state.get("recipe_result_key") != recipe_key:
    st.session_state.pop("recipe_result", None)
if st.button("🍳 이 재료로 요리 추천받기", type="primary", disabled=not ingredients, key="recipe_go"):
    with st.spinner("보유 재료로 만들 음식과 조리 순서를 정리하고 있습니다…"):
        recommended = call_api({"ingredients": ingredients, "servings": int(servings),
                                "pantry": pantry, "cuisine": cuisine}, path="/recipes")
    if recommended is not None:
        st.session_state["recipe_result"] = recommended
        st.session_state["recipe_result_key"] = recipe_key

recommended = st.session_state.get("recipe_result")
if recommended is not None:
    if recommended.get("note"):
        st.info(recommended["note"])
    dishes = recommended["recipes"]
    if dishes:
        recipe_tabs = st.tabs([dish["name"] for dish in dishes])
        for tab, dish in zip(recipe_tabs, dishes):
            with tab:
                st.subheader(dish["name"])
                st.caption(f'{recommended.get("cuisine", cuisine)} · {recommended["servings"]}인분 · 약 {dish["minutes"]}분')
                st.write(dish["description"])
                ingredient_col, steps_col = st.columns([1, 2], gap="large")
                with ingredient_col:
                    st.markdown("**사용할 재료와 분량**")
                    for item in dish["ingredients"]:
                        kind = "기본 양념" if item["name"] in pantry and item["name"] not in ingredients else "보유 재료"
                        st.write(f'• {item["name"]} — {item["amount"]} ({kind})')
                with steps_col:
                    st.markdown("**만드는 순서**")
                    for number, step in enumerate(dish["steps"], 1):
                        st.write(f"{number}. {step}")
                    if dish.get("tip"):
                        st.success(f'조리 팁 · {dish["tip"]}')
        recipe_text = ["요리 종류: " + recommended.get("cuisine", cuisine), ""]
        for dish in dishes:
            recipe_text.extend([dish["name"], f'{recommended["servings"]}인분 / 약 {dish["minutes"]}분',
                                "재료: " + ", ".join(f'{i["name"]} {i["amount"]}' for i in dish["ingredients"])])
            recipe_text.extend(f"{number}. {step}" for number, step in enumerate(dish["steps"], 1))
            recipe_text.extend(["조리 팁: " + dish.get("tip", ""), ""])
        recipe_text.append(recommended.get("note", ""))
        st.download_button("📥 추천 요리와 만드는 순서 저장", "\n".join(recipe_text).encode("utf-8-sig"),
                           file_name="fridge_recipes.txt", mime="text/plain")
    else:
        st.warning("현재 재료로 추천할 수 있는 요리가 없습니다. 보유한 식재료를 더 입력해 주세요.")
elif not ingredients:
    st.caption("사진을 분석하거나 재료를 직접 입력하면 요리를 추천받을 수 있습니다.")

st.divider()
st.subheader("4. 음식 이름으로 조리법 알아보기")
st.write("만들고 싶은 음식 이름을 입력하면 필요한 재료와 분량, 만드는 순서를 알려드려요.")
st.caption("사진 없이도 사용할 수 있어요. 기본 양념을 포함해 요리에 필요한 전체 재료를 안내합니다.")
dish_name_col, dish_servings_col = st.columns([2, 1])
with dish_name_col:
    dish_name = st.text_input("음식 이름", placeholder="예: 김치찌개, 오므라이스, 마파두부",
                              max_chars=80, key="dish_lookup_name").strip()
with dish_servings_col:
    dish_servings = st.number_input("조리법 인분", min_value=1, max_value=6, value=2, step=1,
                                    key="dish_lookup_servings")

dish_key = (dish_name, int(dish_servings), API_URL)
if st.session_state.get("dish_result_key") != dish_key:
    st.session_state.pop("dish_result", None)
if st.button("📖 조리법 알아보기", type="primary", disabled=not dish_name, key="dish_lookup_go"):
    with st.spinner("필요한 재료와 만드는 순서를 정리하고 있습니다…"):
        dish_result = call_api({"dish_name": dish_name, "servings": int(dish_servings)}, path="/recipe-by-name")
    if dish_result is not None:
        st.session_state["dish_result"] = dish_result
        st.session_state["dish_result_key"] = dish_key

dish_result = st.session_state.get("dish_result")
if dish_result is not None:
    dish = dish_result["recipe"]
    if dish is None:
        st.info(dish_result.get("note") or "음식 이름을 확인하고 다시 입력해 주세요.")
    else:
        st.subheader(dish["name"])
        st.caption(f'{dish_result["servings"]}인분 · 약 {dish["minutes"]}분')
        st.write(dish["description"])
        dish_ingredients_col, dish_steps_col = st.columns([1, 2], gap="large")
        with dish_ingredients_col:
            st.markdown("**필요한 재료와 분량**")
            st.dataframe([{"재료·양념": item["name"], "분량": item["amount"]} for item in dish["ingredients"]],
                         hide_index=True, width="stretch")
        with dish_steps_col:
            st.markdown("**만드는 순서**")
            for number, step in enumerate(dish["steps"], 1):
                st.write(f"{number}. {step}")
            if dish.get("tip"):
                st.success(f'조리 팁 · {dish["tip"]}')
        if dish_result.get("note"):
            st.info(dish_result["note"])
        dish_text = [dish["name"], f'{dish_result["servings"]}인분 / 약 {dish["minutes"]}분', dish["description"],
                     "재료: " + ", ".join(f'{item["name"]} {item["amount"]}' for item in dish["ingredients"])]
        dish_text.extend(f"{number}. {step}" for number, step in enumerate(dish["steps"], 1))
        dish_text.extend(["조리 팁: " + dish.get("tip", ""), dish_result.get("note", "")])
        st.download_button("📥 이 조리법 저장", "\n".join(dish_text).encode("utf-8-sig"),
                           file_name="recipe_by_name.txt", mime="text/plain")

st.divider()
st.caption("식품의 종류와 위치를 찾는 AI 도우미입니다. 사진만으로 유통기한·신선도·섭취 안전성을 판단할 수 없습니다.")
