import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

import streamlit as st
import pandas as pd
from agent import run_agent
from tools import get_player_stats, predict_salary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="NBA Value Analyzer",
    page_icon="🏀",
    layout="wide"
)

st.markdown("""
<style>
.metric-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid #16213e;
}
.metric-label { color: #888; font-size: 13px; margin-bottom: 6px; }
.metric-value { color: #fff; font-size: 24px; font-weight: bold; }
.verdict-overpaid  { background: #3d1a1a; border: 1px solid #e74c3c; border-radius: 12px; padding: 16px; text-align: center; }
.verdict-fair      { background: #1a3d1a; border: 1px solid #2ecc71; border-radius: 12px; padding: 16px; text-align: center; }
.verdict-underpaid { background: #1a1a3d; border: 1px solid #3498db; border-radius: 12px; padding: 16px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🏀 NBA Player Value Analyzer")
st.caption("선수 스탯 기반 연봉 가치 분석 시스템 | Random Forest + Claude AI")
st.divider()

col1, col2 = st.columns([2, 1])
with col1:
    player_name = st.text_input("선수 이름 (영문)", placeholder="예: LeBron James, Stephen Curry")
with col2:
    year = st.selectbox("시즌", options=list(range(2026, 2019, -1)), index=0)

salary_df = None
try:
    salary_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'nba_salaries_2025.csv'))
except:
    pass

actual_salary = None
if player_name and salary_df is not None:
    match = salary_df[salary_df['player_name'].str.contains(
        player_name.split()[-1], case=False, na=False
    )]
    if not match.empty:
        actual_salary = int(match.iloc[0]['salary_2526'])
        st.info(f"HoopsHype 2025-26 연봉 자동 조회: **${actual_salary:,}**")

if actual_salary is None:
    actual_salary_input = st.number_input(
        "실제 연봉 (USD) — HoopsHype 데이터에 없는 경우 직접 입력",
        min_value=0, value=0, step=100000, format="%d"
    )
    if actual_salary_input > 0:
        actual_salary = actual_salary_input

analyze_btn = st.button("분석 시작", type="primary", use_container_width=True)

if analyze_btn and player_name:
    with st.spinner(f"{player_name} 스탯 조회 중..."):
        try:
            stats = get_player_stats(player_name, year=year)
        except Exception as e:
            st.error(f"선수 조회 실패: {e}")
            st.stop()

    with st.spinner("ML 모델 예측 중..."):
        result = predict_salary(stats, actual_salary=actual_salary if actual_salary else None)

    st.subheader(f"{stats['player_name']} — {stats['season']}")
    info_cols = st.columns(4)
    info_cols[0].metric("포지션", stats['pos'])
    info_cols[1].metric("나이", f"{stats['Age']}세")
    info_cols[2].metric("경력", f"{stats['experience']}년")
    info_cols[3].metric("출전", f"{stats['gp']}경기")

    st.divider()

    st.subheader("주요 스탯 (경기당)")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("PTS", stats['PTS'])
    s2.metric("TRB", stats['TRB'])
    s3.metric("AST", stats['AST'])
    s4.metric("STL", stats['STL'])
    s5.metric("BLK", stats['BLK'])
    s6.metric("TOV", stats['TOV'])

    s7, s8, s9 = st.columns(3)
    s7.metric("FG%", f"{stats['FG%']*100:.1f}%")
    s8.metric("3P%", f"{stats['3P%']*100:.1f}%")
    s9.metric("FT%", f"{stats['FT%']*100:.1f}%")

    st.divider()

    st.subheader("연봉 분석")
    c1, c2, c3 = st.columns(3)
    c1.metric("예측 적정 연봉", f"${result['predicted_salary']:,}")

    if actual_salary:
        delta = actual_salary - result['predicted_salary']
        delta_pct = (delta / result['predicted_salary']) * 100
        c2.metric("실제 연봉", f"${actual_salary:,}",
                  delta=f"{delta_pct:+.1f}%",
                  delta_color="inverse")
        c3.metric("Ratio", f"{result['ratio']:.3f}")

        st.divider()
        if result['verdict'] == 'Overpaid (먹튀)':  # ratio > 1.35
            st.markdown(f"""<div class="verdict-overpaid">
                <h2 style="color:#e74c3c;margin:0">🔴 먹튀 (Overpaid)</h2>
                <p style="color:#ccc;margin:6px 0 0">실제 연봉이 적정가의 {result['ratio']:.1%} 수준</p>
            </div>""", unsafe_allow_html=True)
        elif result['verdict'] == 'Underpaid (저평가)':
            st.markdown(f"""<div class="verdict-underpaid">
                <h2 style="color:#3498db;margin:0">🔵 저평가 (Underpaid)</h2>
                <p style="color:#ccc;margin:6px 0 0">실제 연봉이 적정가의 {result['ratio']:.1%} 수준</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="verdict-fair">
                <h2 style="color:#2ecc71;margin:0">🟢 적정 (Fair Value)</h2>
                <p style="color:#ccc;margin:6px 0 0">실제 연봉이 적정가의 {result['ratio']:.1%} 수준</p>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("AI 분석 리포트")
        with st.spinner("Claude AI 분석 중..."):
            prompt = (
                f"{stats['player_name']}의 {stats['season']}시즌을 분석해줘. "
                f"실제 연봉은 ${actual_salary:,}이고 ML 예측 적정 연봉은 ${result['predicted_salary']:,}이야. "
                f"판정은 {result['verdict']}이고 ratio는 {result['ratio']}야. "
                f"스탯: PTS {stats['PTS']}, TRB {stats['TRB']}, AST {stats['AST']}, "
                f"STL {stats['STL']}, BLK {stats['BLK']}, FG% {stats['FG%']:.3f}, "
                f"3P% {stats['3P%']:.3f}, 경력 {stats['experience']}년, 나이 {stats['Age']}세. "
                f"판정 근거를 구체적으로 설명해줘."
            )
            ai_report = run_agent(prompt)
        st.markdown(ai_report)
    else:
        c2.metric("실제 연봉", "미입력")
        c3.metric("Ratio", "—")
