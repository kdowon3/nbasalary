import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

import streamlit as st
import pandas as pd
import numpy as np
from agent import run_agent
from tools import get_player_stats, predict_salary, YEAR_MAX_SALARY, LEAGUE_AVG_PTS, FEATURES

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
st.caption("선수 스탯 기반 연봉 가치 분석 시스템")
st.divider()

tab1, tab2 = st.tabs(["선수 검색", "스탯 직접 입력 (신인/가상 선수)"])

# ── Tab 1: 선수 검색 ──────────────────────────────────────────────
with tab1:
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
            min_value=0, value=0, step=100000, format="%d",
            key="tab1_salary"
        )
        if actual_salary_input > 0:
            actual_salary = actual_salary_input

    analyze_btn = st.button("분석 시작", type="primary", use_container_width=True, key="tab1_btn")

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
            if result['verdict'] == 'Overpaid (먹튀)':
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
                    f"판정 기준: ratio > 1.35 먹튀, 0.80~1.35 적정, < 0.80 저평가. 이 기준으로만 설명해줘. "
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

# ── Tab 2: 스탯 직접 입력 ─────────────────────────────────────────
with tab2:
    st.caption("드래프트 예정 신인이나 가상의 선수 스탯을 입력해 적정 연봉을 예측합니다.")

    st.subheader("기본 정보")
    bi1, bi2, bi3 = st.columns(3)
    with bi1:
        t2_age = st.number_input("나이 (Age)", min_value=18, max_value=45, value=22)
    with bi2:
        t2_experience = st.number_input("경력 (experience) — NBA 활동 연수, 신인이면 0", min_value=0, max_value=25, value=0)
    with bi3:
        t2_year = st.selectbox("시즌 연도", options=list(range(2026, 2019, -1)), index=0, key="tab2_year")

    st.subheader("경기 출전")
    pg1, pg2, pg3 = st.columns(3)
    with pg1:
        t2_g = st.number_input("출전 경기 수 (G)", min_value=1, max_value=82, value=60)
    with pg2:
        t2_gs = st.number_input("선발 출전 경기 수 (GS) — 선발로 나온 경기 수", min_value=0, max_value=82, value=40)
    with pg3:
        t2_mp = st.number_input("경기당 출전 시간 (MP) — 분 단위", min_value=1.0, max_value=48.0, value=28.0, step=0.1)

    st.subheader("주요 스탯 (경기당)")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        t2_pts = st.number_input("득점 (PTS)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
    with sc2:
        t2_trb = st.number_input("리바운드 (TRB)", min_value=0.0, max_value=25.0, value=5.0, step=0.1)
    with sc3:
        t2_ast = st.number_input("어시스트 (AST)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)

    sc4, sc5, sc6 = st.columns(3)
    with sc4:
        t2_stl = st.number_input("스틸 (STL)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    with sc5:
        t2_blk = st.number_input("블록 (BLK)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    with sc6:
        t2_tov = st.number_input("턴오버 (TOV) — 공 빼앗긴 횟수", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

    st.subheader("슈팅")
    sh1, sh2 = st.columns(2)
    with sh1:
        t2_fga = st.number_input("경기당 야투 시도 (FGA)", min_value=0.1, max_value=35.0, value=12.0, step=0.1)
    with sh2:
        t2_fta = st.number_input("경기당 자유투 시도 (FTA)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)

    predict_btn = st.button("적정 연봉 예측", type="primary", use_container_width=True, key="tab2_btn")

    if predict_btn:
        year_max = YEAR_MAX_SALARY.get(t2_year, 55761216)
        league_avg = LEAGUE_AVG_PTS.get(t2_year, 18.0)

        value_composite = t2_pts + t2_trb + t2_ast + t2_stl + t2_blk - t2_tov
        gs_rate = t2_gs / t2_g
        pts_per_mp = t2_pts / t2_mp if t2_mp > 0 else 0
        pts_vs_avg = t2_pts / league_avg if league_avg > 0 else 1.0
        usage_proxy = t2_fga + t2_fta * 0.44 + t2_tov

        stats_manual = {
            'player_name': '가상 선수',
            'season': f"{t2_year-1}-{str(t2_year)[-2:]}",
            'year': t2_year,
            'value_composite': round(value_composite, 4),
            'experience': t2_experience,
            'GS_rate': round(gs_rate, 4),
            'Age': t2_age,
            'PTS_vs_avg': round(pts_vs_avg, 4),
            'PTS_growth': 0.0,
            'usage_proxy': round(usage_proxy, 4),
            'MP': round(t2_mp, 1),
            'PTS_per_MP': round(pts_per_mp, 4),
            'Year': t2_year,
            'year_max_salary': year_max,
            'PTS': round(t2_pts, 1),
            'TRB': round(t2_trb, 1),
            'AST': round(t2_ast, 1),
            'STL': round(t2_stl, 1),
            'BLK': round(t2_blk, 1),
            'TOV': round(t2_tov, 1),
        }

        result2 = predict_salary(stats_manual)

        st.divider()
        st.subheader("예측 결과")
        r1, r2, r3 = st.columns(3)
        r1.metric("예측 적정 연봉", f"${result2['predicted_salary']:,}")
        r2.metric("연봉 비율 (salary_pct)", f"{result2['salary_pct']*100:.1f}%", help="최고 연봉 대비 비율")
        r3.metric("시즌 최고 연봉 기준", f"${year_max:,}")

        st.info(
            f"입력 스탯 기준으로 {t2_year-1}-{str(t2_year)[-2:]} 시즌 적정 연봉은 "
            f"**${result2['predicted_salary']:,}** 입니다. "
            f"(리그 최고 연봉 ${year_max:,}의 {result2['salary_pct']*100:.1f}% 수준)"
        )
