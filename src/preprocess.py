import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

YEAR_MAX_SALARY = {
    2000: 17142000, 2001: 19600000, 2002: 22400000, 2003: 25200000,
    2004: 28000000, 2005: 29464000, 2006: 20000000, 2007: 21000000,
    2008: 23751934, 2009: 24751934, 2010: 23034375, 2011: 24806250,
    2012: 25244493, 2013: 30453805, 2014: 30453805, 2015: 23500000,
    2016: 25000000, 2017: 30963450, 2018: 37457154, 2019: 37457154,
    2020: 40231758, 2021: 52411485, 2022: 52938707, 2023: 50968059,
    2024: 53458234, 2025: 55761216,
}

FEATURES = [
    # 기본 스탯
    'Age', 'G', 'MP',
    # per-minute 스탯
    'PTS_per_MP', 'TRB_per_MP', 'AST_per_MP',
    # 효율 지표
    'AST_TOV', 'STL_BLK', 'value_composite',
    'FG%', '3P%', 'FT%', 'eFG%',
    'ORB', 'DRB', 'PF',
    # 포지션/연도
    'pos_encoded', 'Year',
    # 신규 피처
    'GS_rate',       # 선발 출장 비율 (중요도 지표)
    '3PA_rate',      # 3점 시도 비율 (플레이 스타일)
    'scoring_eff',   # 득점 효율 (PTS / FGA)
    'usage_proxy',   # 볼 점유율 추정 (FGA + FTA*0.44 + TOV)
    'prime_age',     # 전성기 여부 (25-29세)
    'experience',    # 리그 경력 연수
    'PTS_vs_avg',    # 리그 평균 대비 득점 비율
    'PTS_growth',    # 전년 대비 득점 성장률
]

TARGET = 'salary_pct'


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 필터링
    df = df[df['G'] >= 20].copy()
    df = df[df['MP'] >= 15].copy()

    # 포지션 단순화
    df['Pos'] = df['Pos'].str.split('-').str[0]

    # 결측치 처리
    df['3P%'] = df['3P%'].fillna(0)
    for col in ['FG%', '2P%', 'eFG%', 'FT%']:
        df[col] = df.groupby('Pos')[col].transform(lambda x: x.fillna(x.median()))

    # 연봉 정규화
    df['year_max_salary'] = df['Year'].map(YEAR_MAX_SALARY)
    df[TARGET] = df['Salary'] / df['year_max_salary']

    # 기존 파생 피처
    df['PTS_per_MP']      = df['PTS'] / df['MP']
    df['TRB_per_MP']      = df['TRB'] / df['MP']
    df['AST_per_MP']      = df['AST'] / df['MP']
    df['AST_TOV']         = df['AST'] / (df['TOV'] + 0.1)
    df['STL_BLK']         = df['STL'] + df['BLK']
    df['value_composite'] = df['PTS'] + df['TRB'] + df['AST'] + df['STL'] + df['BLK'] - df['TOV']

    # 신규 파생 피처
    df['GS_rate']     = df['GS'] / df['G']
    df['3PA_rate']    = df['3PA'] / df['FGA'].replace(0, np.nan).fillna(0)
    df['scoring_eff'] = df['PTS'] / df['FGA'].replace(0, np.nan).fillna(df['PTS'])
    df['usage_proxy'] = df['FGA'] + df['FTA'] * 0.44 + df['TOV']
    df['prime_age']   = df['Age'].between(25, 29).astype(int)

    # 경력 연수 (해당 선수의 첫 시즌 기준)
    df['experience'] = df['Year'] - df.groupby('Player')['Year'].transform('min')

    # 리그 평균 대비 득점
    league_avg_pts = df.groupby('Year')['PTS'].transform('mean')
    df['PTS_vs_avg'] = df['PTS'] / league_avg_pts

    # 전년 대비 득점 성장률 (누수 없음 — 같은 선수 전년도 데이터)
    df_sorted = df.sort_values(['Player', 'Year'])
    df['PTS_growth'] = (
        df_sorted.groupby('Player')['PTS']
        .pct_change()
        .fillna(0)
        .clip(-1, 2)
    )
    # sort 원복
    df = df.sort_index()

    # 포지션 인코딩
    le = LabelEncoder()
    df['pos_encoded'] = le.fit_transform(df['Pos'])

    return df


def split_data(df: pd.DataFrame, random_state: int = 42):
    from sklearn.model_selection import train_test_split

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    df = load_and_preprocess('data/nba_stats_salaries.csv')
    print(f'전처리 완료: {df.shape}, 피처 수: {len(FEATURES)}')
    print(df[FEATURES + [TARGET]].head())
