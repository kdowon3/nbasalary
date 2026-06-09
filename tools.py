import os
import joblib
import numpy as np
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import (
    playerdashboardbyyearoveryear,
    playergamelog,
    commonplayerinfo,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YEAR_MAX_SALARY = {
    2000: 17142000, 2001: 19600000, 2002: 22400000, 2003: 25200000,
    2004: 28000000, 2005: 29464000, 2006: 20000000, 2007: 21000000,
    2008: 23751934, 2009: 24751934, 2010: 23034375, 2011: 24806250,
    2012: 25244493, 2013: 30453805, 2014: 30453805, 2015: 23500000,
    2016: 25000000, 2017: 30963450, 2018: 37457154, 2019: 37457154,
    2020: 40231758, 2021: 52411485, 2022: 52938707, 2023: 50968059,
    2024: 53458234, 2025: 55761216,
}

LEAGUE_AVG_PTS = {
    2000: 15.2, 2001: 15.0, 2002: 14.9, 2003: 14.8, 2004: 14.2,
    2005: 14.9, 2006: 15.4, 2007: 15.5, 2008: 15.6, 2009: 15.4,
    2010: 15.2, 2011: 15.0, 2012: 14.9, 2013: 15.1, 2014: 15.4,
    2015: 15.6, 2016: 16.1, 2017: 16.5, 2018: 16.8, 2019: 17.1,
    2020: 16.9, 2021: 17.2, 2022: 17.4, 2023: 17.8, 2024: 18.1,
    2025: 18.3,
}

POS_MAP = {
    'Guard': 'PG', 'Guard-Forward': 'SG', 'Forward-Guard': 'SF',
    'Forward': 'SF', 'Forward-Center': 'PF', 'Center-Forward': 'PF',
    'Center': 'C', 'F': 'SF', 'G': 'PG', 'C': 'C',
}
POS_ENCODE = {'C': 0, 'PF': 1, 'PG': 2, 'SF': 3, 'SG': 4}

FEATURES = [
    'value_composite', 'experience', 'GS_rate', 'Age',
    'PTS_vs_avg', 'PTS_growth', 'usage_proxy', 'MP',
    'PTS_per_MP', 'Year',
]


def _find_player_id(name: str) -> dict:
    results = players.find_players_by_full_name(name)
    if not results:
        all_players = players.get_players()
        results = [p for p in all_players if name.lower() in p['full_name'].lower()]
    if not results:
        raise ValueError(f"선수를 찾을 수 없습니다: {name}")
    active = [p for p in results if p['is_active']]
    return active[0] if active else results[0]


def _get_stats_from_gamelog(player_id: int, season_str: str) -> dict:
    gl = playergamelog.PlayerGameLog(player_id=player_id, season=season_str)
    df = gl.player_game_log.get_data_frame()
    if df.empty:
        raise ValueError(f"게임로그 데이터가 없습니다: {season_str}")

    gp = len(df)
    gs = df['START_POSITION'].notna().sum() if 'START_POSITION' in df.columns else gp

    totals = {
        'MIN': df['MIN'].sum(), 'PTS': df['PTS'].sum(), 'REB': df['REB'].sum(),
        'AST': df['AST'].sum(), 'STL': df['STL'].sum(), 'BLK': df['BLK'].sum(),
        'TOV': df['TOV'].sum(), 'OREB': df['OREB'].sum(), 'DREB': df['DREB'].sum(),
        'PF': df['PF'].sum(), 'FGM': df['FGM'].sum(), 'FGA': df['FGA'].sum(),
        'FG3M': df['FG3M'].sum(), 'FG3A': df['FG3A'].sum(),
        'FTM': df['FTM'].sum(), 'FTA': df['FTA'].sum(),
    }

    fga_total = totals['FGA']
    fta_total = totals['FTA']
    fg_pct  = totals['FGM'] / fga_total if fga_total > 0 else 0.0
    fg3_pct = totals['FG3M'] / totals['FG3A'] if totals['FG3A'] > 0 else 0.0
    ft_pct  = totals['FTM'] / fta_total if fta_total > 0 else 0.0
    efg_pct = (totals['FGM'] + 0.5 * totals['FG3M']) / fga_total if fga_total > 0 else fg_pct

    return {
        'gp': gp, 'gs': gs,
        'min_total': totals['MIN'],
        'pts': totals['PTS'] / gp, 'trb': totals['REB'] / gp,
        'ast': totals['AST'] / gp, 'stl': totals['STL'] / gp,
        'blk': totals['BLK'] / gp, 'tov': totals['TOV'] / gp,
        'orb': totals['OREB'] / gp, 'drb': totals['DREB'] / gp,
        'pf': totals['PF'] / gp,
        'fgm': totals['FGM'] / gp, 'fga': fga_total / gp,
        'fg3a': totals['FG3A'] / gp, 'fta': fta_total / gp,
        'fg3m': totals['FG3M'] / gp,
        'fg_pct': fg_pct, 'fg3_pct': fg3_pct, 'ft_pct': ft_pct, 'efg_pct': efg_pct,
    }


def get_player_stats(player_name: str, year: int = 2026) -> dict:
    player = _find_player_id(player_name)
    player_id = player['id']
    season_str = f"{year-1}-{str(year)[-2:]}"

    if season_str == '2025-26':
        s = _get_stats_from_gamelog(player_id, season_str)
        gp = s['gp'];  gs = s['gs']
        mp_per_game = s['min_total'] / gp
        pts = s['pts']; trb = s['trb']; ast = s['ast']
        stl = s['stl']; blk = s['blk']; tov = s['tov']
        orb = s['orb']; drb = s['drb']; pf  = s['pf']
        fgm = s['fgm']; fga = s['fga']; fg3a = s['fg3a']
        fta = s['fta']; fg3m = s['fg3m']
        fg_pct = s['fg_pct']; fg3_pct = s['fg3_pct']
        ft_pct = s['ft_pct']; efg_pct = s['efg_pct']

        dash_prev = playerdashboardbyyearoveryear.PlayerDashboardByYearOverYear(
            player_id=player_id, season='2024-25'
        )
        df_all_prev = dash_prev.by_year_player_dashboard.get_data_frame()
        experience = len(df_all_prev)

        prev_season_str = '2024-25'
        prev_row = df_all_prev[df_all_prev['GROUP_VALUE'] == prev_season_str]
        if not prev_row.empty and prev_row.iloc[0]['GP'] > 0:
            prev_gp = prev_row.iloc[0]['GP']
            prev_pts = prev_row.iloc[0]['PTS'] / prev_gp
            pts_growth = np.clip((pts - prev_pts) / (prev_pts + 0.1), -1, 2)
        else:
            pts_growth = 0.0
    else:
        season_str_key = season_str

        dash = playerdashboardbyyearoveryear.PlayerDashboardByYearOverYear(
            player_id=player_id, season=season_str_key
        )
        df_all = dash.by_year_player_dashboard.get_data_frame()

        row = df_all[df_all['GROUP_VALUE'] == season_str_key]
        if row.empty:
            raise ValueError(f"{player['full_name']}의 {season_str} 시즌 데이터가 없습니다.")
        row = row.iloc[0]

        gp       = row['GP']
        min_total = row['MIN']
        mp_per_game = min_total / gp

        pts = row['PTS'] / gp
        trb = row['REB'] / gp
        ast = row['AST'] / gp
        stl = row['STL'] / gp
        blk = row['BLK'] / gp
        tov = row['TOV'] / gp
        orb = row['OREB'] / gp
        drb = row['DREB'] / gp
        pf  = row['PF'] / gp
        gs  = row.get('GS', gp)

        fgm  = row['FGM'] / gp
        fga  = row['FGA'] / gp
        fg3a = row['FG3A'] / gp
        fta  = row['FTA'] / gp
        fg3m = row['FG3M'] / gp

        fg_pct  = row['FG_PCT']
        fg3_pct = float(row['FG3_PCT']) if row['FG3_PCT'] else 0.0
        ft_pct  = row['FT_PCT']
        efg_pct = (fgm + 0.5 * fg3m) / fga if fga > 0 else fg_pct

        experience = max(0, len(df_all) - 1)

        prev_season_str = f"{year-2}-{str(year-1)[-2:]}"
        prev_row = df_all[df_all['GROUP_VALUE'] == prev_season_str]
        if not prev_row.empty and prev_row.iloc[0]['GP'] > 0:
            prev_pts = prev_row.iloc[0]['PTS'] / prev_row.iloc[0]['GP']
            pts_growth = np.clip((pts - prev_pts) / (prev_pts + 0.1), -1, 2)
        else:
            pts_growth = 0.0

    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    info_df = info.common_player_info.get_data_frame()
    pos_raw     = info_df['POSITION'].iloc[0]
    pos         = POS_MAP.get(pos_raw, 'SF')
    pos_encoded = POS_ENCODE.get(pos, 3)
    age         = int(info_df['SEASON_EXP'].iloc[0]) + 19

    pts_per_mp      = pts / mp_per_game if mp_per_game > 0 else 0
    trb_per_mp      = trb / mp_per_game if mp_per_game > 0 else 0
    ast_per_mp      = ast / mp_per_game if mp_per_game > 0 else 0
    ast_tov         = ast / (tov + 0.1)
    stl_blk         = stl + blk
    value_composite = pts + trb + ast + stl + blk - tov

    gs_rate     = gs / gp
    fga3_rate   = fg3a / fga if fga > 0 else 0
    scoring_eff = pts / fga if fga > 0 else 0
    usage_proxy = fga + fta * 0.44 + tov
    prime_age   = 1 if 25 <= age <= 29 else 0

    league_avg = LEAGUE_AVG_PTS.get(year, 18.0)
    pts_vs_avg = pts / league_avg if league_avg > 0 else 1.0

    year_max = YEAR_MAX_SALARY.get(year, 55761216)

    stats = {
        'player_name': player['full_name'],
        'season': season_str,
        'year': year,
        'gp': int(gp),
        'mp_per_game': round(mp_per_game, 1),
        'pos': pos,
        'Age': age,
        'G': int(gp),
        'MP': round(mp_per_game, 1),
        'PTS_per_MP': round(pts_per_mp, 4),
        'TRB_per_MP': round(trb_per_mp, 4),
        'AST_per_MP': round(ast_per_mp, 4),
        'AST_TOV': round(ast_tov, 4),
        'STL_BLK': round(stl_blk, 4),
        'value_composite': round(value_composite, 4),
        'FG%': round(fg_pct, 4),
        '3P%': round(fg3_pct, 4),
        'FT%': round(ft_pct, 4),
        'eFG%': round(efg_pct, 4),
        'ORB': round(orb, 4),
        'DRB': round(drb, 4),
        'PF': round(pf, 4),
        'pos_encoded': pos_encoded,
        'Year': year,
        'GS_rate': round(gs_rate, 4),
        '3PA_rate': round(fga3_rate, 4),
        'scoring_eff': round(scoring_eff, 4),
        'usage_proxy': round(usage_proxy, 4),
        'prime_age': prime_age,
        'experience': experience,
        'PTS_vs_avg': round(pts_vs_avg, 4),
        'PTS_growth': round(float(pts_growth), 4),
        'PTS': round(pts, 1),
        'TRB': round(trb, 1),
        'AST': round(ast, 1),
        'STL': round(stl, 1),
        'BLK': round(blk, 1),
        'TOV': round(tov, 1),
        'year_max_salary': year_max,
    }
    return stats


def predict_salary(stats: dict, actual_salary: int = None) -> dict:
    model  = joblib.load(os.path.join(BASE_DIR, 'model', 'salary_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'model', 'scaler.pkl'))

    X = np.array([[stats[f] for f in FEATURES]])
    X_scaled = scaler.transform(X)
    salary_pct = float(model.predict(X_scaled)[0])

    year_max = stats.get('year_max_salary', 55761216)
    predicted_salary = int(salary_pct * year_max)

    result = {
        'player_name': stats.get('player_name'),
        'season': stats.get('season'),
        'predicted_salary': predicted_salary,
        'salary_pct': round(salary_pct, 4),
    }

    if actual_salary:
        ratio = actual_salary / predicted_salary
        if ratio > 1.35:
            verdict = 'Overpaid (먹튀)'
            color = 'red'
        elif ratio < 0.80:
            verdict = 'Underpaid (저평가)'
            color = 'blue'
        else:
            verdict = 'Fair Value (적정)'
            color = 'green'

        result.update({
            'actual_salary': actual_salary,
            'ratio': round(ratio, 3),
            'verdict': verdict,
            'color': color,
        })

    return result
