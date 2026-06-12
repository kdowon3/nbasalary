import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

from preprocess import load_and_preprocess, split_data, FEATURES, TARGET

POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C']

# 포지션별 핵심 스탯 정의
POS_KEY_STATS = {
    'PG': ['AST_per_MP', 'AST_TOV', '3PA_rate', '3P%', 'usage_proxy'],
    'SG': ['PTS_per_MP', '3PA_rate', '3P%', 'scoring_eff', 'usage_proxy'],
    'SF': ['value_composite', 'PTS_per_MP', 'TRB_per_MP', 'scoring_eff'],
    'PF': ['TRB_per_MP', 'value_composite', 'STL_BLK', 'ORB', 'DRB'],
    'C':  ['TRB_per_MP', 'STL_BLK', 'ORB', 'DRB', 'FG%'],
}

HGB_PARAMS = {
    'max_iter':          [300, 500, 700],
    'learning_rate':     [0.01, 0.02, 0.05],
    'max_depth':         [3, 4, 5],
    'min_samples_leaf':  [20, 30, 50],
    'l2_regularization': [0.1, 1.0, 5.0, 10.0],
    'max_leaf_nodes':    [15, 20, 31],
}


def report(name, model, X_tr, y_tr, X_te, y_te, ym_te):
    tr_r2 = r2_score(y_tr, model.predict(X_tr))
    pred  = model.predict(X_te)
    te_r2 = r2_score(y_te, pred)
    mae   = mean_absolute_error(y_te.values * ym_te, pred * ym_te)
    gap   = tr_r2 - te_r2
    print(f"  [{name:20s}]  Train={tr_r2:.4f}  Test={te_r2:.4f}  gap={gap:.4f}  MAE=${mae:,.0f}")
    return te_r2, mae


# ── Method A: 포지션별 개별 모델 ────────────────────────────
def method_a(df, X_train, y_train, X_val, y_val, X_test, y_test):
    print("=" * 65)
    print("  METHOD A: 포지션별 개별 모델")
    print("=" * 65)

    pos_train = df.loc[X_train.index, 'Pos']
    pos_val   = df.loc[X_val.index,   'Pos']
    pos_test  = df.loc[X_test.index,  'Pos']
    ym_test   = df.loc[X_test.index,  'year_max_salary'].values

    models_a, scalers_a = {}, {}
    all_preds = pd.Series(index=X_test.index, dtype=float)
    all_true  = y_test.copy()

    for pos in POSITIONS:
        tr_mask = pos_train == pos
        va_mask = pos_val   == pos
        te_mask = pos_test  == pos

        X_tr = X_train[tr_mask]; y_tr = y_train[tr_mask]
        X_va = X_val[va_mask];   y_va = y_val[va_mask]
        X_te = X_test[te_mask];  y_te = y_test[te_mask]
        ym   = df.loc[X_te.index, 'year_max_salary'].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        X_tv = np.vstack([X_tr_s, X_va_s])
        y_tv = np.concatenate([y_tr, y_va])

        search = RandomizedSearchCV(
            HistGradientBoostingRegressor(random_state=42, early_stopping=True, validation_fraction=0.1),
            HGB_PARAMS, n_iter=40, cv=5, scoring='r2',
            n_jobs=-1, random_state=42, verbose=0,
        )
        search.fit(X_tv, y_tv)
        best = search.best_estimator_

        report(f"A-{pos}", best, X_tr_s, y_tr, X_te_s, y_te, ym)
        all_preds.loc[X_te.index] = best.predict(X_te_s)
        models_a[pos]  = best
        scalers_a[pos] = scaler

    # 전체 통합 성능
    ym_all = df.loc[X_test.index, 'year_max_salary'].values
    overall_r2  = r2_score(all_true, all_preds)
    overall_mae = mean_absolute_error(all_true.values * ym_all, all_preds.values * ym_all)
    print(f"\n  [A 전체 통합]  Test R²={overall_r2:.4f}  MAE=${overall_mae:,.0f}")
    return models_a, scalers_a, overall_r2, overall_mae


# ── Method B: Interaction 피처 추가 단일 모델 ───────────────
def method_b(df, X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 65)
    print("  METHOD B: Interaction 피처 추가 단일 모델")
    print("=" * 65)

    def add_interaction(X, pos_series):
        X = X.copy()
        pos_enc = pd.get_dummies(pos_series, prefix='pos')
        for pos in POSITIONS:
            col = f'pos_{pos}'
            if col not in pos_enc.columns:
                continue
            for stat in POS_KEY_STATS[pos]:
                if stat in X.columns:
                    X[f'{pos}_{stat}'] = X[stat] * pos_enc[col].values
        return X

    pos_train = df.loc[X_train.index, 'Pos']
    pos_val   = df.loc[X_val.index,   'Pos']
    pos_test  = df.loc[X_test.index,  'Pos']
    ym_test   = df.loc[X_test.index,  'year_max_salary'].values

    X_tr_int = add_interaction(X_train, pos_train)
    X_va_int = add_interaction(X_val,   pos_val)
    X_te_int = add_interaction(X_test,  pos_test)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_int)
    X_va_s = scaler.transform(X_va_int)
    X_te_s = scaler.transform(X_te_int)

    X_tv = np.vstack([X_tr_s, X_va_s])
    y_tv = np.concatenate([y_train, y_val])

    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=42, early_stopping=True, validation_fraction=0.1),
        HGB_PARAMS, n_iter=60, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0,
    )
    search.fit(X_tv, y_tv)
    best = search.best_estimator_

    print(f"  Best params: {search.best_params_}")
    r2, mae = report("B-단일(interaction)", best, X_tr_s, y_train, X_te_s, y_test, ym_test)
    return best, scaler, X_te_int.columns.tolist(), r2, mae


# ── 현재 모델 베이스라인 ─────────────────────────────────────
def baseline(df, X_train, y_train, X_test, y_test):
    model  = joblib.load('model/salary_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)
    ym_te  = df.loc[X_test.index, 'year_max_salary'].values
    r2, mae = report("현재 모델(HGB)", model, X_tr_s, y_train, X_te_s, y_test, ym_te)
    return r2, mae


def main():
    df = load_and_preprocess('data/nba_stats_salaries.csv')
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    print("\n" + "=" * 65)
    print("  BASELINE")
    print("=" * 65)
    base_r2, base_mae = baseline(df, X_train, y_train, X_test, y_test)

    models_a, scalers_a, a_r2, a_mae = method_a(df, X_train, y_train, X_val, y_val, X_test, y_test)
    model_b, scaler_b, feat_b, b_r2, b_mae = method_b(df, X_train, y_train, X_val, y_val, X_test, y_test)

    # 최종 비교
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  현재 모델 (baseline) : R²={base_r2:.4f}  MAE=${base_mae:,.0f}")
    print(f"  Method A (개별 모델) : R²={a_r2:.4f}  MAE=${a_mae:,.0f}")
    print(f"  Method B (interaction): R²={b_r2:.4f}  MAE=${b_mae:,.0f}")

    results = {
        'baseline': (base_r2, base_mae, None, None),
        'method_a': (a_r2,    a_mae,    models_a, scalers_a),
        'method_b': (b_r2,    b_mae,    model_b,  scaler_b),
    }
    best = max(['baseline','method_a','method_b'], key=lambda k: results[k][0])
    print(f"\n  => 최고 성능: {best}")

    if best == 'method_a':
        os.makedirs('model', exist_ok=True)
        for pos in POSITIONS:
            joblib.dump(models_a[pos],  f'model/model_{pos}.pkl')
            joblib.dump(scalers_a[pos], f'model/scaler_{pos}.pkl')
        joblib.dump('method_a', 'model/model_type.pkl')
        print("  포지션별 모델 저장 완료.")
    elif best == 'method_b':
        joblib.dump(model_b,  'model/salary_model.pkl')
        joblib.dump(scaler_b, 'model/scaler.pkl')
        joblib.dump(feat_b,   'model/features.pkl')
        joblib.dump('method_b', 'model/model_type.pkl')
        print("  Interaction 모델 저장 완료.")
    else:
        print("  기존 모델 유지.")


if __name__ == '__main__':
    main()
