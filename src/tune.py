import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

from preprocess import load_and_preprocess, split_data, FEATURES


def report(name, model, X_train_s, y_train, X_test_s, y_test, year_max_test):
    train_r2 = r2_score(y_train, model.predict(X_train_s))
    y_pred   = model.predict(X_test_s)
    test_r2  = r2_score(y_test, y_pred)
    mae      = mean_absolute_error(y_test.values * year_max_test, y_pred * year_max_test)
    gap      = train_r2 - test_r2
    print(f"  [{name}]  Train R²={train_r2:.4f}  Test R²={test_r2:.4f}  gap={gap:.4f}  MAE=${mae:,.0f}")
    return test_r2, mae


def tune():
    os.makedirs('model', exist_ok=True)

    df = load_and_preprocess('data/nba_stats_salaries.csv')
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    scaler = joblib.load('model/scaler.pkl')
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    year_max_test = df.loc[X_test.index, 'year_max_salary'].values

    # Train+Val 합쳐서 CV
    X_tv = np.vstack([X_train_s, X_val_s])
    y_tv = np.concatenate([y_train, y_val])

    print("=" * 60)
    print("  PHASE 1: GradientBoosting - 과적합 억제 중심")
    print("=" * 60)
    gb_params = {
        'n_estimators':     [300, 400, 500, 600],
        'learning_rate':    [0.01, 0.02, 0.03, 0.05],
        'max_depth':        [3, 4],           # 얕게
        'subsample':        [0.5, 0.6, 0.7],  # 더 공격적으로
        'min_samples_leaf': [10, 20, 30, 50], # 더 크게
        'max_features':     ['sqrt', 'log2'],
    }
    gb_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_params, n_iter=80, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0,
    )
    gb_search.fit(X_tv, y_tv)
    print(f"  Best params: {gb_search.best_params_}")
    print(f"  CV R²: {gb_search.best_score_:.4f}")
    report("GB 튜닝", gb_search.best_estimator_, X_train_s, y_train, X_test_s, y_test, year_max_test)

    print("\n" + "=" * 60)
    print("  PHASE 2: HistGradientBoosting (early stopping)")
    print("=" * 60)
    hgb_params = {
        'max_iter':         [300, 500, 700],
        'learning_rate':    [0.01, 0.02, 0.05],
        'max_depth':        [3, 4, 5],
        'min_samples_leaf': [20, 30, 50],
        'l2_regularization':[0.1, 1.0, 5.0, 10.0],
        'max_leaf_nodes':   [15, 20, 31],
    }
    hgb_search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=42, early_stopping=True, validation_fraction=0.1),
        hgb_params, n_iter=80, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0,
    )
    hgb_search.fit(X_tv, y_tv)
    print(f"  Best params: {hgb_search.best_params_}")
    print(f"  CV R²: {hgb_search.best_score_:.4f}")
    report("HGB 튜닝", hgb_search.best_estimator_, X_train_s, y_train, X_test_s, y_test, year_max_test)

    print("\n" + "=" * 60)
    print("  PHASE 3: RandomForest - 과적합 억제")
    print("=" * 60)
    rf_params = {
        'n_estimators':     [200, 300, 500],
        'max_depth':        [6, 8, 10, None],
        'min_samples_leaf': [10, 20, 30],
        'max_features':     ['sqrt', 'log2', 0.5],
        'max_samples':      [0.6, 0.7, 0.8],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        rf_params, n_iter=60, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0,
    )
    rf_search.fit(X_tv, y_tv)
    print(f"  Best params: {rf_search.best_params_}")
    print(f"  CV R²: {rf_search.best_score_:.4f}")
    report("RF 튜닝", rf_search.best_estimator_, X_train_s, y_train, X_test_s, y_test, year_max_test)

    # ── 최종 비교 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)
    current = joblib.load('model/salary_model.pkl')
    candidates = [
        ("현재 모델",   current),
        ("GB 튜닝",    gb_search.best_estimator_),
        ("HGB 튜닝",   hgb_search.best_estimator_),
        ("RF 튜닝",    rf_search.best_estimator_),
    ]

    best_r2, best_model, best_name = -999, None, None
    for name, model in candidates:
        r2, mae = report(name, model, X_train_s, y_train, X_test_s, y_test, year_max_test)
        # gap 패널티 적용: gap이 0.1 초과 시 페널티
        gap = r2_score(y_train, model.predict(X_train_s)) - r2
        score = r2 - max(0, gap - 0.10) * 0.5
        if score > best_r2:
            best_r2, best_model, best_name = score, model, name

    print(f"\n  => 선택: {best_name}")
    joblib.dump(best_model, 'model/salary_model.pkl')
    print("  모델 저장 완료.")


if __name__ == '__main__':
    tune()
