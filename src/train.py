import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from preprocess import load_and_preprocess, split_data, FEATURES, YEAR_MAX_SALARY


def evaluate(name, model, X, y, year_max_salaries):
    preds = model.predict(X)
    avg_max = year_max_salaries.values.mean()

    mae_pct  = mean_absolute_error(y, preds)
    mse_pct  = mean_squared_error(y, preds)
    rmse_pct = mse_pct ** 0.5
    r2       = r2_score(y, preds)

    mae_dollar  = mae_pct  * avg_max
    mse_dollar  = mse_pct  * (avg_max ** 2)
    rmse_dollar = rmse_pct * avg_max

    print(f'[{name}]')
    print(f'  MAE  : {mae_pct:.4f} ({mae_dollar:>12,.0f} USD)')
    print(f'  MSE  : {mse_pct:.6f} ({mse_dollar:>12,.0f} USD²)')
    print(f'  RMSE : {rmse_pct:.4f} ({rmse_dollar:>12,.0f} USD)')
    print(f'  R²   : {r2:.4f}')
    return r2


def train():
    os.makedirs('model', exist_ok=True)

    df = load_and_preprocess('data/nba_stats_salaries.csv')
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    val_ym  = df.loc[X_val.index,  'year_max_salary']
    test_ym = df.loc[X_test.index, 'year_max_salary']
    train_ym = df.loc[X_train.index, 'year_max_salary']

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # GridSearchCV로 RandomForest 최적 하이퍼파라미터 탐색
    print('=== GridSearchCV (RandomForest) ===')
    param_grid = {
        'n_estimators':     [200, 300, 500],
        'max_depth':        [8, 10, 12, None],
        'min_samples_leaf': [5, 10, 15],
        'max_features':     ['sqrt', 0.5],
        'max_samples':      [0.7, 0.8, 1.0],
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train_s, y_train)
    print(f'Best params : {gs.best_params_}')
    print(f'Best CV R2  : {gs.best_score_:.4f}')
    tuned_rf = gs.best_estimator_

    # 비교 모델 (모두 수업 범위 내)
    models = {
        'Ridge':              Ridge(alpha=1.0),
        'SGDRegressor':       SGDRegressor(max_iter=1000, random_state=42),
        'RandomForest(base)': RandomForestRegressor(
                                  n_estimators=300, max_depth=10,
                                  min_samples_leaf=10, max_features='sqrt',
                                  max_samples=0.7, random_state=42),
        'RandomForest(tuned)': tuned_rf,
    }

    print('\n=== Validation ===')
    best_r2, best_model, best_name = -999, None, None
    for name, model in models.items():
        if name != 'RandomForest(tuned)':
            model.fit(X_train_s, y_train)
        r2 = evaluate(name, model, X_val_s, y_val, val_ym)
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, model, name

    print(f'\n최적 모델: {best_name}')
    print('\n=== Test ===')
    evaluate(best_name, best_model, X_test_s, y_test, test_ym)

    train_r2 = r2_score(y_train, best_model.predict(X_train_s))
    test_r2  = r2_score(y_test,  best_model.predict(X_test_s))
    print(f'Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f} | Gap: {train_r2-test_r2:.4f}')

    if hasattr(best_model, 'feature_importances_'):
        importances = sorted(
            zip(FEATURES, best_model.feature_importances_),
            key=lambda x: -x[1]
        )
        print('\n=== Feature Importance (Top 10) ===')
        for feat, imp in importances[:10]:
            print(f'  {feat}: {imp:.4f}')

    joblib.dump(best_model, 'model/salary_model.pkl')
    joblib.dump(scaler,     'model/scaler.pkl')
    joblib.dump(FEATURES,   'model/features.pkl')
    print('\n모델 저장 완료: model/')


if __name__ == '__main__':
    train()
