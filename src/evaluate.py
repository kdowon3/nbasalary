import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import load_and_preprocess, split_data, FEATURES, YEAR_MAX_SALARY

def evaluate():
    df = load_and_preprocess('data/nba_stats_salaries.csv')
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    model = joblib.load('model/salary_model.pkl')
    scaler = joblib.load('model/scaler.pkl')

    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    splits = {
        'Train (80%)': (X_train_s, y_train, df.loc[X_train.index]),
        'Val   (10%)': (X_val_s,   y_val,   df.loc[X_val.index]),
        'Test  (10%)': (X_test_s,  y_test,  df.loc[X_test.index]),
    }

    print("=" * 60)
    print("  MODEL PERFORMANCE REPORT")
    print("=" * 60)

    all_results = {}
    for name, (X_s, y_true, subset) in splits.items():
        y_pred_pct = model.predict(X_s)
        year_max   = subset['year_max_salary'].values

        # salary_pct 기준
        mae_pct  = mean_absolute_error(y_true, y_pred_pct)
        rmse_pct = np.sqrt(mean_squared_error(y_true, y_pred_pct))
        r2       = r2_score(y_true, y_pred_pct)

        # 달러 기준
        y_true_usd = y_true.values * year_max
        y_pred_usd = y_pred_pct    * year_max
        mae_usd    = mean_absolute_error(y_true_usd, y_pred_usd)
        rmse_usd   = np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))

        # MAPE
        mape = np.mean(np.abs((y_true_usd - y_pred_usd) / (y_true_usd + 1))) * 100

        print(f"\n[{name}]  n={len(y_true)}")
        print(f"  R²        : {r2:.4f}")
        print(f"  MAE       : ${mae_usd:>12,.0f}  ({mae_pct:.4f} pct)")
        print(f"  RMSE      : ${rmse_usd:>12,.0f}  ({rmse_pct:.4f} pct)")
        print(f"  MAPE      : {mape:.2f}%")

        all_results[name] = {
            'y_true_usd': y_true_usd,
            'y_pred_usd': y_pred_usd,
            'y_pred_pct': y_pred_pct,
            'y_true_pct': y_true.values,
            'r2': r2, 'mae_usd': mae_usd,
        }

    # ── Verdict 분포 (Test set) ──────────────────────────────
    test_true = all_results['Test  (10%)']['y_true_usd']
    test_pred = all_results['Test  (10%)']['y_pred_usd']
    ratios = test_true / test_pred

    overpaid   = (ratios > 1.30).sum()
    fair       = ((ratios >= 0.80) & (ratios <= 1.30)).sum()
    underpaid  = (ratios < 0.80).sum()
    total      = len(ratios)

    print("\n" + "=" * 60)
    print("  VERDICT DISTRIBUTION  (Test 2023-2025)")
    print("=" * 60)
    print(f"  Overpaid  (>1.30) : {overpaid:>4}  ({overpaid/total*100:.1f}%)")
    print(f"  Fair Value        : {fair:>4}  ({fair/total*100:.1f}%)")
    print(f"  Underpaid (<0.80) : {underpaid:>4}  ({underpaid/total*100:.1f}%)")

    # ── Feature Importance ───────────────────────────────────
    if hasattr(model, 'feature_importances_'):
        print("\n" + "=" * 60)
        print("  FEATURE IMPORTANCE (Top 10)")
        print("=" * 60)
        imps = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
        for feat, imp in imps[:10]:
            bar = '#' * int(imp * 100)
            print(f"  {feat:<20} {imp:.4f}  {bar}")

    # ── 실제 vs 예측 산점도 ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Actual vs Predicted Salary (USD)', fontsize=14)

    for ax, (name, res) in zip(axes, all_results.items()):
        true_m = res['y_true_usd'] / 1e6
        pred_m = res['y_pred_usd'] / 1e6
        ax.scatter(true_m, pred_m, alpha=0.3, s=10)
        lim = max(true_m.max(), pred_m.max()) * 1.05
        ax.plot([0, lim], [0, lim], 'r--', linewidth=1)
        ax.set_xlabel('Actual ($M)')
        ax.set_ylabel('Predicted ($M)')
        ax.set_title(f"{name}\nR²={res['r2']:.3f}  MAE=${res['mae_usd']/1e6:.1f}M")

    plt.tight_layout()
    plt.savefig('model/performance_plot.png', dpi=120)
    print("\n  Plot saved: model/performance_plot.png")
    print("=" * 60)


if __name__ == '__main__':
    evaluate()
