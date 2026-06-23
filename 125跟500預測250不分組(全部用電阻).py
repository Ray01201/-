import numpy as np
import pandas as pd
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

# =========================
# USER SETTINGS
# =========================
EXCEL_PATH = r"C:\專題\raw data_正125,500.csv"
TEST_EXCEL_PATH = r"C:\專題\raw data正250.csv"
TARGET_COL = "Output 1"

CATEGORICAL_COLS = ["特徵值1", "特徵值2", "特徵值3", "特徵值4", "特徵值7", "特徵值8", "特徵值24"]
NUMERIC_COLS = [
    "特徵值5", "特徵值6", "特徵值9", "特徵值10", "特徵值11", "特徵值12",
    "特徵值13", "特徵值14", "特徵值15", "特徵值16", "特徵值17", "特徵值18",
    "特徵值19", "特徵值20", "特徵值21", "特徵值22", "特徵值23"
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

RANDOM_STATE = 36
N_SPLITS = 5  # H2O Grid 內部的 Cross-Validation 折數

def to_ratio_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    has_pct = s.str.contains("%", na=False)
    s_no_pct = s.str.replace("%", "", regex=False).str.strip()
    num = pd.to_numeric(s_no_pct, errors="coerce")
    num.loc[has_pct] = num.loc[has_pct] / 100.0
    return num

def main():
    # 1) 初始化 H2O 集群 (使用所有可用 CPU 核心)
    print("🚀 正在啟動 H2O 集群...")
    h2o.init(nthreads=-1, max_mem_size="8G") # 可依記憶體大小調整 max_mem_size
    h2o.no_progress()

    # 2) 讀取訓練資料
    df = pd.read_csv(EXCEL_PATH, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()

    # 3) 清理：% 符號轉換
    for col in FEATURE_COLS:
        if col in df.columns and df[col].astype(str).str.contains("%", na=False).any():
            df[col] = to_ratio_series(df[col])

    # 4) 數值欄強制轉數字
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) Y 資料清洗與偵錯
    y_raw_series = df[TARGET_COL].astype(str)
    y_str = y_raw_series.str.strip()
    y_str = y_str.str.replace(r"\((.*?)\)", r"-\1", regex=True)
    y_str = y_str.str.replace(",", "", regex=False)
    y_str = y_str.str.replace(r"(?i)m?A(mps)?", "", regex=True)
    y_str = y_str.str.replace(r"[^\d\.\-]", "", regex=True)

    y = pd.to_numeric(y_str, errors="coerce")
    y = y.abs()

    # 💡 核心安全檢查：特徵值9(電壓)必須大於0
    df["特徵值9"] = pd.to_numeric(df["特徵值9"], errors="coerce")
    keep = (~y.isna()) & (~df["特徵值9"].isna()) & (df["特徵值9"] > 0)
    df = df.loc[keep].copy()
    y = y.loc[keep].copy()
    
    print(f"清理完成！最終有效用於訓練的樣本數: {len(df)} 筆\n")
    if len(df) == 0:
        print("❌ 錯誤：沒有任何有效數據可用於訓練！")
        return

    # 6) 計算目標變數：等效電導 (電流 / 電壓)
    # H2O 內部會直接操作 H2OFrame，這裡先在 pandas 計算好
    df["Target_Conductance"] = y / df["特徵值9"]

    # 7) 缺值處理 (維持你原本的邏輯，但只針對 FEATURE_COLS)
    numeric_medians = df[NUMERIC_COLS].median()
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(numeric_medians)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("<NA>")

    # 8) 將 Pandas DataFrame 轉換為 H2OFrame
    # 💡 注意：我們不進行 One-hot encoding，讓 H2O 自動處理類別變數
    train_hf = h2o.H2OFrame(df[FEATURE_COLS + ["Target_Conductance"]])
    
    # 強制將 H2OFrame 中的類別欄位設定為 enum (Factor) 型態
    for col in CATEGORICAL_COLS:
        train_hf[col] = train_hf[col].asfactor()

    print("=== H2O LightGBM 參數調優配置 ===")
    print(f"驗證模式: {N_SPLITS}-Fold 交叉驗證")
    print(f"優化目標: 最小化 MAPE\n")

    # 9) 定義 LightGBM (H2O 中使用 tree_method="hist" 的 GBM 來模擬 LightGBM 效能)
    # 並且我們直接將 stopping_metric 設為 MAPE
    lgbm_backend = H2OGradientBoostingEstimator(
        tree_method="hist",          # 使用直方圖演算法 (LightGBM 核心技術)
        score_each_iteration=True,
        nfolds=N_SPLITS,             # 5-fold CV
        seed=RANDOM_STATE,
        stopping_metric="mape",      # 💡 根據 MAPE 決定何時停止
        stopping_rounds=5,
        stopping_tolerance=0.001
    )

    # 10) 設定超參數搜尋網格 (Grid Search)
    hyper_params = {
        'ntrees': [100, 200, 300, 500],
        'max_depth': [4, 6, 8, 10],
        'learn_rate': [0.01, 0.05, 0.1, 0.15],
        'sample_rate': [0.7, 0.8, 0.9, 1.0],          # 類似 LightGBM 的 bagging_fraction
        'col_sample_rate': [0.7, 0.8, 0.9, 1.0],      # 類似 LightGBM 的 feature_fraction
        'min_rows': [5, 10, 20]                       # 類似 LightGBM 的 min_data_in_leaf
    }

    # 隨向搜尋策略：最多跑 30 組模型，或者總時間不超過 600 秒
    search_criteria = {
        'strategy': 'RandomDiscrete',
        'max_models': 30,
        'max_runtime_secs': 600,
        'seed': RANDOM_STATE
    }

    print("-> 開始執行 H2O Grid Search 尋找最佳 LightGBM 參數...")
    grid = H2OGridSearch(
        model=lgbm_backend,
        hyper_params=hyper_params,
        grid_id="lgbm_grid_mape",
        search_criteria=search_criteria
    )
    
    grid.train(
        x=FEATURE_COLS,
        y="Target_Conductance",
        training_frame=train_hf
    )

    # 11) 依據 CV 階段的 MAPE 排序，選出最棒的模型
    # H2O 內部計算的 MAPE 是小數（例如 0.05 代表 5%）
    grid_perf = grid.get_grid(sort_by="mape", decreasing=False)
    print("\n" + "="*20 + " Grid Search 結果排行 (前 5 名) " + "="*20)
    print(grid_perf)

    best_model = grid_perf.models[0]
    print(f"\n🏆 最佳模型參數:")
    print(best_model.actual_params)

    # 12) 讀取外部測試檔
    print("\n" + "="*25 + " 讀取外部測試檔並進行整批預測 " + "="*25)
    try:
        try:
            df_user = pd.read_csv(TEST_EXCEL_PATH, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            print("💡 提示：測試檔非 UTF-8 編碼，切換至 CP950 (ANSI/Big5) 編碼讀取...")
            df_user = pd.read_csv(TEST_EXCEL_PATH, encoding="cp950", low_memory=False)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到測試檔案，請確認路徑是否正確：{TEST_EXCEL_PATH}")
        h2o.cluster().shutdown()
        return

    df_user.columns = df_user.columns.str.strip()
    
    # 保持輸出 DataFrame 結構
    existing_features = [c for c in FEATURE_COLS if c in df_user.columns]
    output_df = df_user[existing_features].copy()
    
    # 測試檔清理
    for col in FEATURE_COLS:
        if col in df_user.columns and df_user[col].astype(str).str.contains("%", na=False).any():
            df_user[col] = to_ratio_series(df_user[col])
            
    for col in NUMERIC_COLS:
        if col in df_user.columns:
            df_user[col] = pd.to_numeric(df_user[col], errors="coerce")
            
    # 測試檔缺值填補 (使用訓練集的 Median)
    df_user[NUMERIC_COLS] = df_user[NUMERIC_COLS].fillna(numeric_medians)
    df_user[CATEGORICAL_COLS] = df_user[CATEGORICAL_COLS].fillna("<NA>")
    
    # 轉換為 H2OFrame 預測 (不需要對齊 One-hot 欄位，H2O 會自動對齊類別名稱)
    test_hf = h2o.H2OFrame(df_user[FEATURE_COLS])
    for col in CATEGORICAL_COLS:
        test_hf[col] = test_hf[col].asfactor()

    print(f"成功載入測試數據共計 {len(df_user)} 筆樣本，開始進行最佳模型預測...")
    
    # 預測等效電導
    pred_conductance_hf = best_model.predict(test_hf)
    # 轉回 Pandas Series
    pred_conductance = pred_conductance_hf.as_data_frame()["predict"].values
    
    # 💡 乘以電壓（特徵值9）還原成電流
    v_test_user = df_user["特徵值9"].astype(float).values
    output_df["Pred_LGBM_H2O_Optimized"] = pred_conductance * v_test_user
        
    # 匯出結果
    output_filename = "test_predictions_h2o_lgbm.csv"
    output_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*50)
    print(f"🎉 測試集預測完畢！結果已成功儲存至: {output_filename}")
    print(f"該檔案保留了原始特徵，並在右側新增了 H2O 調參後最優化 MAPE 的 LightGBM 預測結果。")
    print("="*50)

    # 關閉 H2O
    h2o.cluster().shutdown()

if __name__ == "__main__":
    main()