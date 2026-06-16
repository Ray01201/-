import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# 核心平滑與整合演算法
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# =========================
# USER SETTINGS
# =========================
EXCEL_PATH = r"C:\專題\raw data_正125,500.csv"
TEST_EXCEL_PATH = r"C:\專題\raw data正250.csv"  # 💡 新測試集 CSV 的路徑
TARGET_COL = "Output 1"

CATEGORICAL_COLS = ["特徵值1", "特徵值2", "特徵值3", "特徵值4", "特徵值7", "特徵值8", "特徵值24"]
NUMERIC_COLS = [
    "特徵值5", "特徵值6", "特徵值9", "特徵值10", "特徵值11", "特徵值12",
    "特徵值13", "特徵值14", "特徵值15", "特徵值16", "特徵值17", "特徵值18",
    "特徵值19", "特徵值20", "特徵值21", "特徵值22", "特徵值23"
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

RANDOM_STATE = 36
N_SPLITS = 5

def get_12_models():
    """
    建立 4 種演算法 × 3 種複雜度級距 (Simple, Base, Strong) = 12 組模型
    """
    models = {}
    
    # 1. 支撐向量迴規系列 (SVR)
    models["SVR_simple"] = SVR(kernel="rbf", C=0.1, epsilon=0.2)
    models["SVR_base"]   = SVR(kernel="rbf", C=1.0, epsilon=0.1)
    models["SVR_strong"] = SVR(kernel="rbf", C=10.0, epsilon=0.02)
    
    # 2. 隨機森林系列 (Random Forest)
    models["RF_simple"]  = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
    models["RF_base"]    = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    models["RF_strong"]  = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)
    
    # 3. XGBoost 系列 (XGBoost)
    models["XGB_simple"] = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1)
    models["XGB_base"]   = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
    models["XGB_strong"] = XGBRegressor(n_estimators=200, max_depth=9, learning_rate=0.15, random_state=RANDOM_STATE, n_jobs=-1)
    
    # 4. LightGBM 系列 (LightGBM) - 保持原本附檔的參數設定，僅在後面改變訓練與預測邏輯
    models["LGBM_simple"] = LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    models["LGBM_base"]   = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    models["LGBM_strong"] = LGBMRegressor(n_estimators=200, max_depth=9, learning_rate=0.15, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    
    return models

def to_ratio_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    has_pct = s.str.contains("%", na=False)
    s_no_pct = s.str.replace("%", "", regex=False).str.strip()
    num = pd.to_numeric(s_no_pct, errors="coerce")
    num.loc[has_pct] = num.loc[has_pct] / 100.0
    return num

def main():
    # 1) Read data
    df = pd.read_csv(EXCEL_PATH, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()

    # 2) 清理：特徵欄位中出現 % 的值
    for col in FEATURE_COLS:
        if col in df.columns and df[col].astype(str).str.contains("%", na=False).any():
            df[col] = to_ratio_series(df[col])

    # 3) 數值欄強制轉數字
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4) 整合：強效清洗與偵錯照妖鏡邏輯
    y_raw_series = df[TARGET_COL].astype(str)
    y_str = y_raw_series.str.strip()

    y_str = y_str.str.replace(r"\((.*?)\)", r"-\1", regex=True)
    y_str = y_str.str.replace(",", "", regex=False)
    y_str = y_str.str.replace(r"(?i)m?A(mps)?", "", regex=True)
    y_str = y_str.str.replace(r"[^\d\.\-]", "", regex=True)

    y = pd.to_numeric(y_str, errors="coerce")
    
    is_bad = y.isna() & (y_raw_series.str.upper() != "NAN") & (y_raw_series.str.strip() != "")
    bad_count = is_bad.sum()
    
    if bad_count > 0:
        print(f"\n⚠️ 發現真正無法轉換的怪異 Y 資料共 {bad_count} 筆。以下抽樣前 15 筆展示：")
        debug_df = pd.DataFrame({
            "原始欄位數值": y_raw_series[is_bad],
            "清理後的文字": y_str[is_bad]
        })
        print(debug_df.head(15).to_string())
        print("-" * 50)

    y = y.abs()

    # 💡 額外安全檢查：為了符合電導模型，特徵值9(電壓)必須大於0
    # 避免在 LightGBM 計算 y / 電壓 時發生除以零或 NaN
    df["特徵值9"] = pd.to_numeric(df["特徵值9"], errors="coerce")
    
    keep = (~y.isna()) & (~df["特徵值9"].isna()) & (df["特徵值9"] > 0)
    df = df.loc[keep].copy()
    y = y.loc[keep].copy()
    
    print(f"清理完成！最終有效用於訓練的樣本數: {len(df)} 筆\n")

    if len(df) == 0:
        print("❌ 錯誤：沒有任何有效數據可用於訓練，請檢查 Y 資料格式或特徵值9是否有大於0的數值！")
        return

    # 5) X 缺值處理
    X_raw = df[FEATURE_COLS].copy()
    
    numeric_medians = X_raw[NUMERIC_COLS].median()
    X_raw[NUMERIC_COLS] = X_raw[NUMERIC_COLS].fillna(numeric_medians)
    X_raw[CATEGORICAL_COLS] = X_raw[CATEGORICAL_COLS].fillna("<NA>")

    groups = df.index.to_series().astype(str)

    # One-hot encoding
    X = pd.get_dummies(X_raw, columns=CATEGORICAL_COLS, drop_first=False)
    trained_features_columns = X.columns.tolist()

    print("=== Configuration Summary (No Grouping Control) ===")
    print(f"Unique design groups: {groups.nunique()} (Every row is a separate group)")
    print(f"X shape after one-hot: {X.shape}\n")

    # 6) 初始化 12 組模型與評估字典
    all_models = get_12_models()
    model_results = []
    final_trained_models = {}

    # 7) 開始對 12 組模型各自進行 Group 5-Fold 交叉驗證
    gkf = GroupKFold(n_splits=N_SPLITS)

    for model_name, model_obj in all_models.items():
        print(f"Running Cross-Validation for: {model_name}...")
        
        fold_r2, fold_rmse, fold_mape = [], [], []
        
        # 判斷目前模型是否為新邏輯的 LightGBM
        is_lgbm = "LGBM" in model_name
        
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 💡 新增邏輯：如果是 LGBM，目標變數轉換為電導 (y / 電壓)
            if is_lgbm:
                # 這裡要抓取未標準化前的「特徵值9」作為除數
                v_train = X_train["特徵值9"]
                y_train_target = y_train / v_train
            else:
                y_train_target = y_train

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            try:
                from sklearn.base import clone
                fold_model = clone(model_obj)
                fold_model.fit(X_train_scaled, y_train_target)
                
                # 預測
                pred_out = fold_model.predict(X_test_scaled)

                # 💡 新增邏輯：如果是 LGBM，預測出來的是電導，需乘回電壓還原成電流
                if is_lgbm:
                    v_test = X_test["特徵值9"]
                    y_pred = pred_out * v_test
                else:
                    y_pred = pred_out

                fold_r2.append(r2_score(y_test, y_pred))
                fold_rmse.append(mean_squared_error(y_test, y_pred) ** 0.5)
                fold_mape.append(mean_absolute_percentage_error(y_test, y_pred) * 100)
            except Exception as e:
                print(f"Fold error in {model_name}: {e}")
                continue

        if fold_r2:
            model_results.append({
                "Model Name": model_name,
                "Avg R2": np.mean(fold_r2),
                "Avg RMSE": np.mean(fold_rmse),
                "Avg MAPE(%)": np.mean(fold_mape)
            })
        
        # --- 全量訓練階段 ---
        print(f"-> Training final model for {model_name} on ALL valid data...")
        global_scaler = StandardScaler()
        X_scaled_full = global_scaler.fit_transform(X)
        
        # 💡 全量訓練：如果是 LGBM，同樣使用電導轉換
        if is_lgbm:
            y_full_target = y / X["特徵值9"]
        else:
            y_full_target = y

        final_model = clone(model_obj)
        final_model.fit(X_scaled_full, y_full_target)
        
        final_trained_models[model_name] = {
            "model": final_model,
            "scaler": global_scaler
        }

    # 8) 輸出 12 組模型最終排名摘要表
    summary_df = pd.DataFrame(model_results)
    summary_df = summary_df.sort_values(by="Avg R2", ascending=False).reset_index(drop=True)

    print("\n" + "="*20 + " 12 Models CV Performance Summary (LGBM Physics Optimized) " + "="*20)
    print(summary_df.to_string(index=True, formatters={
        "Avg R2": "{:.6f}".format,
        "Avg RMSE": "{:.6f}".format,
        "Avg MAPE(%)": "{:.2f}%".format,
    }))
    
    summary_df.to_csv("cv5_12_models_trees_summary.csv", index=False, encoding="utf-8-sig")
    print("\nSaved summary to 'cv5_12_models_trees_summary.csv'")

    # ==================================================
    # 9) 讀取外部測試檔，並批次整合輸出為 CSV 檔
    # ==================================================
    print("\n" + "="*25 + " 讀取外部測試檔並進行整批預測 " + "="*25)
    
    try:
        try:
            df_user = pd.read_csv(TEST_EXCEL_PATH, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            print("💡 提示：測試檔非 UTF-8 編碼，切換至 CP950 (ANSI/Big5) 編碼讀取...")
            df_user = pd.read_csv(TEST_EXCEL_PATH, encoding="cp950", low_memory=False)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到測試檔案，請確認路徑是否正確：{TEST_EXCEL_PATH}")
        return

    df_user.columns = df_user.columns.str.strip()
    
    existing_features = [c for c in FEATURE_COLS if c in df_user.columns]
    output_df = df_user[existing_features].copy()
    
    for col in FEATURE_COLS:
        if col in df_user.columns and df_user[col].astype(str).str.contains("%", na=False).any():
            df_user[col] = to_ratio_series(df_user[col])
            
    for col in NUMERIC_COLS:
        if col in df_user.columns:
            df_user[col] = pd.to_numeric(df_user[col], errors="coerce")
            
    # 缺值自動補齊
    df_user[NUMERIC_COLS] = df_user[NUMERIC_COLS].fillna(numeric_medians)
    df_user[CATEGORICAL_COLS] = df_user[CATEGORICAL_COLS].fillna("<NA>")
    
    # 進行 One-hot encoding 並對齊訓練特徵維度
    df_user_encoded = pd.get_dummies(df_user[FEATURE_COLS], columns=CATEGORICAL_COLS)
    for col in trained_features_columns:
        if col not in df_user_encoded.columns:
            df_user_encoded[col] = 0
            
    df_user_encoded = df_user_encoded[trained_features_columns]

    print(f"成功載入測試數據共計 {len(df_user)} 筆樣本，開始批次進行預測...")
    
    # 備份測試集中的電壓欄位，供 LGBM 還原電流使用
    v_test_user = df_user_encoded["特徵值9"].astype(float)

    # 依交叉驗證 R2 表現從高到低的順序，將各模型的預測結果併入 output_df 中
    for model_name in summary_df["Model Name"]:
        model_pack = final_trained_models[model_name]
        mdl = model_pack["model"]
        scl = model_pack["scaler"]
        
        # 依該模型的縮放器進行標準化
        df_user_scaled = scl.transform(df_user_encoded)
        
        # 預測
        raw_pred = mdl.predict(df_user_scaled)
        
        # 💡 新增邏輯：如果是 LGBM，預測結果是電導，必須乘以電壓還原成電流
        if "LGBM" in model_name:
            output_df[f"Pred_{model_name}"] = raw_pred * v_test_user
        else:
            output_df[f"Pred_{model_name}"] = raw_pred
        
    # 將整合後的結果導出為新的 CSV 檔案
    output_filename = "test_predictions_result.csv"
    output_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*50)
    print(f"🎉 測試集預測完畢！結果已成功打包儲存至: {output_filename}")
    print(f"該檔案保留了原始特徵，並在右側依 R2 表現排序新增了 12 個模型的預測結果欄位。")
    print("="*50)

if __name__ == "__main__":
    main()