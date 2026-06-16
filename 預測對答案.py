import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

# =========================
# USER SETTINGS
# =========================
PRED_CSV_PATH = "test_predictions_result.csv"  # 剛剛產出的預測結果檔
TRUE_CSV_PATH = r"C:\專題\raw data正250.csv"      # 包含正確答案的原始測試檔
TARGET_COL = "Output 1"                         # 正確答案的欄位名稱

def load_csv_with_fallback(path):
    """自動嘗試編碼讀取 CSV"""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp950", low_memory=False)

def main():
    print("正在讀取預測結果與正確答案...")
    try:
        df_pred = load_csv_with_fallback(PRED_CSV_PATH)
        df_true = load_csv_with_fallback(TRUE_CSV_PATH)
    except FileNotFoundError as e:
        print(f"❌ 錯誤：找不到檔案，請確認路徑是否正確。\n{e}")
        return

    # 清理欄位名稱空白
    df_pred.columns = df_pred.columns.str.strip()
    df_true.columns = df_true.columns.str.strip()

    # 檢查正確答案欄位是否存在
    if TARGET_COL not in df_true.columns:
        print(f"❌ 錯誤：在答案檔中找不到標題為 '{TARGET_COL}' 的欄位！")
        print(f"目前答案檔擁有的欄位有：{list(df_true.columns)}")
        return

    # 清理正確答案 (Y) 的格式（比照主程式的強效清洗邏輯）
    y_raw_series = df_true[TARGET_COL].astype(str).str.strip()
    y_str = y_raw_series.str.replace(r"\((.*?)\)", r"-\1", regex=True)
    y_str = y_str.str.replace(",", "", regex=False)
    y_str = y_str.str.replace(r"(?i)m?A(mps)?", "", regex=True)
    y_str = y_str.str.replace(r"[^\d\.\-]", "", regex=True)
    
    y_true = pd.to_numeric(y_str, errors="coerce").abs()

    # 確保兩份表格長度一致
    if len(df_pred) != len(df_true):
        print(f"⚠️ 警告：預測檔 ({len(df_pred)} 筆) 與 答案檔 ({len(df_true)} 筆) 筆數不一致！")
        print("將依據最小筆數進行強制對齊（請確保兩份檔案的資料順序是完全相同的）。")
        min_len = min(len(df_pred), len(df_true))
        df_pred = df_pred.iloc[:min_len].copy()
        y_true = y_true.iloc[:min_len].copy()

    # 找出所有以 Pred_ 開頭的模型預測欄位
    pred_cols = [col for col in df_pred.columns if col.startswith("Pred_")]
    
    if not pred_cols:
        print("❌ 錯誤：在預測檔中找不到任何以 'Pred_' 開頭的模型預測欄位！")
        return

    # 移除正確答案中有 NaN (無法轉換) 的資料列，避免計算 MAPE 時出錯
    valid_mask = ~y_true.isna()
    if not valid_mask.all():
        bad_rows_count = (~valid_mask).sum()
        print(f"💡 提示：移除了答案檔中無法轉換為數字的怪異資料共 {bad_rows_count} 筆。")
        y_true = y_true[valid_mask]
        df_pred = df_pred[valid_mask]

    # 額外防錯：如果答案裡面有完全等於 0 的數字，計算 MAPE 會除以 0 導致無限大 (inf)
    if (y_true == 0).any():
        print("⚠️ 警告：正確答案中包含 0，這會導致傳統 MAPE 計算出現無限大 (inf)。")
        print("程序將自動為分母加上極小值 (1e-8) 以防出錯。")
        # 自定義 MAPE 防分母為 0
        def safe_mape(y_t, y_p):
            return np.mean(np.abs((y_t - y_p) / (y_t + 1e-8))) * 100
        mape_func = safe_mape
    else:
        mape_func = lambda y_t, y_p: mean_absolute_percentage_error(y_t, y_p) * 100

    # 遍歷每個模型計算 MAPE
    mape_results = []
    for col in pred_cols:
        model_name = col.replace("Pred_", "")  # 還原原本的模型名稱
        y_pred = df_pred[col]
        
        try:
            mape_val = mape_func(y_true, y_pred)
            mape_results.append({
                "Model Name": model_name,
                "Test MAPE(%)": mape_val
            })
        except Exception as e:
            print(f"計算模型 {model_name} 的 MAPE 時發生錯誤: {e}")

    # 轉成 DataFrame 並依據 MAPE 從小到大排序（誤差越小表現越好）
    summary_mape_df = pd.DataFrame(mape_results)
    summary_mape_df = summary_mape_df.sort_values(by="Test MAPE(%)", ascending=True).reset_index(drop=True)

    # 終端機輸出結果
    print("\n" + "="*20 + " 測試集 (250筆) 實際 MAPE 評估結果 " + "="*20)
    print(summary_mape_df.to_string(index=True, formatters={
        "Test MAPE(%)": "{:.2f}%".format
    }))
    print("="*70)

    # 導出成 CSV 檔案報告
    output_filename = "test_mape_evaluation_summary.csv"
    summary_mape_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"\n🎉 評估完成！各模型的 MAPE 排名已儲存至: {output_filename}")

if __name__ == "__main__":
    main()