import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor  # Scikit-learn 的 GBR
from xgboost import XGBRegressor                        # XGBoost

# 1. 讀取 Excel 檔案
df = pd.read_excel(r"C:\\Users\Asus\Downloads\\merged_file.xlsx")

# 2. 處理類別型欄位 (Categorical Data)
#將文字編碼，轉為 0 與 1
# 假設 'device_type' 和 'location' 是類別欄位
df_processed = pd.get_dummies(df, columns=['特徵值1', '特徵值2','特徵值3','特徵值4','特徵值5','特徵值7','特徵值8','特徵值15','特徵值24'])
#df_processed = pd.get_dummies(df, columns=['特徵值8'])

# 3. 定義特徵 (X) 與目標 (y)
# 假設你要預測的目標欄位名稱是 'max_current'

#X = df_processed.drop(['Unnamed: 0','Output 1','Output 2','Output 3','Output 4','Output 5','特徵值1', '特徵值2','特徵值3','特徵值4','特徵值5','特徵值6','特徵值7','特徵值10','特徵值11','特徵值12', '特徵值13','特徵值14','特徵值15','特徵值16','特徵值17','特徵值18','特徵值19','特徵值20','特徵值21','特徵值22','特徵值23','特徵值24'], axis=1) # 除了目標以外的所有欄位都是特徵          # 特徵：特徵值8 和 特徵值9
X = df_processed.drop(['Unnamed: 0','Output 1','Output 2','Output 3','Output 4','Output 5'], axis=1) # 除了目標以外的所有欄位都是特徵          # 特徵：特徵值8 和 特徵值9
y = df_processed['Output 1'] 
y = abs(y)             # 目標：最大電流

# 4. 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ==========================================
# 5a. 建立並訓練「梯度提升迴歸模型 (Gradient Boosting)」
# ==========================================
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# 進行預測與評估
y_pred_gb = gb_model.predict(X_test)
print("--- Gradient Boosting 評估結果 ---")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_gb)*100:.2f}%")
print(f"R2 Score: {r2_score(y_test, y_pred_gb):.2f}")

# ==========================================
# 5b. 建立並訓練「XGBoost 迴歸模型」
# ==========================================
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# 進行預測與評估
y_pred_xgb = xgb_model.predict(X_test)
print("\n--- XGBoost 評估結果 ---")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_xgb)*100:.2f}%")
print(f"R2 Score: {r2_score(y_test, y_pred_xgb):.2f}")

# ==========================================
# 8. 查看 XGBoost 的特徵重要性 (範例)
# ==========================================
xgb_importances = xgb_model.feature_importances_
xgb_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_importances}).sort_values(by='Importance', ascending=False)
print("\n--- XGBoost 特徵重要性 (前10名) ---")
print(xgb_importance_df.head(10))