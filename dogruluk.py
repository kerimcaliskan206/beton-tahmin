import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score

# 1. Excel verisini oku ve temizle
df = pd.read_excel('Concrete_Data.xls')
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace(',', '.')
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values   

# 2. Yapay zekayı arka planda hızlıca test için eğit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=200, random_state=42)
model.fit(X_train, y_train)

# 3. Doğruluk (F1-Score) Hesapla
y_pred = model.predict(X_test)
y_test_sinif = np.where(y_test >= 30, 1, 0)
y_pred_sinif = np.where(y_pred >= 30, 1, 0)

f1_sonucu = f1_score(y_test_sinif, y_pred_sinif)

# 4. Terminale yazdır
print("\n" + "="*45)
print(f"SİSTEMİN DOĞRULUK ORANI (F1-SCORE): {round(f1_sonucu, 4)}")
print("="*45 + "\n")