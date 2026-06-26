import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

app = Flask(__name__)

# ==========================================
# 1. FIREBASE BAĞLANTISI
# ==========================================
try:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Sistem Durumu: Firebase Bulut Veritabanına Başarıyla Bağlanıldı!")
except Exception as e:
    print(f"Firebase Bağlantı Hatası: {e}")
    exit()

# ==========================================
# 2. VERİ TEMİZLEME VE MODELİ EĞİTME SÜRECİ
# ==========================================
print("-" * 50)
print("Sistem Durumu: Yapı Yapay Zekası için veriler optimize ediliyor...")

# Excel'i genel olarak yüklüyoruz
df = pd.read_excel('Concrete_Data.xls')

# Garanti yöntem: Eğer hücrelerde virgül yüzünden metin algılanan varsa temizleyip sayıya (float) zorluyoruz
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace(',', '.')
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Boş veya hatalı satırlar oluştuysa onları siliyoruz
df = df.dropna()

X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model mimarisi
model = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=200, random_state=42)
model.fit(X_train, y_train)
print("Sistem Durumu: Yapay sinir ağı gerçekçi değerlerle eğitildi! Web sunucusu açılıyor...")
print("-" * 50)

# ==========================================
# 3. WEB ROTALARI VE TAHMİN KAYDI
# ==========================================
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Form verilerini al
    input_features = [float(x) for x in request.form.values()]
    final_features = np.array([input_features])
    
    # Canlı ölçeklendirme ve gerçekçi tahmin
    final_features_scaled = scaler.transform(final_features)
    tahmin = model.predict(final_features_scaled)
    sonuc = max(0.0, round(tahmin[0], 2))
    
    # ------------------------------------------
    # FIREBASE BULUTUNA VERİ YAZMA ALANI
    # ------------------------------------------
    try:
        veri_paketi = {
            "cimento": input_features[0],
            "curuf": input_features[1],
            "ucucu_kul": input_features[2],
            "su": input_features[3],
            "super_akiskanlastirici": input_features[4],
            "kaba_agrega": input_features[5],
            "ince_agrega": input_features[6],
            "beton_yasi": input_features[7],
            "tahmin_edilen_dayanim": sonuc,
            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        # Firestore'a yazma emri
        db.collection("tahminler").add(veri_paketi)
        print(f"Sistem Durumu: Tahmin sonucu ({sonuc} MPa) başarıyla Firebase bulutuna kaydedildi!")
    except Exception as e:
        print(f"Veri Firebase'e yazılırken hata oluştu: {e}")
    # ------------------------------------------
    
    return render_template('index.html', prediction=sonuc)

if __name__ == "__main__":
    app.run(debug=True, port=5001)