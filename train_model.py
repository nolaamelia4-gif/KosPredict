
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Membaca dataset
df = pd.read_csv("kos_madiun.csv")

# Mengubah data AC menjadi angka
df["ac"] = df["ac"].map({
    "tidak": 0,
    "ya": 1
})

# Menentukan fitur dan target
X = df[["jarak_km", "ac", "rating"]]
y = df["harga"]

# Membagi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Membuat model Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Melatih model
model.fit(X_train, y_train)

# Menguji model
prediksi = model.predict(X_test)

print("Model berhasil dilatih!")
print("Jumlah data latih:", len(X_train))
print("Jumlah data uji:", len(X_test))
print("Mean Absolute Error:", mean_absolute_error(y_test, prediksi))
print("R2 Score:", r2_score(y_test, prediksi))

# Menyimpan model
lokasi_model = Path("prediksi_kos") / "model_kos.joblib"
joblib.dump(model, lokasi_model)

print("Model berhasil disimpan di:", lokasi_model)
