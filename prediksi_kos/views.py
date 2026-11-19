
import pandas as pd
import joblib
from pathlib import Path
from django.shortcuts import render

# Menentukan lokasi file model
LOKASI_MODEL = Path(__file__).resolve().parent / "model_kos.joblib"

# Memuat model Machine Learning
model = joblib.load(LOKASI_MODEL)


def home(request):
    hasil = None
    error = None

    if request.method == "POST":
        try:
            jarak = float(request.POST.get("jarak_km"))
            ac = request.POST.get("ac")
            rating = float(request.POST.get("rating"))

            # Validasi data input
            if jarak < 0 or not 1 <= rating <= 5:
                raise ValueError

            if ac not in ["ya", "tidak"]:
                raise ValueError

            # Mengubah pilihan AC menjadi angka
            ac_angka = 1 if ac == "ya" else 0

            # Menyiapkan data untuk prediksi
            data_baru = pd.DataFrame([{
                "jarak_km": jarak,
                "ac": ac_angka,
                "rating": rating
            }])

            # Melakukan prediksi harga kos
            hasil = round(float(model.predict(data_baru)[0]), 2)

        except (ValueError, TypeError):
            error = "Mohon isi semua data dengan benar."

    return render(request, "prediksi_kos/home.html", {
        "hasil": hasil,
        "error": error
    })
