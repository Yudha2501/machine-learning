# nama Sang Putu Yudha Mahendra
# nim 2415344022
# kelas 5A TO
# Machine learning

# Mission-06A — Linear Regression on ESP32

## Predicting Machine Temperature with MicroPython

**Mata kuliah:** Machine Learning  
**Topik:** Regresi linear sederhana, training, inference, dan evaluasi pada embedded system  
**Perangkat:** ESP32 (firmware MicroPython), kabel USB, laptop dengan Thonny IDE  
**Perangkat tambahan:** Tidak diperlukan sensor, breadboard, atau koneksi internet  
**Estimasi durasi:** 80–100 menit  
**Prasyarat:** Dasar Python, regresi linear, MAE, dan penggunaan Thonny untuk menjalankan MicroPython pada ESP32

> **Misi:** Biasanya Anda melatih dan menguji model machine learning di laptop. Dalam praktikum ini, jalankan prediksi dan perhitungan koefisien regresi langsung di ESP32. Semua nilai temperatur adalah **data sintetik**, bukan hasil pembacaan sensor.

## 1. Tujuan pembelajaran

Setelah praktikum, mahasiswa mampu:

1. Menjelaskan perbedaan **training** dan **inference**.
2. Mengimplementasikan model regresi linear sederhana pada ESP32 tanpa `numpy`, `pandas`, atau `scikit-learn`.
3. Menghitung koefisien slope dan intercept dari data training langsung pada ESP32.
4. Menguji model pada data yang **tidak digunakan saat training** dan menghitung *Mean Absolute Error* (MAE).
5. Mengidentifikasi risiko ekstrapolasi serta keterbatasan model regresi linear.

## 2. Studi kasus

Sebuah mesin hipotetis mengalami kenaikan temperatur seiring bertambahnya waktu operasi. Kita ingin memperkirakan temperatur berdasarkan waktu operasi.

| Variabel | Simbol | Satuan | Peran |
|---|---|---|---|
| Waktu operasi | `x` | menit | Fitur/input |
| Temperatur mesin | `y` | °C | Target/output |


Model regresi linear sederhana:

$$
\hat{y} = b_0 + b_1 x
$$

Keterangan:

- $b_0$: **intercept**, temperatur prediksi saat $x = 0$.
- $b_1$: **slope**, perubahan temperatur prediksi untuk tambahan satu menit operasi.
- $x$: waktu operasi mesin (menit).
- $\hat{y}$: temperatur hasil prediksi model (°C).


**Penting:** Hubungan linear ini hanya ilustrasi pembelajaran. Temperatur mesin nyata dapat berubah secara nonlinear dan dipengaruhi kondisi lingkungan, beban, dan pendinginan.

### Dataset sintetik

Berikut data **training** yang disimpan langsung di dalam kode ESP32:

| Waktu (menit) | Temperatur (°C) |
|---:|---:|
| 0 | 28.0 |
| 1 | 29.7 |
| 2 | 31.5 |
| 3 | 32.8 |
| 4 | 34.9 |
| 5 | 36.1 |
| 6 | 38.2 |
| 7 | 39.6 |
| 8 | 41.4 |

Data **testing** (tidak digunakan untuk menghitung koefisien):

| Waktu (menit) | Temperatur (°C) |
|---:|---:|
| 1.5 | 30.6 |
| 3.5 | 33.9 |
| 5.5 | 37.4 |
| 7.5 | 40.5 |

> Dataset disusun sebagai simulasi pembelajaran, bukan bukti pengukuran mesin. Pemisahan train-test dipakai untuk mengevaluasi prediksi pada titik waktu yang belum dilihat model. Karena ini dataset kecil, MAE belum membuktikan kemampuan generalisasi pada mesin nyata.

## 3. Persiapan

1. Sambungkan ESP32 ke laptop melalui USB.
2. Jalankan **Thonny** dan pilih interpreter **MicroPython (ESP32)** serta port serial yang benar.
3. Pastikan Shell Thonny dapat menjalankan `print("Hello ESP32")`.
4. Buat berkas baru untuk setiap eksperimen. Jalankan dari Thonny atau simpan berkas ke ESP32.
5. Untuk menghentikan `while True`, tekan **Stop** di Thonny atau `Ctrl+C` pada Shell.

**Catatan:** Kode di bawah hanya menggunakan fungsi bawaan Python/MicroPython. Tidak diperlukan instalasi paket ML pada ESP32.

---

## 4. Eksperimen A — Inference dengan koefisien yang sudah diketahui (15 menit)

Pada eksperimen ini kita **belum melakukan training di ESP32**. Anggap model linear berikut sudah diperoleh dari proses training sebelumnya:

$$
\hat{T} = 28.0 + 1.7 t
$$

Buat berkas **`01_inference.py`**:

```python
# Mission-06A, Eksperimen A
# Inference pada ESP32 (koefisien sudah diberikan)

b0 = 28.0       # intercept (derajat C)
b1 = 1.7        # slope (derajat C per menit)

def predict(minutes):
    return b0 + b1 * minutes

for minutes in [0, 2, 4, 6, 8]:
    predicted_temp = predict(minutes)
    print("Time:", minutes, "min | Prediction:",
          round(predicted_temp, 2), "C")
```

**Amati:** ESP32 dapat menghasilkan prediksi tanpa melakukan training atau membaca sensor.

**Pertanyaan cepat:** Apakah `b0 = 28.0` dan `b1 = 1.7` dalam kode di atas dipelajari ESP32 dari dataset? **Tidak.** Keduanya sengaja diberikan sebagai parameter model pada eksperimen A.

---

## 5. Eksperimen B — Training regresi linear langsung di ESP32 (25 menit)

Sekarang ESP32 sendiri menghitung koefisien regresi dengan metode **ordinary least squares (OLS)** menggunakan data training.

Untuk pasangan data $(x_i, y_i)$:

$$
b_1 = \frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sum_i(x_i-\bar{x})^2}
$$

$$
b_0 = \bar{y} - b_1\bar{x}
$$

Buat berkas **`02_training.py`**:

```python
# Mission-06A, Eksperimen B
# Training regresi linear sederhana langsung di ESP32

train_data = [
    (0, 28.0),
    (1, 29.7),
    (2, 31.5),
    (3, 32.8),
    (4, 34.9),
    (5, 36.1),
    (6, 38.2),
    (7, 39.6),
    (8, 41.4)
]

def fit_linear_regression(data):
    n = len(data)
    if n < 2:
        raise ValueError("Need at least two training samples")

    mean_x = sum(row[0] for row in data) / n
    mean_y = sum(row[1] for row in data) / n

    numerator = sum(
        (x - mean_x) * (y - mean_y) for x, y in data
    )
    denominator = sum(
        (x - mean_x) ** 2 for x, y in data
    )

    if denominator == 0:
        raise ValueError("All training x values are identical")

    b1 = numerator / denominator
    b0 = mean_y - b1 * mean_x
    return b0, b1

def predict(x, b0, b1):
    return b0 + b1 * x

b0, b1 = fit_linear_regression(train_data)

print("Intercept:", round(b0, 4))
print("Slope:", round(b1, 4))

for minutes in [0, 2, 4, 6, 8]:
    prediction = predict(minutes, b0, b1)
    print("Time:", minutes, "min | Prediction:",
          round(prediction, 2), "C")
```

**Diskusikan:**

- Bandingkan `b0` dan `b1` dari eksperimen B dengan angka yang diberikan dalam eksperimen A.
- Mengapa hasilnya tidak harus identik?
- Pada kode ini, bagian manakah yang merupakan **training** dan bagian manakah yang merupakan **inference**?


**Hasil Run Eksperimen B**
```text
Intercept: 28.0222
Slope: 1.6667
Time: 0 min | Prediction: 28.02 C
Time: 2 min | Prediction: 31.36 C
Time: 4 min | Prediction: 34.69 C
Time: 6 min | Prediction: 38.02 C
Time: 8 min | Prediction: 41.36 C
>>>
```

**1. Perbandingan b0 dan b1 Eksperimen A dan B**

Hasil Eksperimen B:
- Intercept (b0) = 28.0222
- Slope (b1) = 1.6667

Persamaan regresi linear:

y = 28.0222 + 1.6667x

Nilai b0 menunjukkan perkiraan suhu awal sebesar 28.0222°C ketika waktu 0 menit. Nilai b1 menunjukkan bahwa suhu diperkirakan meningkat sekitar 1.6667°C setiap menit.

Nilai tersebut perlu dibandingkan dengan angka Eksperimen A untuk mengetahui perbedaannya.

**2. Mengapa hasilnya tidak harus identik?**

Hasil tidak harus identik karena Eksperimen A dan B mungkin menggunakan metode perhitungan atau parameter yang berbeda. Eksperimen B menghitung b0 dan b1 langsung dari data training menggunakan metode Ordinary Least Squares (OLS). Perbedaan data dan pembulatan angka juga dapat menyebabkan hasil berbeda.

**3. Bagian training dan inference**

**Training:**

Bagian training terdapat pada pemanggilan fungsi:

```python
b0, b1 = fit_linear_regression(train_data)
```

Fungsi `fit_linear_regression()` menghitung nilai b0 dan b1 berdasarkan data training.

**Inference:**

Bagian inference terdapat pada fungsi:

```python
def predict(x, b0, b1):
    return b0 + b1 * x
```

Fungsi `predict()` menggunakan parameter yang telah diperoleh dari training untuk memprediksi suhu pada waktu tertentu.

**Kesimpulan**

Eksperimen B menunjukkan bahwa regresi linear sederhana dapat dilatih dengan menghitung parameter b0 dan b1 dari data. Training digunakan untuk memperoleh parameter model, sedangkan inference digunakan untuk menghasilkan prediksi berdasarkan parameter tersebut.

---

## 6. Eksperimen C — Mengukur error pada data testing (15 menit)

Jalankan kode **`02_training.py`** terlebih dahulu, lalu **tambahkan** bagian berikut ke **akhir berkas yang sama**. Jangan menggunakan data testing untuk menghitung `b0` dan `b1`.

```python
# Tambahkan di akhir 02_training.py

test_data = [
    (1.5, 30.6),
    (3.5, 33.9),
    (5.5, 37.4),
    (7.5, 40.5)
]

absolute_errors = []

print("\n--- TEST RESULTS ---")
for minutes, actual in test_data:
    predicted = predict(minutes, b0, b1)
    error = abs(actual - predicted)
    absolute_errors.append(error)

    print("t =", minutes,
          "| Actual =", actual,
          "| Predicted =", round(predicted, 2),
          "| Abs error =", round(error, 2))

mae = sum(absolute_errors) / len(absolute_errors)
print("MAE test:", round(mae, 4), "C")
```

Rumus MAE:

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

**Tugas mahasiswa:** catat koefisien model, nilai prediksi tiap titik testing, dan MAE. Jelaskan makna satuan °C pada MAE.
### Hasil Run Eksperimen C

```text
MPY: soft reboot
Intercept: 28.0222
Slope: 1.6667
Time: 0 min | Prediction: 28.02 C
Time: 2 min | Prediction: 31.36 C
Time: 4 min | Prediction: 34.69 C
Time: 6 min | Prediction: 38.02 C
Time: 8 min | Prediction: 41.36 C

--- TEST RESULTS ---
t = 1.5 | Actual = 30.6 | Predicted = 30.52 | Abs error = 0.08
t = 3.5 | Actual = 33.9 | Predicted = 33.86 | Abs error = 0.04
t = 5.5 | Actual = 37.4 | Predicted = 37.19 | Abs error = 0.21
t = 7.5 | Actual = 40.5 | Predicted = 40.52 | Abs error = 0.02
MAE test: 0.0889 C
>>>
```

### Analisis Hasil Eksperimen C

**1. Koefisien model**

- Intercept (b0) = 28.0222
- Slope (b1) = 1.6667

Persamaan regresi linear:

y = 28.0222 + 1.6667x

**2. Analisis hasil testing**

Pengujian dilakukan menggunakan empat data testing. Hasil prediksi dibandingkan dengan suhu aktual untuk menghitung absolute error.

- Pada waktu 1.5 menit, error sebesar 0.08 °C.
- Pada waktu 3.5 menit, error sebesar 0.04 °C.
- Pada waktu 5.5 menit, error sebesar 0.21 °C.
- Pada waktu 7.5 menit, error sebesar 0.02 °C.

**3. Nilai MAE**

MAE test = 0.0889 °C.

MAE dihitung dengan menjumlahkan seluruh absolute error, kemudian membaginya dengan jumlah data testing, yaitu empat sampel.

**4. Makna satuan °C pada MAE**

Satuan °C menunjukkan bahwa MAE mengukur rata-rata besar selisih absolut antara suhu aktual dan suhu prediksi dalam derajat Celsius. Semakin kecil MAE, semakin dekat prediksi model dengan suhu aktual pada data pengujian.

**Kesimpulan**

Eksperimen C menunjukkan bahwa model regresi linear yang dilatih pada Eksperimen B menghasilkan MAE sebesar 0.0889 °C pada empat data testing. Hasil tersebut menunjukkan bahwa prediksi model cukup dekat dengan nilai aktual pada sampel pengujian ini.
---

## 7. Eksperimen D — Prediksi interaktif dari terminal serial (10 menit)

Masih di **`02_training.py`**, boleh tambahkan bagian berikut di akhir setelah evaluasi. ESP32 akan menerima input waktu operasi melalui Shell Thonny.

```python
# Input manual dari Shell Thonny
# Ketik q untuk keluar

while True:
    user_input = input("Waktu operasi (menit), atau q: ")

    if user_input.strip().lower() == "q":
        print("Program selesai")
        break

    try:
        minutes = float(user_input)
        if minutes < 0:
            print("Waktu tidak boleh negatif")
            continue

        predicted = predict(minutes, b0, b1)
        print("Predicted temperature:",
              round(predicted, 2), "C")

        if minutes > 8:
            print("WARNING: extrapolation beyond training range")

    except ValueError:
        print("Masukkan angka atau q")
```

**Coba input:** `2.5`, `6.5`, `8`, `12`, dan `20` menit. Apakah prediksi pada 20 menit dapat langsung dipercaya? **Tidak.** Prediksi tersebut berada di luar rentang training (0–8 menit), sehingga bersifat ekstrapolasi dan belum divalidasi.

---

### Hasil Run Eksperimen D

```text
MPY: soft reboot
Intercept: 28.0222
Slope: 1.6667
Time: 0 min | Prediction: 28.02 C
Time: 2 min | Prediction: 31.36 C
Time: 4 min | Prediction: 34.69 C
Time: 6 min | Prediction: 38.02 C
Time: 8 min | Prediction: 41.36 C

--- TEST RESULTS ---
t = 1.5 | Actual = 30.6 | Predicted = 30.52 | Abs error = 0.08
t = 3.5 | Actual = 33.9 | Predicted = 33.86 | Abs error = 0.04
t = 5.5 | Actual = 37.4 | Predicted = 37.19 | Abs error = 0.21
t = 7.5 | Actual = 40.5 | Predicted = 40.52 | Abs error = 0.02
MAE test: 0.0889 C

Waktu operasi (menit), atau q: 2.5
Predicted temperature: 32.19 C

Waktu operasi (menit), atau q: 6.5
Predicted temperature: 38.86 C

Waktu operasi (menit), atau q: 8
Predicted temperature: 41.36 C

Waktu operasi (menit), atau q: 12
Predicted temperature: 48.02 C
WARNING: extrapolation beyond training range

Waktu operasi (menit), atau q: 20
Predicted temperature: 61.36 C
WARNING: extrapolation beyond training range
```

### Analisis Hasil

**1. Koefisien model**

- Intercept (b0) = 28.0222
- Slope (b1) = 1.6667

Persamaan regresi linear:

y = 28.0222 + 1.6667x

**2. Hasil prediksi input manual**

| Waktu operasi (menit) | Prediksi suhu (°C) | Keterangan |
|---:|---:|---|
| 2.5 | 32.19 | Dalam rentang training |
| 6.5 | 38.86 | Dalam rentang training |
| 8 | 41.36 | Batas maksimum training |
| 12 | 48.02 | Ekstrapolasi |
| 20 | 61.36 | Ekstrapolasi |

**3. Mengapa muncul WARNING pada 12 dan 20 menit?**

Data training hanya mencakup waktu operasi 0–8 menit. Oleh karena itu, input 12 dan 20 menit berada di luar rentang training. Kondisi ini disebut ekstrapolasi.

Model tetap menghitung prediksi menggunakan persamaan regresi linear, tetapi hasilnya belum divalidasi dengan data aktual pada rentang waktu tersebut.

**4. Apakah prediksi pada 20 menit dapat dipercaya?**

Prediksi 61.36 °C pada waktu 20 menit tidak dapat langsung dipercaya sebagai suhu aktual. Model mengasumsikan kenaikan suhu berlangsung secara linear, padahal kondisi sebenarnya dapat dipengaruhi oleh pendinginan, perubahan beban, dan lingkungan.

Untuk memastikan keandalannya, diperlukan pengukuran suhu aktual pada waktu operasi yang lebih panjang.

### Kesimpulan Eksperimen D

Eksperimen D berhasil menunjukkan bahwa ESP32 dapat menerima input waktu operasi melalui Shell Thonny dan menghasilkan prediksi suhu berdasarkan model regresi linear yang telah dilatih.

Sistem memberikan peringatan ketika input melebihi 8 menit. Peringatan ini membantu pengguna mengenali bahwa prediksi berada di luar rentang data training dan belum tentu sesuai dengan kondisi sebenarnya.

---

## 8. Eksperimen E — Tantangan modifikasi dataset (15 menit)

Ubah **hanya data training**, lalu lakukan training dan evaluasi ulang dengan **test_data yang sama**. Hasilnya dicatat dalam tabel.

1. **Skenario 1:** Gunakan 9 data training asli.
2. **Skenario 2:** Hapus data training pada `t = 2`, `t = 4`, dan `t = 6`.
3. **Skenario 3:** Gunakan data training asli, tetapi ubah satu nilai temperatur, yaitu pada `t = 4` dari `34.9` menjadi `45.0` °C (outlier buatan).

| Skenario | Jumlah data training | Intercept (`b0`) | Slope (`b1`) | MAE testing (°C) |
|---|---:|---:|---:|---:|
| 1 — Asli | 9 |28.0222| 1.6667 |0.0889|
| 2 — Data dikurangi | 6 |27.9385|1.6654 |0.1673 |
| 3 — Satu outlier | 9 | 29.1444 |1.6667 | 1.0444 |

**Analisis:** Apakah dataset yang lebih banyak **selalu** meningkatkan kinerja? Mengapa satu outlier dapat memengaruhi hasil OLS? Jelaskan berdasarkan hasil, bukan asumsi.
## Eksperimen E — Tantangan Modifikasi Dataset

### 1. Skenario 1 — Dataset Asli

Hasil training menggunakan 9 data training asli:

```text
Intercept: 28.0222
Slope: 1.6667
MAE test: 0.0889 C
```

Dataset asli menghasilkan intercept sebesar 28.0222 dan slope sebesar 1.6667. Nilai MAE testing sebesar 0.0889 °C menunjukkan rata-rata kesalahan absolut prediksi pada empat data testing.

### 2. Skenario 2 — Data Training Dikurangi

Hasil training setelah data pada waktu 2, 4, dan 6 menit dihapus:

```text
Intercept: 27.9385
Slope: 1.6654
MAE test: 0.1673 C
```

Jumlah data training berkurang dari 9 menjadi 6 data. Nilai MAE meningkat dari 0.0889 °C menjadi 0.1673 °C. Hal ini menunjukkan bahwa penghapusan data pada percobaan ini menyebabkan hasil prediksi menjadi kurang akurat dibandingkan dataset asli.

### 3. Skenario 3 — Satu Outlier

Hasil training setelah temperatur pada waktu 4 menit diubah dari 34.9 °C menjadi 45.0 °C:

```text
Intercept: 29.1444
Slope: 1.6667
MAE test: 1.0444 C
```

Jumlah data training tetap 9, tetapi satu nilai temperatur diubah menjadi outlier buatan. Nilai MAE meningkat dari 0.0889 °C menjadi 1.0444 °C. Hal ini menunjukkan bahwa perubahan satu nilai temperatur dapat memengaruhi hasil prediksi model.

### 4. Analisis

**A. Apakah dataset yang lebih banyak selalu meningkatkan kinerja?**

Tidak selalu. Pada skenario 1, sembilan data training asli menghasilkan MAE sebesar 0.0889 °C. Pada skenario 2, setelah tiga data dihapus, MAE meningkat menjadi 0.1673 °C. Hasil ini menunjukkan bahwa pengurangan data training pada eksperimen tersebut meningkatkan kesalahan prediksi. Namun, jumlah data saja tidak menjamin kinerja model yang lebih baik karena kualitas dan keterwakilan data juga berpengaruh.

**B. Mengapa satu outlier dapat memengaruhi hasil OLS?**

Metode Ordinary Least Squares (OLS) mencari garis regresi yang meminimalkan jumlah kuadrat selisih antara nilai aktual dan prediksi. Outlier dapat mengubah posisi garis regresi sehingga prediksi pada data testing ikut berubah.

Pada skenario 3, intercept berubah dari 28.0222 menjadi 29.1444, sedangkan slope tetap 1.6667. Hal ini terjadi karena temperatur pada waktu 4 menit diubah, tepat pada rata-rata waktu dataset training asli. Perubahan tersebut memengaruhi intercept tanpa mengubah slope.

**5. Kesimpulan**

Skenario 1 menghasilkan MAE terkecil sebesar 0.0889 °C. Skenario 2 menghasilkan MAE sebesar 0.1673 °C, sedangkan skenario 3 menghasilkan MAE terbesar sebesar 1.0444 °C. Berdasarkan hasil eksperimen, dataset asli memberikan prediksi paling dekat dengan nilai aktual pada empat sampel testing yang digunakan. Pengujian ini juga menunjukkan bahwa pengurangan data dan keberadaan outlier dapat memengaruhi kinerja regresi linear.
## 9. Ringkasan: Apa yang benar-benar dijalankan di ESP32?

| Bagian | Eksperimen A | Eksperimen B–E |
|---|---|---|
| Dataset training di ESP32 | Tidak diperlukan | Ya |
| Menghitung `b0` dan `b1` di ESP32 | Tidak | Ya |
| Inference/prediksi di ESP32 | Ya | Ya |
| Evaluasi pada test set | Tidak | Ya (C dan E) |
| Sensor fisik | Tidak | Tidak |
| Library machine learning | Tidak | Tidak |

**Catatan ilmiah:** Praktikum ini mendemonstrasikan *on-device training* skala kecil dan *on-device inference* pada ESP32. Ini adalah pengantar *embedded ML/TinyML*. Tidak berarti metode OLS dengan data besar atau semua jenis training ML cocok dilakukan pada ESP32.

## 10. Laporan singkat mahasiswa

Kumpulkan laporan 1–2 halaman yang memuat:

1. Foto/screenshot keluaran Thonny dari eksperimen B, C, dan D.

2. Persamaan regresi hasil training ESP32 serta MAE testing.
3. Tabel hasil eksperimen E.
4. Jawaban tiga pertanyaan refleksi berikut:
   - Apa perbedaan training dan inference dalam praktikum ini?
   - Apakah sebuah prediksi tetap bisa dihitung meskipun berada di luar rentang training? Mengapa prediksi itu perlu diwaspadai?
   - Apa keuntungan dan keterbatasan melakukan training model sederhana langsung di ESP32 dibanding hanya melakukan inference?

## 11. Kriteria keberhasilan

- [ ] ESP32 menjalankan prediksi menggunakan koefisien tetap.
- [ ] ESP32 berhasil menghitung slope dan intercept dari data training.
- [ ] ESP32 menghasilkan MAE menggunakan data testing yang terpisah.
- [ ] Mahasiswa mencoba input interaktif dan mengidentifikasi ekstrapolasi.
- [ ] Mahasiswa membandingkan hasil tiga skenario dataset.

---

