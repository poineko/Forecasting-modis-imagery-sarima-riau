# Peramalan Suhu Permukaan Darat (LST) di Wilayah Riau Menggunakan Data Satelit MODIS dan Metode SARIMA

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Dash](https://img.shields.io/badge/Plotly_Dash-3.0%2B-008DE4?logo=plotly&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Time_Series-SARIMA-0D9488)
![NASA MODIS](https://img.shields.io/badge/Dataset-NASA_MODIS_MOD11C3-EA580C)
![Accuracy](https://img.shields.io/badge/Test_MAPE-%3C_3%25_(Sangat_Akurat)-16A34A)


Sistem analitik geospasial *end-to-end* dan *dashboard* web interaktif untuk memodelkan serta meramalkan tren **Suhu Permukaan Darat (*Land Surface Temperature* / LST)** pada **12 kabupaten/kota di Provinsi Riau** menggunakan arsip citra satelit **NASA MODIS** (2000–2020) dan pemodelan deret waktu **SARIMA (*Seasonal Autoregressive Integrated Moving Average*)**.

<video src="https://github.com/poineko/Forecasting-modis-imagery-sarima-riau/blob/master/demo-sarima.mp4?raw=true" controls width="100%"></video>
---

## Daftar Isi
1. [Latar Belakang & Tujuan Penelitian](#1-latar-belakang--tujuan-penelitian)
2. [Arsitektur Alur Kerja (End-to-End Pipeline)](#2-arsitektur-alur-kerja-end-to-end-pipeline)
3. [Sumber Data & Pra-pemrosesan Geospasial](#3-sumber-data--pra-pemrosesan-geospasial)
4. [Formulasi Matematis & Metodologi SARIMA](#4-formulasi-matematis--metodologi-sarima)
5. [Hasil Evaluasi Model & Analisis Spasial-Temporal](#5-hasil-evaluasi-model--analisis-spasial-temporal)
6. [Fitur Aplikasi Web Interaktif (`Sistem Peramalan Sarima`)](#6-fitur-aplikasi-web-interaktif-sistem-peramalan-sarima)
7. [Struktur Repositori](#7-struktur-repositori)
8. [Panduan Instalasi & Penggunaan](#8-panduan-instalasi--penggunaan)
9. [Stack Teknologi](#9-stack-teknologi)

---

## 1. Latar Belakang & Tujuan Penelitian

### Permasalahan
Perubahan iklim global meningkatkan urgensi pemantauan suhu permukaan darat (*Land Surface Temperature* / LST) sebagai salah satu indikator utama iklim. Provinsi Riau—yang didominasi ekosistem dataran rendah dan lahan gambut luas—sangat rentan terhadap anomali pemanasan permukaan yang berkorelasi erat dengan kekeringan serta kebakaran hutan dan lahan (karhutla). Namun, pemantauan suhu permukaan secara komprehensif di seluruh wilayah administratif mengalami kendala:
- **Keterbatasan Spasial Stasiun Terestrial:** Stasiun pengamatan cuaca permukaan (BMKG) tidak tersebar merata di seluruh kecamatan maupun area pedalaman/gambut.
- **Kebutuhan Proyeksi Dini:** Pemangku kebijakan memerlukan estimasi tren suhu jangka menengah hingga panjang beserta batas ketidakpastiannya (*confidence interval*) untuk perencanaan strategi adaptasi dan mitigasi iklim.

### Tujuan
1. Mengonversi dan mengagregasi data citra satelit termal **NASA MODIS** menjadi deret waktu suhu permukaan bulanan yang tervalidasi pada **12 wilayah administratif kabupaten/kota di Provinsi Riau**.
2. Membangun model peramalan musiman **SARIMA$(p,d,q)(P,D,Q)_{12}$** yang optimal untuk masing-masing karakteristik iklim mikro wilayah.
3. Mengembangkan **sistem visualisasi interaktif berbasis web** agar peneliti maupun pemangku kepentingan dapat melakukan eksplorasi data, uji stasioneritas, diagnostik model, hingga simulasi peramalan secara mandiri.

---

## 2. Arsitektur Alur Kerja (*End-to-End Pipeline*)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    1. AKUISISI & PRA-PEMROSESAN SPASIAL                  │
│  NASA MODIS MOD11C3 (HDF/NetCDF, 2000–2023)                             │
│  ──► Ekstraksi Band LST_Day_CMG ──► Konversi Kelvin ke Celcius (°C)     │
│  ──► Clipping Shapefile 12 Kab/Kota Riau ──► Agregasi Rata-rata Bulanan │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 2. EKSPLORASI & UJI STASIONERITAS DERET WAKTU            │
│  Split Data: Train (Feb 2000–Des 2020) | Test (Jan 2021–Jul 2023)       │
│  ──► Uji Augmented Dickey-Fuller (ADF) ──► Differencing (d=1, D=1, s=12)│
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│             3. IDENTIFIKASI, ESTIMASI & DIAGNOSTIK MODEL SARIMA          │
│  Analisis Korelogram ACF & PACF ──► Estimasi MLE Kandidat Model         │
│  ──► Seleksi AIC Terkecil ──► Uji White Noise (Ljung-Box) & Normalitas  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              4. EVALUASI AKURASI & VISUALISASI INTERAKTIF WEB            │
│  Evaluasi Out-of-Sample (MAE, MSE, RMSE, MAPE < 3%)                     │
│  ──► Dashboard Multi-Page Plotly Dash (Proyeksi + 95% Confidence Band)  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Sumber Data & Pra-pemrosesan Geospasial

| Atribut Data | Spesifikasi |
| :--- | :--- |
| **Produk Satelit** | NASA MODIS Terra/Aqua Land Surface Temperature & Emissivity Monthly (**MOD11C3** v061) |
| **Resolusi Spasial** | Climate Modeling Grid (CMG) $0,05^\circ \times 0,05^\circ$ ($\approx 5,6\text{ km} \times 5,6\text{ km}$) |
| **Parameter Utama** | `LST_Day_CMG` (Suhu Permukaan Darat Siang Hari) |
| **Formula Konversi** | $T(^\circ\text{C}) = (\text{DN} \times 0,02) - 273,15$ |
| **Rentang Temporal** | Februari 2000 – Juli 2023 (**282 titik observasi bulanan** per wilayah) |
| **Pembagian Data** | **Data Latih:** Feb 2000 – Des 2020 (251 bulan / 89%)<br>**Data Uji:** Jan 2021 – Jul 2023 (31 bulan / 11%) |
| **Cakupan Wilayah** | **12 Kabupaten/Kota di Provinsi Riau:** Kota Pekanbaru, Kota Dumai, Kab. Kampar, Kab. Siak, Kab. Bengkalis, Kab. Pelalawan, Kab. Indragiri Hulu, Kab. Indragiri Hilir, Kab. Kuantan Singingi, Kab. Rokan Hulu, Kab. Rokan Hilir, dan Kab. Kepulauan Meranti. |

---

## 4. Formulasi Matematis & Metodologi SARIMA

Karena suhu permukaan darat memiliki pola fluktuasi tahunan yang berulang setiap 12 bulan ($s = 12$), digunakan model **SARIMA$(p, d, q)(P, D, Q)_{12}$** yang dirumuskan sebagai:

$$\Phi_P(B^{12}) \phi_p(B) (1 - B)^d (1 - B^{12})^D Y_t = \Theta_Q(B^{12}) \theta_q(B) \varepsilon_t$$

Keterangan:
- $Y_t$: Nilai suhu permukaan darat ($^\circ\text{C}$) pada bulan ke-$t$, dan $B$: operator *backshift* ($B Y_t = Y_{t-1}$).
- $(p, d, q)$: Orde komponen non-musiman (*Autoregressive*, *Differencing*, *Moving Average*).
- $(P, D, Q)_{12}$: Orde komponen musiman dengan periode $s = 12$ bulan.
- $\varepsilon_t$: Komponen galat (*white noise*) yang berdistribusi normal independen $\mathcal{N}(0, \sigma^2)$.

### Tahapan Pemodelan Box-Jenkins
1. **Uji Stasioneritas (ADF Test):** Memastikan deret waktu stasioner terhadap rata-rata ($p\text{-value} < 0,05$) melalui *differencing* non-musiman ($d$) dan musiman ($D$).
2. **Identifikasi Orde (ACF & PACF):** Mengamati lag signifikan pada korelogram untuk menentukan kandidat parameter $(p,q)$ pada lag awal serta $(P,Q)$ pada lag kelipatan musiman ($12, 24, 36$).
3. **Estimasi Parameter & Seleksi Model:** Mengestimasi parameter menggunakan *Maximum Likelihood Estimation* (MLE) dan memilih model dengan **AIC (*Akaike Information Criterion*)** minimum.
4. **Verifikasi Diagnostik Residual:**
   - **Uji Ljung-Box:** Memastikan tidak ada autokorelasi tersisa pada residual ($p\text{-value} > 0,05$).
   - **Normalitas Residual:** Memeriksa sebaran galat menggunakan *Quantile-Quantile (Q-Q) Plot*.

---

## 5. Hasil Evaluasi Model & Analisis Spasial-Temporal

### Perbandingan Model Terbaik & Akurasi Data Uji (2021–2023)
Seluruh model wilayah mencapai tingkat akurasi tinggi dengan nilai **MAPE (*Mean Absolute Percentage Error*) di bawah 3%**:

| No | Kabupaten / Kota | Karakteristik Geografis | Model Optimal $(p,d,q)(P,D,Q)_{12}$ | Test MAPE (%) | Kategori Akurasi |
| :-: | :--- | :--- | :--- | :-: | :-: |
| 1 | **Kab. Indragiri Hilir** | Pesisir & Estuari | $\text{SARIMA}(1,1,1)(1,1,1)_{12}$ | **1,35%** | Sangat Akurat |
| 2 | **Kab. Kepulauan Meranti** | Kepulauan & Gambut Pesisir | $\text{SARIMA}(1,1,1)(0,1,1)_{12}$ | **1,74%** | Sangat Akurat |
| 3 | **Kab. Kampar** | Perbukitan & Dataran Tengah | $\text{SARIMA}(1,1,1)(1,1,1)_{12}$ | **1,86%** | Sangat Akurat |
| 4 | **Kota Pekanbaru** | Kawasan Perkotaan (*Urban*) | $\text{SARIMA}(1,1,1)(1,1,1)_{12}$ | **1,92%** | Sangat Akurat |
| 5 | **Kab. Indragiri Hulu** | Dataran Rendah & Perbukitan | $\text{SARIMA}(1,1,1)(1,1,1)_{12}$ | **1,98%** | Sangat Akurat |
| 6 | **Kab. Pelalawan** | Ekosistem Gambut Tengah | $\text{SARIMA}(1,1,1)(1,1,1)_{12}$ | **2,04%** | Sangat Akurat |
| 7 | **Kab. Kuantan Singingi** | Perbukitan Barat Daya | $\text{SARIMA}(1,1,1)(1,1,1)_{12}$ | **2,07%** | Sangat Akurat |
| 8 | **Kab. Siak** | Dataran Rendah & Gambut | $\text{SARIMA}(1,1,1)(0,1,1)_{12}$ | **2,11%** | Sangat Akurat |
| 9 | **Kab. Bengkalis** | Pesisir & Pulau Gambut | $\text{SARIMA}(1,1,1)(1,1,1)_{12}$ | **2,15%** | Sangat Akurat |
| 10 | **Kab. Rokan Hulu** | Perbukitan Barat Laut | $\text{SARIMA}(1,1,1)(1,1,1)_{12}$ | **2,19%** | Sangat Akurat |
| 11 | **Kab. Rokan Hilir** | Pesisir Utara & Gambut | $\text{SARIMA}(1,1,1)(1,1,1)_{12}$ | **2,34%** | Sangat Akurat |
| 12 | **Kota Dumai** | Pesisir Industri & Perkotaan | $\text{SARIMA}(1,1,1)(1,1,1)_{12}$ | **2,48%** | Sangat Akurat |

### Temuan Utama
- **Pola Musiman Konsisten:** Puncak suhu permukaan darat di wilayah Riau secara konsisten terjadi pada periode musim kemarau (**Mei–September**), sedangkan suhu terendah terjadi pada puncak musim penghujan (**November–Januari**).
- **Variasi Spasial:** Wilayah dengan tutupan lahan terbangun dan area industri pesisir (seperti **Kota Pekanbaru** dan **Kota Dumai**) menunjukkan rata-rata baseline suhu permukaan yang lebih tinggi serta amplitudo fluktuasi yang lebih tajam dibanding wilayah pesisir estuari (**Indragiri Hilir**).

---

## 6. Fitur Aplikasi Web Interaktif (`Sistem Peramalan Sarima`)

Aplikasi web dibangun menggunakan arsitektur *multi-page* **Plotly Dash** yang memandu pengguna menjalankan seluruh siklus pemodelan Box-Jenkins secara interaktif:

| Tahap / Halaman | Route | Fungsionalitas Utama |
| :--- | :--- | :--- |
| **Beranda** | `/` | Ringkasan arsitektur sistem, latar belakang penelitian, dan panduan alur penggunaan. |
| **1-Pemilihan Data Wilayah** | `/step1` | Seleksi interaktif 12 kabupaten/kota di Riau serta inspeksi grafik deret waktu historis (2000–2023). |
| **2-Stationarity** | `/step2` | Komputasi otomatis statistik uji **ADF** (*ADF Stat* & *p-value*) dan simulasi transformasi *differencing* (orde 0–5) secara *real-time*. |
| **3-Identifikasi Model** | `/step3` | Visualisasi korelogram **ACF** dan **PACF** interaktif dengan kontrol jumlah *lag* dinamis dan garis batas signifikansi 95% ($\pm 1,96/\sqrt{n}$). |
| **4-Estimasi Parameter** | `/step4` | Pelatihan dan komparasi hingga **3 kandidat model SARIMA** sekaligus, lengkap dengan ringkasan koefisien, nilai **AIC**, uji autokorelasi residual **Ljung-Box**, **Q-Q Plot**, serta tabel evaluasi **MAE, MSE, dan RMSE**. |
| **5-Peramalan** | `/step5` | Simulasi proyeksi suhu permukaan masa depan (skala bulanan maupun tahunan) lengkap dengan pita **95% Confidence Interval** serta tabel data hasil prediksi yang dapat dieksplorasi. |

---

## 7. Struktur Repositori

```text
Forecasting-modis-imagery-sarima-riau/
├── Sistem Peramalan Sarima/           # Aplikasi Web Interaktif (Plotly Dash)
│   ├── app.py                         # Entry point utama & konfigurasi router multi-page
│   ├── pages/                         # Modul halaman setiap tahapan metodologi SARIMA
│   │   ├── home.py                    # Halaman beranda & dokumentasi singkat
│   │   ├── step1.py                   # Pemilihan wilayah & eksplorasi deret waktu historis
│   │   ├── step2.py                   # Uji stasioneritas (ADF) & transformasi differencing
│   │   ├── step3.py                   # Identifikasi orde parameter (ACF & PACF)
│   │   ├── step4.py                   # Estimasi parameter & diagnostik residual
│   │   └── step5.py                   # Peramalan akhir, interval kepercayaan & tabel output
│   ├── assets/                        # Styling tema gelap (CSS kustom) & layout grafik Plotly
│   │   ├── fig_layout.py              # Konfigurasi standar visualisasi grafik
│   │   ├── acf_pacf_plots.py          # Generator grafik korelogram residual
│   │   └── fix_dark_dropdown.css      # Penyesuaian tema gelap untuk Dash 3+
│   ├── data/                          # Dataset rata-rata LST bulanan Riau (2000–2023)
│   │   └── Suhu_Permukaan_Riau.xlsx   # Data teragregasi 12 kabupaten/kota
│   └── requirements.txt               # Daftar dependensi Python
└── README.md                          # Dokumentasi utama proyek
```

---

## 8. Panduan Instalasi & Penggunaan

### Prasyarat
- **Python 3.10 atau lebih baru** (teruji pada Python 3.10 – 3.14)
- **pip** (*Python package manager*)

### Langkah Menjalankan Secara Lokal

1. **Clone repositori ini:**
   ```bash
   git clone [https://github.com/poineko/Forecasting-modis-imagery-sarima-riau.git](https://github.com/poineko/Forecasting-modis-imagery-sarima-riau.git)
   cd "Forecasting-modis-imagery-sarima-riau/Sistem Peramalan Sarima"
   ```

2. **Buat dan aktifkan *virtual environment*:**
   ```bash
   python -m venv venv

   # Linux / macOS / WSL:
   source venv/bin/activate

   # Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   ```

3. **Install seluruh dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan server aplikasi Dash:**
   ```bash
   python app.py
   ```

5. **Akses aplikasi di browser:**
   Buka `http://127.0.0.1:8050` untuk mulai menggunakan *dashboard* peramalan.

---

## 9. Stack Teknologi

| Kategori | Teknologi & Pustaka |
| :--- | :--- |
| **Bahasa Pemrograman** | Python 3 |
| **Pengolahan Data & Geospasial** | Pandas, NumPy, Xarray, Rasterio, GeoPandas, OpenPyXL |
| **Pemodelan Statistik & Deret Waktu** | Statsmodels (`SARIMAX`, `adfuller`, `acorr_ljungbox`), Scikit-Learn, SciPy |
| **Antarmuka Web & Visualisasi** | Plotly Dash, Dash Bootstrap Components, Plotly Graph Objects |

---

## Kontribusi & Sitasi

Jika repositori atau metodologi ini bermanfaat untuk penelitian maupun pengembangan sistem pemantauan iklim Anda, silakan berikan bintang (*star*) pada repositori ini atau cantumkan sitasi berikut:

> **Riawi, M. I. A.** (2024). *Peramalan Suhu Permukaan Darat di Wilayah Riau Menggunakan Data Satelit MODIS dengan Metode SARIMA*. Program Studi Teknik Informatika, Universitas Islam Riau.
