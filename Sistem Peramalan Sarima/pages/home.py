import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Home', title='SARIMA | Home')

layout = dbc.Container([
    # Title Section
    dbc.Row([
        dbc.Col([
            html.H3('Selamat Datang'),
            html.P(html.B('App Overview'), className='par')
        ], width=12, className='row-titles')
    ]),
    
    # Guidelines Section
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([
            # Step 1: Pemilihan Data Wilayah
            html.P([
                html.B('1) Pemilihan Data Wilayah'),
                html.Br(),
                'Pada tahap ini, pengguna memilih wilayah di Provinsi Riau untuk melihat data suhu permukaan. Prosesnya meliputi beberapa langkah sebagai berikut:',
                html.Br(), html.Br(),
                'a) Data Suhu Permukaan Riau: Data ini diambil dari file Excel berisi suhu rata-rata bulanan untuk kota dan kabupaten di Riau, termasuk wilayah seperti Bengkalis, Indragiri Hilir, Pekanbaru, dan lainnya. Setiap baris data mencakup informasi Tahun, Bulan, dan suhu rata-rata di wilayah tersebut.',
                html.Br(), html.Br(),
                'b) Pemeriksaan Nilai Kosong: Sistem memeriksa kolom Tahun dan Bulan untuk memastikan tidak ada nilai kosong (missing values) yang dapat menyebabkan kesalahan dalam pemrosesan data. Jika ditemukan nilai kosong, sistem akan menghentikan proses dan menampilkan pesan peringatan.',
                html.Br(), html.Br(),
                'c) Penggabungan Kolom Tahun dan Bulan: Data dalam kolom Tahun dan Bulan digabungkan menjadi kolom baru bernama "Tanggal". Tanggal ini diatur sebagai format tanggal dengan format YYYY-MM-DD dan diset sebagai indeks untuk memungkinkan visualisasi berkelanjutan berdasarkan waktu.',
                html.Br(), html.Br(),
                'd) Pemilihan Wilayah: Pengguna dapat memilih wilayah tertentu menggunakan opsi radio button. Setiap pilihan pada radio button akan memperbarui grafik suhu rata-rata untuk wilayah yang dipilih.',
                html.Br(), html.Br(),
                'e) Penyimpanan Data Terpilih: Data dari wilayah yang dipilih disimpan dalam format JSON untuk memudahkan akses di halaman lain atau untuk analisis lebih lanjut.'
            ], className='guide'),

            # Step 2: Stationarity
            html.P([
                html.B('2) Stationarity'),
                html.Br(),
                'Pada tahap ini, dilakukan pengecekan ke-stasioneran data suhu permukaan untuk wilayah yang dipilih. Stasioneritas data diperlukan agar data memenuhi asumsi ARIMA atau SARIMA. Langkah-langkah yang dilakukan pada tahap ini adalah sebagai berikut:',
                html.Br(), html.Br(),
                'a) Uji Augmented Dickey-Fuller (ADF): Data suhu permukaan yang telah dipilih melalui tahap sebelumnya diambil, lalu dilakukan uji ADF untuk memeriksa apakah data stasioner atau tidak.',
                html.Br(),
                'b) Plot Data Asli: Grafik garis dari data suhu permukaan asli ditampilkan untuk membantu pengguna memahami pola data sebelum dilakukan differencing.',
                html.Br(),
                'c) Differencing: Jika data tidak stasioner, pengguna dapat memilih tingkat differencing menggunakan dropdown untuk membuat data menjadi stasioner.'
            ], className='guide'),

            # Step 3: Identifikasi Model
            html.P([
                html.B('3) Identifikasi Model'),
                html.Br(),
                'Pada tahap ini, identifikasi model dilakukan melalui analisis ACF dan PACF. Proses identifikasi ini bertujuan untuk menentukan komponen AR, MA, SAR, dan SMA dalam model SARIMA.',
                html.Br(), html.Br(),
                'a) Plot ACF dan PACF: Data suhu permukaan yang telah melalui tahap differencing digunakan sebagai input dalam analisis ini.',
                html.Br(),
                'b) Garis Signifikansi: Garis horizontal ditambahkan pada plot ACF dan PACF sebagai garis signifikansi.',
                html.Br(),
                'c) Interpretasi Plot: Berdasarkan teori dasar ARIMA/SARIMA yang telah ditetapkan, pengguna dapat menentukan nilai p, q, P, dan Q sesuai pola yang teridentifikasi.'
            ], className='guide'),

            # Step 4: Estimasi Parameter
            html.P([
                html.B('4) Estimasi Parameter'),
                html.Br(),
                'Pada tahap ini, dilakukan seleksi model SARIMA berdasarkan parameter yang dipilih pengguna dan data suhu permukaan yang telah diolah. Proses estimasi parameter meliputi:',
                html.Br(), html.Br(),
                
                'a) Seleksi Tahun Awal dan Akhir: Pengguna memilih rentang waktu (tahun awal dan akhir) untuk subset data suhu yang akan digunakan dalam proses pemodelan.',
                html.Br(), html.Br(),
                
                'b) Pemilihan Parameter Model: Pengguna dapat memilih nilai parameter berikut untuk tiga model SARIMA yang akan dibandingkan:',
                html.Br(),
                html.Ul([
                    html.Li('p (Autoregressive Order)'),
                    html.Li('d (Differencing Order)'),
                    html.Li('q (Moving Average Order)'),
                    html.Li('P (Seasonal Autoregressive Order)'),
                    html.Li('D (Seasonal Differencing Order)'),
                    html.Li('Q (Seasonal Moving Average Order)'),
                    html.Li('s (Seasonal Period, default = 12)'),
                ]),
                html.Br(),
                
                'c) Estimasi Model: Setelah parameter dipilih, pengguna dapat menjalankan estimasi model SARIMA dengan menekan tombol "Estimate Parameters". Hasil estimasi mencakup:',
                html.Br(),
                html.Ul([
                    html.Li('Ringkasan model SARIMA untuk setiap kombinasi parameter.'),
                    html.Li('Nilai kriteria evaluasi seperti AIC dan BIC untuk membandingkan model.'),
                    html.Li('Residuals (selisih antara data asli dan data prediksi).'),
                    html.Li('Fitted values (nilai prediksi yang sesuai dengan data training).'),
                ]),
                html.Br(),

                'd) Diagnostik Model: Setelah estimasi model, pengguna dapat melakukan pengecekan diagnostik dengan menekan tombol "Check Diagnostic". Analisis diagnostik meliputi:',
                html.Br(),
                html.Ul([
                    html.Li('Plot residuals untuk memeriksa asumsi normalitas.'),
                    html.Li('Perhitungan metrik diagnostik seperti p-value Ljung-Box.'),
                    html.Li('Tabel ringkasan metrik diagnostik untuk setiap model.'),
                ]),
                html.Br(),
                
                'e) Output Estimasi: Hasil akhir dari tahap ini mencakup teks ringkasan model, tabel metrik diagnostik, serta grafik residuals dan nilai fitted untuk memvisualisasikan hasil estimasi parameter.'
            ], className='guide'),


            # Step 5: Peramalan
            html.P([
                html.B('5) Peramalan'),
                html.Br(),
                'Pada tahap ini, dilakukan peramalan menggunakan model SARIMA yang telah difit dengan parameter yang dipilih sebelumnya.',
                html.Br(), 'Pengguna dapat memilih rentang tahun (mulai dan akhir) untuk peramalan serta nilai parameter SARIMA seperti p, d, q, P, D, dan Q.',
                html.Br(), 'Model SARIMA digunakan untuk memprediksi suhu rata-rata (°C) untuk periode waktu yang ditentukan.',
                html.Br(), 'Data yang digunakan untuk peramalan adalah data suhu permukaan yang telah diproses dan diperoleh dari proses differencing untuk memastikan kestasioneran.',
                html.Br(), 'Setelah pemodelan, hasil peramalan akan ditampilkan bersama dengan interval kepercayaan (CI) 95% untuk memvisualisasikan rentang peramalan yang diharapkan.',
                html.Br(), 'Pengguna juga dapat melihat grafik data asli dan hasil peramalan yang menunjukkan perbandingan antara data asli, data uji, dan hasil prediksi.',
                html.Br(), 'Selain itu, tabel hasil peramalan akan ditampilkan, yang mencakup nilai-nilai yang diprediksi untuk setiap periode waktu yang telah dipilih.'
            ], className='guide'),
        ], width=8),
        dbc.Col([], width=2)
    ])
])
