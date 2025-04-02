# Strategi Bursa 2025

Projek Python ini menganalisis trend harga saham menggunakan regresi hierarki (`Bambi`) dan regresi robust (`RANSAC`), dengan penskalaan data automatik dan kuantifikasi ketidakpastian.

## Ciri-ciri Utama

* **Pemodelan Bayesian:** Menggunakan `Bambi` untuk model hierarki (`Close ~ 1 + (1|Ticker) + (Month|Year:Ticker)`).
* **Ketahanan Terhadap Pencilan:** Melaksanakan `RANSACRegressor` untuk anggaran trend robust.
* **Pemilihan Saham:** Menggunakan `linregress` dari SciPy untuk memilih saham berdasarkan analisis cerun data EPS dan DPS.
* **Pra-pemprosesan Boleh Skala:** Standardisasi per-ticker dengan `sklearn.preprocessing`.
* **Kuantifikasi Ketidakpastian:** Menjana 94% HDI untuk ramalan.
* **Pengikisan Data Automatik:** PlayWright
* **Pemprosesan Berbilang:** Pool dan ThreadPoolExecutor

## Struktur Projek
project_structure.txt

## Kebergantungan
### requirements.txt
* arviz==0.21.0
* bambi==0.15.0
* beautifulsoup4==4.13.3
* numpy==2.2.4
* pandas==2.2.3
* playwright==1.51.0
* scikit_learn==1.6.1
* scipy==1.15.2
* tabulate==0.9.0
* yfinance==0.2.55

## Pemasangan

1.  Klon repositori: `git clone https://github.com/sithsenz/Bursa2025.git`
2.  Buat persekitaran maya: `python3 -m venv myenv`
3.  Aktifkan persekitaran maya: `source myenv/bin/activate`
4.  Pasang kebergantungan: `pip install -r requirements.txt`
5.  Pasang kebergantungan sistem untuk Playwright:
    ```bash
    sudo apt-get install -y python3-dev libxml2-dev libxslt-dev
    playwright install --with-deps
    ```

## Penggunaan

1.  Jalankan `menyimpan_laman_htm.py` untuk memuat turun data laman web.
2.  Jalankan `melombong_data.py` untuk menganalisis data saham.
3.  Jalankan `menilai_saham.py` untuk membuat ramalan saham.

## Sumber Data

Data diperoleh dari KLSEScreener dan Yahoo Finance.

## Nota Tambahan

* Kamus ticker dalam `menilai_saham.py` perlu dikemas kini dengan kamus yang dihasilkan dari `melombong_data.py`.
* Gunakan persekitaran maya untuk mengelakkan konflik kebergantungan.
* Pastikan anda telah memasang semua kebergantungan sistem untuk Playwright.