'''
Analisis Saham Menggunakan Model Bayesian Hierarchical dengan Bambi.

File ini merupakan langkah terakhir dalam siri analisis data saham. Ia menggunakan
model Bayesian Hierarchical yang dibangunkan dengan Bambi untuk meramal harga saham
dan memberikan cadangan pembelian berdasarkan ramalan tersebut.

Langkah-langkah utama yang dilakukan:
1. Meminta input tahun, bulan, draw_tune, dan target_accept untuk ramalan.
2. Memuat data saham dari file HTML yang telah diproses sebelumnya menggunakan
   modul 'pelombong'.
3. Melakukan penskalaan data harga saham menggunakan StandardScaler.
4. Membangun dan melatih model Bayesian Hierarchical menggunakan Bambi.
5. Meringkas hasil penemuan model menggunakan ArviZ.
6. Menentukan saham dengan tren naik dan turun berdasarkan ringkasan model.
7. Meramal harga saham untuk tahun dan bulan yang ditentukan.
8. Mengembalikan skala harga ramalan ke skala asal.
9. Menghitung harga beli yang disarankan.
10. Merekodkan harga semasa saham untuk perbandingan.
11. Menambahkan maklumat tren tahunan dan bulanan saham.
12. Mencetak hasil analisis dalam format jadual menggunakan tabulate.
13. Menyenaraikan saham dengan harga di bawah harga berpatutan.

Fungsi dan modul yang digunakan:
- bambi: Membangun dan melatih model Bayesian Hierarchical.
- arviz: Meringkas hasil pembinaan model.
- ast: Mengubah string literal dari file .env menjadi dictionary.
- dotenv: Memuatkan variabel lingkungan dari file Bursa.env.
- sklearn.preprocessing.StandardScaler: Melakukan penskalaan data.
- tabulate: Mencetak data dalam format jadual.
- pelombongan.pelombong.dapatkan_data_saham: Memuat data saham dari file HTML.

Catatan:
- Kamus 'ticker' di awal file perlu dikemaskini dengan kamus yang dihasilkan dari
file 'melombong_data.py'.

Output:
- Jadual "Analisis Keseluruhan" yang berisi nama saham, ticker, tahun, bulan, harga ramalan,
  batas bawah, batas atas, harga beli, harga semasa, serta tren tahunan dan bulanan saham.
- Jadual "Peluang" yang berisi saham-saham dengan harga semasa di bawah batas bawah ramalan.
'''

import arviz as az
import ast
import bambi as bmb
import numpy as np
import os
import pandas as pd

from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate


from pelombongan import pelombong


if __name__ == "__main__":
    tahun: int = input("   Tahun untuk diramal = ")
    bulan: int = input("   Bulan untuk diramal = ")
    draw_tune: int = eval(input("   Bilangan draw dan tune (biasanya 4000) = "))
    target_accept: float = eval(input("   Nilai target_accept (biasanya 0.95) = "))

    load_dotenv("bursa.env")
    ticker: dict = ast.literal_eval(os.environ['ticker'])

    data: pd.DataFrame = pelombong.dapatkan_data_saham(ticker)

# Menyimpan harga saham terkini untuk kegunaan seterusnya
    df_semasa: pd.DataFrame = data[data["Date"]==data["Date"].max()][["Ticker", "Close"]]
    harga_semasa: dict = dict(zip(df_semasa["Ticker"], df_semasa["Close"]))

# Melaraskan data Close kepada skala piawai
    kamus_penskala: dict = dict()

    for saham, kumpulan in data.groupby("Ticker", observed=False):
        penskala = StandardScaler()
        kamus_penskala[saham] = penskala
        data.loc[kumpulan.index, "harga_piawai"] = penskala.fit_transform(kumpulan[["Close"]])


# Membina dan melatih model
    model = bmb.Model(
        formula="harga_piawai ~ 1 + (1|Ticker) + (bulan | tahun : Ticker)",
        data=data,
        noncentered=False,
    )

    idata= model.fit(draws=draw_tune, tune=draw_tune, cores=4, target_accept=target_accept)

# Meringkaskan penemuan daripada model
    ringkasan = az.summary(idata)

# Menentukan tren tahunan dan bulanan saham
    tahun_positif = (ringkasan[(ringkasan["hdi_3%"] > 0)
        & (ringkasan.index.str.startswith("1"))
        & (ringkasan.index.str.contains(tahun))
    ])
    tahun_ini_naik = [x.split(":")[-1].strip("]") for x in tahun_positif.index]

    tahun_negatif = (ringkasan[(ringkasan["hdi_97%"] < 0)
        & (ringkasan.index.str.startswith("1"))
        & (ringkasan.index.str.contains(tahun))
    ])
    tahun_ini_jatuh = [x.split(":")[-1].strip("]") for x in tahun_negatif.index]

    bulan_positif = (ringkasan[(ringkasan["hdi_3%"] > 0)
        & (ringkasan.index.str.startswith("bulan"))
        & (ringkasan.index.str.contains(tahun))
    ])
    bulan_naik = [x.split(":")[-1].strip("]") for x in bulan_positif.index]

    bulan_negatif = (ringkasan[(ringkasan["hdi_97%"] < 0)
        & (ringkasan.index.str.startswith("bulan"))
        & (ringkasan.index.str.contains(tahun))
    ])
    bulan_jatuh = [x.split(":")[-1].strip("]") for x in bulan_negatif.index]

# Menghasilkan ramalan saham mengikut model
    ramalan: pd.DataFrame = bmb.interpret.predictions(
        model,
        idata,
        ["tahun", "bulan", "Ticker"]
    )

# Mengehadkan paparan ramalan kepada bulan dan tahun yang dikehendaki
    ramalan = ramalan[(ramalan["tahun"]==eval(tahun)) & (ramalan["bulan"]==eval(bulan))]

# Menambah rekod nama saham sebagai rujukan
    ramalan["nama"] = ramalan["Ticker"].map(ticker)

# Melaraskan harga saham kembali kepada skala asal
    for saham, kumpulan in ramalan.groupby("Ticker", observed=False):
        penskala = kamus_penskala[saham]
        ramalan.loc[kumpulan.index, ["harga_ramalan", "bawah", "atas"]] = (
            penskala.inverse_transform(
                kumpulan[["estimate", "lower_3.0%", "upper_97.0%"]]
            ))

# Menentukan harga beli di mana harga ramalan adalah 1.05 x harga beli    
    ramalan["harga_beli"] = ramalan["harga_ramalan"] / 1.05

# Memaparkan tren tahunan dan bulanan saham
    ramalan["year_up"] = np.where(
        ramalan["Ticker"].isin(tahun_ini_naik), "naik", "-"
    )

    ramalan["year_down"] = np.where(
        ramalan["Ticker"].isin(tahun_ini_jatuh), "jatuh", "-"
    )

    ramalan["tren_tahun"] = (ramalan["year_up"]
                        + "_"
                        + ramalan["year_down"]).str.strip("-")
    
    ramalan["month_up"] = np.where(
        ramalan["Ticker"].isin(bulan_naik), "naik", "-"
    )

    ramalan["month_down"] = np.where(
        ramalan["Ticker"].isin(bulan_jatuh), "jatuh", "-"
    )

    ramalan["tren_bulan"] = (ramalan["month_up"]
                        + "_"
                        + ramalan["month_down"]).str.strip("-")

# Harga semasa menjadi type string selepas operasi zip di atas
# Menukarkan type harga semasa kembali kepada float64
    ramalan["harga_semasa"] = ramalan["Ticker"].map(harga_semasa).astype("float64")

# Mengatur semula jadual ramalan
    ramalan = ramalan[[
        "nama",
        "Ticker",
        "tahun",
        "bulan",
        "harga_ramalan",
        "bawah",
        "atas",
        "harga_beli",
        "harga_semasa",
        "tren_tahun",
        "tren_bulan",
    ]]

# Memaparkan jadual ramalan dalam format yang senang dibaca
    print("")
    print("-"*100)
    print("Analisis Keseluruhan")
    print(tabulate(
        ramalan,
        headers="keys",
        showindex=False,
        tablefmt="fancy_grid",
        floatfmt=".3f",
        stralign="center",
    ))

# Mengatur jadual peluang sebagai saham yang berpotensi untuk dibeli
    peluang: pd.DataFrame = ramalan[ramalan["harga_semasa"] < ramalan["bawah"]]

# Memaparkan jadual peluang dalam format yang senang dibaca
    print("")
    print("-"*100)
    print("Peluang")
    print(tabulate(
        peluang,
        headers="keys",
        showindex=False,
        tablefmt="fancy_grid",
        floatfmt=".3f",
        stralign="center",
    ))