'''
Analisis Saham Menggunakan Model Bayesian Hierarchical dengan Bambi.

File ini merupakan langkah terakhir dalam siri analisis data saham. Ia menggunakan
model Bayesian Hierarchical yang dibangunkan dengan Bambi untuk meramal harga saham
dan memberikan cadangan pembelian berdasarkan ramalan tersebut.

Langkah-langkah utama yang dilakukan:
1. Meminta input tahun dan bulan untuk ramalan.
2. Memuat data saham dari file HTML yang telah diproses sebelumnya menggunakan
modul 'pelombong'.
3. Melakukan penskalaan data harga saham menggunakan StandardScaler.
4. Membangun dan melatih model Bayesian Hierarchical menggunakan Bambi.
5. Meringkas hasil pembinaan model menggunakan ArviZ.
6. Menentukan saham dengan tren naik dan turun berdasarkan ringkasan model.
7. Meramal harga saham untuk tahun dan bulan yang ditentukan.
8. Mengembalikan skala harga ramalan ke skala asal.
9. Menghitung harga beli yang disarankan.
10. Menentukan keputusan beli berdasarkan perbandingan harga beli dan batas bawah ramalan.
11. Menambahkan maklumat tren saham.
12. Mencetak hasil analisis dalam format jadual menggunakan tabulate.

Fungsi dan modul yang digunakan:
- bambi: Membangun dan melatih model Bayesian Hierarchical.
- arviz: Meringkas hasil pembinaan model.
- sklearn.preprocessing.StandardScaler: Melakukan penskalaan data.
- tabulate: Mencetak data dalam format jadual.
- pelombongan.pelombong.dapatkan_data_saham: Memuat data saham dari file HTML.

Catatan:
- Kamus 'ticker' di awal file perlu dikemaskini dengan kamus yang dihasilkan dari
file 'melombong_data.py'.

Output:
- Jadual yang berisi nama saham, ticker, tahun, bulan, harga ramalan, batas bawah,
batas atas, harga beli, keputusan beli, dan tren saham.
'''

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from pelombongan import pelombong


if __name__ == "__main__":
    tahun: int = input("   Tahun untuk diramal = ")
    bulan: int = input("   Bulan untuk diramal = ")

    ticker: dict = {'0101.KL': 'TMCLIFE', '6262.KL': 'INNO', '5246.KL': 'WPRTS', '0002.KL': 'KOTRA', '4006.KL': 'ORIENT', '0128.KL': 'FRONTKN', '0032.KL': 'REDTONE', '0106.KL': 'REXIT', '2828.KL': 'CIHLDG', '0001.KL': 'SCOMNET', '0040.KL': 'OPENSYS', '2836.KL': 'CARLSBG', '7100.KL': 'UCHITEC', '6483.KL': 'KENANGA', '0157.KL': 'FOCUSP', '5819.KL': 'HLBANK', '0151.KL': 'KGB', '9296.KL': 'RCECAP', '2658.KL': 'AJI', '1996.KL': 'KRETAM', '8079.KL': 'LEESK', '3255.KL': 'HEIM', '9172.KL': 'FPI', '1066.KL': 'RHBBANK', '3689.KL': 'F&N', '3069.KL': 'MFCB', '5109.KL': 'YTLREIT', '0012.KL': '3A'}

    data: pd.DataFrame = pelombong.dapatkan_data_saham(ticker)

    df_semasa: pd.DataFrame = data[data["Date"]==data["Date"].max()][["Ticker", "Close"]]
    harga_semasa: dict = dict(zip(df_semasa["Ticker"], df_semasa["Close"]))

    kamus_penskala: dict = dict()

    for saham, kumpulan in data.groupby("Ticker", observed=False):
        penskala = StandardScaler()
        kamus_penskala[saham] = penskala
        data.loc[kumpulan.index, "harga_piawai"] = penskala.fit_transform(kumpulan[["Close"]])


    model = bmb.Model(
        formula="harga_piawai ~ 1 + (1|Ticker) + (bulan | tahun : Ticker)",
        data=data,
        noncentered=False,
    )

    idata= model.fit(draws=2000, tune=2000, cores=4, target_accept=0.9)

    ringkasan = az.summary(idata)

    positif = ringkasan[(ringkasan["hdi_3%"] > 0) & (ringkasan.index.str.contains(tahun))]
    tahun_ini_naik = [x.split(":")[-1].strip("]") for x in positif.index]

    negatif = ringkasan[(ringkasan["hdi_97%"] < 0) & (ringkasan.index.str.contains(tahun))]
    tahun_ini_jatuh = [x.split(":")[-1].strip("]") for x in negatif.index]

    ramalan: pd.DataFrame = bmb.interpret.predictions(
        model,
        idata,
        ["tahun", "bulan", "Ticker"]
    )

    ramalan = ramalan[(ramalan["tahun"]==eval(tahun)) & (ramalan["bulan"]==eval(bulan))]

    ramalan["nama"] = ramalan["Ticker"].map(ticker)

    for saham, kumpulan in ramalan.groupby("Ticker", observed=False):
        penskala = kamus_penskala[saham]
        ramalan.loc[kumpulan.index, ["harga_ramalan", "bawah", "atas"]] = (
            penskala.inverse_transform(
                kumpulan[["estimate", "lower_3.0%", "upper_97.0%"]]
            ))
    
    ramalan["harga_beli"] = ramalan["harga_ramalan"] / 1.05

    ramalan["up_trend"] = np.where(
        ramalan["Ticker"].isin(tahun_ini_naik), "naik", "-"
    )

    ramalan["down_trend"] = np.where(
        ramalan["Ticker"].isin(tahun_ini_jatuh), "jatuh", "-"
    )

    ramalan["trend"] = (ramalan["up_trend"]
                        + "_"
                        + ramalan["down_trend"]).str.strip("-")
    
    ramalan["harga_semasa"] = ramalan["Ticker"].map(harga_semasa).astype("float64")

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
        "trend",
    ]]

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

    peluang: pd.DataFrame = ramalan[ramalan["harga_semasa"] < ramalan["bawah"]]

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