'''
Analisis Saham dari Laman HTML yang Dimuat Turun dan Penentuan Saham 'Bagus'.

File ini merupakan langkah kedua dalam siri analisis data saham. Ia memproses file-file
HTML yang telah dimuat turun sebelumnya (dari langkah pertama) yang mengandungi
data kewangan saham. File ini menggunakan teknik pelombongan data dan analisis
statistik untuk menentukan saham-saham yang dianggap 'bagus' berdasarkan nilai cerun
yang dihitung daripada data EPS dan DPS.

Langkah-langkah utama yang dilakukan:
1. Membaca laman-laman HTML dari folder 'laman_saham/'.
2. Mengekstrak data nama, kod, EPS, dan DPS daripada setiap laman HTML.
3. Melakukan pra-pemprosesan data, termasuk penapisan data berdasarkan tahun.
4. Menghitung nilai cerun untuk EPS dan DPS menggunakan regresi linear (RANSAC).
5. Menentukan nilai cerun akhir saham sebagai hasil darab cerun EPS dan DPS.
6. Cerun akhir (hasil darab) hanya akan positif jika kedua-dua cerun EPS dan DPS adalah positif.
7. Menyaring saham-saham dengan nilai cerun akhir positif (dianggap 'bagus').
8. Mencetak kamus yang mengandungi kod dan nama saham untuk saham-saham 'bagus'.

Kamus yang dicetak perlu disalin dan digunakan dalam file 'menilai_saham.py'
untuk langkah analisis selanjutnya.

Fungsi utama dalam file ini:
    utama(laman: str) -> tuple:
        Menganalisis data saham dari laman HTML dan mengembalikan kod, nama dan cerun saham.

Catatan:
    - File ini menggunakan modul 'pelombong' untuk ekstrak data HTML dan 'regresi'
      untuk analisis statistik.
    - Variabel global 'tahun_ini' digunakan untuk penapisan data berdasarkan tahun.
    - Pengiraan cerun hanya dilakukan jika data mencukupi (lebih dari 10 titik data).
    - Nilai alpha digunakan dalam fungsi 'dapatkan_min_cerun' dari modul 'regresi'.
'''


import pandas as pd


from bs4 import BeautifulSoup
from glob import glob
from multiprocessing import Pool


from analisis_stat import regresi
from pelombongan import pelombong


tahun_ini: int = eval(input("   Tahun ini = "))

semua_laman: list = glob("laman_saham/*.htm")

min_inlier: int = eval(input("   min_inlier (~7) = "))


def utama(laman):
    '''
    Menganalisis data saham dari file HTML dan mengembalikan nama, kod, dan nilai cerun saham.

    Fungsi ini membaca file HTML yang berisi data saham, mengekstrak nama dan kod saham,
    dan menghitung nilai cerun berdasarkan data EPS dan DPS. Fungsi ini menggunakan
    regresi linear untuk menentukan cerun, dan menyaring data berdasarkan tahun.

    Args:
        laman (str): Alamat ke file HTML yang berisi data saham.

    Returns:
        tuple: Tuple yang berisi kod (str), nama (str) dan cerun saham (float).

    Catatan:
        - Fungsi ini menggunakan modul 'pelombong' dan 'regresi' untuk ekstrak dan analisis data.
        - Variabel global 'tahun_ini' digunakan untuk menyaring data tahun.
        - Nilai cerun dihitung hanya jika terdapat lebih dari 10 titik data yang sah.
        - Nilai cerun EPS dan DPS diatur ke 0.0 jika data tidak mencukupi atau tidak sah.
        - Nilai alpha digunakan dalam fungsi 'dapatkan_min_cerun' dari modul 'regresi'.

    Contoh:
        Jika file 'saham.html' berisi data saham yang sah, fungsi ini akan mengembalikan:
        ('Nama Saham', 'Kod Saham', 0.15)

        Jika data tidak valid atau tidak mencukupi, fungsi ini akan mengembalikan:
        ('Nama Saham', 'Kod Saham', 0.0)
    '''
    cerun_eps: float = 0.
    cerun_dps: float = 0.
    alpha: float =0.05

    with open(laman, mode="r", encoding="utf-8") as l:
        kandungan = l.read()
    
    sup: BeautifulSoup = BeautifulSoup(kandungan, "html.parser")

    nama, kod = pelombong.dapatkan_nama_saham(sup)

    if nama == kod:
        return ("error", "error", 0.)
    
    kod = f'{kod}.KL'
    df: pd.DataFrame = pelombong.dapatkan_data_eps_dps(sup)

    if len(df) > 10:
        df = df.drop(axis="index", index=df[df["fy"] >= tahun_ini].index)
        df = df.drop(axis="index", index=df[df["fy"] < (tahun_ini-12)].index)
        dfe_inlier: pd.DataFrame = regresi.dapatkan_inlier(df, "fy", "eps")

        if len(dfe_inlier) > min_inlier:
            cerun_eps = regresi.dapatkan_min_cerun(dfe_inlier, "fy", "eps", alpha)

    if cerun_eps > 0.:
        dfd_inlier: pd.DataFrame = regresi.dapatkan_inlier(df, "fy", "dps")
    
        if len(dfd_inlier) > min_inlier:
            cerun_dps = regresi.dapatkan_min_cerun(dfd_inlier, "fy", "dps", alpha)
        
    cerun: float = cerun_eps * cerun_dps

    saham: tuple = (kod, nama, cerun)

    return saham


if __name__ == "__main__":
    with Pool() as p:
        semua_saham: list = p.map(utama, semua_laman)
    
    saham_bagus: dict = {kod: nama for kod, nama, cerun in semua_saham if cerun > 0}
    bil_saham_bagus: int = len(saham_bagus)

    print(f'''
 Terdapat {bil_saham_bagus} saham yang bagus seperti berikut:

{saham_bagus}

 Salin kamus saham yang bagus di atas ke dalam file menilai_saham.py .

Nota:
  Sasarkan bilangan saham bagus antara 20 hingga 50.
  Jika bilangan saham bagus > 50, naikkan nilai min_inlier.
  Jika bilangan saham bagus < 20, kurangkan nilai min_inlier.
    ''')