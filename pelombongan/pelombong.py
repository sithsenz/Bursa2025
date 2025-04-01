import pandas as pd
import yfinance as yf


from bs4 import BeautifulSoup
from glob import glob
from playwright.sync_api import sync_playwright


def dapatkan_semua_url(laman_screener: str) -> set:
    '''
    Mendapatkan semua URL yang sepadan daripada file laman_screener.html .

    Fungsi ini membaca kandungan file laman_screener.html yang diberikan,
    mencari semua tag 'a', menapis URL yang mengandungi rentetan tertentu,
    sepertimana dalam sbhgn_href dan mengembalikan set URL unik.

    Args:
        laman_screener (str): Alamat file laman_screener.html .

    Returns:
        set: Set URL unik yang sepadan dengan kriteria penapisan.

    Contoh:
        Jika file 'screener.html' mengandungi pautan seperti:
        <a href="https://www.klsescreener.com/v2/stocks/view/ABC">ABC</a>
        <a href="https://www.example.com/page">Page</a>
        <a href="https://www.klsescreener.com/v2/stocks/view/XYZ">XYZ</a>

        maka, memanggil dapatkan_semua_url('screener.html') akan mengembalikan:
        {'https://www.klsescreener.com/v2/stocks/view/ABC',
        'https://www.klsescreener.com/v2/stocks/view/XYZ'}
    '''
    with open(laman_screener, "r") as laman:
        kandungan_laman = laman.read()
    
    sup = BeautifulSoup(kandungan_laman, "html.parser")

# dapatkan semua url daripada semua tag a.
    semua_a: list = sup.find_all("a")
    semua_href: list = [str(l.get("href")) for l in semua_a]

# simpan hanya url yang sepadan sahaja.
    sbhgn_href: str = "https://www.klsescreener.com/v2/stocks/view/"
    semua_url: list = [h for h in semua_href if sbhgn_href in h]    
    semua_url_yang_unik: set = set(semua_url)

    return semua_url_yang_unik


def simpan_laman(url) -> None:
    '''
    Menyimpan sumber halaman web dari URL yang diberikan ke dalam file HTML menggunakan
    Playwright.

    Fungsi ini mengambil URL, memuatkannya menggunakan Playwright, dan menyimpan sumber
    halaman yang dimuatkan ke dalam file HTML dengan nama yang berdasarkan nombor stok
    yang diekstrak dari URL. Fungsi ini juga mencetak kemajuan penyimpanan ke konsol.

    Args:
        url (str): URL halaman web yang akan disimpan.

    Returns:
        None: Fungsi ini tidak mengembalikan nilai.

    Contoh:
        Jika url = "https://www.klsescreener.com/v2/stocks/view/1234/ABC",
        maka file yang disimpan akan bernama "1234.htm" di dalam folder "laman_saham".

    Catatan:
        - Fungsi ini menggunakan Playwright untuk memuat dan menyimpan halaman web.
        - Fungsi ini memanggil fungsi 'dapatkan_semua_url' untuk mendapatkan daftar semua URL.
        - Fungsi ini mencetak kemajuan penyimpanan ke konsol dalam bentuk peratusan.
    '''

    semua_url: set = dapatkan_semua_url("screener_htm/Screener.html")
    
    jumlah_url: int = len(semua_url)

    nombor_stok: str = url.split("view/")[1].split("/")[0]

    nama_file_laman: str = f'laman_saham/{nombor_stok}.htm'

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url=url, timeout=30000)
        page_source = page.content()

        with open(nama_file_laman, "w", encoding="utf-8") as halaman:
            halaman.write(page_source)
        
        browser.close()
    
    bil_laman: int = len(glob("laman_saham/*.htm"))
    peratus_siap: float = bil_laman / jumlah_url

    print(f'   {bil_laman} / {jumlah_url} = {peratus_siap:.2%}', end="\r")


def dapatkan_nama_saham(sup: BeautifulSoup) -> tuple:
    '''
    Mengekstrak nama saham dan kod saham daripada objek BeautifulSoup.

    Fungsi ini mengambil objek BeautifulSoup yang mewakili laman sesawang dan
    mengambil tajuk halaman untuk mengekstrak nama saham dan kod saham.
    Tajuk halaman dijangka dalam format "Nama Saham: (Kod Saham)".

    Args:
        sup (BeautifulSoup): Objek BeautifulSoup yang mengandungi kandungan laman sesawang.

    Returns:
        tuple: Tuple yang mengandungi nama saham (str) dan kod saham (str).

    Contoh:
        Jika tajuk halaman ialah "ABC Berhad: (1234)", maka fungsi ini akan mengembalikan
        ("ABC Berhad", "1234").
    '''
    _tajuk: str = sup.find("title")
    _tajuk = _tajuk.text.strip()
    nama: str = _tajuk.split(":")[0]
    _kod: str = _tajuk.split("(")[-1]
    kod: str = _kod.split(")")[0]

    return nama, kod


def dapatkan_harga(sup: BeautifulSoup) -> float:
    '''
    Mengekstrak harga saham daripada objek BeautifulSoup.

    Fungsi ini mencari elemen HTML dengan ID "price" dalam objek BeautifulSoup
    dan mengembalikan harga saham sebagai nilai float.

    Args:
        sup (BeautifulSoup): Objek BeautifulSoup yang mengandungi kandungan laman sesawang.

    Returns:
        float: Harga saham yang diekstrak.

    Contoh:
        Jika elemen dengan ID "price" mengandungi teks "12.34", maka fungsi ini akan
        mengembalikan 12.34.
    '''
    _harga: str = sup.find(attrs={"id": "price"})
    harga: float = float(_harga.text.strip())

    return harga


def dapatkan_data_eps_dps(sup: BeautifulSoup) -> list:
    '''
    Mengekstrak data EPS dan DPS dari objek BeautifulSoup dan mengembalikan DataFrame Pandas.

    Fungsi ini mengambil objek BeautifulSoup yang mewakili laman web dengan jadual laporan
    kewangan, mengekstrak data EPS dan DPS untuk setiap FY, dan mengembalikan data tersebut
    dalam bentuk DataFrame Pandas.

    Args:
        sup (BeautifulSoup): Objek BeautifulSoup yang mengandungi kandungan HTML.

    Returns:
        pd.DataFrame: DataFrame Pandas yang mengandungi lajur 'fy', 'eps', dan 'dps'.
                      Data dikumpulkan berdasarkan fy, dan nilai EPS dan DPS dijumlahkan
                      untuk setiap tahun.

    Catatan:
        - Fungsi ini mencari jadual dengan kelas
        "financial_reports table table-hover table-sm table-theme" dalam objek BeautifulSoup.
        - fy diekstrak dari lajur ke-8 (indeks 7), dan 4 digit terakhir diekstrak
        dan ditukar kepada integer.
        - EPS diekstrak dari lajur pertama (indeks 0), dan ditukar kepada float.
        - DPS diekstrak dari lajur ke-2 (indeks 1), dan ditukar kepada float.
        - Data dikumpulkan berdasarkan fy, dan EPS dan DPS dijumlahkan untuk setiap tahun.
        - Fungsi ini mengubah data dalam jadual dengan float menggunakan fungsi eval().
    '''

    df_data: pd.DataFrame = pd.DataFrame(columns=["fy", "eps", "dps"])

    _jadual = sup.find("table", attrs={
        "class": "financial_reports table table-hover table-sm table-theme"
    })

    for baris in _jadual.tbody.find_all("tr"):
        lajur= baris.find_all("td")

        if len(lajur) > 1:
            f = eval(lajur[7].text.strip()[-4:])
            e = eval(lajur[0].text.strip())
            d = eval(lajur[1].text.strip())

            df_data.loc[len(df_data)] = (f, e, d)
    
    data = df_data.groupby(["fy"], as_index=False).sum()
    
    return data


def dapatkan_data_saham(ticker: dict) -> pd.DataFrame:
    '''
    Mengambil data saham dari Yahoo Finance dan mengembalikan DataFrame Pandas.

    Fungsi ini mengambil kamus ticker (nama saham) sebagai input, mengambil data
    saham dari Yahoo Finance selama 3 tahun terakhir, dan mengembalikan data
    tersebut dalam bentuk DataFrame Pandas.

    Args:
        ticker (dict): Kamus di mana kunci adalah ticker saham (str) dan
        nilai adalah nama saham (str).

    Returns:
        pd.DataFrame: DataFrame Pandas yang berisi data saham dengan lajur:
            - Date (datetime): Tarikh data saham.
            - Ticker (str): Ticker saham.
            - nama (str): Nama saham.
            - tahun (int): Tahun data dari Date.
            - bulan (int): Bulan data dari Date.
            - Close (float): Harga tutup saham.

    Catatan:
        - Fungsi ini menggunakan modul `yfinance` untuk mengambil data saham.
        - Data diambil untuk tempoh 3 tahun terakhir.
        - Data di-stack dan di-reset index untuk merapikan format.
        - Lajur 'nama', 'tahun', dan 'bulan' ditambahkan untuk memudahkan analisis.
    '''
    data: pd.DataFrame = yf.Tickers([*ticker]).download(period="3y")
    data = data.stack(future_stack=True).reset_index()
    data["nama"] = data["Ticker"].map(ticker)
    data["tahun"] = data["Date"].dt.year
    data["bulan"] = data["Date"].dt.month
    data = data[["Date", "Ticker", "nama", "tahun", "bulan", "Close"]]

    return data