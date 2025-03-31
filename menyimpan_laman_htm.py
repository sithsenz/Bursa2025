'''
Muat Turun Laman Saham dari KLSEScreener dan Simpan sebagai Fail HTML.

Fail ini berfungsi untuk memuat turun data laman saham daripada KLSEScreener
dan menyimpannya sebagai fail HTML di dalam folder 'laman_saham'. Fail ini
adalah langkah pertama dalam siri analisis data saham dan perlu dijalankan
secara berkala (setiap 3 bulan, pada hari Sabtu antara 8-14hb).

Laman saham yang dimuat turun akan digunakan dalam fail 'melombong_data.py' untuk
analisis selanjutnya.

Langkah-langkah yang dilakukan:
1. Menghapuskan fail HTML lama dalam folder 'laman_saham'.
2. Membaca URL daripada fail 'screener_htm/Screener.html'.
3. Memuat turun dan menyimpan setiap laman saham sebagai fail HTML.
4. Mencetak ringkasan jumlah laman saham yang berjaya dan bermasalah.
'''


import os


from concurrent.futures import ThreadPoolExecutor
from glob import glob


from modulam import pencatit_masa
from pelombongan import pelombong


if __name__ == "__main__":
    with pencatit_masa.mencatit_masa():
# hapuskan semula laman.html yang lama dalam folder laman_saham
        semua_laman_htm_lama: list = glob("laman_saham/*.htm")    
        for h in semua_laman_htm_lama: os.remove(h)

# laman screener disimpan sebagai Screener.html
        laman_screener: str = "screener_htm/Screener.html"

# dapatkan set semua url.
        semua_url: set = pelombong.dapatkan_semua_url(laman_screener)
        jumlah_url: int = len(semua_url)

# simpan semua laman.
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(pelombong.simpan_laman, semua_url)

        semua_laman_baharu: list = glob("laman_saham/*.htm")
        jumlah_laman_baharu: int = len(semua_laman_baharu)
        bil_laman_stok_bermasalah: int = jumlah_url - jumlah_laman_baharu

    print(f'''
Terdapat {bil_laman_stok_bermasalah} laman stok bermasalah.
Semua {jumlah_laman_baharu} / {jumlah_url} laman stok selesai disimpan.
Sila teruskan ke file melombong_data.py .
    ''')