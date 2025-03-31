import time


from contextlib import contextmanager


@contextmanager
def mencatit_masa():
    '''
    Pengurus konteks untuk mengukur dan mencetak tempoh masa pelaksanaan blok kod.

    Fungsi ini menggunakan pengurus konteks `@contextmanager` untuk mengukur tempoh masa
    yang diambil untuk melaksanakan blok kod di dalam konteks `with`. Ia mencetak masa
    mula dan tempoh masa pelaksanaan dalam format yang mudah dibaca.

    Contoh:
        with mencatit_masa():
            # Kod yang tempoh masa pelaksanaannya ingin diukur
            time.sleep(5)  # Simulasi operasi yang mengambil masa 5 saat

        Output:
        10:00:00 Mula menyimpan laman saham...

        -----Tamat----- 0 minit 5.0 saat -----
    '''
    masa_mula: time.time = time.time()
    str_masa_mula: str = time.strftime("%H:%M:%S", time.localtime(masa_mula))
    print()
    print(f'{str_masa_mula} Mula menyimpan laman saham...')
    print()
    yield
    masa_tamat: time.time = time.time()
    tempoh_masa: float = masa_tamat - masa_mula
    (minit, saat) = divmod(tempoh_masa, 60)
    str_tempoh: str = f'{minit:.0f} minit {saat:.1f} saat'
    print()
    print()
    print(f'-----Tamat----- {str_tempoh} -----')