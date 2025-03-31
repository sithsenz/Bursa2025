import numpy as np
import pandas as pd
import warnings


from scipy.stats import linregress, t
from sklearn.linear_model import RANSACRegressor as ransac
from sklearn.exceptions import UndefinedMetricWarning


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def dapatkan_inlier(df: pd.DataFrame, x: str, y: str) -> np.array:
    '''
    Mengembalikan DataFrame yang mengandungi hanya inlier dari model RANSAC.

    Fungsi ini melatih model RANSAC (RANdom SAmple Consensus) pada data yang diberikan
    dalam DataFrame dan mengembalikan DataFrame baru yang mengandungi hanya titik data
    yang dianggap inlier oleh model.

    Args:
        df (pd.DataFrame): DataFrame Pandas yang mengandungi data.
        x (str): Nama lajur dalam df yang akan digunakan sebagai input untuk model RANSAC.
        y (str): Nama lajur dalam df yang akan digunakan sebagai output untuk model RANSAC.

    Returns:
        pd.DataFrame: DataFrame yang mengandungi hanya inlier.

    Contoh:
        Jika df mengandungi data dengan kolom 'fy' dan 'eps', dan kita ingin
        mendapatkan inlier berdasarkan hubungan linear antara 'fy' dan 'eps',
        maka kita boleh memanggil:

        dapatkan_inlier(df, 'fy', 'eps')

        Fungsi ini akan mengembalikan DataFrame baru yang mengandungi hanya baris-baris
        di mana nilai 'fy' dan 'eps' sesuai dengan model linear yang ditentukan
        oleh RANSAC.
    '''
    model: ransac = ransac().fit(df[[x]], df[y])
    df_baru: pd.DataFrame = df[model.inlier_mask_]

    return df_baru


def dapatkan_min_cerun(df: pd.DataFrame, x: str, y: str, alpha: float) -> float:
    n: int = len(df)
    ts: float = abs(t.ppf(alpha/2, n-2))
    
    model: linregress = linregress(df[x], df[y])
    cerun: float = model.slope    
    st_error: float = model.stderr * ts
    min: float = cerun - st_error

    return min