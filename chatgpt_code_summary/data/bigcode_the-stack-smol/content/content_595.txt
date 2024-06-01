# 電子レンジ

def get_E_Elc_microwave_d_t(P_Elc_microwave_cook_rtd, t_microwave_cook_d_t):
    """時刻別消費電力量を計算する
    
    Parameters
    ----------
    P_Elc_microwave_cook_rtd : float
        調理時の定格待機電力, W
        
    t_microwave_cook_d_t : ndarray(N-dimensional array)
        1年間の全時間の調理時間を格納したND配列, h
        d日t時の調理時間が年開始時から8760個連続して格納されている
        
    Returns
    ----------
    E_Elc_microwave_d_t : ndarray(N-dimensional array)
        1年間の全時間の消費電力量を格納したND配列, Wh
        d日t時の消費電力量が年開始時から8760個連続して格納されている
    """
    
    P_Elc_microwave_cook = get_P_Elc_microwave_cook(P_Elc_microwave_cook_rtd)
    
    E_Elc_microwave_d_t = P_Elc_microwave_cook * t_microwave_cook_d_t
    E_Elc_microwave_d_t = E_Elc_microwave_d_t * 10**(-3)
    
    return E_Elc_microwave_d_t


def get_P_Elc_microwave_cook(P_Elc_microwave_rtd):
    """調理時の消費電力を計算する
    
    Parameters
    ----------
    P_Elc_microwave_cook_rtd : float
        調理時の定格待機電力, W
        
    Returns
    ----------
    P_Elc_microwave_cook : float
        調理時の消費電力, W
    """
    
    P_Elc_microwave_cook = 0.9373 * P_Elc_microwave_rtd
        
    return P_Elc_microwave_cook



