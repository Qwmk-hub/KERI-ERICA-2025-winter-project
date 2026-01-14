import numpy as np
from scipy.stats import entropy

def extract_fft_features(daily_seq, top_k=3):
    """주어진 시퀀스(28일)에 대해 FFT 특징 추출"""
    n = len(daily_seq)
    # DC Offset(평균) 제거 후 FFT 수행하여 주기성 강조
    detrended_seq = daily_seq - np.mean(daily_seq)
    fft_vals = np.fft.rfft(detrended_seq)
    amplitudes = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n)
    
    features = {}
    
    # 상위 K개 진폭 및 해당 주파수
    top_indices = np.argsort(amplitudes)[-top_k:][::-1]
    for j, idx in enumerate(top_indices):
        features[f'fft_amp_{j}'] = amplitudes[idx]
        features[f'fft_freq_{j}'] = freqs[idx]
        
    # Spectral 특성
    psd = amplitudes**2
    features['total_power'] = np.sum(psd)
    
    # Spectral Entropy (에너지 분포의 불확실성)
    psd_norm = psd / (np.sum(psd) + 1e-9)
    features['spectral_entropy'] = entropy(psd_norm)
    
    # Band Power (저주파/고주파 에너지 비중)
    mid_idx = len(psd) // 2
    features['low_freq_power'] = np.sum(psd[:mid_idx])
    features['high_freq_power'] = np.sum(psd[mid_idx:])
    
    return features