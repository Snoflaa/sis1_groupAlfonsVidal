import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def load_audio(path):
    """Carga un archivo de audio (mono o estéreo) y devuelve señal y frecuencia de muestreo."""
    x, fs = sf.read(path, always_2d=False)
    if hasattr(x, "ndim") and x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype('float32'), int(fs)

def save_audio(path, x, fs):
    """Guarda un archivo de audio WAV."""
    sf.write(path, x.astype('float32'), fs)
    return path

def plot_signals(x, fs, t_start=None, t_end=None, label=None):
    """Grafica una señal entre t_start y t_end (en segundos)."""
    t = np.arange(len(x)) / fs
    if t_start is not None or t_end is not None:
        t_start = 0.0 if t_start is None else t_start
        t_end = t[-1] if t_end is None else t_end
        sel = (t >= t_start) & (t <= t_end)
        t, x = t[sel], x[sel]
    plt.figure(figsize=(10,3))
    plt.plot(t, x)
    if label:
        plt.title(label)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
