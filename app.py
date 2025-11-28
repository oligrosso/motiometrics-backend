import os
import io
import csv
import socket
import threading
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import deque
from datetime import datetime
import scipy.signal as signal
from scipy.signal import butter, filtfilt, hilbert
from spectrum import pburg

app = Flask(__name__)
CORS(app)  # Permite que GitHub Pages hable con Render

# --- LÓGICA DE ANÁLISIS EN VIVO (Adaptada de leer_datos_prueba.py) ---

# Buffer en memoria para datos en tiempo real
live_data = {
    "yaw": deque(maxlen=50),
    "pitch": deque(maxlen=50),
    "roll": deque(maxlen=50),
    "timestamps": deque(maxlen=50)
}
is_recording = False
udp_thread = None

def udp_listener():
    """Escucha datos UDP en segundo plano y actualiza el buffer"""
    global is_recording
    UDP_IP = "0.0.0.0"
    UDP_PORT = 4210
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((UDP_IP, UDP_PORT))
        sock.settimeout(1.0)
        print(f"Escuchando UDP en {UDP_PORT}...")
    except Exception as e:
        print(f"Error bind UDP: {e}")
        return

    while is_recording:
        try:
            data, addr = sock.recvfrom(1024)
            values = [float(x) for x in data.decode('utf-8').strip().split(',')]
            # Asumiendo formato: Yaw, Pitch, Roll, Ax, Ay, Az
            if len(values) >= 3:
                timestamp = datetime.now().strftime("%H:%M:%S")
                live_data["yaw"].append(values[0])
                live_data["pitch"].append(values[1])
                live_data["roll"].append(values[2])
                live_data["timestamps"].append(timestamp)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Error recibiendo UDP: {e}")
            break
    sock.close()

@app.route('/api/leer_datos', methods=['POST'])
def leer_datos():
    """Inicia/Detiene la escucha o devuelve datos actuales"""
    global is_recording, udp_thread
    
    action = request.json.get('action')
    
    if action == 'start':
        if not is_recording:
            is_recording = True
            # Limpiar buffers
            live_data["yaw"].clear()
            live_data["pitch"].clear()
            live_data["roll"].clear()
            live_data["timestamps"].clear()
            
            udp_thread = threading.Thread(target=udp_listener)
            udp_thread.daemon = True
            udp_thread.start()
        return jsonify({"status": "started", "message": "Escuchando sensor..."})
    
    elif action == 'stop':
        is_recording = False
        return jsonify({"status": "stopped"})
        
    elif action == 'poll':
        # Devolver los datos actuales del buffer
        return jsonify({
            "labels": list(live_data["timestamps"]),
            "yaw": list(live_data["yaw"]),
            "pitch": list(live_data["pitch"]),
            "roll": list(live_data["roll"])
        })
    
    return jsonify({"error": "Acción no válida"}), 400

# --- LÓGICA DE ANÁLISIS CSV (Adaptada de analizar_datos.py) ---
def filtro_pb(senial, sr, low, high):
    nyq = 0.5 * sr
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, senial)

def detectar_temblor(df, SR, mostrar_pasos=False):
    # Filtro pasabanda 3.5-7.5 Hz (temblor PD)
    yaw = filtro_pb(df['Yaw'], SR, 3.5, 7.5)
    pitch = filtro_pb(df['Pitch'], SR, 3.5, 7.5)
    roll = filtro_pb(df['Roll'], SR, 3.5, 7.5)
    
    # Envolvente Hilbert
    env_yaw = np.abs(hilbert(yaw))
    env_pitch = np.abs(hilbert(pitch))
    env_roll = np.abs(hilbert(roll))
    
    # Suavizado (filtro pasa bajo)
    fc = 1  # Hz
    nyq = 0.5 * SR
    b, a = butter(2, fc/nyq, btype='low')
    env_yaw = filtfilt(b, a, env_yaw)
    env_pitch = filtfilt(b, a, env_pitch)
    env_roll = filtfilt(b, a, env_roll)
    
    # Umbral (media + 1.5 * std)
    umbral_yaw = np.mean(env_yaw) + 1.5 * np.std(env_yaw)
    umbral_pitch = np.mean(env_pitch) + 1.5 * np.std(env_pitch)
    umbral_roll = np.mean(env_roll) + 1.5 * np.std(env_roll)
    
    # Detección (1 si supera umbral)
    temblores = {
        'Yaw': (env_yaw > umbral_yaw).astype(int),
        'Pitch': (env_pitch > umbral_pitch).astype(int),
        'Roll': (env_roll > umbral_roll).astype(int)
    }
    
    df_filt = df.copy()
    df_filt['Yaw'] = yaw
    df_filt['Pitch'] = pitch
    df_filt['Roll'] = roll
    
    return temblores, df_filt, yaw, pitch, roll

def cuantificar_temblor(df, SR, temblores, graph=False):
    win_len = int(SR * 1)  # Ventana de 1 seg
    win_step = int(SR * 0.5)  # Paso de 0.5 seg
    
    # RMS por ventana para cada eje
    rms_yaw = []
    rms_pitch = []
    rms_roll = []
    
    for i in range(0, len(df) - win_len + 1, win_step):
        chunk_yaw = df['Yaw'].iloc[i:i+win_len]
        chunk_pitch = df['Pitch'].iloc[i:i+win_len]
        chunk_roll = df['Roll'].iloc[i:i+win_len]
        
        rms_yaw.append(np.sqrt(np.mean(chunk_yaw**2)))
        rms_pitch.append(np.sqrt(np.mean(chunk_pitch**2)))
        rms_roll.append(np.sqrt(np.mean(chunk_roll**2)))
    
    rms_yaw = np.array(rms_yaw)
    rms_pitch = np.array(rms_pitch)
    rms_roll = np.array(rms_roll)
    
    # RMS combinado
    rms_ypr = np.sqrt(rms_yaw**2 + rms_pitch**2 + rms_roll**2)
    
    # Episodios de temblor (donde al menos un eje detecta temblor)
    episodios = []
    for eje in temblores.values():
        diffs = np.diff(np.concatenate([[0], eje, [0]]))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            if e - s > SR * 2:  # Mínimo 2 seg
                episodios.append((s, e))
    
    return rms_ypr, episodios

def frecuencia_temblor(df, episodios, SR):
    if not episodios:
        return [], 0  # No hay episodios
    
    # PSD por episodio (Burg)
    orden = 15
    psd_interp = []
    freqs_std = np.linspace(0, SR/2, 1000)  # Frecuencias estándar
    
    for start, end in episodios:
        seg_yaw = df['Yaw'].iloc[start:end]
        seg_pitch = df['Pitch'].iloc[start:end]
        seg_roll = df['Roll'].iloc[start:end]
        
        # PSD por eje y promedio
        psd_yaw = pburg(seg_yaw, order=orden, NFFT=512, sampling=SR)
        psd_pitch = pburg(seg_pitch, order=orden, NFFT=512, sampling=SR)
        psd_roll = pburg(seg_roll, order=orden, NFFT=512, sampling=SR)
        
        psd_avg = (psd_yaw.psd + psd_pitch.psd + psd_roll.psd) / 3
        freqs_original = psd_yaw.frequencies()
        
        # Interpolar
        psd = np.interp(freqs_std, freqs_original, psd_avg)
        psd_interp.append(psd)
    
    # PSD promedio global
    psd_mean = np.mean(psd_interp, axis=0)
    
    # Frecuencia dominante
    idx_max = np.argmax(psd_mean)
    f_dom_mean = freqs_std[idx_max]
    
    # Frecuencias dominantes por episodio (para retorno original, pero no lo usamos)
    frecuencias = [freqs_std[np.argmax(psd)] for psd in psd_interp]
    
    return frecuencias, f_dom_mean, freqs_std, psd_mean  # Agregamos freqs_std y psd_mean para el gráfico

def procesar_csv_logic(stream):
    df = pd.read_csv(stream, sep=",", encoding="latin1", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    for col in ['Yaw','Pitch','Roll','Ax','Ay','Az']:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    df = df.dropna(subset=['Yaw','Pitch','Roll','Ax','Ay','Az'])

    diffs = df['Timestamp'].diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    SR = 1 / diffs.mean()
    SR = int(round(SR))

    # Llamadas a las nuevas funciones (reemplaza el filtro y FFT simple)
    temblores, df_filt, yaw, pitch, roll = detectar_temblor(df, SR)
    rms_ypr, episodios = cuantificar_temblor(df, SR, temblores)
    frecuencias, f_dom_mean, freqs_std, psd_mean = frecuencia_temblor(df_filt, episodios, SR)  # Usamos df_filt

    # Métricas actualizadas con Burg
    psd_pico = np.max(psd_mean) if len(psd_mean) > 0 else 0

    # Datos para gráficos (diezmo si es grande)
    factor_diezmo = 1 if len(df) < 1000 else int(len(df)/1000)

    return {
        "metricas": {
            "frecuencia_dominante": round(float(f_dom_mean), 2),
            "psd_pico": round(float(psd_pico), 2),
            "sr": SR
        },
        "graficos": {
            "tiempo": df['Timestamp'].astype(str).iloc[::factor_diezmo].tolist(),
            "rms": rms_ypr[::factor_diezmo].tolist() if len(rms_ypr) > 0 else [],  # RMS como antes
            "freq_x": freqs_std.tolist(),  # Ahora freqs_std de Burg
            "freq_y": psd_mean.tolist()    # Ahora psd_mean de Burg
        }
    }

@app.route('/api/analizar_datos', methods=['POST'])
def analizar_datos_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Procesamos el archivo en memoria
        stream = io.StringIO(file.stream.read().decode("UTF-8"), newline=None)
        resultados = procesar_csv_logic(stream)
        return jsonify(resultados)
    except Exception as e:
        print(f"Error procesando CSV: {e}")
        return jsonify({"error": str(e)}), 500
    
# --- RUTA DE PRUEBA (Health Check) ---
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "message": "MotioMetrics Backend is running!",
        "endpoints": [
            "POST /api/leer_datos",
            "POST /api/analizar_datos"
        ]
    })

if __name__ == '__main__':
    # Render usa la variable PORT
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)