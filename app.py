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
# Nota: Asegúrate de tener 'spectrum' en requirements.txt si usas pburg
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

def procesar_csv_logic(file_stream):
    # 1. Cargar Datos
    df = pd.read_csv(file_stream, sep=",", encoding="latin1", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    
    # Limpieza básica (igual que tu script)
    cols_req = ['Yaw','Pitch','Roll']
    for col in cols_req:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    df = df.dropna(subset=cols_req)
    
    # Calcular SR (Frecuencia de muestreo)
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        diffs = df['Timestamp'].diff().dropna().dt.total_seconds()
        diffs = diffs[diffs > 0]
        SR = int(round(1 / diffs.mean())) if not diffs.empty else 50
    except:
        SR = 50 # Fallback

    # 2. Análisis Espectral (Simplificado para JSON)
    # Filtro Pasa Bandas 3.5 - 7.5 Hz (Tu lógica de cuantificar_temblor)
    def filtro_pb(senial, sr, low, high):
        nyq = 0.5 * sr
        b, a = butter(4, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, senial)

    yaw_band = filtro_pb(df['Yaw'], SR, 3.5, 7.5)
    pitch_band = filtro_pb(df['Pitch'], SR, 3.5, 7.5)
    roll_band = filtro_pb(df['Roll'], SR, 3.5, 7.5)
    
    # RMS Combinado
    rms_ypr = np.sqrt(yaw_band**2 + pitch_band**2 + roll_band**2)
    
    # Frecuencia Dominante (usando FFT simple para velocidad)
    # Promediamos los 3 ejes
    promedio_ejes = (df['Yaw'] + df['Pitch'] + df['Roll']) / 3
    N = len(promedio_ejes)
    T = 1.0 / SR
    yf = np.fft.fft(promedio_ejes.values)
    xf = np.fft.fftfreq(N, T)[:N//2]
    yf_magnitude = 2.0 / N * np.abs(yf[0:N//2])
    
    # Buscar pico entre 3 y 10 Hz
    mask = (xf >= 3) & (xf <= 10)
    xf_band = xf[mask]
    yf_band = yf_magnitude[mask]
    
    f_dom = 0
    amp_peak = 0
    if len(yf_band) > 0:
        idx_max = np.argmax(yf_band)
        f_dom = xf_band[idx_max]
        amp_peak = yf_band[idx_max]
    
    # --- LÓGICA DE DETECCIÓN DE TEMBLOR ---
    # Basado en tu script original: si f_dom está en rango y amp supera umbral
    # Umbral ajustado (en tu script original usabas 0.05 o 0.5 dependiendo del método)
    # Usaremos 0.1 como un valor seguro para FFT simple.
    #tiene_temblor = False
    #if 3.5 <= f_dom <= 7.5 and amp_peak > 0.05:
    #    tiene_temblor = True

    # Datos para gráficos (Reducimos puntos si es muy grande para no saturar el JSON)
    factor_diezmo = 1 if len(df) < 1000 else int(len(df)/1000)
    
    return {
        "metricas": {
            "frecuencia_dominante": round(float(f_dom), 2),
            "psd_pico": round(float(amp_peak), 2),
            "sr": SR
        },
        "graficos": {
            "tiempo": df['Timestamp'].astype(str).iloc[::factor_diezmo].tolist(),
            "rms": rms_ypr[::factor_diezmo].tolist(),
            "freq_x": xf_band.tolist(),
            "freq_y": yf_band.tolist()
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