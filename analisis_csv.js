// URL del backend en Render
const API_URL = "https://motiometrics-backend.onrender.com";

const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const spinner = document.getElementById('spinner');

// Event Listeners para Drag & Drop
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length) handleFileUpload(files[0]);
});

// Event Listener para Input File manual
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFileUpload(e.target.files[0]);
});

// Lógica Principal de Subida y Análisis
async function handleFileUpload(file) {
    if (!file.name.endsWith('.csv')) {
        alert("Por favor sube un archivo .csv válido");
        return;
    }

    // UI Feedback
    dropzone.classList.add('loading');
    dropzone.innerHTML = `Analizando <strong>${file.name}</strong>...`;
    spinner.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/api/analizar_datos`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Error en el análisis del servidor');

        const data = await response.json();
        mostrarResultados(data);

        // Feedback Éxito
        dropzone.classList.remove('loading');
        dropzone.classList.add('success');
        dropzone.innerHTML = `<strong>${file.name}</strong> analizado correctamente.`;

    } catch (error) {
        console.error(error);
        dropzone.classList.remove('loading');
        dropzone.classList.add('error');
        dropzone.textContent = "Error al procesar el archivo.";
    } finally {
        spinner.style.display = 'none';
    }
}

function mostrarResultados(data) {
    // 1. Actualizar Métricas Numéricas
    document.getElementById('valor-f-dom').textContent = `${data.metricas.frecuencia_dominante} Hz`;
    document.getElementById('valor-psd-peak').textContent = data.metricas.psd_pico;

    // 2. Renderizar Gráfico de Frecuencia (FFT)
    const traceFreq = {
        x: data.graficos.freq_x,
        y: data.graficos.freq_y,
        type: 'scatter',
        mode: 'lines',
        name: 'Espectro',
        line: { color: '#0284c7' }
    };
    
    const layoutFreq = {
        title: 'Espectro de Frecuencia (FFT)',
        margin: { t: 30, b: 30, l: 40, r: 20 },
        xaxis: { title: 'Hz' },
        yaxis: { title: 'Amplitud' }
    };
    
    Plotly.newPlot('chartFreqAmp', [traceFreq], layoutFreq);

    // 3. Renderizar Gráfico RMS en el tiempo
    const traceRMS = {
        x: data.graficos.tiempo, // Timestamps simplificados
        y: data.graficos.rms,
        type: 'scatter',
        mode: 'lines',
        name: 'RMS Combinado',
        line: { color: '#dc2626' }
    };

    const layoutRMS = {
        title: 'Energía del Temblor (RMS) en el tiempo',
        margin: { t: 30, b: 30, l: 40, r: 20 },
        xaxis: { title: 'Tiempo' },
        yaxis: { title: 'Amplitud RMS' }
    };

    Plotly.newPlot('chartRMSTime', [traceRMS], layoutRMS);
}