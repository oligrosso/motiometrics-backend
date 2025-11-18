// URL del backend en Render (DEBES CAMBIARLA CUANDO TENGAS EL LINK REAL DE RENDER)
const API_URL = "https://motiometrics-backend.onrender.com";
// const API_URL = "http://127.0.0.1:5000"; // Para pruebas locales

let chart;
let pollingInterval;
let isConnected = false;

// 1. Inicializar gráfico Chart.js
function initChart() {
    const ctx = document.getElementById('liveChartYPR').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [], // Timestamps
            datasets: [
                { label: 'Yaw', data: [], borderColor: 'red', tension: 0.1, pointRadius: 0 },
                { label: 'Pitch', data: [], borderColor: 'orange', tension: 0.1, pointRadius: 0 },
                { label: 'Roll', data: [], borderColor: 'green', tension: 0.1, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false, // Desactivar animación para mejorar rendimiento en vivo
            scales: {
                y: { min: -180, max: 180 }
            }
        }
    });
}

// 2. Función para conectar/desconectar
document.getElementById('btnConnect').addEventListener('click', async () => {
    const btn = document.getElementById('btnConnect');
    const statusText = document.getElementById('connectionStatus');
    const dashboard = document.getElementById('liveDashboard');
    const prompt = document.getElementById('connectionPrompt');

    if (!isConnected) {
        // INICIAR CONEXIÓN
        try {
            const response = await fetch(`${API_URL}/api/leer_datos`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'start' })
            });
            
            if (response.ok) {
                isConnected = true;
                btn.textContent = "⏹ Detener Conexión";
                btn.classList.replace('btn-primary', 'btn');
                statusText.textContent = "Estado: Conectado";
                prompt.style.display = 'none';
                dashboard.style.display = 'grid'; // Mostrar dashboard
                
                // Iniciar Polling (pedir datos cada 100ms)
                startPolling();
            }
        } catch (error) {
            console.error("Error conectando:", error);
            alert("No se pudo conectar con el servidor. Asegúrate que el backend esté corriendo.");
        }
    } else {
        // DETENER CONEXIÓN
        stopPolling();
        try {
            await fetch(`${API_URL}/api/leer_datos`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'stop' })
            });
        } catch (e) { console.error(e); }

        isConnected = false;
        btn.textContent = "🛜 Conectar MotioSensor";
        btn.classList.replace('btn', 'btn-primary');
        statusText.textContent = "Estado: Desconectado";
    }
});

// 3. Función de Polling (Pedir datos constantemente)
function startPolling() {
    pollingInterval = setInterval(async () => {
        try {
            const res = await fetch(`${API_URL}/api/leer_datos`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'poll' })
            });
            const data = await res.json();
            
            if (data.labels) {
                updateChart(data);
            }
        } catch (error) {
            console.error("Error en polling:", error);
        }
    }, 100); // Cada 100ms
}

function stopPolling() {
    clearInterval(pollingInterval);
}

// 4. Actualizar Gráfico
function updateChart(data) {
    chart.data.labels = data.labels;
    chart.data.datasets[0].data = data.yaw;
    chart.data.datasets[1].data = data.pitch;
    chart.data.datasets[2].data = data.roll;
    chart.update();
}

// Inicializar al cargar
document.addEventListener('DOMContentLoaded', initChart);