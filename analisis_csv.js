// URL del backend en Render
const API_URL = "https://motiometrics-backend.onrender.com"; 

const { jsPDF } = window.jspdf; // Importar jsPDF desde la ventana global

const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const spinner = document.getElementById('spinner');
const btnExport = document.getElementById('btnExport');

// Variables globales para guardar estado
let datosAnalisis = null; // Aquí guardaremos la respuesta del backend
let observaciones = [];   // Aquí acumularemos las observaciones

// --- 1. MANEJO DE ARCHIVOS (Drag & Drop) ---

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

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFileUpload(e.target.files[0]);
});

async function handleFileUpload(file) {
    if (!file.name.endsWith('.csv')) {
        alert("Por favor sube un archivo .csv válido");
        return;
    }

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
        datosAnalisis = data; // Guardamos datos para el reporte
        mostrarResultados(data);

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
    // Actualizar Métricas UI
    document.getElementById('valor-f-dom').textContent = `${data.metricas.frecuencia_dominante} Hz`;
    document.getElementById('valor-psd-peak').textContent = data.metricas.psd_pico;

    // --- LÓGICA DE TEXTO DINÁMICO ---
    //const textoEstado = document.getElementById('texto-estado-temblor');
    
    //if (data.metricas.tiene_temblor) {
    //    textoEstado.textContent = "Valores promedio calculados durante episodios de temblor.";
    //    textoEstado.style.color = "var(--muted)"; // Gris normal
    //    textoEstado.style.fontWeight = "normal";
    //} else {
    //    textoEstado.textContent = "El paciente no presentó episodios de temblor significativos.";
    //    textoEstado.style.color = "var(--success)"; // Verde para indicar "todo bien"
    //    textoEstado.style.fontWeight = "600";
    //}

    // Gráfico 1: PSD vs Frecuencia (con Burg)
    Plotly.newPlot('chartFreqAmp', [{
        x: data.graficos.freq_x,
        y: data.graficos.freq_y,
        type: 'scatter',
        mode: 'lines',
        name: 'PSD Promedio',
        line: {color: 'rgb(0, 132, 199)'}
    }], {
        title: 'PSD vs Frecuencia (Método de Burg)',
        xaxis: {title: 'Frecuencia (Hz)', range: [0, 15]},
        yaxis: {title: 'PSD'},
        height: 400,
        margin: {t: 40, b: 40, l: 40, r: 20}
    });
    
    const layoutFreq = {
        title: 'Espectro de Frecuencia (FFT)',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 40, b: 40, l: 50, r: 20 },
        xaxis: { title: 'Frecuencia (Hz)' },
        yaxis: { title: 'Amplitud' },
        showlegend: false
    };
    
    Plotly.newPlot('chartFreqAmp', [traceFreq], layoutFreq);

    // Gráfico 2: RMS
    const traceRMS = {
        x: data.graficos.tiempo,
        y: data.graficos.rms,
        type: 'scatter',
        mode: 'lines',
        name: 'RMS Combinado',
        line: { color: '#dc2626', width: 1.5 }
    };

    const layoutRMS = {
        title: 'Energía del Temblor (RMS) en el tiempo',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 40, b: 40, l: 50, r: 20 },
        xaxis: { title: 'Tiempo' },
        yaxis: { title: 'Amplitud RMS' },
        showlegend: false
    };

    Plotly.newPlot('chartRMSTime', [traceRMS], layoutRMS);
}

// --- 2. GESTIÓN DE OBSERVACIONES ---

document.getElementById('btnAddObservation').addEventListener('click', () => {
    const desc = document.getElementById('obsDescription').value;
    const start = document.getElementById('obsStartTime').value;
    const end = document.getElementById('obsEndTime').value;

    if (!desc) {
        alert("Escribe una descripción para la observación.");
        return;
    }

    const obs = {
        descripcion: desc,
        inicio: start || "--:--",
        fin: end || "--:--"
    };

    observaciones.push(obs);
    renderObservaciones();
    
    // Limpiar inputs
    document.getElementById('obsDescription').value = "";
    document.getElementById('obsStartTime').value = "";
    document.getElementById('obsEndTime').value = "";
});

function renderObservaciones() {
    const list = document.getElementById('observationList');
    if (observaciones.length === 0) {
        list.innerHTML = "No hay observaciones.";
        return;
    }
    
    let html = '<ul style="padding-left: 20px; margin: 0;">';
    observaciones.forEach((obs, index) => {
        html += `<li style="margin-bottom: 4px;">
            <strong>[${obs.inicio} - ${obs.fin}]</strong> ${obs.descripcion}
        </li>`;
    });
    html += '</ul>';
    list.innerHTML = html;
}

// --- 3. GENERACIÓN Y EXPORTACIÓN DE PDF ---

btnExport.addEventListener('click', async () => {
    if (!window.jspdf) {
        alert("Error: La librería jsPDF no se cargó correctamente. Recarga la página.");
        return;
    }

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF("p", "mm", "a4"); // Formato A4 vertical
    const pageHeight = doc.internal.pageSize.height;

    // Obtener datos actuales del formulario
    const pName = document.getElementById('pName').value || "Paciente";
    const pId = document.getElementById('pId').value || "---";
    const pAge = document.getElementById('pAge').value || "--";
    const pGender = document.getElementById('pGender').value || "--";
    const today = new Date().toLocaleDateString().replace(/\//g, '-'); // Formato dd-mm-yyyy

    // --- PÁGINA 1: TEXTO Y DATOS ---
    
    // Encabezado Azul
    doc.setFillColor(16, 44, 89); // Azul oscuro corporativo
    doc.rect(0, 0, 210, 25, 'F'); // Barra superior
    
    // LOGO (Usamos el logo oculto específico para PDF)
    try {
        // CAMBIO: Buscamos el ID del logo oculto
        const logoImg = document.getElementById('pdf-logo-hidden');
        if (logoImg && logoImg.complete) {
            const logoData = getBase64Image(logoImg);
            doc.addImage(logoData, 'PNG', 10, 5, 15, 15); // Logo en x=10
        }
    } catch(e) { console.log("Logo no disponible para PDF", e); }

    // Título
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(20);
    doc.setFont("helvetica", "bold");
    doc.text("MotioMetrics", 28, 17); // Texto desplazado a la derecha del logo
    
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.text("Informe Clínico de Temblor", 200, 17, { align: "right" });


    let y = 40;
    
    // Datos Paciente
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.text("Datos del Paciente", 10, y);
    doc.line(10, y+2, 200, y+2);
    y += 8;

    doc.setFontSize(11);
    doc.setFont("helvetica", "normal");
    doc.text(`Nombre: ${pName}`, 10, y);
    doc.text(`ID: ${pId}`, 110, y);
    y += 6;
    doc.text(`Edad: ${pAge}`, 10, y);
    doc.text(`Género: ${pGender}`, 110, y);
    y += 6;
    doc.text(`Fecha reporte: ${new Date().toLocaleDateString()}`, 10, y);

    y += 12;

    // Métricas
    if (datosAnalisis) {
        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.text("Métricas Principales", 10, y);
        doc.line(10, y+2, 200, y+2);
        y += 8;

        // Caja destacada
        doc.setFillColor(240, 249, 255);
        doc.rect(10, y, 190, 20, 'F');
        
        doc.setFontSize(12);
        doc.setTextColor(16, 44, 89);
        doc.text(`Frecuencia Dominante: ${datosAnalisis.metricas.frecuencia_dominante} Hz`, 15, y+8);
        doc.text(`Pico de Potencia (PSD): ${datosAnalisis.metricas.psd_pico}`, 15, y+15);
        
        y += 28;
    }

    // Observaciones
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.text("Observaciones Clínicas", 10, y);
    doc.line(10, y+2, 200, y+2);
    y += 8;

    doc.setFontSize(11);
    doc.setFont("helvetica", "normal");
    
    if (observaciones.length === 0) {
        doc.setTextColor(100);
        doc.text("No se registraron observaciones adicionales.", 10, y);
        y += 10;
    } else {
        observaciones.forEach(obs => {
            const linea = `• [${obs.inicio} - ${obs.fin}]: ${obs.descripcion}`;
            const splitText = doc.splitTextToSize(linea, 180);
            doc.text(splitText, 10, y);
            y += (6 * splitText.length);
        });
    }

    // --- PÁGINA 2: GRÁFICOS ---
    if (datosAnalisis) {
        doc.addPage();
        
        doc.setFillColor(16, 44, 89);
        doc.rect(0, 0, 210, 15, 'F');
        doc.setTextColor(255);
        doc.setFontSize(10);
        doc.text("MotioMetrics - Gráficos", 10, 10);

        let imgY = 25;
        const chartHeight = 75;

        try {
            // Gráfico 1
            if (document.getElementById('chartFreqAmp').data) {
                const img1 = await Plotly.toImage(document.getElementById('chartFreqAmp'), {format: 'png', width: 800, height: 400});
                doc.addImage(img1, 'PNG', 15, imgY, 180, chartHeight);
                imgY += chartHeight + 10;
            }
            
            // Gráfico 2
            if (document.getElementById('chartRMSTime').data) {
                // Nueva página si no cabe
                if ((imgY + chartHeight) > (pageHeight - 10)) {
                     doc.addPage();
                     imgY = 20;
                }
                
                const img2 = await Plotly.toImage(document.getElementById('chartRMSTime'), {format: 'png', width: 800, height: 400});
                doc.addImage(img2, 'PNG', 15, imgY, 180, chartHeight);
            }
        } catch (err) {
            console.error("Error capturando gráficos", err);
        }
    }
    
    const safeName = pName.replace(/\s+/g, '_');
    doc.save(`${safeName}_${today}_Motio.pdf`);
});

// Función auxiliar para convertir imagen HTML (logo) a Base64
function getBase64Image(img) {
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    return canvas.toDataURL("image/png");
}