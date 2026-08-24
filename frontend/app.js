/**
 * Frontend JavaScript - Real Estate Predictor
 * Maneja la interacción con la API y el renderizado de oportunidades.
 */

const API_BASE = '';
const PAGE_SIZE = 25;
const REFRESH_INTERVAL = 30000; // 30 segundos

let currentPage = 0;
let currentSort = { field: 'diferencia_porcentual', asc: false };
let refreshTimer = null;

// === Inicialización ===
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkHealth();
    loadOportunidades();
    startAutoRefresh();
});

function setupEventListeners() {
    document.getElementById('btn-refresh').addEventListener('click', () => {
        loadOportunidades();
    });

    document.getElementById('btn-prev').addEventListener('click', () => {
        if (currentPage > 0) {
            currentPage--;
            loadOportunidades();
        }
    });

    document.getElementById('btn-next').addEventListener('click', () => {
        currentPage++;
        loadOportunidades();
    });

    document.getElementById('filter-clasificacion').addEventListener('change', () => {
        currentPage = 0;
        loadOportunidades();
    });

    document.getElementById('filter-barrio').addEventListener('input', debounce(() => {
        currentPage = 0;
        loadOportunidades();
    }, 300));

    // Sorting
    document.querySelectorAll('th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const field = th.dataset.sort;
            if (currentSort.field === field) {
                currentSort.asc = !currentSort.asc;
            } else {
                currentSort.field = field;
                currentSort.asc = false;
            }
            updateSortIndicators();
            loadOportunidades();
        });
    });

    // Modal
    document.getElementById('modal-close').addEventListener('click', closeModal);
    document.getElementById('explain-modal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) {
            closeModal();
        }
    });
}

// === API Calls ===
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        updateStatus('connected', `API v${data.version} | Modelo: ${data.modelo}`);
    } catch (error) {
        updateStatus('error', 'Error de conexión con la API');
    }
}

async function loadOportunidades() {
    const clasificacion = document.getElementById('filter-clasificacion').value;
    const barrio = document.getElementById('filter-barrio').value;

    const params = new URLSearchParams({
        limit: PAGE_SIZE,
        offset: currentPage * PAGE_SIZE,
    });

    if (clasificacion) params.append('clasificacion', clasificacion);
    if (barrio) params.append('barrio', barrio);

    try {
        const response = await fetch(`${API_BASE}/oportunidades?${params}`);
        if (!response.ok) throw new Error('Error al cargar oportunidades');

        const oportunidades = await response.json();
        renderOportunidades(oportunidades);
        updatePagination(oportunidades.length);
        updateLastUpdate();
    } catch (error) {
        console.error('Error:', error);
        renderError('Error al cargar las oportunidades');
    }
}

async function loadExplicacion(id) {
    const modal = document.getElementById('explain-modal');
    const title = document.getElementById('modal-title');
    const summary = document.getElementById('explanation-summary');
    const details = document.getElementById('explanation-details');

    title.textContent = `Explicación - Publicación ${id}`;
    summary.innerHTML = '';
    details.innerHTML = '<p class="loading">Calculando explicación...</p>';
    modal.classList.add('active');

    try {
        const response = await fetch(`${API_BASE}/oportunidades/${id}/explain`);
        if (!response.ok) throw new Error('Error al cargar explicación');

        const explicacion = await response.json();
        renderExplicacion(explicacion);
    } catch (error) {
        console.error('Error:', error);
        details.innerHTML = '<p class="error">Error al calcular la explicación</p>';
    }
}

// === Render Functions ===
function renderOportunidades(oportunidades) {
    const tbody = document.getElementById('oportunidades-body');

    if (oportunidades.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" class="loading">No se encontraron oportunidades</td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = oportunidades.map(o => `
        <tr>
            <td>
                <span class="badge badge-${o.clasificacion || 'sin_clasificar'}">
                    ${formatClasificacion(o.clasificacion)}
                </span>
            </td>
            <td>${o.barrio || '-'}</td>
            <td>${o.tipo_propiedad || '-'}</td>
            <td>${formatCurrency(o.precio_usd)}</td>
            <td>${formatCurrency(o.precio_predicho_usd)}</td>
            <td class="${getDiferenciaClass(o.diferencia_porcentual)}">
                ${formatPorcentaje(o.diferencia_porcentual)}
            </td>
            <td>
                <button class="btn btn-sm btn-primary" onclick="loadExplicacion('${o.id}')">
                    Explicar
                </button>
                ${o.link ? `<a href="${o.link}" target="_blank" class="btn btn-sm btn-secondary" style="margin-left: 4px;">Ver</a>` : ''}
            </td>
        </tr>
    `).join('');
}

function renderExplicacion(explicacion) {
    const summary = document.getElementById('explanation-summary');
    const details = document.getElementById('explanation-details');

    // Summary
    summary.innerHTML = `
        <div class="price">Precio predicho: $${formatNumber(explicacion.precio_predicho_usd)}</div>
        <div class="resumen">${explicacion.resumen}</div>
    `;

    // Feature contributions
    const maxContrib = Math.max(...explicacion.features.map(f => Math.abs(f.contribucion)));

    details.innerHTML = `
        <h3 style="margin-bottom: 1rem; font-size: 1rem;">Contribuciones SHAP por feature</h3>
        ${explicacion.features.map(f => {
            const width = (Math.abs(f.contribucion) / maxContrib) * 50;
            const isPositive = f.contribucion > 0;
            return `
                <div class="feature-bar">
                    <span class="feature-name">${f.nombre}</span>
                    <span class="feature-value">${formatNumber(f.valor)}</span>
                    <div class="feature-bar-container">
                        <div class="feature-bar-fill ${isPositive ? 'positive' : 'negative'}"
                             style="width: ${width}%"></div>
                    </div>
                    <span class="feature-impact ${isPositive ? 'positive' : 'negative'}">
                        ${isPositive ? '+' : ''}${f.contribucion_usd.toFixed(1)}%
                    </span>
                </div>
            `;
        }).join('')}
        <div style="margin-top: 1rem; font-size: 0.8rem; color: var(--text-muted);">
            <strong>Valor base SHAP:</strong> ${explicacion.base_shap.toFixed(4)} (log precio de referencia)
        </div>
    `;
}

function renderError(message) {
    const tbody = document.getElementById('oportunidades-body');
    tbody.innerHTML = `
        <tr>
            <td colspan="7" class="loading" style="color: var(--danger);">${message}</td>
        </tr>
    `;
}

// === Helper Functions ===
function formatCurrency(value) {
    if (value === null || value === undefined) return '-';
    return '$' + formatNumber(value);
}

function formatNumber(value) {
    if (value === null || value === undefined) return '-';
    return Math.round(value).toLocaleString('es-AR');
}

function formatPorcentaje(value) {
    if (value === null || value === undefined || isNaN(value)) return '-';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toFixed(1)}%`;
}

function formatClasificacion(clasificacion) {
    if (!clasificacion) return 'Sin clasificar';
    return clasificacion.replace('_', ' ');
}

function getDiferenciaClass(diferencia) {
    if (diferencia === null || diferencia === undefined || isNaN(diferencia)) return '';
    if (diferencia > 10) return 'diferencia-positiva';
    if (diferencia < -10) return 'diferencia-negativa';
    return 'diferencia-cero';
}

function updateStatus(status, text) {
    const dot = document.getElementById('status-indicator');
    const textEl = document.getElementById('status-text');

    dot.className = 'status-dot ' + status;
    textEl.textContent = text;
}

function updateLastUpdate() {
    const el = document.getElementById('last-update');
    const now = new Date();
    el.textContent = `Última actualización: ${now.toLocaleTimeString('es-AR')}`;
}

function updatePagination(count) {
    const pageInfo = document.getElementById('page-info');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');

    pageInfo.textContent = `Página ${currentPage + 1}`;
    btnPrev.disabled = currentPage === 0;
    btnNext.disabled = count < PAGE_SIZE;
}

function updateSortIndicators() {
    document.querySelectorAll('th.sortable').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
        if (th.dataset.sort === currentSort.field) {
            th.classList.add(currentSort.asc ? 'sort-asc' : 'sort-desc');
        }
    });
}

function closeModal() {
    document.getElementById('explain-modal').classList.remove('active');
}

function startAutoRefresh() {
    refreshTimer = setInterval(() => {
        loadOportunidades();
        checkHealth();
    }, REFRESH_INTERVAL);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
