const API_URL = "http://127.0.0.1:8000/analytics/ml/train";

let r2ChartInstance = null;
let rmseChartInstance = null;

async function trainModel(method) {
    const response = await fetch(`${API_URL}?method=${method}`, {
        method: "POST"
    });

    const data = await response.json();

    if (method === "baseline") {
        renderBaseline(data);
    } else {
        renderAutoML(data);
    }
}

/* =========================
   BASELINE
========================= */

function renderBaseline(data) {
    document.getElementById("modelText").innerText =
        `Baseline model: ${data.model}
R²: ${data.r2}
RMSE: ${data.rmse}`;

    clearTable();
    clearCharts();
}

/* =========================
   AutoML
========================= */

function renderAutoML(data) {
    document.getElementById("modelText").innerText =
        `AutoML использует несколько алгоритмов машинного обучения.
Лучшей моделью выбрана:
${data.best_model}`;

    renderTable(data.leaderboard);
    renderCharts(data.visualization);
}

/* =========================
   TABLE
========================= */

function renderTable(rows) {
    const tbody = document.querySelector("#leaderboardTable tbody");
    tbody.innerHTML = "";

    rows.forEach(row => {
        const tr = document.createElement("tr");

        tr.innerHTML = `
            <td>${row.Model}</td>
            <td>${row.R2}</td>
            <td>${row.RMSE}</td>
            <td>${row.MAE}</td>
        `;

        tbody.appendChild(tr);
    });
}

function clearTable() {
    document.querySelector("#leaderboardTable tbody").innerHTML = "";
}

/* =========================
   CHARTS
========================= */

function renderCharts(vis) {
    const ctxR2 = document.getElementById("r2Chart");
    const ctxRMSE = document.getElementById("rmseChart");

    if (r2ChartInstance) r2ChartInstance.destroy();
    if (rmseChartInstance) rmseChartInstance.destroy();

    r2ChartInstance = new Chart(ctxR2, {
        type: "bar",
        data: {
            labels: vis.x,
            datasets: [{
                label: "R²",
                data: vis.r2
            }]
        }
    });

    rmseChartInstance = new Chart(ctxRMSE, {
        type: "bar",
        data: {
            labels: vis.x,
            datasets: [{
                label: "RMSE",
                data: vis.rmse
            }]
        }
    });
}

function clearCharts() {
    if (r2ChartInstance) r2ChartInstance.destroy();
    if (rmseChartInstance) rmseChartInstance.destroy();
}
