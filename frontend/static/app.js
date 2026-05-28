const config = window.FINSTREAM_CONFIG || {};
const apiBaseUrl = (config.apiBaseUrl || "http://localhost:8000").replace(/\/$/, "");

const form = document.getElementById("predictForm");
const newsInput = document.getElementById("newsInput");
const predictButton = document.getElementById("predictButton");
const loadingSpinner = document.getElementById("loadingSpinner");
const errorBox = document.getElementById("errorBox");
const emptyState = document.getElementById("emptyState");
const resultArea = document.getElementById("resultArea");
const sentimentCard = document.getElementById("sentimentCard");
const sentimentLabel = document.getElementById("sentimentLabel");
const confidenceScore = document.getElementById("confidenceScore");
const themeToggle = document.getElementById("themeToggle");
const sampleChips = document.querySelectorAll(".sample-chip");
const csvFileInput = document.getElementById("csvFile");
const browseCsvButton = document.getElementById("browseCsvButton");
const uploadCsvButton = document.getElementById("uploadCsvButton");
const reportIdInput = document.getElementById("reportId");
const selectedCsvLabel = document.getElementById("selectedCsvLabel");
const batchError = document.getElementById("batchError");
const batchSummary = document.getElementById("batchSummary");
const downloadReportLink = document.getElementById("downloadReportLink");
const batchTableSection = document.getElementById("batchTableSection");
const batchResultsTableBody = document.getElementById("batchResultsTableBody");
const batchPagination = document.getElementById("batchPagination");
const batchResultCount = document.getElementById("batchResultCount");
const batchTableMeta = document.getElementById("batchTableMeta");

let chartInstance = null;
let batchPredictions = [];
let batchSummaryData = null;
let batchCurrentPage = 1;
const batchPageSize = 5;

const sampleData = [
  { label: "Bullish", value: 0.46 },
  { label: "Neutral", value: 0.30 },
  { label: "Bearish", value: 0.24 },
];

function setLoading(isLoading) {
  predictButton.disabled = isLoading;
  loadingSpinner.classList.toggle("d-none", !isLoading);
  predictButton.querySelector(".btn-label").textContent = isLoading ? "Analyzing" : "Predict";
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("d-none");
}

function hideError() {
  errorBox.classList.add("d-none");
  errorBox.textContent = "";
}

function sentimentClass(label) {
  const normalized = (label || "neutral").toLowerCase();
  if (normalized.includes("bull")) return "bullish";
  if (normalized.includes("bear") || normalized.includes("neg")) return "bearish";
  return "neutral";
}

function updateChart(label, confidence) {
  const bullish = label === "bullish" ? confidence : label === "neutral" ? confidence * 0.35 : Math.max(0.1, 1 - confidence);
  const bearish = label === "bearish" ? confidence : label === "neutral" ? confidence * 0.35 : Math.max(0.1, 1 - confidence);
  const neutral = Math.max(0.1, 1 - Math.abs(bullish - bearish));
  const dataset = [bullish, neutral, bearish].map((value) => Number(Math.max(0.05, Math.min(0.95, value)).toFixed(2)));

  if (chartInstance) {
    chartInstance.data.datasets[0].data = dataset;
    chartInstance.update();
    return;
  }

  const context = document.getElementById("sentimentChart");
  chartInstance = new Chart(context, {
    type: "doughnut",
    data: {
      labels: sampleData.map((item) => item.label),
      datasets: [{
        data: dataset,
        backgroundColor: ["#22c55e", "#f59e0b", "#ef4444"],
        borderWidth: 0,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true,
      cutout: "72%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: getComputedStyle(document.documentElement).getPropertyValue("--text").trim(),
            usePointStyle: true,
            padding: 18,
          },
        },
      },
    },
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function sentimentBadgeClass(label) {
  const normalized = (label || "unknown").toLowerCase();
  if (normalized.includes("bull")) return "bg-success";
  if (normalized.includes("bear")) return "bg-danger";
  if (normalized.includes("neu")) return "bg-warning text-dark";
  return "bg-secondary";
}

function renderBatchPagination(totalItems) {
  const totalPages = Math.max(1, Math.ceil(totalItems / batchPageSize));
  const currentPage = Math.min(Math.max(batchCurrentPage, 1), totalPages);
  batchCurrentPage = currentPage;

  batchPagination.innerHTML = `
    <li class="page-item ${currentPage === 1 ? "disabled" : ""}">
      <button class="page-link" data-page="prev" type="button">Previous</button>
    </li>
    ${Array.from({ length: totalPages }, (_, index) => {
      const pageNumber = index + 1;
      return `
        <li class="page-item ${pageNumber === currentPage ? "active" : ""}">
          <button class="page-link" data-page="${pageNumber}" type="button">${pageNumber}</button>
        </li>
      `;
    }).join("")}
    <li class="page-item ${currentPage === totalPages ? "disabled" : ""}">
      <button class="page-link" data-page="next" type="button">Next</button>
    </li>
  `;

  batchPagination.querySelectorAll(".page-link").forEach((button) => {
    button.addEventListener("click", () => {
      const page = button.dataset.page;
      if (page === "prev" && batchCurrentPage > 1) {
        batchCurrentPage -= 1;
      } else if (page === "next" && batchCurrentPage < totalPages) {
        batchCurrentPage += 1;
      } else if (!Number.isNaN(Number(page))) {
        batchCurrentPage = Number(page);
      }
      renderBatchTable();
    });
  });
}

function renderBatchTable() {
  if (!batchPredictions.length) {
    batchTableSection.classList.add("d-none");
    return;
  }

  const totalItems = batchPredictions.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / batchPageSize));
  const start = (batchCurrentPage - 1) * batchPageSize;
  const pageItems = batchPredictions.slice(start, start + batchPageSize);

  batchResultsTableBody.innerHTML = pageItems.map((item) => `
    <tr>
      <td>${item.row_number}</td>
      <td class="batch-message-cell">${escapeHtml(item.message)}</td>
      <td><span class="badge ${sentimentBadgeClass(item.predicted_label)}">${escapeHtml(item.predicted_label)}</span></td>
      <td>${(Number(item.confidence) * 100).toFixed(2)}%</td>
    </tr>
  `).join("");

  batchResultCount.textContent = `${totalItems} rows`;
  batchTableMeta.textContent = `Showing ${start + 1}-${Math.min(start + pageItems.length, totalItems)} of ${totalItems} rows • Page ${batchCurrentPage} of ${totalPages}`;
  batchTableSection.classList.remove("d-none");
  renderBatchPagination(totalItems);
}

function renderResult(payload) {
  const label = (payload.label || "neutral").toLowerCase();
  const confidence = Number(payload.confidence || 0);
  sentimentCard.className = `sentiment-card ${sentimentClass(label)}`;
  sentimentLabel.textContent = label;
  confidenceScore.textContent = `${Math.round(confidence * 100)}%`;
  emptyState.classList.add("d-none");
  resultArea.classList.remove("d-none");
  resultArea.classList.add("fade-in");
  updateChart(label, confidence);
}

async function readApiResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const rawText = await response.text();

  if (contentType.includes("application/json")) {
    try {
      return rawText ? JSON.parse(rawText) : {};
    } catch (error) {
      return { detail: rawText || "Invalid JSON response from server." };
    }
  }

  return { detail: rawText || "Unexpected non-JSON response from server." };
}

async function handleSubmit(event) {
  event.preventDefault();
  hideError();

  const text = newsInput.value.trim();
  if (!text) {
    showError("Please enter a financial news headline or paragraph.");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch(`${apiBaseUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await readApiResponse(response);

    if (!response.ok) {
      throw new Error(data.detail || "Unable to analyze the provided text.");
    }

    renderResult(data);
  } catch (error) {
    showError(error.message || "Something went wrong while calling the prediction service.");
  } finally {
    setLoading(false);
  }
}

sampleChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    newsInput.value = chip.dataset.sample || "";
    newsInput.focus();
  });
});

themeToggle.addEventListener("click", () => {
  const root = document.documentElement;
  const current = root.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  themeToggle.innerHTML = next === "dark"
    ? '<i class="fa-solid fa-moon"></i>'
    : '<i class="fa-solid fa-sun"></i>';
});

form.addEventListener("submit", handleSubmit);
updateChart("neutral", 0.5);

async function handleCsvUpload() {
  batchError.classList.add("d-none");
  batchSummary.classList.add("d-none");
  downloadReportLink.classList.add("d-none");
  batchTableSection.classList.add("d-none");

  const file = csvFileInput.files && csvFileInput.files[0];
  if (!file) {
    batchError.textContent = "Please select a CSV file to upload.";
    batchError.classList.remove("d-none");
    return;
  }

  const form = new FormData();
  form.append("file", file);
  const rid = (reportIdInput && reportIdInput.value && reportIdInput.value.trim()) || "";
  if (rid) form.append("report_id", rid);

  uploadCsvButton.disabled = true;
  uploadCsvButton.textContent = "Analyzing...";

  try {
    const resp = await fetch(`${apiBaseUrl}/analyze-csv`, {
      method: "POST",
      body: form,
    });

    const data = await readApiResponse(resp);
    if (!resp.ok) throw new Error(data.detail || "CSV analysis failed");

    const s = data.summary;
    batchSummaryData = s;
    batchPredictions = Array.isArray(data.predictions) ? data.predictions : [];
    batchCurrentPage = 1;
    batchSummary.innerHTML = `
      <div class="card p-3 batch-summary-card">
        <div class="d-flex justify-content-between align-items-start gap-3 flex-wrap">
          <div>
            <div class="text-uppercase small text-secondary fw-bold mb-2">Batch Report</div>
            <div><strong>Report ID:</strong> ${escapeHtml(s.report_id)}</div>
            <div><strong>Detected column:</strong> ${escapeHtml(s.detected_text_column)}</div>
            <div><strong>Rows analyzed:</strong> ${s.analyzed_rows}</div>
          </div>
          <div class="batch-net-card">
            <div class="text-uppercase small text-secondary fw-bold">Sentiment Net</div>
            <div class="h3 mb-0 ${s.net_sentiment > 0 ? "text-success" : s.net_sentiment < 0 ? "text-danger" : "text-warning"}">
              ${s.net_sentiment}
            </div>
            <div class="small text-secondary">${escapeHtml(s.net_sentiment_label)}</div>
          </div>
        </div>
        <hr>
        <div class="row g-2 small">
          <div class="col-md-3"><strong>Bullish:</strong> ${s.bullish_count}</div>
          <div class="col-md-3"><strong>Neutral:</strong> ${s.neutral_count}</div>
          <div class="col-md-3"><strong>Bearish:</strong> ${s.bearish_count}</div>
          <div class="col-md-3"><strong>Avg Confidence:</strong> ${(Number(s.average_confidence) * 100).toFixed(2)}%</div>
        </div>
      </div>
    `;
    batchSummary.classList.remove("d-none");
    downloadReportLink.href = s.report_pdf_url;
    downloadReportLink.classList.remove("d-none");
    renderBatchTable();
  } catch (err) {
    batchError.textContent = err.message || String(err);
    batchError.classList.remove("d-none");
  } finally {
    uploadCsvButton.disabled = false;
    uploadCsvButton.textContent = "Analyze CSV";
  }
}

uploadCsvButton.addEventListener("click", handleCsvUpload);

browseCsvButton.addEventListener("click", () => {
  csvFileInput.click();
});

csvFileInput.addEventListener("change", () => {
  const file = csvFileInput.files && csvFileInput.files[0];
  selectedCsvLabel.textContent = file ? file.name : "No file selected";
});
