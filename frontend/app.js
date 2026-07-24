/* Entropy self-estimation lab - frontend logic.
 *
 * If the FastAPI backend is reachable, the RUN button launches a live
 * experiment. Otherwise the page falls back to demo mode and renders the
 * bundled validation results (demo_results.json).
 */

const CONDITION_COLORS = {
  "self|pre": "#4ade80",
  "self|post": "#16a34a",
  "cross|pre": "#f472b6",
  "cross|post": "#be185d",
};

let scatterChart = null;
let barChart = null;
let backendAvailable = false;

async function init() {
  try {
    const res = await fetch("/api/health", { signal: AbortSignal.timeout(3000) });
    backendAvailable = res.ok;
  } catch {
    backendAvailable = false;
  }

  if (backendAvailable) {
    await loadConfig();
    document.getElementById("run-btn").addEventListener("click", launchRun);
    // Render the demo results as a starting view if present.
    loadDemo({ silent: true });
  } else {
    document.getElementById("demo-notice").classList.remove("hidden");
    document.getElementById("run-btn").disabled = true;
    document.getElementById("run-btn").textContent = "Backend not available";
    loadDemo({ silent: false });
  }
  loadPairwise();
  loadGradient();
}

let gradientChart = null;

async function loadGradient() {
  try {
    const res = await fetch("gradient_inversion.json");
    if (!res.ok) return;
    const data = await res.json();
    const labels = data.conditions;
    const orig = labels.map((l) => data.original[l].acc);
    const lit = labels.map((l) => data.literate[l].acc);
    const ns = labels.map((l) => `n=${data.literate[l].n}`);
    if (gradientChart) gradientChart.destroy();
    gradientChart = new Chart(document.getElementById("gradient-bars"), {
      data: {
        labels,
        datasets: [
          { type: "bar", label: "Original prompt", data: orig, backgroundColor: "#f472b6" },
          { type: "bar", label: "Entropy-literate prompt", data: lit, backgroundColor: "#5b8cff" },
          {
            type: "line",
            label: "Chance (50%)",
            data: labels.map(() => 0.5),
            borderColor: "#9aa7bf",
            borderDash: [6, 6],
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        scales: { y: { min: 0, max: 1 } },
        plugins: {
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => ns[ctx.dataIndex],
            },
          },
        },
      },
    });
  } catch (err) {
    console.error("No gradient data", err);
  }
}

let pairwiseChart = null;

async function loadPairwise() {
  // With a backend, one endpoint resolves the best available file
  // server-side (no 404 probing). In standalone demo mode (no backend)
  // fall back to probing the static files directly.
  const sources = ["/api/pairwise_results"];
  for (const file of [
    "pairwise_api_results.json",
    "pairwise_3b_results.json",
    "pairwise_results.json",
  ]) {
    sources.push(file);
  }
  for (const file of sources) {
    try {
      const res = await fetch(file);
      if (!res.ok) continue;
      const data = await res.json();
      const labels = data.models ? Object.values(data.models).join(" vs ") : "";
      const tag = labels ? ` (${labels})` : "";
      const heading = document.querySelector("#pairwise-bars")
        ?.closest(".panel")
        ?.querySelector("h2");
      if (heading && !heading.textContent.includes(tag)) {
        heading.textContent += tag;
      }
      renderPairwise(data.analysis || {});
      return;
    } catch (err) {
      console.error("No pairwise results in " + file, err);
    }
  }
}

function renderPairwise(analysis) {
  const rel = analysis.by_relation || {};
  const order = ["self", "cross"];
  const labels = [];
  const accs = [];
  const meta = [];
  for (const key of order) {
    const stats = rel[key];
    if (!stats) continue;
    labels.push(key);
    accs.push(stats.accuracy);
    meta.push(stats);
  }
  if (!labels.length) return;

  if (pairwiseChart) pairwiseChart.destroy();
  pairwiseChart = new Chart(document.getElementById("pairwise-bars"), {
    data: {
      labels,
      datasets: [
        {
          type: "bar",
          label: "Accuracy",
          data: accs,
          backgroundColor: ["#4ade80", "#f472b6"],
        },
        {
          type: "line",
          label: "Chance (50%)",
          data: labels.map(() => 0.5),
          borderColor: "#9aa7bf",
          borderDash: [6, 6],
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 0, max: 1, title: { display: true, text: "proportion correct" } },
      },
      plugins: {
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => {
              if (ctx.datasetIndex !== 0) return "";
              const s = meta[ctx.dataIndex];
              const gap = s.by_gap || {};
              const ord = s.by_order || {};
              return [
                `n=${s.n} · binomial p=${s.binom_p !== null ? s.binom_p.toFixed(4) : "-"}`,
                `large gap: ${fmtAcc(gap.large)} · small gap: ${fmtAcc(gap.small)}`,
                `correct=A: ${fmtAcc(ord.high_A)} · correct=B: ${fmtAcc(ord.high_B)}`,
              ];
            },
          },
        },
      },
    },
  });

  const comp = analysis.self_vs_cross || {};
  const detail = document.getElementById("pairwise-detail");
  detail.textContent =
    comp.p !== null && comp.p !== undefined
      ? `Self vs cross contrast (proportions z test): z=${comp.z.toFixed(2)}, p=${comp.p.toFixed(3)}`
      : "";
}

function fmtAcc(stats) {
  if (!stats || stats.accuracy === null || stats.accuracy === undefined) return "-";
  return `${(stats.accuracy * 100).toFixed(0)}% (n=${stats.n})`;
}

// Model keys of the active engine (filled from /api/config; the
// fallback keeps the static demo page working without a backend).
let MODEL_KEYS = ["qwen", "smollm"];

async function loadConfig() {
  try {
    const cfg = await (await fetch("/api/config")).json();
    const keys = Object.keys(cfg.models || {});
    if (keys.length >= 2) MODEL_KEYS = keys;
    const labelA = cfg.models[MODEL_KEYS[0]] || "Model A";
    const labelB = cfg.models[MODEL_KEYS[1]] || "Model B";
    document.getElementById("label-a").textContent = labelA;
    document.getElementById("label-b").textContent = labelB;
    document.getElementById("gen-label-a").textContent = labelA;
    document.getElementById("gen-label-b").textContent = labelB;
    if (cfg.engine) {
      const badge = document.getElementById("engine-badge");
      badge.textContent =
        cfg.engine.name === "api"
          ? `remote engine: ${cfg.engine.api_base}`
          : `${cfg.engine.name} engine`;
      badge.hidden = false;
    }
  } catch (err) {
    // Demo mode (no backend): keep the bundled labels.
    console.warn("No backend config available, using demo defaults", err);
  }
}

async function loadDemo({ silent }) {
  try {
    const res = await fetch("demo_results.json");
    if (!res.ok) return;
    const data = await res.json();
    render(data);
  } catch (err) {
    if (!silent) console.error("No demo results available", err);
  }
}

function readControls() {
  const generators = [];
  if (document.getElementById("gen-a").checked) generators.push(MODEL_KEYS[0]);
  if (document.getElementById("gen-b").checked) generators.push(MODEL_KEYS[1]);
  const relations = [];
  if (document.getElementById("rel-self").checked) relations.push("self");
  if (document.getElementById("rel-cross").checked) relations.push("cross");
  const timings = [];
  if (document.getElementById("tim-pre").checked) timings.push("pre");
  if (document.getElementById("tim-post").checked) timings.push("post");
  return {
    generators,
    relations,
    timings,
    temp_min: parseFloat(document.getElementById("temp-min").value),
    temp_max: parseFloat(document.getElementById("temp-max").value),
    reps: parseInt(document.getElementById("reps").value, 10),
    use_selected: document.getElementById("qset-selected").checked,
    reveal_identity: document.getElementById("reveal-identity").checked,
    max_new_tokens: 48,
  };
}

async function launchRun() {
  const body = readControls();
  if (!body.generators.length || !body.relations.length || !body.timings.length) {
    alert("Select at least one generator, one relation and one timing.");
    return;
  }
  const btn = document.getElementById("run-btn");
  btn.disabled = true;
  btn.textContent = "Running...";
  document.getElementById("progress").classList.remove("hidden");

  const { run_id } = await (
    await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
  ).json();

  await pollRun(run_id);

  const results = await (await fetch(`/api/runs/${run_id}/results`)).json();
  render(results);
  btn.disabled = false;
  btn.textContent = "Run experiment";
}

async function pollRun(runId) {
  const fill = document.getElementById("progress-fill");
  const text = document.getElementById("progress-text");
  for (;;) {
    const status = await (await fetch(`/api/runs/${runId}`)).json();
    if (status.status === "error") {
      text.textContent = "Error: " + status.error;
      throw new Error(status.error);
    }
    if (status.progress && status.progress.total > 0) {
      const pct = (100 * status.progress.done) / status.progress.total;
      fill.style.width = pct.toFixed(0) + "%";
      text.textContent = `${status.progress.done}/${status.progress.total} - ${status.progress.current}`;
    }
    if (status.status === "done") {
      fill.style.width = "100%";
      return;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
}

function render(data) {
  renderScatter(data.trials || []);
  renderBars(data.analysis?.pooled || {});
  renderComparisons(data.analysis?.comparisons || {});
  renderTable(data.trials || []);
}

function renderComparisons(comparisons) {
  const el = document.getElementById("comparisons");
  const parts = [];
  for (const [key, comp] of Object.entries(comparisons)) {
    if (comp.p === null || comp.p === undefined) continue;
    const label = key.replace("|", " · ");
    parts.push(`${label}: z=${comp.z.toFixed(2)}, p=${comp.p.toFixed(3)}`);
  }
  el.textContent = parts.length
    ? "Self vs cross contrast (Fisher z): " + parts.join("  |  ")
    : "";
}

function conditionKey(trial) {
  return `${trial.relation}|${trial.timing}`;
}

function renderScatter(trials) {
  const usable = trials.filter((t) => t.estimate !== null && t.estimate !== undefined);
  const groups = {};
  for (const t of usable) {
    const key = conditionKey(t);
    groups[key] = groups[key] || [];
    groups[key].push({ x: t.estimate, y: t.true_entropy });
  }
  const datasets = Object.entries(groups).map(([key, points]) => ({
    label: key.replace("|", " / "),
    data: points,
    backgroundColor: CONDITION_COLORS[key] || "#999",
    pointRadius: 5,
    pointHoverRadius: 7,
  }));

  if (scatterChart) scatterChart.destroy();
  scatterChart = new Chart(document.getElementById("scatter"), {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      scales: {
        x: {
          title: { display: true, text: "Verbalized estimate (0-10)" },
          min: 0,
          max: 10,
        },
        y: { title: { display: true, text: "True entropy (nats)" } },
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: (ctx) =>
              `${ctx.dataset.label}: est=${ctx.parsed.x.toFixed(1)}, H=${ctx.parsed.y.toFixed(2)}`,
          },
        },
      },
    },
  });
}

function renderBars(pooled) {
  const order = ["self|pre", "self|post", "cross|pre", "cross|post"];
  const labels = [];
  const trialR = [];
  const questionR = [];
  const tooltipMeta = [];
  for (const key of order) {
    const stats = pooled[key];
    if (!stats || stats.pearson_r === null || stats.pearson_r === undefined) continue;
    labels.push(key.replace("|", " / "));
    trialR.push(stats.pearson_r);
    questionR.push(stats.question_level ? stats.question_level.pearson_r : null);
    tooltipMeta.push({ n: stats.n, ci: stats.pearson_ci95, p: stats.pearson_p });
  }
  if (barChart) barChart.destroy();
  barChart = new Chart(document.getElementById("bars"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Per trial",
          data: trialR,
          backgroundColor: labels.map((l) => CONDITION_COLORS[l.replace(" / ", "|")]),
        },
        {
          label: "Per question",
          data: questionR,
          backgroundColor: "rgba(154,167,191,0.55)",
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { min: -1, max: 1, title: { display: true, text: "correlation estimate vs. true entropy" } },
      },
      plugins: {
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => {
              if (ctx.datasetIndex !== 0) return "";
              const meta = tooltipMeta[ctx.dataIndex];
              if (!meta) return "";
              const ci = meta.ci
                ? `95% CI [${meta.ci[0].toFixed(2)}, ${meta.ci[1].toFixed(2)}]`
                : "95% CI -";
              return `n=${meta.n} · ${ci} · p=${meta.p !== null ? meta.p.toFixed(3) : "-"}`;
            },
          },
        },
      },
    },
  });
}

function renderTable(trials) {
  const tbody = document.querySelector("#trials-table tbody");
  tbody.innerHTML = "";
  for (const t of trials) {
    const row = document.createElement("tr");
    const cells = [
      t.question_id,
      t.generator === MODEL_KEYS[0] ? "A" : "B",
      `${t.relation}-${t.timing}`,
      t.temperature.toFixed(2),
      t.true_entropy.toFixed(3),
      t.estimate === null || t.estimate === undefined ? "-" : t.estimate.toFixed(1),
    ];
    for (const value of cells) {
      const td = document.createElement("td");
      td.textContent = value;
      row.appendChild(td);
    }
    tbody.appendChild(row);
  }
}

init();
