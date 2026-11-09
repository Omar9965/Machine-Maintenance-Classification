/**
 * Machine Predictive Maintenance — Monitoring Dashboard
 *
 * Renders JSON from the API. Contains no business logic beyond
 * client-side recomputation of flagged status and metrics when the
 * threshold slider moves (scores are stored per row by the server).
 *
 * Data flow:
 *   load() → GET /api/v1/dashboard/config
 *          → GET /api/v1/dashboard/batches?include_rows=true (rebuild state)
 *          → poll loop (chained setTimeout, never overlapping)
 */

(function () {
    'use strict';

    // --- State -------------------------------------------------------------
    var state = {
        config: null,           // {threshold, poll_ms, batch_size_default, batch_size_max}
        batches: [],            // [{batch_id, size, counts, rows: [{...}]}]
        playing: false,
        inFlight: false,
        threshold: 0.8966,
        batchSize: 50,
        replayComplete: false,
        histogramChart: null,
        prChart: null,
        modeChart: null,
        totalAvailable: 0
    };

    var MODE_LIST = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'UNKNOWN'];
    var HISTOGRAM_BINS = 25;
    var API = '/api/v1';

    // --- DOM refs ------------------------------------------------------------
    function $(id) { return document.getElementById(id); }

    // --- Init ----------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', function () {
        bindControls();
        load();
    });

    function bindControls() {
        $('playBtn').addEventListener('click', function () { togglePlay(); });
        $('resetBtn').addEventListener('click', function () { resetSimulation(); });
        $('retryBtn').addEventListener('click', function () {
            hideError();
            poll();
        });
        $('thresholdSlider').addEventListener('input', function (e) {
            state.threshold = parseFloat(e.target.value);
            $('thresholdValue').textContent = state.threshold.toFixed(4);
            renderAll();
        });
        $('batchSizeSelect').addEventListener('change', function (e) {
            state.batchSize = parseInt(e.target.value, 10);
        });
    }

    // --- Initial load ----------------------------------------------------------
    function load() {
        showLoading(true);
        fetchJson(API + '/dashboard/config').then(function (config) {
            state.config = config;
            state.threshold = config.threshold;
            state.batchSize = config.batch_size_default;
            $('thresholdSlider').value = config.threshold;
            $('thresholdValue').textContent = config.threshold.toFixed(4);
            fillBatchSizes(config);
            return fetchJson(API + '/dashboard/batches?include_rows=true').then(function (batches) {
                state.batches = batches;
                return fetchJson(API + '/dashboard/summary').then(function (summary) {
                    applySummary(summary);
                    state.replayComplete = summary.replay_complete;
                    showLoading(false);
                    renderAll();
                    if (state.replayComplete) {
                        stopPlaying();
                        showComplete();
                    } else {
                        startPlaying();
                    }
                });
            });
        }).catch(showError);
    }

    function fillBatchSizes(config) {
        var sizes = [];
        [10, 20, 50, 100, config.batch_size_default, config.batch_size_max].forEach(function (s) {
            if (s >= 1 && s <= config.batch_size_max && sizes.indexOf(s) === -1) sizes.push(s);
        });
        sizes.sort(function (a, b) { return a - b; });
        var select = $('batchSizeSelect');
        select.textContent = '';
        sizes.forEach(function (s) {
            var opt = document.createElement('option');
            opt.value = String(s);
            opt.textContent = s;
            if (s === state.batchSize) opt.selected = true;
            select.appendChild(opt);
        });
    }

    // --- Poll loop (chained setTimeout, never overlapping) -----------------------
    function startPlaying() {
        if (state.playing) return;
        state.playing = true;
        $('playBtnText').textContent = 'Pause';
        poll();
    }

    function stopPlaying() {
        state.playing = false;
        $('playBtnText').textContent = 'Play';
    }

    function togglePlay() {
        if (state.playing) stopPlaying();
        else startPlaying();
    }

    function poll() {
        if (!state.playing || state.inFlight || state.replayComplete) return;
        state.inFlight = true;
        fetchJson(API + '/simulation/next?size=' + state.batchSize, { method: 'POST' })
            .then(function (data) {
                state.inFlight = false;
                applySummary(data.summary);
                if (data.batch && data.batch.size > 0) {
                    upsertBatch(data.batch);
                }
                renderAll();
                if (data.summary.replay_complete) {
                    state.replayComplete = true;
                    stopPlaying();
                    showComplete();
                    return;
                }
                window.setTimeout(poll, state.config.poll_ms);
            })
            .catch(function (err) {
                state.inFlight = false;
                stopPlaying();
                showError(err);
            });
    }

    // --- Server actions --------------------------------------------------------
    function resetSimulation() {
        fetchJson(API + '/simulation/reset', { method: 'POST' }).then(function (summary) {
            state.replayComplete = false;
            hideError();
            hideComplete();
            applySummary(summary);
            state.batches = [];
            renderAll();
            startPlaying();
        }).catch(showError);
    }

    function applySummary(summary) {
        if (summary && typeof summary.total_available === 'number') {
            state.totalAvailable = summary.total_available;
        }
    }

    function upsertBatch(batch) {
        for (var i = 0; i < state.batches.length; i++) {
            if (state.batches[i].batch_id === batch.batch_id) {
                state.batches[i] = batch;
                return;
            }
        }
        state.batches.push(batch);
    }

    // --- Client-side metrics (current threshold) --------------------------------
    function allRows() {
        var rows = [];
        state.batches.forEach(function (b) {
            if (b.rows) b.rows.forEach(function (r) { rows.push(r); });
        });
        return rows;
    }

    function computeMetrics(rows) {
        var tp = 0, fp = 0, fn = 0, tn = 0;
        rows.forEach(function (r) {
            var flagged = r.score >= state.threshold;
            var actual = !!r.actual_failure;
            if (flagged && actual) tp++;
            else if (flagged && !actual) fp++;
            else if (!flagged && actual) fn++;
            else tn++;
        });
        var precision = (tp + fp) > 0 ? tp / (tp + fp) : null;
        var recall = (tp + fn) > 0 ? tp / (tp + fn) : null;
        var f1 = (precision !== null && recall !== null && precision + recall > 0)
            ? 2 * precision * recall / (precision + recall) : null;
        return { tp: tp, fp: fp, fn: fn, tn: tn, precision: precision, recall: recall, f1: f1 };
    }

    function latestBatch() {
        return state.batches.length ? state.batches[state.batches.length - 1] : null;
    }

    // --- Rendering ---------------------------------------------------------------
    function renderAll() {
        var rows = allRows();
        var metrics = computeMetrics(rows);
        renderReplayStatus(rows.length, metrics);
        renderKpis(metrics);
        renderFlaggedTable(rows.length, metrics);
        renderCharts(rows, metrics);
    }

    function renderReplayStatus(scored, metrics) {
        var total = state.totalAvailable || 2000;
        $('replayCounts').textContent = scored + ' / ' + total;
        $('replayProgressFill').style.width =
            (total > 0 ? Math.min(100, (scored / total) * 100) : 0) + '%';
    }

    function renderKpis(metrics) {
        $('kpiScored').textContent = String(metrics.tp + metrics.fp + metrics.fn + metrics.tn);
        var latest = latestBatch();
        var flaggedLatest = 0, latestSize = 0;
        if (latest && latest.rows) {
            latestSize = latest.rows.length;
            latest.rows.forEach(function (r) {
                if (r.score >= state.threshold) flaggedLatest++;
            });
        }
        $('kpiFlagged').textContent = String(flaggedLatest);
        var flagRate = latestSize > 0 ? (flaggedLatest / latestSize * 100) : 0;
        $('kpiFlagRate').textContent = flagRate.toFixed(0) + '%';

        var p = $('kpiPrecision');
        var r = $('kpiRecall');
        setTextOrDash(p, metrics.precision, 'kpi-green');
        setTextOrDash(r, metrics.recall, 'kpi-green');
    }

    function setTextOrDash(el, value, cssClass) {
        if (value === null || value === undefined) {
            el.textContent = '–';
            el.classList.add('kpi-null');
            el.classList.remove(cssClass);
        } else {
            el.textContent = value.toFixed(3);
            el.classList.remove('kpi-null');
            el.classList.add(cssClass);
        }
    }

    function fmtReading(v, digits) {
        if (typeof v !== 'number') return String(v);
        return v.toFixed(digits !== undefined ? digits : 1);
    }

    function renderFlaggedTable(totalScored, metrics) {
        var tbody = $('flaggedTableBody');
        tbody.textContent = '';
        var latest = latestBatch();
        var flaggedRows = [];
        if (latest && latest.rows) {
            latest.rows.forEach(function (r) {
                if (r.score >= state.threshold) flaggedRows.push(r);
            });
        }

        if (flaggedRows.length === 0) {
            var tr0 = document.createElement('tr');
            var td0 = document.createElement('td');
            td0.colSpan = 9;
            td0.className = 'table-empty';
            td0.textContent = 'No flagged machines in the latest batch.';
            tr0.appendChild(td0);
            tbody.appendChild(tr0);
            return;
        }

        flaggedRows.forEach(function (r) {
            var actual = !!r.actual_failure;
            var isTruePositive = actual;

            var tr = document.createElement('tr');
            if (isTruePositive) tr.classList.add('row-tp');

            appendTextCell(tr, String(r.udi), 'machine-id');
            appendTextCell(tr, r.product_id);
            appendTextCell(tr, r.type);

            appendTextCell(tr, fmtReading(r.torque_nm));
            appendTextCell(tr, fmtReading(r.rotational_speed_rpm, 0));
            appendTextCell(tr, fmtReading(r.tool_wear_min, 0));

            var scoreTd = document.createElement('td');
            var strong = document.createElement('strong');
            strong.textContent = r.score.toFixed(3);
            scoreTd.appendChild(strong);
            if (r.score >= state.threshold && r.score < 0.95) {
                scoreTd.style.color = 'var(--accent-amber)';
            }
            tr.appendChild(scoreTd);

            var factorsTd = document.createElement('td');
            factorsTd.className = 'factor-list';
            var factorText = (r.top_factors || []).map(function (f) {
                return f.feature + ' +' + f.contribution.toFixed(2);
            }).join(', ');
            factorsTd.textContent = factorText || '–';
            tr.appendChild(factorsTd);

            var outcomeTd = document.createElement('td');
            outcomeTd.appendChild(makeBadge(isTruePositive));
            tr.appendChild(outcomeTd);

            tbody.appendChild(tr);
        });
    }

    function appendTextCell(tr, text, cssClass) {
        var td = document.createElement('td');
        if (cssClass) td.className = cssClass;
        td.textContent = text;
        tr.appendChild(td);
    }

    function makeBadge(isTruePositive) {
        var span = document.createElement('span');
        if (isTruePositive) {
            span.className = 'badge badge-caught';
            span.textContent = 'true failure';
        } else {
            span.className = 'badge badge-false-alarm';
            span.textContent = 'false alarm';
        }
        return span;
    }

    // --- Charts -------------------------------------------------------------------
    function renderCharts(rows, metrics) {
        renderHistogram(rows);
        renderPrChart();
        renderModeChart(rows);
    }

    function renderHistogram(rows) {
        var scores = rows.map(function (r) { return r.score; });
        var binCounts = new Array(HISTOGRAM_BINS).fill(0);
        scores.forEach(function (s) {
            var b = Math.min(HISTOGRAM_BINS - 1, Math.max(0, Math.floor(s * HISTOGRAM_BINS)));
            binCounts[b]++;
        });
        var points = [];
        for (var i = 0; i < HISTOGRAM_BINS; i++) {
            points.push({ x: (i + 0.5) / HISTOGRAM_BINS, y: binCounts[i] });
        }

        var canvas = $('histogramChart');
        var ctx = canvas.getContext('2d');
        if (state.histogramChart) {
            state.histogramChart.data.datasets[0].data = points;
            state.histogramChart.update();
            return;
        }

        var thresholdPlugin = {
            id: 'thresholdLine',
            afterDatasetsDraw: function (chart) {
                var xAxis = chart.scales.x;
                var pixel = xAxis.getPixelForValue(state.threshold);
                if (xAxis.min <= state.threshold && state.threshold <= xAxis.max) {
                    var yAxis = chart.scales.y;
                    var area = chart.chartArea;
                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(pixel, area.top);
                    ctx.lineTo(pixel, area.bottom);
                    ctx.strokeStyle = '#f59e0b';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 4]);
                    ctx.stroke();
                    ctx.restore();
                }
            }
        };

        state.histogramChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'machines',
                    data: points,
                    backgroundColor: 'rgba(59, 130, 246, 0.25)',
                    borderColor: '#3b82f6',
                    borderWidth: 1,
                    fill: true,
                    tension: 0.35,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                        type: 'linear',
                        min: 0,
                        max: 1,
                        title: { display: true, text: 'risk score' },
                        ticks: { stepSize: 0.25 }
                    },
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'machines' }
                    }
                }
            },
            plugins: [thresholdPlugin]
        });
    }

    function renderPrChart() {
        var labels = state.batches.map(function (b) { return b.batch_id; });
        var precision = [], recall = [];
        state.batches.forEach(function (b) {
            if (!b.rows) { precision.push(null); recall.push(null); return; }
            var tp = 0, fp = 0, fn = 0;
            b.rows.forEach(function (r) {
                var flagged = r.score >= state.threshold;
                if (flagged && r.actual_failure) tp++;
                else if (flagged && !r.actual_failure) fp++;
                else if (!flagged && r.actual_failure) fn++;
            });
            precision.push((tp + fp) > 0 ? tp / (tp + fp) : null);
            recall.push((tp + fn) > 0 ? tp / (tp + fn) : null);
        });

        var ctx = $('prChart').getContext('2d');
        if (state.prChart) {
            state.prChart.data.labels = labels;
            state.prChart.data.datasets[0].data = precision;
            state.prChart.data.datasets[1].data = recall;
            state.prChart.update();
            return;
        }

        state.prChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'precision',
                        data: precision,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16,185,129,0.15)',
                        tension: 0.25,
                        spanGaps: true,
                        pointRadius: 2
                    },
                    {
                        label: 'recall',
                        data: recall,
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139,92,246,0.15)',
                        tension: 0.25,
                        spanGaps: true,
                        pointRadius: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { title: { display: true, text: 'batch' } },
                    y: { min: 0, max: 1, beginAtZero: true }
                }
            }
        });
    }

    function renderModeChart(rows) {
        var counts = {};
        MODE_LIST.forEach(function (m) { counts[m] = { caught: 0, missed: 0 }; });
        rows.forEach(function (r) {
            if (!r.actual_failure) return;
            var modes = (r.failure_modes && r.failure_modes.length) ? r.failure_modes : ['UNKNOWN'];
            var flagged = r.score >= state.threshold;
            modes.forEach(function (m) {
                if (!counts[m]) return;
                if (flagged) counts[m].caught++;
                else counts[m].missed++;
            });
        });

        var ctx = $('modeChart').getContext('2d');
        if (state.modeChart) {
            state.modeChart.data.labels = MODE_LIST;
            state.modeChart.data.datasets[0].data = MODE_LIST.map(function (m) { return counts[m].caught; });
            state.modeChart.data.datasets[1].data = MODE_LIST.map(function (m) { return counts[m].missed; });
            state.modeChart.update();
            return;
        }

        state.modeChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: MODE_LIST,
                datasets: [
                    {
                        label: 'caught',
                        data: MODE_LIST.map(function (m) { return counts[m].caught; }),
                        backgroundColor: 'rgba(16,185,129,0.7)',
                        stack: 'modes'
                    },
                    {
                        label: 'missed',
                        data: MODE_LIST.map(function (m) { return counts[m].missed; }),
                        backgroundColor: 'rgba(239,68,68,0.7)',
                        stack: 'modes'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { stacked: true },
                    y: { stacked: true, beginAtZero: true }
                }
            }
        });
    }

    // --- Banners -------------------------------------------------------------------
    function showLoading(show) {
        $('dashLoading').style.display = show ? 'flex' : 'none';
    }

    function showComplete() {
        $('completeBanner').classList.remove('banner-hidden');
        var scored = state.totalAvailable || 2000;
        $('completeText').textContent =
            'All ' + scored + ' machines have been replayed. Press Reset to run a new simulation.';
    }

    function hideComplete() {
        $('completeBanner').classList.add('banner-hidden');
    }

    function showError(err) {
        console.error(err);
        $('errorMsg').textContent = 'Connection problem: ' + (err && err.message ? err.message : 'unknown error');
        $('errorBanner').classList.remove('banner-hidden');
    }

    function hideError() {
        $('errorBanner').classList.add('banner-hidden');
    }

    function fetchJson(url, options) {
        return fetch(url, options).then(function (resp) {
            if (!resp.ok) {
                return resp.json().catch(function () { return {}; }).then(function (body) {
                    var message = (body && body.detail) ? body.detail : 'HTTP ' + resp.status;
                    throw new Error(message);
                });
            }
            return resp.json();
        });
    }
})();
