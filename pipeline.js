/* Decay-risk pipeline in the browser.
   Same steps as debris_core.py: CSV -> Silver rules C0-C14 -> Gold tables ->
   stratified 80/20 split -> Linear Regression, Logistic Regression and a
   Random Forest, with thresholds chosen on out-of-fold training predictions.
   Pure JavaScript, no DOM, so it can be tested in Node. */
(function (root) {
  "use strict";

  var ANALYSIS_UTC = Date.UTC(2026, 8, 26);
  var DAY = 86400000;
  var MU = 398600.4418;
  var EPS = 1e-4;
  var SEED = 42;
  var TRAIN_FRACTION = 0.8;
  var HIGH_TOP = 0.10;
  var PERIOD_TOL = 0.01;
  var N_TREES = 300;
  var MAX_DEPTH = 10;
  var MIN_LEAF = 3;
  var RIDGE = 1e-6;

  var TYPES = ["DEBRIS", "PAYLOAD", "ROCKET BODY"];
  var ORBITS = ["GEO", "HEO", "LEO", "MEO"];
  var COUNTRIES = ["CHINA", "EU", "INDIA", "JAPAN", "OTHER", "RUSSIA", "USA"];
  var CATS = [["type", TYPES], ["orbit_class", ORBITS], ["source_country", COUNTRIES]];
  var FEATURE_NUMERIC = ["perigee_km", "eccentricity", "inclination_deg", "log_rcs", "object_age_years"];

  var RAW_REQUIRED = [
    "type", "object_name", "orbit_class", "apogee_km", "perigee_km", "inclination_deg",
    "source_country", "eccentricity", "semi_major_axis_km", "period_minutes", "mean_motion",
    "speed_km_s", "rcs_m2", "launch_date", "launch_year", "decay_risk", "object_id"
  ];
  var NUMERIC = [
    "apogee_km", "perigee_km", "inclination_deg", "eccentricity", "semi_major_axis_km",
    "period_minutes", "mean_motion", "speed_km_s", "rcs_m2", "launch_year", "decay_risk"
  ];
  var SILVER_COLUMNS = [
    "object_id", "type", "orbit_class", "source_country", "perigee_km", "apogee_km",
    "inclination_deg", "eccentricity", "semi_major_axis_km", "period_minutes", "rcs_m2",
    "log_rcs", "rcs_outlier", "launch_date", "object_age_years", "altitude_band",
    "decay_risk", "log_decay_risk", "high_decay_risk"
  ];
  // Values pandas treats as missing when it reads a CSV.
  var NA_STRINGS = {
    "": 1, "#N/A": 1, "#N/A N/A": 1, "#NA": 1, "-1.#IND": 1, "-1.#QNAN": 1, "-NaN": 1,
    "-nan": 1, "1.#IND": 1, "1.#QNAN": 1, "<NA>": 1, "N/A": 1, "NA": 1, "NULL": 1,
    "NaN": 1, "None": 1, "n/a": 1, "nan": 1, "null": 1
  };

  function DataError(message) {
    this.name = "DataError";
    this.message = message;
  }
  DataError.prototype = Object.create(Error.prototype);
  DataError.prototype.constructor = DataError;

  // ------------------------------------------------------------ helpers ----
  function tick() { return new Promise(function (resolve) { setTimeout(resolve, 0); }); }
  function numAsc(a, b) { return a - b; }
  function clip01(v) { return Math.min(1, Math.max(0, v)); }
  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  // Python's round(): halves go to the even neighbour.
  function pyRound(x) {
    var f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  // Linear interpolation, the default for pandas and numpy quantiles.
  function quantile(sorted, q) {
    var n = sorted.length;
    if (!n) return NaN;
    var pos = (n - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  function mulberry32(a) {
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), a | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffle(arr, rng) {
    for (var i = arr.length - 1; i > 0; i--) {
      var j = Math.floor(rng() * (i + 1));
      var t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  // ------------------------------------------------------------- md5 ----
  // Needed to reproduce the Python split, which hashes "seed:object_id" with MD5.
  var MD5_S = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21];
  var MD5_K = [];
  for (var mk = 0; mk < 64; mk++) MD5_K[mk] = Math.floor(Math.abs(Math.sin(mk + 1)) * 4294967296) >>> 0;

  function rotl(x, c) { return ((x << c) | (x >>> (32 - c))) >>> 0; }

  function md5Hex(str) {
    var msg = new TextEncoder().encode(str);
    var len = msg.length;
    var total = ((((len + 8) >> 6) + 1) * 64);
    var buf = new Uint8Array(total);
    buf.set(msg);
    buf[len] = 0x80;
    var dv = new DataView(buf.buffer);
    var bits = len * 8;
    dv.setUint32(total - 8, bits >>> 0, true);
    dv.setUint32(total - 4, Math.floor(bits / 4294967296) >>> 0, true);

    var a0 = 0x67452301, b0 = 0xefcdab89, c0 = 0x98badcfe, d0 = 0x10325476;
    var M = new Array(16);
    for (var off = 0; off < total; off += 64) {
      for (var j = 0; j < 16; j++) M[j] = dv.getUint32(off + 4 * j, true);
      var A = a0, B = b0, C = c0, D = d0;
      for (var i = 0; i < 64; i++) {
        var F, g;
        if (i < 16) { F = (B & C) | (~B & D); g = i; }
        else if (i < 32) { F = (D & B) | (~D & C); g = (5 * i + 1) % 16; }
        else if (i < 48) { F = B ^ C ^ D; g = (3 * i + 5) % 16; }
        else { F = C ^ (B | ~D); g = (7 * i) % 16; }
        F = (F + A + MD5_K[i] + M[g]) >>> 0;
        A = D; D = C; C = B;
        B = (B + rotl(F, MD5_S[i])) >>> 0;
      }
      a0 = (a0 + A) >>> 0; b0 = (b0 + B) >>> 0; c0 = (c0 + C) >>> 0; d0 = (d0 + D) >>> 0;
    }
    var out = "";
    [a0, b0, c0, d0].forEach(function (w) {
      for (var k = 0; k < 4; k++) out += ((w >>> (8 * k)) & 255).toString(16).padStart(2, "0");
    });
    return out;
  }

  // ---------------------------------------------------------- csv ----
  function parseCSV(text) {
    if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
    var rows = [], row = [], field = "", i = 0, n = text.length, inQuotes = false;
    while (i < n) {
      var c = text[i];
      if (inQuotes) {
        if (c === '"') {
          if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
          inQuotes = false; i++; continue;
        }
        field += c; i++; continue;
      }
      if (c === '"') { inQuotes = true; i++; continue; }
      if (c === ",") { row.push(field); field = ""; i++; continue; }
      if (c === "\r" || c === "\n") {
        row.push(field); field = "";
        rows.push(row); row = [];
        i += (c === "\r" && text[i + 1] === "\n") ? 2 : 1;
        continue;
      }
      field += c; i++;
    }
    if (inQuotes) throw new DataError("The file has a quoted field that is never closed.");
    if (field !== "" || row.length) { row.push(field); rows.push(row); }
    return rows.filter(function (r) { return !(r.length === 1 && r[0] === ""); });
  }

  function nullIfNA(s) {
    if (s === undefined) return null;
    if (Object.prototype.hasOwnProperty.call(NA_STRINGS, s)) return null;
    return s;
  }

  // Bronze: read every field as text, as the Bronze notebook does.
  function loadCSV(text) {
    var table = parseCSV(text);
    if (!table.length) throw new DataError("The file is empty.");
    var header = table[0].map(function (h) { return h.trim().toLowerCase(); });
    var rows = [];
    for (var k = 1; k < table.length; k++) {
      var r = table[k];
      if (r.length > header.length) {
        throw new DataError("The file has a malformed line: line " + (k + 1) + " has " +
          r.length + " fields, but the header has " + header.length + ".");
      }
      var obj = {};
      for (var j = 0; j < header.length; j++) obj[header[j]] = j < r.length ? nullIfNA(r[j]) : null;
      rows.push(obj);
    }
    var missing = RAW_REQUIRED.filter(function (c) { return header.indexOf(c) < 0; });
    if (missing.length) throw new DataError("Missing columns: " + missing.join(", "));
    return { header: header, rows: rows };
  }

  var NUM_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
  function toNumber(s) {
    if (s === null || s === undefined) return null;
    var t = String(s).trim();
    if (!NUM_RE.test(t)) return null;
    var v = Number(t);
    return isFinite(v) ? v : null;
  }

  var ISO_RE = /^(\d{4})-(\d{1,2})-(\d{1,2})/;
  function toDayMs(s) {
    if (s === null || s === undefined) return null;
    var t = String(s).trim();
    var m = ISO_RE.exec(t);
    if (m) {
      var y = +m[1], mo = +m[2], d = +m[3];
      var ms = Date.UTC(y, mo - 1, d);
      var check = new Date(ms);
      if (check.getUTCMonth() !== mo - 1 || check.getUTCDate() !== d) return null;
      return ms;
    }
    var parsed = Date.parse(t);
    if (isNaN(parsed)) return null;
    var dt = new Date(parsed);
    return Date.UTC(dt.getFullYear(), dt.getMonth(), dt.getDate());
  }

  function isoDay(ms) { return new Date(ms).toISOString().slice(0, 10); }

  function inRange(v, a, b) { return v !== null && v >= a && v <= b; }

  // ------------------------------------------------------- silver ----
  // Rules C0 to C14, in the order of debris_core.clean_silver.
  var C10_RULES = [
    ["perigee_not_above_100km", function (r) { return r.perigee_km !== null && r.perigee_km > 100; }],
    ["apogee_below_perigee", function (r) { return r.apogee_km !== null && r.perigee_km !== null && r.apogee_km >= r.perigee_km; }],
    ["eccentricity_out_of_range", function (r) { return inRange(r.eccentricity, 0, 0.9999); }],
    ["inclination_out_of_range", function (r) { return inRange(r.inclination_deg, 0, 180); }],
    ["decay_risk_out_of_range", function (r) { return inRange(r.decay_risk, 0, 1); }],
    ["rcs_not_positive", function (r) { return r.rcs_m2 !== null && r.rcs_m2 > 0; }],
    ["launch_year_before_1957", function (r) { return r.launch_year !== null && r.launch_year >= 1957; }],
    ["launch_date_after_analysis_date", function (r) { return r.launch_ms !== null && r.launch_ms <= ANALYSIS_UTC; }]
  ];

  function cleanSilver(rawRows, header) {
    var df = [];
    var rejects = [];
    var audit = [];
    var i;

    rawRows.forEach(function (r) {
      var o = {};
      NUMERIC.forEach(function (c) { o[c] = null; });
      o.object_id = r.object_id;
      o.object_name = r.object_name;
      o.type = r.type;
      o.orbit_class = r.orbit_class;
      o.source_country = r.source_country;
      o.src = r;
      o.launch_raw = r.launch_date;
      o.launch_year_raw = r.launch_year;
      o.key = JSON.stringify(header.map(function (h) { return r[h]; }));
      df.push(o);
    });

    function log(step, rowsIn, note) {
      audit.push({ step: step, rows_in: rowsIn, rows_out: df.length, rows_removed: rowsIn - df.length, note: note });
    }
    function reject(pred, step, reason) {
      var keep = [];
      for (var k = 0; k < df.length; k++) {
        if (pred(df[k])) rejects.push({ step: step, reason: reason });
        else keep.push(df[k]);
      }
      df = keep;
    }
    function dedupe(keyOf, step, reason) {
      var seen = Object.create(null);
      reject(function (r) {
        var key = keyOf(r);
        if (seen[key]) return true;
        seen[key] = true;
        return false;
      }, step, reason);
    }

    var n = df.length;
    log("C0 read CSV", n, "every line parsed with " + header.length + " fields");

    n = df.length;
    dedupe(function (r) { return r.key; }, "C2", "exact_duplicate");
    log("C2 remove exact duplicate rows", n, "all columns equal");

    n = df.length;
    dedupe(function (r) { return r.object_id === null ? "\u0001null" : r.object_id; }, "C3", "duplicate_object_id");
    log("C3 unique object_id", n, "object_id is the catalog key");

    n = df.length;
    df.forEach(function (r) {
      ["type", "orbit_class", "source_country"].forEach(function (c) {
        if (r[c] !== null) r[c] = r[c].trim().toUpperCase();
      });
    });
    log("C4 clean category text", n, "trim and uppercase");

    n = df.length;
    var renames = { "ROCKET_BODY": "ROCKET BODY", "RB": "ROCKET BODY", "R": "ROCKET BODY", "B": "ROCKET BODY" };
    df.forEach(function (r) { if (r.type !== null && renames.hasOwnProperty(r.type)) r.type = renames[r.type]; });
    var countryCounts = Object.create(null), countryTotal = 0;
    df.forEach(function (r) {
      if (r.source_country !== null) { countryCounts[r.source_country] = (countryCounts[r.source_country] || 0) + 1; countryTotal++; }
    });
    var rare = Object.keys(countryCounts).filter(function (c) { return countryCounts[c] / countryTotal < 0.01; });
    df.forEach(function (r) { if (r.source_country !== null && rare.indexOf(r.source_country) >= 0) r.source_country = "OTHER"; });
    log("C5 merge synonyms and rare countries", n, "rocket body variants merged; countries under 1% become OTHER");

    n = df.length;
    df.forEach(function (r) {
      NUMERIC.forEach(function (c) { r[c] = toNumber(r.src[c]); });
    });
    log("C6 coerce numeric columns", n, "unparseable values become null");

    n = df.length;
    df.forEach(function (r) {
      r.launch_ms = toDayMs(r.launch_raw);
      r.launch_year = r.launch_ms !== null ? new Date(r.launch_ms).getUTCFullYear() : toNumber(r.launch_year_raw);
    });
    log("C7 parse dates, reconcile year", n, "launch_year follows launch_date");

    n = df.length;
    reject(function (r) { return ORBITS.indexOf(r.orbit_class) < 0; }, "C8", "unknown_orbit");
    reject(function (r) { return TYPES.indexOf(r.type) < 0; }, "C8", "unknown_type");
    log("C8 orbit and type values present", n, "unknown orbit or type removed");

    n = df.length;
    var imputed = df.filter(function (r) { return r.rcs_m2 === null; }).length;
    var byType = {};
    df.forEach(function (r) { if (r.rcs_m2 !== null) (byType[r.type] = byType[r.type] || []).push(r.rcs_m2); });
    var medianByType = {};
    Object.keys(byType).forEach(function (t) { medianByType[t] = quantile(byType[t].slice().sort(numAsc), 0.5); });
    df.forEach(function (r) { if (r.rcs_m2 === null && medianByType.hasOwnProperty(r.type)) r.rcs_m2 = medianByType[r.type]; });
    reject(function (r) { return r.decay_risk === null; }, "C9", "missing_decay_risk");
    log("C9 fill and reject missing values", n, imputed + " missing RCS values filled by type median");

    n = df.length;
    var keep10 = [];
    df.forEach(function (r) {
      var failed = null;
      for (var k = 0; k < C10_RULES.length; k++) {
        if (!C10_RULES[k][1](r)) { failed = C10_RULES[k][0]; break; }
      }
      if (failed !== null) rejects.push({ step: "C10", reason: failed });
      else keep10.push(r);
    });
    df = keep10;
    log("C10 remove impossible values", n, "first failing rule recorded for each rejected row");

    n = df.length;
    var worst = 0;
    df.forEach(function (r) {
      if (r.semi_major_axis_km === null || r.period_minutes === null) { r.rel_err = NaN; return; }
      var calc = 2 * Math.PI * Math.sqrt(Math.pow(r.semi_major_axis_km, 3) / MU) / 60;
      r.rel_err = Math.abs(r.period_minutes - calc) / calc;
      if (isFinite(r.rel_err) && r.rel_err > worst) worst = r.rel_err;
    });
    reject(function (r) { return !(r.rel_err <= PERIOD_TOL); }, "C11", "period_mismatch_kepler");
    log("C11 Kepler period check", n, "largest relative period error " + worst.toExponential(2) + " (speed check is in the Spark notebook)");

    n = df.length;
    var rcsSorted = df.filter(function (r) { return r.rcs_m2 !== null; }).map(function (r) { return r.rcs_m2; }).sort(numAsc);
    var q1 = quantile(rcsSorted, 0.25), q3 = quantile(rcsSorted, 0.75), iqr = q3 - q1;
    var outliers = 0;
    df.forEach(function (r) {
      r.rcs_outlier = r.rcs_m2 !== null && (r.rcs_m2 < q1 - 1.5 * iqr || r.rcs_m2 > q3 + 1.5 * iqr);
      if (r.rcs_outlier) outliers++;
      r.log_rcs = r.rcs_m2 !== null && r.rcs_m2 > 0 ? Math.log10(r.rcs_m2) : null;
    });
    log("C12 outlier flag and log of RCS", n, outliers + " IQR outliers kept and flagged");

    n = df.length;
    df.forEach(function (r) {
      r.object_age_years = Math.floor((ANALYSIS_UTC - r.launch_ms) / DAY) / 365.25;
      var base = Math.floor(r.perigee_km / 100) * 100;
      r.altitude_band = base + "-" + (base + 100);
      r.log_decay_risk = Math.log10(r.decay_risk + EPS);
    });
    var decaySorted = df.map(function (r) { return r.decay_risk; }).sort(numAsc);
    var p90 = quantile(decaySorted, 1 - HIGH_TOP);
    df.forEach(function (r) { r.high_decay_risk = r.decay_risk >= p90 ? 1 : 0; });
    log("C13 derived features", n, "altitude band, log target, high-risk flag (top " + Math.round(HIGH_TOP * 100) + "%)");

    n = df.length;
    var silver = df.map(function (r) {
      return {
        object_id: r.object_id, type: r.type, orbit_class: r.orbit_class, source_country: r.source_country,
        perigee_km: r.perigee_km, apogee_km: r.apogee_km, inclination_deg: r.inclination_deg,
        eccentricity: r.eccentricity, semi_major_axis_km: r.semi_major_axis_km,
        period_minutes: r.period_minutes, rcs_m2: r.rcs_m2, log_rcs: r.log_rcs,
        rcs_outlier: r.rcs_outlier, launch_date: isoDay(r.launch_ms),
        object_age_years: r.object_age_years, altitude_band: r.altitude_band,
        decay_risk: r.decay_risk, log_decay_risk: r.log_decay_risk,
        high_decay_risk: r.high_decay_risk
      };
    });
    log("C14 final Silver schema", n, "final columns");

    var summary = {};
    rejects.forEach(function (x) {
      var key = x.step + "|" + x.reason;
      summary[key] = summary[key] || { step: x.step, reason: x.reason, count: 0 };
      summary[key].count++;
    });
    return {
      silver: silver,
      audit: audit,
      rejectCount: rejects.length,
      rejectSummary: Object.keys(summary).map(function (k) { return summary[k]; })
        .sort(function (a, b) { return a.step < b.step ? -1 : a.step > b.step ? 1 : 0; })
    };
  }

  // -------------------------------------------------------- gold ----
  function goldTables(silver) {
    var groups = {}, countries = {};
    silver.forEach(function (r) {
      var key = [r.altitude_band, r.orbit_class, r.type].join("\u0001");
      var g = groups[key];
      if (!g) g = groups[key] = { altitude_band: r.altitude_band, orbit_class: r.orbit_class, type: r.type, decays: [], high: 0, logs: [] };
      g.decays.push(r.decay_risk); g.high += r.high_decay_risk;
      if (r.log_rcs !== null) g.logs.push(r.log_rcs);
      var c = countries[r.source_country];
      if (!c) c = countries[r.source_country] = { source_country: r.source_country, decays: [], high: 0 };
      c.decays.push(r.decay_risk); c.high += r.high_decay_risk;
    });
    var mean = function (a) { return a.length ? a.reduce(function (s, v) { return s + v; }, 0) / a.length : NaN; };
    var summary = Object.keys(groups).sort().map(function (k) {
      var g = groups[k];
      return {
        altitude_band: g.altitude_band, orbit_class: g.orbit_class, type: g.type,
        objects: g.decays.length,
        median_decay_risk: quantile(g.decays.slice().sort(numAsc), 0.5),
        high_risk_share: g.high / g.decays.length,
        mean_log_rcs: mean(g.logs)
      };
    });
    var byCountry = Object.keys(countries).sort().map(function (k) {
      var c = countries[k];
      return {
        source_country: c.source_country, objects: c.decays.length, high_risk_objects: c.high,
        median_decay_risk: quantile(c.decays.slice().sort(numAsc), 0.5)
      };
    });
    return { summary: summary, country: byCountry };
  }

  // ------------------------------------------------------ split ----
  function stratifiedSplit(silver, fraction, seed) {
    var n = silver.length;
    var hashes = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      hashes[i] = parseInt(md5Hex(seed + ":" + silver[i].object_id).slice(0, 12), 16);
    }
    var isTrain = new Uint8Array(n);
    [0, 1].forEach(function (cls) {
      var idx = [];
      for (var k = 0; k < n; k++) if (silver[k].high_decay_risk === cls) idx.push(k);
      idx.sort(function (a, b) { return (hashes[a] - hashes[b]) || (a - b); });
      var nTrain = pyRound(fraction * idx.length);
      for (var m = 0; m < nTrain; m++) isTrain[idx[m]] = 1;
    });
    return isTrain;
  }

  // ------------------------------------------------ design matrix ----
  function designColumns(dropFirst) {
    var cols = FEATURE_NUMERIC.slice();
    CATS.forEach(function (c) {
      var levels = dropFirst ? c[1].slice(1) : c[1];
      levels.forEach(function (lv) { cols.push(c[0] + "=" + lv); });
    });
    return cols;
  }

  function designRow(r, cols) {
    return cols.map(function (name) {
      var eq = name.indexOf("=");
      if (eq < 0) return r[name];
      return r[name.slice(0, eq)] === name.slice(eq + 1) ? 1 : 0;
    });
  }

  function fitScaler(X) {
    var n = X.length, d = X[0].length, mean = new Array(d).fill(0), scale = new Array(d).fill(0);
    for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) mean[j] += X[i][j];
    for (j = 0; j < d; j++) mean[j] /= n;
    for (i = 0; i < n; i++) for (j = 0; j < d; j++) { var t = X[i][j] - mean[j]; scale[j] += t * t; }
    for (j = 0; j < d; j++) { scale[j] = Math.sqrt(scale[j] / n); if (!(scale[j] > 0)) scale[j] = 1; }
    return { mean: mean, scale: scale };
  }

  function applyScaler(sc, x) {
    return x.map(function (v, j) { return (v - sc.mean[j]) / sc.scale[j]; });
  }

  function dot(a, b) {
    var s = 0;
    for (var i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  // Solve A x = b by Gaussian elimination with partial pivoting.
  function solve(A, b) {
    var n = b.length, M = [], i, j, k;
    for (i = 0; i < n; i++) M.push(A[i].slice().concat([b[i]]));
    for (k = 0; k < n; k++) {
      var p = k;
      for (i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
      var tmp = M[k]; M[k] = M[p]; M[p] = tmp;
      var piv = M[k][k] || 1e-12;
      for (i = k + 1; i < n; i++) {
        var f = M[i][k] / piv;
        if (f === 0) continue;
        for (j = k; j <= n; j++) M[i][j] -= f * M[k][j];
      }
    }
    var x = new Array(n).fill(0);
    for (i = n - 1; i >= 0; i--) {
      var s = M[i][n];
      for (j = i + 1; j < n; j++) s -= M[i][j] * x[j];
      x[i] = s / (M[i][i] || 1e-12);
    }
    return x;
  }

  // Linear Regression on standardised features (ordinary least squares).
  function fitLinear(Z, y) {
    var n = Z.length, d = Z[0].length, i, j, k;
    var ybar = y.reduce(function (s, v) { return s + v; }, 0) / n;
    var A = [], b = new Array(d).fill(0);
    for (j = 0; j < d; j++) A.push(new Array(d).fill(0));
    for (i = 0; i < n; i++) {
      var z = Z[i], r = y[i] - ybar;
      for (j = 0; j < d; j++) {
        b[j] += z[j] * r;
        for (k = 0; k < d; k++) A[j][k] += z[j] * z[k];
      }
    }
    for (j = 0; j < d; j++) A[j][j] += 1e-9;
    return { coef: solve(A, b), intercept: ybar };
  }

  // Logistic Regression by Newton steps (IRLS) with a very small ridge term,
  // which is close to the unregularised fit that the Python code requests.
  function fitLogistic(Z, y) {
    var n = Z.length, d = Z[0].length, p = d + 1, i, j, k, it;
    var X = Z.map(function (z) { return [1].concat(z); });
    // Penalised log-likelihood (numerically stable form).
    function loglik(w) {
      var ll = 0;
      for (var r = 0; r < n; r++) {
        var eta = dot(w, X[r]);
        var softplus = eta > 0 ? eta + Math.log1p(Math.exp(-eta)) : Math.log1p(Math.exp(eta));
        ll += y[r] * eta - softplus;
      }
      var pen = 0;
      for (var q = 0; q < p; q++) pen += w[q] * w[q];
      return ll - 0.5 * RIDGE * pen;
    }
    var w = new Array(p).fill(0);
    var current = loglik(w);
    for (it = 0; it < 200; it++) {
      var H = [], g = new Array(p).fill(0);
      for (j = 0; j < p; j++) H.push(new Array(p).fill(0));
      for (i = 0; i < n; i++) {
        var x = X[i];
        var mu = sigmoid(dot(w, x));
        var wt = mu * (1 - mu) + 1e-12;
        var resid = y[i] - mu;
        for (j = 0; j < p; j++) {
          g[j] += x[j] * resid;
          for (k = 0; k < p; k++) H[j][k] += wt * x[j] * x[k];
        }
      }
      for (j = 0; j < p; j++) { g[j] -= RIDGE * w[j]; H[j][j] += RIDGE; }
      var dir = solve(H, g);
      // Backtracking: halve the Newton step until the log-likelihood rises.
      var t = 1, accepted = false, next = null, nextLL = current;
      while (t > 1e-10) {
        next = w.map(function (v, q) { return v + t * dir[q]; });
        nextLL = loglik(next);
        if (nextLL >= current) { accepted = true; break; }
        t /= 2;
      }
      if (!accepted) break;
      var moved = 0;
      for (j = 0; j < p; j++) moved = Math.max(moved, Math.abs(next[j] - w[j]));
      var gain = nextLL - current;
      w = next; current = nextLL;
      if (moved < 1e-9 || gain < 1e-12) break;
    }
    return { intercept: w[0], coef: w.slice(1) };
  }

  function logisticProb(model, Z) {
    return Z.map(function (z) { return sigmoid(model.intercept + dot(model.coef, z)); });
  }

  // ------------------------------------------------------ metrics ----
  function rankAUC(y, s) {
    var n = s.length, idx = [], i;
    for (i = 0; i < n; i++) idx.push(i);
    idx.sort(function (a, b) { return s[a] - s[b]; });
    var rank = new Float64Array(n);
    for (i = 0; i < n;) {
      var j = i;
      while (j + 1 < n && s[idx[j + 1]] === s[idx[i]]) j++;
      var avg = (i + j + 2) / 2;
      for (var k = i; k <= j; k++) rank[idx[k]] = avg;
      i = j + 1;
    }
    var nPos = 0, sumPos = 0;
    for (i = 0; i < n; i++) if (y[i] === 1) { nPos++; sumPos += rank[i]; }
    var nNeg = n - nPos;
    if (nPos === 0 || nNeg === 0) return NaN;
    return (sumPos - nPos * (nPos + 1) / 2) / (nPos * nNeg);
  }

  function averagePrecision(y, s) {
    var n = s.length, idx = [], i;
    for (i = 0; i < n; i++) idx.push(i);
    idx.sort(function (a, b) { return s[b] - s[a]; });
    var P = 0;
    for (i = 0; i < n; i++) if (y[i] === 1) P++;
    if (P === 0) return NaN;
    var tp = 0, fp = 0, prevRecall = 0, ap = 0;
    for (i = 0; i < n;) {
      var t = s[idx[i]];
      while (i < n && s[idx[i]] === t) {
        if (y[idx[i]] === 1) tp++; else fp++;
        i++;
      }
      var prec = tp / (tp + fp), rec = tp / P;
      ap += (rec - prevRecall) * prec;
      prevRecall = rec;
    }
    return ap;
  }

  // Threshold that maximises F1 over the out-of-fold scores; ties go to the lower threshold.
  function bestF1Threshold(y, p) {
    var n = p.length, idx = [], i;
    for (i = 0; i < n; i++) idx.push(i);
    idx.sort(function (a, b) { return p[b] - p[a]; });
    var P = 0;
    for (i = 0; i < n; i++) if (y[i] === 1) P++;
    var tp = 0, fp = 0, best = -1, bestT = 0.5;
    for (i = 0; i < n;) {
      var t = p[idx[i]];
      while (i < n && p[idx[i]] === t) {
        if (y[idx[i]] === 1) tp++; else fp++;
        i++;
      }
      var prec = tp / (tp + fp), rec = P ? tp / P : 0;
      var f1 = 2 * prec * rec / Math.max(prec + rec, 1e-12);
      if (f1 >= best) { best = f1; bestT = t; }
    }
    return bestT;
  }

  function classificationMetrics(y, p, thr) {
    var tp = 0, fp = 0, tn = 0, fn = 0;
    for (var i = 0; i < y.length; i++) {
      var pred = p[i] >= thr ? 1 : 0;
      if (pred && y[i]) tp++;
      else if (pred && !y[i]) fp++;
      else if (!pred && y[i]) fn++;
      else tn++;
    }
    var precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    var recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    var f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
    return [
      ["ROC-AUC", rankAUC(y, p)], ["PR-AUC", averagePrecision(y, p)], ["F1", f1],
      ["Precision", precision], ["Recall", recall],
      ["TP", tp], ["FP", fp], ["FN", fn], ["TN", tn]
    ];
  }

  function r2(yTrue, yPred) {
    var n = yTrue.length, mean = 0, i;
    for (i = 0; i < n; i++) mean += yTrue[i];
    mean /= n;
    var ssRes = 0, ssTot = 0;
    for (i = 0; i < n; i++) {
      ssRes += (yTrue[i] - yPred[i]) * (yTrue[i] - yPred[i]);
      ssTot += (yTrue[i] - mean) * (yTrue[i] - mean);
    }
    return 1 - ssRes / ssTot;
  }

  // ------------------------------------------------------ forest ----
  function gini(W, W1) {
    if (W <= 0) return 0;
    var q = W1 / W;
    return 1 - q * q - (1 - q) * (1 - q);
  }

  function sampleFeatures(nf, mtry, rng) {
    var pool = [], i;
    for (i = 0; i < nf; i++) pool.push(i);
    for (i = 0; i < mtry; i++) {
      var j = i + Math.floor(rng() * (nf - i));
      var t = pool[i]; pool[i] = pool[j]; pool[j] = t;
    }
    return pool.slice(0, mtry);
  }

  // One classification tree on weighted, bootstrapped rows. X is column-major.
  function buildTree(X, y, w, bag, rng, maxDepth, minLeaf, mtry, imp) {
    var f = [], thr = [], l = [], r = [], p = [];

    function grow(idx, depth) {
      var W = 0, W1 = 0, a, i;
      for (a = 0; a < idx.length; a++) { i = idx[a]; W += w[i]; if (y[i]) W1 += w[i]; }
      var id = f.length;
      f.push(-2); thr.push(0); l.push(-1); r.push(-1); p.push(W > 0 ? W1 / W : 0);
      if (depth >= maxDepth || idx.length < 2 * minLeaf || W1 === 0 || W1 === W) return id;

      var g0 = gini(W, W1), best = null;
      var feats = sampleFeatures(X.length, mtry, rng);
      for (var fi = 0; fi < feats.length; fi++) {
        var j = feats[fi], xj = X[j];
        var order = idx.slice().sort(function (u, v) { return xj[u] - xj[v]; });
        var WL = 0, WL1 = 0, nl = 0, m = order.length;
        for (var k = 0; k < m - 1; k++) {
          var ik = order[k];
          WL += w[ik]; if (y[ik]) WL1 += w[ik]; nl++;
          var v0 = xj[ik], v1 = xj[order[k + 1]];
          if (v0 === v1) continue;
          if (nl < minLeaf || m - nl < minLeaf) continue;
          var WR = W - WL, WR1 = W1 - WL1;
          var gain = W * g0 - WL * gini(WL, WL1) - WR * gini(WR, WR1);
          if (best === null || gain > best.gain) {
            var cut = (v0 + v1) / 2;
            if (cut === v1) cut = v0;
            best = { gain: gain, f: j, thr: cut };
          }
        }
      }
      if (best === null || best.gain <= 1e-12) return id;

      var left = [], right = [];
      for (a = 0; a < idx.length; a++) {
        i = idx[a];
        if (X[best.f][i] <= best.thr) left.push(i); else right.push(i);
      }
      imp[best.f] += best.gain;
      f[id] = best.f; thr[id] = best.thr;
      l[id] = grow(left, depth + 1);
      r[id] = grow(right, depth + 1);
      return id;
    }

    grow(bag, 0);
    return { f: f, t: thr, l: l, r: r, p: p };
  }

  // Random Forest with bootstrap samples, sqrt(features) per split and balanced class weights.
  async function fitForest(X, y, nTrees, maxDepth, seed, onTree) {
    var n = y.length, d = X.length, rng = mulberry32(seed);
    var n0 = 0, n1 = 0, i;
    for (i = 0; i < n; i++) if (y[i]) n1++; else n0++;
    var cw = [n / (2 * n0), n / (2 * n1)];
    var mtry = Math.max(1, Math.floor(Math.sqrt(d)));
    var trees = [], impSum = new Float64Array(d);
    for (var t = 0; t < nTrees; t++) {
      var cnt = new Float64Array(n);
      for (i = 0; i < n; i++) cnt[Math.floor(rng() * n)] += 1;
      var w = new Float64Array(n), bag = [];
      for (i = 0; i < n; i++) if (cnt[i] > 0) { w[i] = cnt[i] * cw[y[i]]; bag.push(i); }
      var impT = new Float64Array(d);
      trees.push(buildTree(X, y, w, bag, rng, maxDepth, MIN_LEAF, mtry, impT));
      var total = 0;
      for (var j = 0; j < d; j++) total += impT[j];
      if (total > 0) for (j = 0; j < d; j++) impSum[j] += impT[j] / total;
      if (onTree && t % 5 === 4) { onTree(t + 1, nTrees); await tick(); }
    }
    var importance = [];
    for (j = 0; j < d; j++) importance.push(impSum[j] / nTrees);
    return { trees: trees, importance: importance };
  }

  function walkTree(tree, x) {
    var node = 0;
    while (tree.f[node] !== -2) {
      node = x[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
    }
    return tree.p[node];
  }

  function forestProb(trees, x) {
    var s = 0;
    for (var k = 0; k < trees.length; k++) s += walkTree(trees[k], x);
    return s / trees.length;
  }

  function stratifiedFolds(y, k, seed) {
    var rng = mulberry32(seed), fold = new Int32Array(y.length);
    [0, 1].forEach(function (cls) {
      var idx = [];
      for (var i = 0; i < y.length; i++) if (y[i] === cls) idx.push(i);
      shuffle(idx, rng);
      idx.forEach(function (ii, j) { fold[ii] = j % k; });
    });
    return fold;
  }

  // ------------------------------------------------------ run ----
  async function run(text, opts) {
    opts = opts || {};
    var progress = opts.onProgress || function () {};
    var nTrees = opts.nTrees || N_TREES;
    var depth = opts.maxDepth || MAX_DEPTH;
    var started = Date.now();
    async function step(label, pct) { progress(label, pct); await tick(); }

    await step("Reading the CSV (Bronze)", 3);
    var loaded = loadCSV(text);

    var rawMissing = 0;
    loaded.rows.forEach(function (row) {
      loaded.header.forEach(function (h) { if (row[h] === null) rawMissing++; });
    });

    await step("Cleaning rules C0 to C14 (Silver)", 8);
    var cleaned = cleanSilver(loaded.rows, loaded.header);
    var silver = cleaned.silver;
    if (silver.length < 100) throw new DataError("Too few rows remain after cleaning to train the models.");

    await step("Gold summary tables", 16);
    var gold = goldTables(silver);

    await step("Stratified 80/20 split", 20);
    var isTrain = stratifiedSplit(silver, TRAIN_FRACTION, SEED);
    var train = [], test = [];
    silver.forEach(function (r, k) { (isTrain[k] ? train : test).push(r); });

    var linCols = designColumns(true), rfCols = designColumns(false);
    var XlinTr = train.map(function (r) { return designRow(r, linCols); });
    var XlinTe = test.map(function (r) { return designRow(r, linCols); });
    var XrfTr = train.map(function (r) { return designRow(r, rfCols); });
    var XrfTe = test.map(function (r) { return designRow(r, rfCols); });
    var yLogTr = train.map(function (r) { return r.log_decay_risk; });
    var yLogTe = test.map(function (r) { return r.log_decay_risk; });
    var yHiTr = Uint8Array.from(train.map(function (r) { return r.high_decay_risk; }));
    var yHiTe = Uint8Array.from(test.map(function (r) { return r.high_decay_risk; }));
    var rawTe = test.map(function (r) { return r.decay_risk; });

    await step("Linear Regression on log10 risk", 24);
    var linScaler = fitScaler(XlinTr);
    var ZTr = XlinTr.map(function (x) { return applyScaler(linScaler, x); });
    var ZTe = XlinTe.map(function (x) { return applyScaler(linScaler, x); });
    var lin = fitLinear(ZTr, yLogTr);
    var predLog = ZTe.map(function (z) { return lin.intercept + dot(lin.coef, z); });
    var predRaw = predLog.map(function (v) { return clip01(Math.pow(10, v) - EPS); });
    var linMetrics = [
      ["R2 log scale", r2(yLogTe, predLog)],
      ["R2 original scale", r2(rawTe, predRaw)],
      ["RMSE", Math.sqrt(rawTe.reduce(function (s, v, k) { return s + Math.pow(v - predRaw[k], 2); }, 0) / rawTe.length)],
      ["MAE", rawTe.reduce(function (s, v, k) { return s + Math.abs(v - predRaw[k]); }, 0) / rawTe.length]
    ];

    // Out-of-fold scores on training rows only, for the two thresholds.
    var folds = stratifiedFolds(yHiTr, 5, SEED);
    var oofLR = new Float64Array(train.length), oofRF = new Float64Array(train.length);
    for (var fold = 0; fold < 5; fold++) {
      var trIdx = [], teIdx = [];
      for (var k = 0; k < train.length; k++) (folds[k] === fold ? teIdx : trIdx).push(k);
      var subX = trIdx.map(function (k) { return XlinTr[k]; });
      var sc = fitScaler(subX);
      var lrF = fitLogistic(subX.map(function (x) { return applyScaler(sc, x); }),
        trIdx.map(function (k) { return yHiTr[k]; }));
      var pl = logisticProb(lrF, teIdx.map(function (k) { return applyScaler(sc, XlinTr[k]); }));
      teIdx.forEach(function (k, m) { oofLR[k] = pl[m]; });

      await step("Random Forest, fold " + (fold + 1) + " of 5 for the threshold", 26 + fold * 10);
      var colsSub = [];
      for (var j = 0; j < rfCols.length; j++) {
        var col = new Float64Array(trIdx.length);
        for (var m2 = 0; m2 < trIdx.length; m2++) col[m2] = XrfTr[trIdx[m2]][j];
        colsSub.push(col);
      }
      var ySub = Uint8Array.from(trIdx.map(function (k) { return yHiTr[k]; }));
      var forestF = await fitForest(colsSub, ySub, nTrees, depth, SEED, function (done, total) {
        progress("Random Forest, fold " + (fold + 1) + " of 5: tree " + done + " of " + total, 26 + fold * 10 + 10 * done / total);
      });
      teIdx.forEach(function (k) { oofRF[k] = forestProb(forestF.trees, XrfTr[k]); });
    }
    var thrLR = bestF1Threshold(yHiTr, oofLR);
    var thrRF = bestF1Threshold(yHiTr, oofRF);

    await step("Logistic Regression, final fit", 80);
    var logistic = fitLogistic(ZTr, yHiTr);
    var pLrTe = logisticProb(logistic, ZTe);

    await step("Random Forest, final fit", 82);
    var colsAll = [];
    for (j = 0; j < rfCols.length; j++) {
      var cAll = new Float64Array(train.length);
      for (var m3 = 0; m3 < train.length; m3++) cAll[m3] = XrfTr[m3][j];
      colsAll.push(cAll);
    }
    var forest = await fitForest(colsAll, yHiTr, nTrees, depth, SEED, function (done, total) {
      progress("Random Forest, final fit: tree " + done + " of " + total, 82 + 12 * done / total);
    });
    var pRfTe = XrfTe.map(function (x) { return forestProb(forest.trees, x); });

    await step("Scoring the test set", 96);
    var metrics = [];
    linMetrics.forEach(function (mv) { metrics.push({ model: "Linear Regression", metric: mv[0], value: mv[1] }); });
    classificationMetrics(yHiTe, pLrTe, thrLR).forEach(function (mv) {
      metrics.push({ model: "Logistic Regression", metric: mv[0], value: mv[1] });
    });
    classificationMetrics(yHiTe, pRfTe, thrRF).forEach(function (mv) {
      metrics.push({ model: "Random Forest", metric: mv[0], value: mv[1] });
    });

    var singleAuc = {};
    FEATURE_NUMERIC.forEach(function (c) { singleAuc[c] = rankAUC(yHiTe, test.map(function (r) { return r[c]; })); });

    var lrCoef = linCols.map(function (name, k) { return { feature: name, coefficient: logistic.coef[k] }; })
      .sort(function (a, b) { return Math.abs(b.coefficient) - Math.abs(a.coefficient); });
    var rfImp = rfCols.map(function (name, k) { return { feature: name, importance: forest.importance[k] }; })
      .sort(function (a, b) { return b.importance - a.importance; });

    var models = {
      meta: {
        source_file: opts.fileName || "uploaded.csv", raw_rows: loaded.rows.length,
        silver_rows: silver.length, rejected_rows: cleaned.rejectCount,
        n_train: train.length, n_test: test.length, analysis_date: "2026-09-26",
        eps: EPS, seed: SEED
      },
      features: { numeric: FEATURE_NUMERIC, types: TYPES, orbits: ORBITS, countries: COUNTRIES },
      linear: { columns: linCols, mean: linScaler.mean, scale: linScaler.scale, coef: lin.coef, intercept: lin.intercept },
      logistic: {
        columns: linCols, mean: linScaler.mean, scale: linScaler.scale,
        coef: logistic.coef, intercept: logistic.intercept, threshold: thrLR
      },
      forest: { columns: rfCols, threshold: thrRF, trees: forest.trees.map(function (t) {
        return { f: t.f, t: t.t, l: t.l, r: t.r, p: t.p };
      }) },
      metrics: metrics,
      single_feature_auc: singleAuc,
      rf_importance: rfImp,
      lr_coef: lrCoef
    };

    await step("Done", 100);
    return {
      rawRows: loaded.rows.length,
      rawColumns: loaded.header.length,
      rawMissing: rawMissing,
      silver: silver,
      audit: cleaned.audit,
      rejectCount: cleaned.rejectCount,
      rejectSummary: cleaned.rejectSummary,
      gold: gold,
      split: {
        train: train.length, test: test.length,
        trainHigh: yHiTr.reduce(function (s, v) { return s + v; }, 0),
        testHigh: yHiTe.reduce(function (s, v) { return s + v; }, 0)
      },
      thresholds: { lr: thrLR, rf: thrRF },
      testRows: test,
      scores: { lr: Array.from(pLrTe), rf: Array.from(pRfTe), actual: Array.from(yHiTe) },
      models: models,
      seconds: (Date.now() - started) / 1000
    };
  }

  var api = {
    run: run, loadCSV: loadCSV, cleanSilver: cleanSilver, goldTables: goldTables,
    stratifiedSplit: stratifiedSplit, md5Hex: md5Hex, bestF1Threshold: bestF1Threshold,
    rankAUC: rankAUC, averagePrecision: averagePrecision,
    constants: { ANALYSIS_UTC: ANALYSIS_UTC, EPS: EPS, SEED: SEED, TYPES: TYPES, ORBITS: ORBITS,
      COUNTRIES: COUNTRIES, FEATURE_NUMERIC: FEATURE_NUMERIC, N_TREES: N_TREES, MAX_DEPTH: MAX_DEPTH }
  };
  root.DebrisPipeline = api;
  root.DebrisError = DataError;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
