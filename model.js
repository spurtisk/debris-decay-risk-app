/* Browser-side inference for the decay-risk models.
   Mirrors debris_core.py: same features, same scaling, same trees.
   No DOM code here, so it can be tested in Node. */
(function (root) {
  "use strict";

  var DAY_MS = 86400000;

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  function clip01(v) { return Math.min(1, Math.max(0, v)); }

  // Feature vector in the same column order the model was trained with.
  // input: { type, orbit_class, source_country, perigee_km, eccentricity,
  //          inclination_deg, rcs_m2, launch_date ("YYYY-MM-DD") }
  function buildRow(model, input) {
    var analysis = Date.UTC(2026, 8, 26);
    var parts = String(input.launch_date).split("-").map(Number);
    var launch = Date.UTC(parts[0], parts[1] - 1, parts[2]);
    var ageDays = Math.floor((analysis - launch) / DAY_MS);
    var values = {
      perigee_km: Number(input.perigee_km),
      eccentricity: Number(input.eccentricity),
      inclination_deg: Number(input.inclination_deg),
      log_rcs: Math.log10(Number(input.rcs_m2)),
      object_age_years: ageDays / 365.25
    };
    var categorical = { type: input.type, orbit_class: input.orbit_class, source_country: input.source_country };
    return model.columns.map(function (name) {
      if (Object.prototype.hasOwnProperty.call(values, name)) return values[name];
      var eq = name.indexOf("=");
      var col = name.slice(0, eq), level = name.slice(eq + 1);
      return categorical[col] === level ? 1 : 0;
    });
  }

  function linearScore(block, x) {
    var s = block.intercept;
    for (var i = 0; i < x.length; i++) {
      s += block.coef[i] * ((x[i] - block.mean[i]) / block.scale[i]);
    }
    return s;
  }

  function walkTree(tree, x) {
    var node = 0;
    while (tree.f[node] !== -2) {
      node = x[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
    }
    return tree.p[node];
  }

  function forestProbability(forest, x) {
    var total = 0;
    for (var i = 0; i < forest.trees.length; i++) total += walkTree(forest.trees[i], x);
    return total / forest.trees.length;
  }

  // Full prediction for one object, plus per-feature contributions for the logistic model.
  function predict(models, input) {
    var xl = buildRow(models.linear, input);
    var xr = buildRow(models.forest_row, input);
    var yhat = linearScore(models.linear, xl);
    var risk = clip01(Math.pow(10, yhat) - models.meta.eps);
    var logit = linearScore(models.logistic, buildRow(models.logistic, input));
    var pLr = sigmoid(logit);
    var pRf = forestProbability(models.forest, xr);

    var contributions = models.logistic.columns.map(function (name, i) {
      var z = (buildRow(models.logistic, input)[i] - models.logistic.mean[i]) / models.logistic.scale[i];
      return { feature: name, contribution: models.logistic.coef[i] * z };
    });
    return {
      predicted_decay_risk: risk,
      log10_risk: yhat,
      p_high_logistic: pLr,
      p_high_forest: pRf,
      class_logistic: pLr >= models.logistic.threshold ? "high" : "normal",
      class_forest: pRf >= models.forest.threshold ? "high" : "normal",
      logistic_contributions: contributions,
      logit: logit
    };
  }

  // Attach the forest's column list to the models object so buildRow can use it.
  function prepare(models) {
    models.forest_row = { columns: models.forest.columns };
    return models;
  }

  root.DebrisModel = { prepare: prepare, predict: predict, buildRow: buildRow };
})(typeof window !== "undefined" ? window : globalThis);
