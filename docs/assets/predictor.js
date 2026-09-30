/* Browser inference for the fitted project models; no prediction server needed. */
(() => {
  "use strict";

  const FEATURES = ["湿度", "湿度变化", "温度", "温度变化", "弦编号"];
  const INPUT_KEYS = ["humidity", "humidity_change", "temperature", "temperature_change"];
  let bundle = null;

  const copy = value => JSON.parse(JSON.stringify(value));
  const finite = value => typeof value === "number" && Number.isFinite(value);

  function verifyBundle(data) {
    if (!data || data.format_version !== 1 || data.default !== "random_forest") {
      throw new Error("模型文件格式无效，请重新加载网页。");
    }
    for (const key of ["random_forest", "ridge"]) {
      const model = data.models?.[key];
      const metadata = data.metadata?.[key];
      if (model?.type !== key || metadata?.key !== key ||
          JSON.stringify(metadata.features) !== JSON.stringify(FEATURES)) {
        throw new Error("模型文件不完整，请重新加载网页。");
      }
    }
    const ridge = data.models.ridge;
    if (![ridge.mean, ridge.scale, ridge.coef].every(
      values => Array.isArray(values) && values.length === 5 && values.every(finite)
    ) || !finite(ridge.intercept) || ridge.scale.some(value => value <= 0) ||
        !Array.isArray(data.models.random_forest.trees) ||
        data.models.random_forest.trees.length === 0) {
      throw new Error("模型参数无效，请重新加载网页。");
    }
  }

  async function load(url = "assets/models.json") {
    // The page passes a relative asset URL, so it also works under /repository/.
    const response = await fetch(url);
    if (!response.ok) throw new Error("模型加载失败，请刷新网页后重试。");
    const data = await response.json();
    verifyBundle(data);
    bundle = data;
    return { default: bundle.default, models: copy(bundle.metadata) };
  }

  function forestPrediction(model, row) {
    // sklearn trees cast incoming features to float32 before comparing against
    // the original float64 split threshold. Keep that boundary behavior here.
    const input = row.map(Math.fround);
    let total = 0;
    for (const tree of model.trees) {
      let node = 0;
      while (tree.children_left[node] !== -1) {
        node = input[tree.feature[node]] <= tree.threshold[node]
          ? tree.children_left[node]
          : tree.children_right[node];
      }
      total += tree.value[node];
    }
    return total / model.trees.length;
  }

  function ridgePrediction(model, row) {
    // StandardScaler transforms first, then Ridge computes the dot product.
    // JavaScript Number retains float64 precision through both operations.
    let value = 0;
    for (let feature = 0; feature < row.length; feature += 1) {
      const scaled = (row[feature] - model.mean[feature]) / model.scale[feature];
      value += scaled * model.coef[feature];
    }
    return value + model.intercept;
  }

  function predict(payload) {
    if (!bundle) throw new Error("模型尚未加载完成，请稍后再试。");
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      throw new Error("请求格式无效。");
    }
    const modelKey = Object.hasOwn(payload, "model") ? payload.model : "random_forest";
    if (typeof modelKey !== "string" || !["random_forest", "ridge"].includes(modelKey)) {
      throw new Error("请选择随机森林或岭回归模型。");
    }
    const values = {};
    for (const key of INPUT_KEYS) {
      if (!finite(payload[key])) throw new Error("请填写四项有效的环境数值。");
      values[key] = payload[key];
    }
    const [h, dh, t, dt] = INPUT_KEYS.map(key => values[key]);
    if (!(h >= 0 && h <= 100 && dh >= -100 && dh <= 100 && h - dh >= 0 && h - dh <= 100)) {
      throw new Error("湿度应在 0～100% 之间，湿度减去变化量也应在此范围内。");
    }
    if (!(t >= -50 && t <= 60 && dt >= -60 && dt <= 60 && t - dt >= -50 && t - dt <= 60)) {
      throw new Error("请检查温度及变化量，当前及前次温度须在 −50～60℃ 范围内。");
    }
    const mode = Object.hasOwn(payload, "mode") ? payload.mode : "forecast";
    if (!["forecast", "measured"].includes(mode)) throw new Error("环境来源无效。");

    const model = bundle.models[modelKey];
    const metadata = bundle.metadata[modelKey];
    const environmentalInput = [h / 100, dh / 100, t, dt];
    const infer = modelKey === "random_forest" ? forestPrediction : ridgePrediction;
    const predictions = Array.from({ length: 21 }, (_, index) => ({
      string: index + 1,
      deviation: infer(model, [...environmentalInput, index + 1]),
    }));
    const warnings = FEATURES.slice(0, 4).flatMap((feature, index) => {
      const [low, high] = metadata.bounds[feature];
      return environmentalInput[index] >= low && environmentalInput[index] <= high
        ? [] : [`${feature}超出历史训练范围，结果仅供探索参考。`];
    });
    let total = 0;
    let absoluteTotal = 0;
    let maxAbsolute = -Infinity;
    let mostAffected = 1;
    for (const { string, deviation } of predictions) {
      total += deviation;
      absoluteTotal += Math.abs(deviation);
      // Strict comparison matches numpy.argmax's first index on ties.
      if (Math.abs(deviation) > maxAbsolute) {
        maxAbsolute = Math.abs(deviation);
        mostAffected = string;
      }
    }
    return {
      predictions,
      inputs: values,
      mode,
      warnings,
      model: modelKey,
      model_name: metadata.name,
      model_params: copy(metadata.params),
      summary: {
        mean: total / 21,
        mean_absolute: absoluteTotal / 21,
        max_absolute: maxAbsolute,
        most_affected: mostAffected,
      },
    };
  }

  window.TingxianModels = Object.freeze({ load, predict });
})();
