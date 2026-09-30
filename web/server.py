"""本地五变量随机森林 / 岭回归预测网页：python web/server.py。"""
import argparse
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from guzheng_future_validation import load_data, FEATURES

ARTIFACT = ROOT / 'output' / 'web_random_forest'
PARAMS = dict(n_estimators=300, max_depth=4, min_samples_leaf=20,
              min_samples_split=2, max_features=0.6, random_state=42, n_jobs=-1)
RIDGE_ALPHA = 316.22776601683796
MODEL_NAMES = {'random_forest': '随机森林', 'ridge': '岭回归'}


def prepare_models():
    data, _, _ = load_data()
    bounds = {key: [float(data[key].min()), float(data[key].max())] for key in FEATURES[:-1]}
    evaluation = pd.read_csv(ROOT / 'output' / 'future_validation' / 'final_metrics.csv').set_index('model')
    models = {
        'random_forest': RandomForestRegressor(**PARAMS),
        'ridge': make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)),
    }
    metadata = {}
    for key, model in models.items():
        model.fit(data[FEATURES], data['偏差'])
        assert list(model.feature_names_in_) == FEATURES and model.n_features_in_ == 5
        folder = ROOT / 'output' / f'web_{key}'
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, folder / 'model.joblib')
        metadata[key] = dict(
            key=key, name=MODEL_NAMES[key],
            model='RandomForestRegressor' if key == 'random_forest' else 'StandardScaler + Ridge',
            features=FEATURES, params=PARAMS if key == 'random_forest' else {'alpha': RIDGE_ALPHA},
            training_rows=len(data), bounds=bounds,
            evaluation={metric: float(evaluation.loc[key, metric]) for metric in ['mae', 'mse', 'r2']},
            description='采用时间滚动验证选出的该类模型配置，使用全部历史数据重训；评估指标来自重训前的最后日期留出实验。',
        )
        (folder / 'metadata.json').write_text(json.dumps(metadata[key], ensure_ascii=False, indent=2), encoding='utf-8')
    return models, metadata


def predict(payload, models, catalog):
    # 无 model 字段时沿用随机森林，兼容保留的原版网页。
    model_key = payload.get('model', 'random_forest')
    if not isinstance(model_key, str) or model_key not in models:
        raise ValueError('请选择随机森林或岭回归模型。')
    model, metadata = models[model_key], catalog[model_key]
    keys = ['humidity', 'humidity_change', 'temperature', 'temperature_change']
    values = {}
    for key in keys:
        raw = payload.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (float, int)) or not math.isfinite(raw):
            raise ValueError('请填写四项有效的环境数值。')
        values[key] = float(raw)
    h, dh, t, dt = [values[key] for key in keys]
    if not (0 <= h <= 100 and -100 <= dh <= 100 and 0 <= h - dh <= 100):
        raise ValueError('湿度应在 0～100% 之间，湿度减去变化量也应在此范围内。')
    if not (-50 <= t <= 60 and -60 <= dt <= 60 and -50 <= t - dt <= 60):
        raise ValueError('请检查温度及变化量，当前及前次温度须在 −50～60℃ 范围内。')
    mode = payload.get('mode', 'forecast')
    if mode not in ('forecast', 'measured'):
        raise ValueError('环境来源无效。')
    inputs = pd.DataFrame([[h / 100, dh / 100, t, dt, string] for string in range(1, 22)], columns=FEATURES)
    results = model.predict(inputs)
    warnings = []
    for key in FEATURES[:-1]:
        low, high = metadata['bounds'][key]
        if not low <= inputs[key].iloc[0] <= high:
            warnings.append(f'{key}超出历史训练范围，结果仅供探索参考。')
    return dict(predictions=[dict(string=i, deviation=float(v)) for i, v in enumerate(results, 1)],
                inputs=values, mode=mode, warnings=warnings,
                model=model_key, model_name=metadata['name'], model_params=metadata['params'],
                summary=dict(mean=float(results.mean()), mean_absolute=float(abs(results).mean()),
                             max_absolute=float(abs(results).max()), most_affected=int(abs(results).argmax()) + 1))


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value, status=200):
        content = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path == '/api/model':
            return self.send_json(self.server.metadata['random_forest'])
        if self.path == '/api/models':
            return self.send_json({'default': 'random_forest', 'models': self.server.metadata})
        pages = {'/': 'qinshi.html', '/index.html': 'qinshi.html', '/qinshi.html': 'qinshi.html', '/original.html': 'index.html'}
        if self.path not in pages:
            return self.send_json({'error': '未找到页面'}, 404)
        content = (ROOT / 'web' / pages[self.path]).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        if self.path != '/api/predict':
            return self.send_json({'error': '接口不存在'}, 404)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('请求大小无效。')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('请求格式无效。')
            result = predict(payload, self.server.models, self.server.metadata)
            self.send_json(result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json({'error': str(exc)}, 400)
        except Exception:
            self.send_json({'error': '模型预测失败，请查看服务终端。'}, 500)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    models, metadata = prepare_models()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.models, server.metadata = models, metadata
    print(f'古筝调音预测已启动：http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == '__main__':
    main()
