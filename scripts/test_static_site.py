"""Checks the standalone Pages build against the original sklearn models.

Run a static server at the repository root, e.g. python -m http.server 8870,
then: python scripts/test_static_site.py --url http://127.0.0.1:8870/docs/
"""
import argparse
import csv
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ['湿度', '湿度变化', '温度', '温度变化', '弦编号']


def cases():
    samples = [dict(humidity=70, humidity_change=10, temperature=26, temperature_change=1),
               dict(humidity=0, humidity_change=-100, temperature=-50, temperature_change=-60),
               dict(humidity=100, humidity_change=100, temperature=60, temperature_change=60)]
    rng = np.random.default_rng(42)
    for _ in range(45):
        h, previous_h = rng.uniform(0, 100, 2)
        t, previous_t = rng.uniform(0, 40, 2)
        samples.append(dict(humidity=float(h), humidity_change=float(h - previous_h),
                            temperature=float(t), temperature_change=float(t - previous_t)))
    forest = joblib.load(ROOT / 'output/web_random_forest/model.joblib')
    # Threshold-adjacent samples expose incorrect float64 vs sklearn float32 routing.
    for estimator in forest.estimators_[:16]:
        for feature, threshold in zip(estimator.tree_.feature, estimator.tree_.threshold):
            if feature not in [0, 1, 2, 3]:
                continue
            for value in [float(np.float32(threshold)), float(np.nextafter(np.float32(threshold), np.float32(-np.inf))), float(np.nextafter(np.float32(threshold), np.float32(np.inf)))]:
                sample = dict(humidity=70.0, humidity_change=0.0, temperature=26.0, temperature_change=0.0)
                key = ['humidity', 'humidity_change', 'temperature', 'temperature_change'][feature]
                sample[key] = value * 100 if feature < 2 else value
                h, dh, t, dt = [sample[k] for k in ['humidity', 'humidity_change', 'temperature', 'temperature_change']]
                if 0 <= h <= 100 and 0 <= h-dh <= 100 and -50 <= t <= 60 and -50 <= t-dt <= 60 and -60 <= dt <= 60:
                    samples.append(sample)
            break
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8870/docs/')
    args = parser.parse_args()
    out = ROOT / 'output/static_site_validation'
    out.mkdir(parents=True, exist_ok=True)
    models = {key: joblib.load(ROOT / 'output' / f'web_{key}' / 'model.joblib') for key in ['random_forest', 'ridge']}
    payloads = cases()
    max_errors = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors, requests = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda req: requests.append((req.method, req.url)))
        page.goto(args.url)
        page.wait_for_function("!document.getElementById('predict').disabled")
        for name, model in models.items():
            max_error = 0.0
            for sample in payloads:
                data = page.evaluate('(payload) => window.TingxianModels.predict(payload)', sample | dict(model=name, mode='forecast'))
                h, dh, t, dt = [sample[k] for k in ['humidity', 'humidity_change', 'temperature', 'temperature_change']]
                x = pd.DataFrame([[h/100, dh/100, t, dt, string] for string in range(1, 22)], columns=FEATURES)
                expected = model.predict(x)
                actual = np.array([row['deviation'] for row in data['predictions']])
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
                max_error = max(max_error, float(np.max(np.abs(actual - expected))))
                assert data['model'] == name and [r['string'] for r in data['predictions']] == list(range(1, 22))
                assert np.isclose(data['summary']['mean_absolute'], np.abs(expected).mean())
            max_errors[name] = max_error
        invalid = [dict(humidity=101), dict(humidity=10, humidity_change=30), dict(temperature=None),
                   dict(model='wrong'), dict(mode='wrong'), dict(humidity=True)]
        for patch in invalid:
            payload = payloads[0] | dict(model='ridge', mode='forecast') | patch
            failure = page.evaluate('(payload) => {try {window.TingxianModels.predict(payload); return false;} catch(e) {return true;}}', payload)
            assert failure
        # Once assets are loaded, input and model changes must work completely offline.
        page.context.set_offline(True)
        page.locator('#example').click()
        page.locator('#predict').click()
        page.wait_for_function("!document.getElementById('download').disabled")
        forest_values = page.locator('#strings').inner_text()
        assert '随机森林' in page.locator('#resultCaption').inner_text()
        page.screenshot(path=str(out / 'desktop.png'), full_page=True)
        page.locator('[data-model="ridge"]').click()
        page.wait_for_function("!document.getElementById('download').disabled && document.getElementById('resultCaption').textContent.includes('岭回归')")
        assert page.locator('#strings').inner_text() != forest_values
        with page.expect_download() as download:
            page.locator('#download').click()
        download.value.save_as(out / 'example_ridge.csv')
        rows = list(csv.DictReader(io.StringIO((out / 'example_ridge.csv').read_text(encoding='utf-8-sig'))))
        assert len(rows) == 21 and {r['模型标识'] for r in rows} == {'ridge'}
        page.locator('[data-model="random_forest"]').click()
        page.locator('[data-model="ridge"]').click()
        page.wait_for_function("!document.getElementById('download').disabled && document.getElementById('resultCaption').textContent.includes('岭回归')")
        assert page.locator('#chart svg rect').count() == 21
        page.locator('[data-scene="dusk"]').click()
        page.screenshot(path=str(out / 'dusk.png'), full_page=True)
        page.locator('[data-scene="morning"]').click()
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(out / 'mobile.png'), full_page=True)
        page.locator('#humidity_change').fill('90')
        page.locator('#predict').click()
        page.wait_for_function("document.getElementById('error').textContent.length > 0")
        assert page.locator('#download').is_disabled()
        assert all(method == 'GET' and '/api/' not in url for method, url in requests), requests
        assert not errors, errors
        browser.close()
    report = dict(cases_per_model=len(payloads), strings_per_case=21, max_absolute_error=max_errors,
                  checked=['sklearn parity', 'float32 tree thresholds', '21 strings', 'input validation', 'offline inference', 'model switching', 'CSV labels', 'mobile layout', 'no API requests'])
    (out / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
