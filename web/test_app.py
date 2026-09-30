"""真实双模型的输入转换、API 路由和 21 弦输出检查。"""
import json
import unittest
import urllib.request
import urllib.error

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from server import predict, FEATURES, ROOT, RIDGE_ALPHA


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models, cls.catalog = {}, {}
        for key in ['random_forest', 'ridge']:
            folder = ROOT / 'output' / f'web_{key}'
            cls.models[key] = joblib.load(folder / 'model.joblib')
            cls.catalog[key] = json.loads((folder / 'metadata.json').read_text())

    def setUp(self):
        self.payload = dict(humidity=70, humidity_change=10, temperature=26, temperature_change=1, mode='forecast')

    def predict(self, patch=None):
        return predict(self.payload | (patch or {}), self.models, self.catalog)

    def test_models_and_percent_conversion(self):
        x = pd.DataFrame([[.7, .1, 26, 1, i] for i in range(1, 22)], columns=FEATURES)
        for key, model in self.models.items():
            with self.subTest(model=key):
                result = self.predict({'model': key})
                self.assertEqual(result['model'], key)
                self.assertEqual(result['model_name'], self.catalog[key]['name'])
                self.assertEqual([r['string'] for r in result['predictions']], list(range(1, 22)))
                np.testing.assert_allclose([r['deviation'] for r in result['predictions']], model.predict(x))
                self.assertEqual(list(model.feature_names_in_), FEATURES)
                self.assertEqual(model.n_features_in_, 5)
        self.assertIsInstance(self.models['random_forest'], RandomForestRegressor)
        self.assertIsInstance(self.models['ridge'], Pipeline)
        self.assertEqual(self.models['ridge'].named_steps['ridge'].alpha, RIDGE_ALPHA)
        # 岭回归与已完成时间验证后保存的全历史模型一致，包括标准化。
        historical = joblib.load(ROOT / 'output/future_validation/future_model.joblib')
        np.testing.assert_allclose(self.models['ridge'].predict(x), historical.predict(x))
        self.assertFalse(np.allclose(self.models['ridge'].predict(x), self.models['random_forest'].predict(x)))

    def test_invalid_inputs_and_model(self):
        patches = [dict(humidity=101), dict(humidity=10, humidity_change=30), dict(temperature=None),
                   dict(humidity=True), dict(mode='invalid'), dict(temperature=float('nan')),
                   dict(model='missing'), dict(model=None), dict(model=['ridge'])]
        for patch in patches:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.predict(patch)

    def test_negative_changes_and_source_modes(self):
        for key in self.models:
            payload = dict(model=key, humidity_change=-10, temperature_change=-2)
            a = self.predict(payload)
            b = self.predict(payload | dict(mode='measured'))
            np.testing.assert_allclose([r['deviation'] for r in a['predictions']], [r['deviation'] for r in b['predictions']])

    def test_training_range_warning_and_legacy_default(self):
        for key in self.models:
            self.assertTrue(self.predict(dict(model=key, temperature=55))['warnings'])
        self.assertEqual(self.predict()['model'], 'random_forest')

    def test_http(self):
        base = 'http://127.0.0.1:8765'
        for path in ['/', '/index.html', '/qinshi.html']:
            with urllib.request.urlopen(base + path) as response:
                page = response.read().decode()
                self.assertIn('qinshi-sprite', page)
                self.assertIn('data-model="ridge"', page)
        with urllib.request.urlopen(base + '/original.html') as response:
            self.assertEqual(response.read(), (ROOT / 'web/index.html').read_bytes())
        with urllib.request.urlopen(base + '/api/models') as response:
            metadata = json.load(response)
            self.assertEqual(set(metadata['models']), {'random_forest', 'ridge'})
        for key in self.models:
            request = urllib.request.Request(base + '/api/predict', data=json.dumps(self.payload | {'model': key}).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request) as response:
                result = json.load(response)
            expected = self.predict({'model': key})
            self.assertEqual(result['model'], key)
            np.testing.assert_allclose([r['deviation'] for r in result['predictions']], [r['deviation'] for r in expected['predictions']])
        for payload in [{}, self.payload | {'model': 'missing'}, self.payload | {'model': ['ridge']}]:
            request = urllib.request.Request(base + '/api/predict', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()


if __name__ == '__main__':
    unittest.main()
