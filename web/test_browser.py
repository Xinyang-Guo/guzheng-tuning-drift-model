"""服务开启后运行；检查桌面、手机、真实预测与下载。"""
from pathlib import Path
import csv
import io
import argparse
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8765')
parser.add_argument('--variant', action='store_true')
args = parser.parse_args()
out = Path(__file__).resolve().parent.parent / 'output' / 'web_random_forest'
if args.variant:
    out = out / 'qinshi_variant'
    out.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(args.url)
    page.wait_for_function("!document.querySelector('[data-model=\"ridge\"]').disabled")
    assert page.locator('[data-model="random_forest"]').get_attribute('aria-pressed') == 'true'
    assert page.locator('#strings .string').count() == 21
    page.screenshot(path=str(out / 'pixel_empty.png'), full_page=True)
    if args.variant:
        page.locator('.hero').screenshot(path=str(out / 'hero.png'))
        page.screenshot(path=str(out / 'character.png'), clip={'x':940,'y':270,'width':135,'height':110}, scale='css')
    assert page.locator('#download').is_disabled()
    page.locator('#example').click()
    page.locator('#predict').click()
    page.wait_for_function("!document.getElementById('download').disabled")
    assert page.locator('#chart svg rect').count() == 21
    assert page.locator('#error').inner_text() == ''
    with page.expect_download() as info:
        page.locator('#download').click()
    info.value.save_as(out / 'example_download.csv')
    assert len((out / 'example_download.csv').read_text(encoding='utf-8-sig').splitlines()) == 22
    page.screenshot(path=str(out / 'desktop.png'), full_page=True)
    forest_values = page.locator('#strings').inner_text()
    assert '随机森林' in page.locator('#resultCaption').inner_text()
    assert '随机森林' in info.value.suggested_filename
    forest_rows = list(csv.DictReader(io.StringIO((out / 'example_download.csv').read_text(encoding='utf-8-sig'))))
    assert len(forest_rows) == 21 and {r['模型标识'] for r in forest_rows} == {'random_forest'}
    page.locator('[data-model="ridge"]').click()
    page.wait_for_function("!document.getElementById('download').disabled && document.getElementById('resultCaption').textContent.includes('岭回归')")
    assert page.locator('#strings .string').count() == 21
    assert page.locator('#chart svg rect').count() == 21
    assert page.locator('#strings').inner_text() != forest_values
    assert '316.23' in page.locator('#modelDescription').text_content()
    assert '岭回归' in page.locator('#modelBadge').inner_text()
    with page.expect_download() as ridge_download:
        page.locator('#download').click()
    ridge_download.value.save_as(out / 'ridge_download.csv')
    assert '岭回归' in ridge_download.value.suggested_filename
    rows = list(csv.DictReader(io.StringIO((out / 'ridge_download.csv').read_text(encoding='utf-8-sig'))))
    assert len(rows) == 21 and {r['模型标识'] for r in rows} == {'ridge'}
    page.screenshot(path=str(out / 'ridge_desktop.png'), full_page=True)
    # 快速切换期间，迟到响应不得把旧模型结果覆盖回来。
    page.locator('[data-model="random_forest"]').click()
    page.locator('[data-model="ridge"]').click()
    page.locator('[data-model="random_forest"]').click()
    page.wait_for_function("!document.getElementById('download').disabled && document.getElementById('resultCaption').textContent.includes('随机森林')")
    assert page.locator('#strings').inner_text() == forest_values
    prediction_before = page.locator('#strings').inner_text()
    page.locator('[data-scene="dusk"]').click()
    assert page.locator('body').get_attribute('data-scene') == 'dusk'
    assert page.locator('#strings').inner_text() == prediction_before
    assert not page.locator('#download').is_disabled()
    page.screenshot(path=str(out / 'pixel_dusk.png'), full_page=True)
    page.locator('[data-scene="morning"]').click()
    page.locator('#humidity').fill('65')
    assert page.locator('#download').is_disabled()
    assert '输入已修改' in page.locator('#resultCaption').inner_text()
    page.locator('[data-mode="measured"]').click()
    page.locator('#predict').click()
    page.wait_for_function("!document.getElementById('download').disabled")
    assert '确定环境' in page.locator('#resultCaption').inner_text()
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.locator('[data-model="ridge"]').click()
    page.wait_for_function("!document.getElementById('download').disabled && document.getElementById('resultCaption').textContent.includes('岭回归')")
    page.screenshot(path=str(out / 'mobile.png'), full_page=True)
    page.locator('#humidity_change').fill('90')
    page.locator('#predict').click()
    page.wait_for_function("document.getElementById('error').textContent.length > 0")
    assert '湿度' in page.locator('#error').inner_text()
    assert page.locator('#download').is_disabled()
    assert not errors, errors
    browser.close()
print('双模型真实预测、自动切换、快速切换、CSV模型标注、输入校验及桌面/手机布局通过；无 JS 异常。')
