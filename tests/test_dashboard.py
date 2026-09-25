"""Click every dashboard control against the real Flask app.

SportMonks is replaced only at the HTTP layer (see tests/mock_sportmonks.py).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from werkzeug.serving import make_server

from laliga.webapp import DEMO_FETCH_MESSAGE, create_app

EXTERNAL = (
    "https://docs.sportmonks.com/v3",
    "https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/includes/scores",
    "https://github.com/Yan8412/-",
)
NAV = ("操作台", "预测结果", "回测结果", "数据状态")
JOB_WAIT_MS = 300_000


class RunningApp:
    def __init__(self, data_dir: Path, *, demo: bool = False) -> None:
        self.app = create_app(data_dir, demo=demo)
        self.server = make_server("127.0.0.1", 0, self.app, threaded=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch(headless=True)
        yield chromium
        chromium.close()


@pytest.fixture
def page(browser) -> Page:
    context = browser.new_context(accept_downloads=True)
    tab = context.new_page()
    tab.set_default_timeout(30_000)
    yield tab
    context.close()


def test_empty_dashboard_every_control_and_external_link(tmp_path: Path, page: Page, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPORTMONKS_API_TOKEN", raising=False)
    monkeypatch.delenv("SPORTMONKS_API_BASE", raising=False)
    app = RunningApp(tmp_path / "empty")
    try:
        page.goto(app.base + "/")
        expect(page.locator("body")).to_have_attribute("data-demo", "0")
        expect(page.locator(".demo-banner")).to_have_count(0)
        expect(page.get_by_text("还没有本地比赛")).to_be_visible()
        expect(page.get_by_text("未设置 SPORTMONKS_API_TOKEN")).to_be_visible()
        assert "北岸联" not in page.content()
        _assert_no_dead_links(page)
        stylesheet = page.request.get(app.base + "/static/app.css")
        assert stylesheet.status == 200 and b"--accent" in stylesheet.body()

        for name in NAV:
            _click_nav(page, name)
            _assert_no_dead_links(page)
            expect(page.locator(".demo-banner")).to_have_count(0)
        _click_nav(page, "操作台")

        _submit(page, "更新数据")
        expect(page.locator("[role=alert]")).to_contain_text("SPORTMONKS_API_TOKEN")
        _click_nav(page, "操作台")

        _submit(page, "训练模型")
        expect(page.locator("[role=alert]")).to_contain_text("没有完场")
        _click_nav(page, "预测结果")
        expect(page.get_by_text("还没有预测")).to_be_visible()
        _click_nav(page, "回测结果")
        expect(page.get_by_text("还没有回测")).to_be_visible()
        _click_nav(page, "数据状态")
        expect(page.get_by_text("比赛表是空的")).to_be_visible()
        expect(page.get_by_text("还没有通过「更新数据」写入赛季清单")).to_be_visible()
        expect(page.get_by_text("还没有训练好的模型")).to_be_visible()
        _click_nav(page, "操作台")

        _submit(page, "运行回测")
        expect(page.locator("[role=alert]")).to_contain_text("没有完场")
        _click_nav(page, "操作台")

        _submit(page, "预测下一轮")
        expect(page.locator("[role=alert]")).to_contain_text("没有未开赛")
        _click_nav(page, "操作台")

        page.locator("input[name=start]").fill("2026-10-10")
        page.locator("input[name=end]").fill("2026-10-01")
        _submit(page, "按日期预测")
        expect(page.locator("[role=alert]")).to_contain_text("结束日期早于开始日期")

        for path, label in (
            ("/predictions.csv", "预测 CSV"),
            ("/predictions.json", "预测 JSON"),
            ("/backtest.csv", "回测 CSV"),
            ("/backtest.json", "回测 JSON"),
        ):
            missing = page.goto(app.base + path)
            assert missing is not None and missing.status == 404
            expect(page.get_by_role("heading", name=f"还没有{label}")).to_be_visible()
            page.get_by_role("link", name="返回操作台").click()
            expect(page.get_by_role("heading", name="当前状态")).to_be_visible()

        unknown = page.goto(app.base + "/fixtures/999999")
        assert unknown is not None and unknown.status == 404
        expect(page.get_by_text("比赛 999999")).to_be_visible()
        page.locator("main").get_by_role("link", name="预测结果").click()
        expect(page.get_by_role("heading", name="预测结果")).to_be_visible()
        page.goto(app.base + "/fixtures/999999")
        page.locator("main").get_by_role("link", name="返回操作台").click()
        expect(page.get_by_role("heading", name="当前状态")).to_be_visible()

        missing_job = page.goto(app.base + "/jobs/does-not-exist")
        assert missing_job is not None and missing_job.status == 404
        expect(page.get_by_text("这个地址没有对应的页面或任务")).to_be_visible()
        page.get_by_role("link", name="返回操作台").click()

        _follow_external(page, app.base + "/")
    finally:
        app.close()


def test_mocked_sportmonks_pipeline(tmp_path: Path, page: Page, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.mock_sportmonks import MockSportMonks, scheduled_day

    mock = MockSportMonks()
    monkeypatch.setenv("SPORTMONKS_API_TOKEN", "dashboard-test-token")
    monkeypatch.setenv("SPORTMONKS_API_BASE", mock.base_url)
    app = RunningApp(tmp_path / "live")
    try:
        page.goto(app.base + "/")
        expect(page.locator("body")).to_have_attribute("data-demo", "0")
        expect(page.locator(".demo-banner")).to_have_count(0)
        expect(page.get_by_text("已检测到 API token")).to_be_visible()
        assert "dashboard-test-token" not in page.content()

        for mode, needle in (("401", "HTTP 401"), ("403", "免费计划"), ("429", "429")):
            mock.set_mode(mode)
            _click_nav(page, "操作台")
            _submit(page, "更新数据")
            expect(page.locator("[role=alert]")).to_contain_text(needle)
            assert "北岸联" not in page.content()

        mock.set_mode("ok")
        _click_nav(page, "操作台")
        _submit(page, "预测下一轮")
        expect(page.locator("[role=alert]")).to_contain_text("完场比赛太少")
        assert any("fixtures/between/" in hit for hit in mock.hits)

        _click_nav(page, "操作台")
        day = scheduled_day()
        page.locator("input[name=start]").fill("1999-01-01")
        page.locator("input[name=end]").fill("1999-01-02")
        _submit(page, "按日期预测")
        expect(page.locator("[role=alert]")).to_contain_text("该日期范围内没有未开赛比赛")

        mock.set_mode("429-once")
        _click_nav(page, "操作台")
        page.locator("input[name=refresh]").check()
        before = len(mock.hits)
        _submit(page, "更新数据")
        expect(page.locator("#log")).to_contain_text("将抓取")
        page.locator("#job-result").click()
        expect(page.get_by_role("heading", name="数据状态")).to_be_visible()
        expect(page.locator("main")).to_contain_text("9001")
        expect(page.locator("main")).to_contain_text("9002")
        assert "dashboard-test-token" in mock.tokens_seen
        assert len(mock.hits) > before
        assert any("cursor=" in hit for hit in mock.hits)
        _assert_cache_hides_token(tmp_path / "live", "dashboard-test-token")

        _click_nav(page, "操作台")
        expect(page.get_by_text("本地比赛")).to_be_visible()
        page.get_by_role("link", name="查看赛季和模型详情").click()
        expect(page.get_by_role("heading", name="SportMonks 赛季清单")).to_be_visible()

        _click_nav(page, "操作台")
        _submit(page, "训练模型")
        page.locator("#job-result").click()
        expect(page.locator("main")).to_contain_text("上次训练截止")
        expect(page.locator("main")).to_contain_text("新加坡")

        _click_nav(page, "操作台")
        _submit(page, "运行回测", timeout=JOB_WAIT_MS)
        expect(page.locator("[role=alert]")).to_contain_text("没有评测任何比赛")

        _click_nav(page, "操作台")
        page.locator("input[name=min_train]").fill("18")
        _submit(page, "运行回测", timeout=JOB_WAIT_MS)
        expect(page.locator("#log")).to_contain_text("回测完成")
        page.locator("#job-result").click()
        expect(page.get_by_role("heading", name="回测结果")).to_be_visible()
        expect(page.get_by_text("全场对数损失")).to_be_visible()
        expect(page.get_by_text("历史频率基准")).to_be_visible()
        _download_contains(page, "下载回测 CSV", "全场对数损失")
        _click_nav(page, "回测结果")
        document = _download_json(page, "下载回测 JSON")
        assert document["model"]["n"] > 0
        assert "baseline" in document

        _click_nav(page, "操作台")
        _submit(page, "预测下一轮", timeout=JOB_WAIT_MS)
        page.locator("#job-result").click()
        expect(page.get_by_role("heading", name="预测结果")).to_be_visible()
        expect(page.locator(".demo-banner")).to_have_count(0)
        row = page.locator("tbody tr").first
        utc = row.locator("td").nth(0).inner_text()
        singapore = row.locator("td").nth(1).inner_text()
        _assert_singapore(utc, singapore)
        home = row.locator("td").nth(2).locator("a")
        away = row.locator("td").nth(3).locator("a")
        home_name = home.inner_text()
        away_name = away.inner_text()
        home.click()
        expect(page.locator("#ft-grid")).to_be_visible()
        expect(page.locator("#ht-grid")).to_be_visible()
        _assert_grid_sums(page, "#ft-grid")
        _assert_grid_sums(page, "#ht-grid")
        assert page.locator("#ft-grid td.top").count() >= 3
        page.get_by_role("link", name="返回预测结果").click()
        page.get_by_role("link", name=away_name).first.click()
        expect(page.get_by_role("heading", name=f"{home_name} 对 {away_name}")).to_be_visible()
        page.get_by_role("link", name="返回预测结果").click()

        csv_text = _download_contains(page, "下载预测 CSV", "kickoff_singapore")
        assert home_name in csv_text
        payload = _download_json(page, "下载预测 JSON")
        assert payload["items"]
        assert payload["items"][0]["ft_matrix"]
        assert abs(sum(payload["items"][0]["ft"].values()) - 1) < 1e-6

        _click_nav(page, "操作台")
        page.locator("input[name=start]").fill(day)
        page.locator("input[name=end]").fill(day)
        _submit(page, "按日期预测", timeout=JOB_WAIT_MS)
        page.locator("#job-result").click()
        expect(page.get_by_role("heading", name="预测结果")).to_be_visible()
        expect(page.locator("main")).to_contain_text(day)
        _assert_no_dead_links(page)
        for name in NAV:
            _click_nav(page, name)
            _assert_no_dead_links(page)
            for href in EXTERNAL:
                expect(page.locator(f'footer a[href="{href}"]')).to_have_count(1)
    finally:
        app.close()
        mock.close()


def test_demo_mode_is_labelled_and_refuses_network(tmp_path: Path, page: Page, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPORTMONKS_API_TOKEN", "should-not-be-sent")
    monkeypatch.setenv("SPORTMONKS_API_BASE", "http://127.0.0.1:9/v3/football")
    app = RunningApp(tmp_path / "demo", demo=True)
    try:
        page.goto(app.base + "/")
        _assert_demo_banner(page)
        expect(page.locator("body")).to_have_attribute("data-demo", "1")
        expect(page.get_by_text("本地比赛")).to_be_visible()

        for name in NAV:
            _click_nav(page, name)
            _assert_demo_banner(page)
        _click_nav(page, "操作台")

        page.locator("input[name=refresh]").check()
        _submit(page, "更新数据")
        expect(page.locator("[role=alert]")).to_contain_text(DEMO_FETCH_MESSAGE[:12])
        _assert_demo_banner(page)
        _click_nav(page, "操作台")

        _submit(page, "训练模型", timeout=JOB_WAIT_MS)
        page.locator("#job-result").click()
        _assert_demo_banner(page)
        expect(page.get_by_text("上次训练截止")).to_be_visible()

        _click_nav(page, "操作台")
        _submit(page, "运行回测", timeout=JOB_WAIT_MS)
        page.locator("#job-result").click()
        _assert_demo_banner(page)
        expect(page.get_by_text("全场对数损失")).to_be_visible()
        _download_contains(page, "下载回测 CSV", "全场对数损失")
        _click_nav(page, "回测结果")
        _download_json(page, "下载回测 JSON")

        _click_nav(page, "操作台")
        _submit(page, "预测下一轮", timeout=JOB_WAIT_MS)
        page.locator("#job-result").click()
        _assert_demo_banner(page)
        expect(page.get_by_text("北岸联").first).to_be_visible()
        page.locator("tbody tr").first.locator("a").first.click()
        _assert_demo_banner(page)
        _assert_grid_sums(page, "#ft-grid")
        page.get_by_role("link", name="返回预测结果").click()

        from tests.mock_sportmonks import scheduled_day

        day = scheduled_day()
        _click_nav(page, "操作台")
        page.locator("input[name=start]").fill(day)
        page.locator("input[name=end]").fill(day)
        _submit(page, "按日期预测", timeout=JOB_WAIT_MS)
        page.locator("#job-result").click()
        _assert_demo_banner(page)
        expect(page.get_by_text("合成示例数据")).to_be_visible()
        _download_contains(page, "下载预测 CSV", "kickoff_utc")
        _click_nav(page, "预测结果")
        _download_json(page, "下载预测 JSON")
    finally:
        app.close()


def _click_nav(page: Page, name: str) -> None:
    page.get_by_role("navigation", name="站内").get_by_role("link", name=name).click()


def _submit(page: Page, button: str, timeout: int = 60_000) -> None:
    page.get_by_role("button", name=button).click()
    page.locator("#job-result, [role=alert]").wait_for(timeout=timeout)


def _assert_no_dead_links(page: Page) -> None:
    hrefs = page.eval_on_selector_all("a", "els => els.map(e => e.getAttribute('href'))")
    assert hrefs, "page has no links"
    for href in hrefs:
        assert href, "empty href"
        assert href != "#"
        assert not href.startswith("javascript:")


def _assert_demo_banner(page: Page) -> None:
    banner = page.locator(".demo-banner")
    expect(banner).to_be_visible()
    expect(banner).to_contain_text("合成示例数据")
    expect(banner).to_contain_text("不是真实西甲")


def _assert_singapore(utc_text: str, singapore_text: str) -> None:
    utc = datetime.strptime(utc_text.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    singapore = datetime.strptime(singapore_text.strip(), "%Y-%m-%d %H:%M")
    assert singapore - utc.replace(tzinfo=None) == timedelta(hours=8)


def _assert_grid_sums(page: Page, selector: str) -> None:
    probs = page.locator(f"{selector} [data-prob]").evaluate_all("els => els.map(e => Number(e.getAttribute('data-prob')))")
    assert probs
    assert abs(sum(probs) - 1) < 1e-3


def _download_contains(page: Page, link_name: str, needle: str) -> str:
    with page.expect_download() as download_info:
        page.get_by_role("link", name=link_name).click()
    download = download_info.value
    text = Path(download.path()).read_text(encoding="utf-8")
    assert needle in text
    return text


def _download_json(page: Page, link_name: str) -> dict:
    text = _download_contains(page, link_name, "{")
    return json.loads(text)


def _follow_external(page: Page, home: str) -> None:
    for href in EXTERNAL:
        response = page.context.request.get(href, timeout=45_000, max_redirects=10)
        assert response.status < 400, (href, response.status)
        page.goto(home)
        link = page.locator(f'footer a[href="{href}"]')
        expect(link).to_be_visible()
        with page.expect_navigation(wait_until="domcontentloaded", timeout=45_000):
            link.click()
        assert page.url.startswith("http")
        assert "127.0.0.1" not in page.url


def _assert_cache_hides_token(data_dir: Path, token: str) -> None:
    cache = data_dir / "cache"
    files = list(cache.glob("*.json"))
    assert files
    for path in files:
        assert token not in path.read_text(encoding="utf-8")
