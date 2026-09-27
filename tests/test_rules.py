"""Price limits, fees, and board lots."""

from datetime import date
from decimal import Decimal

from ashare.config import Settings
from ashare.rules.adjust import ex_rights_reference, parse_corporate_action
from ashare.rules.costs import buy_cash_out, sell_cash_in, stamp_rate
from ashare.rules.limits import board_limit_ratio, classify_board, is_risk_name, limit_prices
from ashare.rules.lots import suggest_shares
from ashare.rules.money import money, to_cents


def test_limit_price_rounds_half_up():
    up, down = limit_prices(Decimal("13.37"), "600000")
    assert up == Decimal("14.71")
    assert down == Decimal("12.03")
    up_quarter, _down = limit_prices(Decimal("10.25"), "000001")
    assert up_quarter == Decimal("11.28")


def test_board_limits():
    assert board_limit_ratio("600519") == Decimal("0.10")
    assert board_limit_ratio("000001") == Decimal("0.10")
    assert board_limit_ratio("300750") == Decimal("0.20")
    assert board_limit_ratio("688981") == Decimal("0.20")
    assert board_limit_ratio("830000") == Decimal("0.30")
    assert classify_board("sh600000") == "main"
    assert classify_board("300001") == "chinext"


def test_risk_names_are_excluded():
    assert is_risk_name("*ST海润")
    assert is_risk_name("ST康美")
    assert is_risk_name("N测试")
    assert is_risk_name("退市某某")
    assert not is_risk_name("平安银行")
    assert not is_risk_name("贵州茅台")


def test_ex_rights_reference_cash_and_bonus():
    assert ex_rights_reference("11.60", "0.249", "0") == Decimal("11.35")
    assert parse_corporate_action({"FHcontent": "10派2.49元"}) == (0.249, 0.0)
    cash, ratio = parse_corporate_action({"FHcontent": "10送5转5派1元"})
    assert cash == 0.1
    assert ratio == 1.0
    assert ex_rights_reference("20", cash, ratio) == Decimal("9.95")


def test_commission_minimum_and_stamp_only_on_sells():
    settings = Settings()
    trade_day = date(2024, 6, 3)
    # 100 shares at 10 CNY. Raw commission is 0.25, so the 5 CNY floor applies.
    cost = buy_cash_out(100, "10", trade_day, settings)
    assert cost == Decimal("1005.01")
    proceeds = sell_cash_in(100, "10", trade_day, settings)
    # Stamp 0.50 + commission 5 + transfer 0.01.
    assert proceeds == Decimal("994.49")
    assert stamp_rate(date(2023, 8, 27), settings) == Decimal("0.001")
    assert stamp_rate(date(2023, 8, 28), settings) == Decimal("0.0005")
    legacy = sell_cash_in(100, "10", date(2023, 8, 27), settings)
    assert legacy == Decimal("993.99")


def test_lots_fit_a_twenty_thousand_account():
    settings = Settings()
    day = date(2024, 6, 3)
    # One slot is 5,000. At 9 CNY that is 500 shares.
    assert suggest_shares("000001", 9, 5000, settings, day) == 500
    # 100 shares of a 30 CNY name costs 3,000, which still fits the slot.
    assert suggest_shares("600000", 30, 5000, settings, day) == 100
    # 100 shares would cost 25,000, above both the slot and a 20,000 account.
    assert suggest_shares("600519", 250, 5000, settings, day) == 0
    # STAR board minimum is 200 shares, bought in 200-share steps.
    assert suggest_shares("688001", 10, 5000, settings, day) == 400
    assert suggest_shares("688001", 10, 5000, settings, day) % 200 == 0
    assert to_cents(money("10.005")) == 1001
