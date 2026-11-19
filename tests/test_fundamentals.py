"""Company filings: every number dated by when it became public. No network: SEC is faked."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tradebot import fundamentals as f


def fact(val, end, filed, start=None, form="10-K", accn=None):
    out = {"end": end, "val": val, "filed": filed, "form": form, "accn": accn or f"acc-{filed}"}
    if start:
        out["start"] = start
    return out


def doc(**tags):
    """A company-facts file. Keys are tag names ('dei__EntityPublicFloat' for another taxonomy)."""
    facts = {}
    for key, values in tags.items():
        taxonomy, name = key.split("__") if "__" in key else ("us-gaap", key)
        facts.setdefault(taxonomy, {})[name] = {"units": {"USD": values}}
    return {"entityName": "Test Co", "facts": facts}


# A company whose fiscal year ends in late September, like Apple.
FY18 = ("2017-10-01", "2018-09-29")
FY19 = ("2018-09-30", "2019-09-28")


# ---------------------------------------------------------------- reading the file

def test_only_listed_tags_from_annual_and_quarterly_reports_are_kept():
    parsed = f.parse_facts(doc(
        NetIncomeLoss=[fact(100, FY18[1], "2018-11-05", FY18[0]),
                       fact(100, FY18[1], "2019-02-01", FY18[0], form="8-K")],       # not a 10-K or 10-Q
        Assets=[fact(500, FY18[1], "2018-11-05"),
                fact(1, FY18[1], "2018-11-05", start="2017-10-01")],                # a balance with a start date
        GoodwillImpairmentLoss=[fact(5, FY18[1], "2018-11-05", FY18[0])],           # a tag we don't use
        dei__EntityPublicFloat=[fact(9000, "2018-03-30", "2018-11-05")]))
    assert sorted(parsed["item"]) == ["assets", "net_income", "public_float"]
    assert parsed.loc[parsed["item"] == "assets", "start"].tolist() == [""]
    assert f.parse_facts(None).empty and f.parse_facts({"facts": {}}).empty


def test_the_quarters_checked_for_new_filings_are_the_last_three_that_ended():
    assert f.recent_quarters(date(2026, 11, 18)) == [(2026, 3), (2026, 2), (2026, 1)]
    assert f.recent_quarters(date(2026, 2, 10), 2) == [(2025, 4), (2025, 3)]


# ---------------------------------------------------------------- four-quarter totals

def known(periods):
    k = f.Known(f.ITEMS["net_income"])
    for start, end, value in periods:
        k.add("NetIncomeLoss", date.fromisoformat(start), date.fromisoformat(end), value)
    return k


def test_four_quarter_total_from_a_full_year_or_from_the_year_so_far():
    k = known([(*FY18, 100.0), ("2017-10-01", "2018-06-30", 70.0), ("2018-09-30", "2019-06-29", 80.0)])
    # nine months to June 2019 (80) + the year to Sept 2018 (100) - the nine months to June 2018 (70)
    assert k.ttm(date(2019, 6, 29)) == pytest.approx(110.0)
    assert k.ttm(date(2018, 9, 29)) == pytest.approx(100.0)
    k.add("NetIncomeLoss", date(2018, 9, 30), date(2019, 9, 28), 120.0)
    assert k.ttm(date(2019, 9, 28)) == pytest.approx(120.0)
    assert known([("2018-09-30", "2019-06-29", 80.0)]).ttm(date(2019, 6, 29)) is None   # last year not known


def test_odd_period_lengths_are_ignored():
    k = known([("2019-01-01", "2019-01-31", 5.0)])          # one month
    assert k.periods == {} and k.latest() is None


# ---------------------------------------------------------------- point in time

def history_for(**tags):
    return f.company_history(f.parse_facts(doc(**tags))).assign(cik=1)


def panel(history, item, days, max_age_days=365):
    return f.item_panels(history, item, {"A": 1}, pd.DatetimeIndex(pd.to_datetime(days)), ["A"], max_age_days)


def test_a_number_counts_from_the_trading_day_after_its_filing_and_a_restatement_from_its_own_filing():
    history = history_for(NetIncomeLoss=[
        fact(100, FY18[1], "2018-11-05", FY18[0]),
        fact(90, FY18[1], "2019-11-01", FY18[0]),         # FY18 restated in the next 10-K
        fact(120, FY19[1], "2019-11-01", FY19[0])])
    days = ["2018-11-05", "2018-11-06", "2019-10-31", "2019-11-01", "2019-11-04"]
    got = panel(history, "net_income", days, max_age_days=400)       # FY18 is 398 days old on 2019-11-01
    assert np.isnan(got["value"]["A"].iloc[0])                       # filed that day: not used yet
    assert got["value"]["A"].tolist()[1:] == [100.0, 100.0, 100.0, 120.0]
    assert got["prev"]["A"].iloc[4] == 90.0                         # last year's figure as restated by then
    assert np.isnan(got["prev"]["A"].iloc[1])                       # nothing a year earlier yet


def test_the_choice_between_tags_uses_only_tags_already_filed():
    # Equity first filed only including minority interests; the preferred tag arrives a year later.
    history = history_for(
        StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest=[fact(60, FY18[1], "2018-11-05")],
        StockholdersEquity=[fact(50, FY18[1], "2019-11-01"), fact(55, FY19[1], "2019-11-01")])
    got = panel(history, "equity", ["2019-06-03", "2019-11-04"])
    assert got["value"]["A"].tolist() == [60.0, 55.0]


def test_revenue_takes_the_largest_tag_filed_for_a_period():
    history = history_for(
        RevenueFromContractWithCustomerExcludingAssessedTax=[fact(80, FY18[1], "2018-11-05", FY18[0])],
        Revenues=[fact(95, FY18[1], "2018-11-05", FY18[0])])        # includes revenue outside contracts
    assert panel(history, "revenue", ["2018-11-06"])["value"]["A"].iloc[0] == 95.0


def test_old_numbers_are_dropped_instead_of_carried_forward():
    history = history_for(Assets=[fact(500, "2018-09-29", "2018-11-05")])
    got = panel(history, "assets", ["2019-06-03", "2019-10-01"], max_age_days=365)
    assert got["value"]["A"].iloc[0] == 500.0 and np.isnan(got["value"]["A"].iloc[1])


# ---------------------------------------------------------------- the nine inputs

def test_the_inputs_on_a_hand_made_company():
    history = history_for(
        NetIncomeLoss=[fact(10, FY18[1], "2018-11-05", FY18[0]), fact(12, FY19[1], "2019-11-01", FY19[0]),
                       fact(10, FY18[1], "2019-11-01", FY18[0])],
        Revenues=[fact(100, FY18[1], "2018-11-05", FY18[0]), fact(110, FY19[1], "2019-11-01", FY19[0]),
                  fact(100, FY18[1], "2019-11-01", FY18[0])],
        NetCashProvidedByUsedInOperatingActivities=[fact(9, FY19[1], "2019-11-01", FY19[0])],
        Assets=[fact(200, FY19[1], "2019-11-01"), fact(160, FY18[1], "2019-11-01")],
        StockholdersEquity=[fact(80, FY19[1], "2019-11-01")],
        dei__EntityPublicFloat=[fact(300, "2019-03-29", "2019-11-01")])
    days = pd.bdate_range("2019-03-28", "2019-11-05")
    close = pd.DataFrame({"A": np.linspace(10, 20, len(days))}, index=days)
    close.loc["2019-03-29", "A"] = 10.0                              # price on the float's date
    close.loc["2019-11-04", "A"] = 15.0
    got = f.compute(history, {"A": 1}, close)
    assert list(got) == f.NAMES
    row = {name: frame.loc["2019-11-04", "A"] for name, frame in got.items()}
    market = 300 * 15.0 / 10.0                                       # the float, grown by the stock's return
    assert row["earnings_yield"] == pytest.approx(12 / market)
    assert row["book_to_market"] == pytest.approx(80 / market)
    assert row["sales_to_price"] == pytest.approx(110 / market)
    assert row["roa"] == pytest.approx(12 / 200)
    assert row["accruals"] == pytest.approx((12 - 9) / 200)
    assert row["equity_to_assets"] == pytest.approx(80 / 200)
    assert row["asset_growth"] == pytest.approx(200 / 160 - 1)
    assert row["sales_growth"] == pytest.approx(110 / 100 - 1)
    assert row["earnings_change"] == pytest.approx((12 - 10) / 200)
    # The 10-K was filed on 2019-11-01, so that day nothing from it is used yet, and the year-old
    # numbers from the 2018 filing are past the one-year limit: every input is unknown.
    assert all(np.isnan(frame.loc["2019-11-01", "A"]) for frame in got.values())


def float_case(floats, assets=1000.0):
    """One company with total assets of `assets` in every 10-K and the given floats (end, value, filed)."""
    history = history_for(
        dei__EntityPublicFloat=[fact(v, end, filed) for end, v, filed in floats],
        Assets=[fact(assets, end, filed) for end, _, filed in floats])
    days = pd.bdate_range("2019-03-28", "2022-12-30")
    close = pd.DataFrame({"A": 10.0}, index=days)
    floats = f.item_panels(history, "public_float", {"A": 1}, days, ["A"], 10 ** 6)
    balance = f.item_panels(history, "assets", {"A": 1}, days, ["A"], 365)["value"]
    return history, close, floats, balance


def test_floats_out_of_line_with_the_balance_sheet_are_held_back_even_twice_in_a_row():
    history, close, floats, balance = float_case([
        ("2019-03-29", 300, "2019-11-01"),
        ("2020-03-31", 4_000_000, "2020-11-02"),              # 4000x assets: a scale error...
        ("2021-03-31", 4_200_000, "2021-11-01"),              # ...repeated the next year
        ("2022-03-31", 330, "2022-11-01")])
    held = []
    market = f.market_value(floats, close, 730, held, balance)["A"]
    assert market.loc["2019-11-04"] == pytest.approx(300.0)
    assert market.loc["2020-11-03"] == pytest.approx(300.0)           # the 2019 float stays in use
    assert np.isnan(market.loc["2021-11-03"])                          # and expires after two years
    assert market.loc["2022-11-02"] == pytest.approx(330.0)
    assert held == [("A", "2020-03-31", 4_000_000.0), ("A", "2021-03-31", 4_200_000.0)]
    notes = {}
    f.compute(history, {"A": 1}, close, notes=notes)
    assert notes["float_held_back"] == held


def test_a_real_change_in_size_is_accepted_at_once():
    _, close, floats, balance = float_case([("2019-03-29", 300, "2019-11-01"),
                                            ("2020-03-31", 30, "2020-11-02")])     # a spin-off: a tenth of the size
    market = f.market_value(floats, close, 730, [], balance)["A"]
    assert market.loc["2020-11-03"] == pytest.approx(30.0)
    # without a balance sheet to compare with, a float is taken as filed
    assert f.market_value(floats, close, 730)["A"].loc["2020-11-03"] == pytest.approx(30.0)


# ---------------------------------------------------------------- following a company to a new SEC number

def test_successor_numbers_are_chained_by_date(tmp_path):
    ciks, successors = tmp_path / "ciks.csv", tmp_path / "next.csv"
    ciks.write_text("symbol,cik\nWRK,10\nAAPL,320193\n", encoding="utf-8")
    successors.write_text("# note\nsymbol,cik,from,note\nWRK,30,2018-11-02,later\nWRK,20,2015-07-01,first\n"
                          "ZZZ,99,2020-01-01,not in the universe\n", encoding="utf-8")
    links = f.read_links(ciks, successors)
    assert links["WRK"] == [(10, None, "2015-07-01"), (20, "2015-07-01", "2018-11-02"), (30, "2018-11-02", None)]
    assert links["AAPL"] == [(320193, None, None)] and "ZZZ" not in links
    assert f.all_ciks(links) == [10, 20, 30, 320193]
    assert f.read_links(ciks, tmp_path / "missing.csv")["WRK"] == [(10, None, None)]


def test_a_successor_takes_over_and_the_old_number_is_ignored_from_its_date():
    old = f.company_history(f.parse_facts(doc(Assets=[
        fact(100, "2018-09-30", "2018-11-01", form="10-Q"),
        fact(7, "2019-06-30", "2019-08-01", form="10-Q")])))       # filed as a subsidiary after the merger
    new = f.company_history(f.parse_facts(doc(Assets=[fact(500, "2019-03-31", "2019-05-01", form="10-Q")])))
    history = pd.concat([old.assign(cik=1), new.assign(cik=2)])
    links = {"A": [(1, None, "2019-03-20"), (2, "2019-03-20", None)]}
    got = f.item_panels(history, "assets", links, pd.to_datetime(["2019-04-01", "2019-05-02", "2019-08-02"]),
                        ["A"], 365)["value"]["A"]
    assert got.tolist() == [100.0, 500.0, 500.0]


def test_negative_equity_has_no_book_to_market_and_unknown_companies_have_nothing():
    history = history_for(StockholdersEquity=[fact(-5, FY19[1], "2019-11-01")],
                          dei__EntityPublicFloat=[fact(300, "2019-03-29", "2019-11-01")])
    days = pd.bdate_range("2019-03-28", "2019-11-05")
    close = pd.DataFrame({"A": 10.0, "B": 20.0}, index=days)
    got = f.compute(history, {"A": 1, "B": 2}, close)
    assert got["book_to_market"]["A"].isna().all() and got["earnings_yield"]["B"].isna().all()


# ---------------------------------------------------------------- downloading

class FakeSec:
    def __init__(self, docs, frames=None, broken=()):
        self.docs, self.frames, self.broken = docs, frames or {}, set(broken)
        self.asked, self.quarters = [], []

    def company_facts(self, cik):
        self.asked.append(cik)
        if cik in self.broken:
            raise ConnectionError("network down")
        return self.docs.get(cik)

    def assets_frame(self, year, quarter):
        self.quarters.append((year, quarter))
        return self.frames.get((year, quarter))


def company(value, filed="2026-10-30"):
    return doc(Assets=[fact(value, "2026-09-30", filed, form="10-Q")])


def test_first_download_then_only_companies_with_a_filing_not_stored_yet(tmp_path):
    conn = f.connect(str(tmp_path / "f.db"))
    sec = FakeSec({1: company(10), 2: company(20), 3: None})
    said = []
    result = f.update(conn, [1, 2, 3], sec, said.append, today=date(2026, 11, 18))
    assert sorted(sec.asked) == [1, 2, 3] and result["empty"] == [3] and sec.quarters == []
    assert len(f.load_history(conn)) == 2

    frames = {(2026, 3): [(1, "acc-2026-11-17"),      # a 10-Q/A we have not stored
                          (2, "acc-2026-10-30"),      # already stored
                          (99, "acc-x")]}             # not one of ours
    sec = FakeSec({1: company(11, "2026-11-17"), 2: company(21)}, frames)
    f.update(conn, [1, 2, 3], sec, said.append, today=date(2026, 11, 20))
    assert sec.asked == [1] and sec.quarters == [(2026, 3), (2026, 2), (2026, 1)]
    assert conn.execute("SELECT value FROM history WHERE cik = 1").fetchone()[0] == 11
    assert any("1 of our companies have a 10-K or 10-Q not stored yet" in line for line in said)

    # A filing that carries total assets but is not a 10-K or 10-Q (so never stored as a fact)
    # triggers one download, and is then remembered.
    frames = {(2026, 3): [(2, "prospectus-1")]}
    sec = FakeSec({2: company(21)}, frames)
    f.update(conn, [1, 2, 3], sec, said.append, today=date(2026, 11, 21))
    assert sec.asked == [2]
    sec = FakeSec({2: company(21)}, frames)
    f.update(conn, [1, 2, 3], sec, said.append, today=date(2026, 11, 22))
    assert sec.asked == []


def test_a_failed_download_is_reported_and_tried_again_next_time(tmp_path):
    conn = f.connect(str(tmp_path / "f.db"))
    f.update(conn, [1], FakeSec({1: company(10)}), lambda line: None, today=date(2026, 11, 18))
    sec = FakeSec({}, broken={2})
    result = f.update(conn, [1, 2], sec, lambda line: None, today=date(2026, 11, 20))
    assert result["failed"] == [2] and sec.asked == [2]              # new to the list, so downloaded
    sec = FakeSec({2: company(5)})
    f.update(conn, [1, 2], sec, lambda line: None, today=date(2026, 11, 21))
    assert sec.asked == [2]                                          # still never stored, so asked again


def test_the_cik_list_is_read_from_the_universe_file(tmp_path):
    path = tmp_path / "ciks.csv"
    path.write_text("# comment\nsymbol,cik\nAAPL,320193\nbrk.b,1067983\n", encoding="utf-8")
    assert f.read_ciks(path) == {"AAPL": 320193, "BRK.B": 1067983}


def test_the_sec_client_names_itself_and_spaces_its_requests():
    class Reply:
        def __init__(self, status, body=""):
            self.status_code, self.text, self._body = status, body, body

        def json(self):
            return json.loads(self._body)

    class Session:
        def __init__(self):
            self.headers, self.urls = {}, []

        def get(self, url, timeout):
            self.urls.append(url)
            if "CY2030" in url:
                return Reply(404)
            if "frames" in url:
                return Reply(200, '{"data": [{"accn": "a-1", "cik": 320193, "end": "2026-09-27", "val": 5}]}')
            return Reply(200, '{"facts": {}}')

    session = Session()
    sec = f.Sec("Jane Doe jane@example.com", spacing_s=0.0, session=session)
    assert session.headers["User-Agent"] == "Jane Doe jane@example.com"
    assert sec.company_facts(320193) == {"facts": {}}
    assert session.urls[0].endswith("CIK0000320193.json")
    assert sec.assets_frame(2026, 3) == [(320193, "a-1")]
    assert session.urls[1].endswith("frames/us-gaap/Assets/USD/CY2026Q3I.json")
    assert sec.assets_frame(2030, 1) is None                          # no such quarter yet
    assert f.Sec(None, session=Session()).session.headers["User-Agent"] == f.DEFAULT_AGENT
