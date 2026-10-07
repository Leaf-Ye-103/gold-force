import io
import json
import sqlite3
import pandas as pd
import pytest
from gold_force.data.ledger import validate_ledger
from gold_force.data.cftc import normalise
from gold_force.data.spdr import parse_archive
from gold_force.data.wgc import parse_mapping, period_end
from gold_force.data.imports import load_table
from gold_force.data.http import HttpClient, SourceError


def cot():
    return pd.DataFrame({"report_date_as_yyyy_mm_dd":["2024-01-02"],"cftc_contract_market_code":["088691"],
                         "open_interest_all":["1,000"],"m_money_positions_long_all":[300],"m_money_positions_short_all":[100],
                         "swap_positions_long_all":[100],"swap__positions_short_all":[250],
                         "prod_merc_positions_long":[50],"prod_merc_positions_short":[150],
                         "other_rept_positions_long":[200],"other_rept_positions_short":[100],
                         "nonrept_positions_long_all":[50],"nonrept_positions_short_all":[100]})


def test_cftc_gold_schema_and_accounting():
    f = normalise(cot())
    assert f.mm_net_pct_oi.iloc[0] == 20
    assert f.swap_net_pct_oi.iloc[0] == -15
    assert f.filter(like="net_pct_oi").sum(axis=1).iloc[0] == 0


def test_cftc_bad_positions_fail():
    f = cot()
    f.loc[0,"m_money_positions_long_all"] = 800
    with pytest.raises(ValueError, match="identity"):
        normalise(f)


def test_spdr_xlsx_header_discovery():
    out = io.BytesIO()
    f = pd.DataFrame({"Date":["2024-01-02","2024-01-03"],"GLD Close":[190,191],
                      "LBMA Gold Price (US$)":[2000,2020],"Total Net Asset Value in the Trust Tonnes":[800,801]})
    with pd.ExcelWriter(out,engine="openpyxl") as writer:
        f.to_excel(writer,index=False,startrow=3)
    result = parse_archive(out.getvalue())
    assert result.gold_price.tolist() == [2000,2020]
    assert result.gld_holdings.tolist() == [800,801]


def test_spdr_rejects_etf_price_as_gold():
    body = b'Date,GLD Close,Total Trust Tonnes\n2024-01-02,190,800\n'
    with pytest.raises(ValueError, match="gold_price"):
        parse_archive(body)


def test_spdr_explicit_share_price_proxy_and_holdings_mode():
    body = b'Date,Closing Price,Tonnes of Gold\n2024-01-02,190,800\n'
    f = parse_archive(body, price_mode="gld_close")
    assert f.gld_price.iloc[0] == 190 and "gold_price" not in f
    holdings = parse_archive(body, price_mode="holdings_only")
    assert set(holdings) == {"date","gld_holdings"}


def test_duplicate_and_wrong_unit_rejected(weekly):
    _, ledger = weekly
    with pytest.raises(ValueError, match="Duplicate"):
        validate_ledger(pd.concat([ledger,ledger.iloc[[0]]]))
    ledger.loc[ledger.series.eq("gold_price"),"unit"] = "usd_per_share"
    with pytest.raises(ValueError,match="unit"):
        validate_ledger(ledger)


def test_sqlite_import_is_read_only(tmp_path):
    path = tmp_path / "data.db"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE sample(value REAL)")
        c.execute("INSERT INTO sample VALUES(1)")
    assert load_table(path,query="SELECT * FROM sample").value.iloc[0] == 1
    with pytest.raises(ValueError,match="SELECT"):
        load_table(path,query="DELETE FROM sample")


def test_wgc_long_import_requires_publication_dates():
    out = io.BytesIO()
    f = pd.DataFrame({"date":["2024Q1","2024Q2"],"available_at":["2024-04-30T12:00:00Z","2024-07-30T12:00:00Z"],"cb":[100,120]})
    f.to_excel(out,index=False)
    cfg={"layout":"long","date_column":"date","available_column":"available_at","series_columns":{"cb_purchase":"cb"}}
    ledger=parse_mapping(out.getvalue(),cfg,"2024-08-01T00:00:00Z")
    assert ledger.observation_date.dt.is_quarter_end.all()
    assert ledger.value.tolist()==[100,120]
    del cfg["available_column"]
    with pytest.raises(ValueError,match="publication"):
        parse_mapping(out.getvalue(),cfg,"2024-08-01T00:00:00Z")


def test_quarter_labels():
    assert period_end("Q2 2024","quarterly") == pd.Timestamp("2024-06-30")
    assert period_end("2024Q2","quarterly") == pd.Timestamp("2024-06-30")


def test_http_403_never_retries(tmp_path,monkeypatch):
    client=HttpClient(tmp_path,retries=2)
    class Reply:
        status_code=403
    calls=[]
    def get(*args,**kwargs):
        calls.append(1)
        return Reply()
    monkeypatch.setattr(client.session,"get",get)
    with pytest.raises(SourceError,match="403"):
        client.get("https://www.gold.org/example.xlsx")
    assert len(calls)==1


def test_cache_integrity(tmp_path,monkeypatch):
    client=HttpClient(tmp_path)
    class Reply:
        status_code=200
        content=b"data"
        headers={"Content-Type":"text/csv"}
        url="https://example.org/data"
        def raise_for_status(self): pass
    calls=[]
    def get(*args,**kwargs):
        calls.append(1)
        return Reply()
    monkeypatch.setattr(client.session,"get",get)
    _,meta=client.get("https://example.org/data")
    client.get("https://example.org/data")
    assert len(calls)==1
    (tmp_path/meta["blob"]).write_bytes(b"corrupted")
    client.get("https://example.org/data")
    assert len(calls)==2
