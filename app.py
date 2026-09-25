"""Stock Research Dashboard — free, self-hosted company research.

Run locally:  streamlit run app.py
Deploy free:  Streamlit Community Cloud (see README.md).

Data sources (all free tiers):
  - yfinance: prices, quotes, profile, holders, options, fallback news
  - SEC EDGAR: companyfacts + submissions (no key; descriptive User-Agent)
  - Finnhub (optional key): news, peer suggestions
  - FRED (optional key): fed funds rate, 10Y yield, CPI

For research/education only — not investment advice. Prices are delayed.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# Page + theme polish
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Research Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .metric-card { background:#151a23; border:1px solid #232b3a; border-radius:12px;
                    padding:12px 14px; margin-bottom:8px; }
      .metric-label { font-size:11px; color:#8b94a7; text-transform:uppercase; letter-spacing:.06em; }
      .metric-value { font-size:20px; font-weight:600; color:#e8edf3; }
      .pos { color:#4ade80; } .neg { color:#f87171; }
      .delayed-note { font-size:11px; color:#8b94a7; }
      div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).parent
WATCHLIST_PATH = APP_DIR / "watchlist.json"

# ----------------------------------------------------------------------------
# Secrets / configuration (env vars or st.secrets; app degrades without keys)
# ----------------------------------------------------------------------------
def get_secret(name: str, default: str = "") -> str:
    """Read a secret from st.secrets first, then environment variables."""
    try:
        if name in st.secrets:
            val = st.secrets[name]
            if val:
                return str(val)
    except Exception:
        pass
    return os.environ.get(name, default)


SEC_EMAIL = get_secret("SEC_EMAIL", "research-dashboard@example.com")
FINNHUB_KEY = get_secret("FINNHUB_API_KEY", "")
FRED_KEY = get_secret("FRED_API_KEY", "")

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def fmt_money(x: Any, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e12:
        return f"{sign}${v/1e12:.{digits}f}T"
    if v >= 1e9:
        return f"{sign}${v/1e9:.{digits}f}B"
    if v >= 1e6:
        return f"{sign}${v/1e6:.{digits}f}M"
    if v >= 1e3:
        return f"{sign}${v/1e3:.{digits}f}K"
    return f"{sign}${v:.{digits}f}"


def fmt_pct(x: Any, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        return f"{float(x)*100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_num(x: Any, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def safe_div(a: Any, b: Any) -> float:
    try:
        a, b = float(a), float(b)
        if b == 0 or np.isnan(a) or np.isnan(b):
            return float("nan")
        return a / b
    except (TypeError, ValueError):
        return float("nan")


def cagr(first: float, last: float, periods: int) -> float:
    """Compound annual growth rate between two positive values."""
    try:
        first, last = float(first), float(last)
        if first <= 0 or last <= 0 or periods <= 0:
            return float("nan")
        return (last / first) ** (1 / periods) - 1
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def csv_download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    if df is None or df.empty:
        return
    st.download_button(
        label=label,
        data=df.to_csv().encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        width="content",
    )


# ============================================================================
# SEC EDGAR data layer
# ============================================================================
SEC_DATA_HOST = "data.sec.gov"
SEC_WWW_HOST = "www.sec.gov"
_last_sec_call = 0.0


def _sec_headers(host: str) -> Dict[str, str]:
    return {
        "User-Agent": f"StockResearchDashboard/1.0 (contact: {SEC_EMAIL})",
        "Accept-Encoding": "gzip, deflate",
        "Host": host,
    }


def sec_get_json(url: str, host: str = SEC_DATA_HOST) -> Dict[str, Any]:
    """GET JSON from SEC EDGAR, respecting the 10 req/sec fair-access limit."""
    global _last_sec_call
    wait = 0.12 - (time.time() - _last_sec_call)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers=_sec_headers(host), timeout=30)
    _last_sec_call = time.time()
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=30 * 24 * 3600, show_spinner=False)
def load_ticker_map() -> Dict[str, Dict[str, Any]]:
    """Map of TICKER -> {cik, title} from SEC company_tickers.json."""
    data = sec_get_json("https://www.sec.gov/files/company_tickers.json", host=SEC_WWW_HOST)
    out: Dict[str, Dict[str, Any]] = {}
    for entry in data.values():
        ticker = str(entry.get("ticker", "")).upper()
        if ticker:
            out[ticker] = {"cik": int(entry["cik_str"]), "title": entry.get("title", "")}
    return out


def ticker_to_cik(ticker: str) -> Optional[int]:
    try:
        return load_ticker_map().get(ticker.upper(), {}).get("cik")
    except Exception:
        return None


def cik_str(cik: int) -> str:
    return str(cik).zfill(10)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_submissions(cik: int) -> Dict[str, Any]:
    """Company submissions JSON: SIC, fiscal year end, recent filings."""
    url = f"https://data.sec.gov/submissions/CIK{cik_str(cik)}.json"
    return sec_get_json(url)


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def get_companyfacts(cik: int) -> Dict[str, Any]:
    """Full XBRL companyfacts JSON for a filer."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str(cik)}.json"
    return sec_get_json(url)


# Tag variants tried in order — filers are inconsistent (e.g. Revenues vs
# RevenueFromContractWithCustomerExcludingAssessedTax).
TAG_SETS: Dict[str, List[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "TotalRevenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "gross_profit": ["GrossProfit", "GrossProfitLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "shares_diluted": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "debt_lt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "debt_current": ["DebtCurrent", "LongTermDebtCurrent"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "assets_current": ["AssetsCurrent"],
    "liab_current": ["LiabilitiesCurrent"],
}


def _pick_tag(facts: Dict[str, Any], names: List[str]) -> Optional[Dict[str, Any]]:
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for name in names:
        tag = us_gaap.get(name)
        if tag and tag.get("units"):
            return tag
    return None


def _frame_to_period(frame: str) -> Optional[Tuple[int, int]]:
    """'CY2023' -> (2023, 0); 'CY2023Q2' -> (2023, 2). None if unparseable.

    The SEC appends 'I' (instant) or 'D' (duration) to some frames
    (e.g. balance-sheet instants like 'CY2024Q3I'); strip those first.
    """
    f = (frame or "").strip()
    if len(f) > 1 and f[-1] in ("I", "D") and f[-2].isdigit():
        f = f[:-1]
    try:
        if f.startswith("CY") and len(f) == 6 and f[2:].isdigit():
            return int(f[2:]), 0
        if (f.startswith("CY") and len(f) == 8 and f[2:6].isdigit()
                and f[6] == "Q" and f[7].isdigit()):
            return int(f[2:6]), int(f[7])
    except (ValueError, IndexError):
        pass
    return None


def _series_from_tag(
    facts: Dict[str, Any],
    names: List[str],
    forms: Tuple[str, ...],
    quarterly: bool,
    limit: int,
    balance_sheet: bool = False,
) -> pd.Series:
    """Extract a deduplicated time series from XBRL facts.

    Keeps the latest-filed value per period (handles 10-K/A amendments).
    Annual duration frames look like CY2023; quarterly like CY2023Q2.
    Balance-sheet instants on a 10-K carry the fiscal year-end quarter
    (e.g. CY2023Q3 for a September year-end), so for those we key by year.
    """
    tag = _pick_tag(facts, names)
    if not tag:
        return pd.Series(dtype=float)
    # Prefer USD; EPS uses USD/shares; shares use plain "shares".
    units = tag.get("units", {})
    unit_key = next((k for k in ("USD", "USD/shares", "shares") if k in units), None)
    if unit_key is None:
        unit_key = next(iter(units), None)
    if unit_key is None:
        return pd.Series(dtype=float)

    best: Dict[Tuple[int, int], Tuple[str, float]] = {}
    for dp in units[unit_key]:
        if dp.get("form") not in forms:
            continue
        parsed = _frame_to_period(str(dp.get("frame", "")))
        if parsed:
            year, q = parsed
            if quarterly:
                if q == 0:
                    continue
                key = (year, q)
            elif balance_sheet:
                key = (year, 0)  # fiscal year-end instant; quarter ignored
            else:
                if q != 0:
                    continue
                key = (year, 0)
        elif balance_sheet and not quarterly:
            # Unframed datapoint: the latest 10-K's balance-sheet instants
            # sometimes carry no frame. Fall back to the fiscal-year field.
            if dp.get("fp") != "FY" or dp.get("dims"):
                continue
            try:
                year = int(dp.get("fy"))
            except (TypeError, ValueError):
                continue
            if not 1900 < year < 2100:
                continue
            key = (year, 0)
        else:
            continue
        try:
            val = float(dp["val"])
        except (TypeError, ValueError, KeyError):
            continue
        filed = str(dp.get("filed", ""))
        if key not in best or filed > best[key][0]:
            best[key] = (filed, val)

    if not best:
        return pd.Series(dtype=float)
    idx = pd.MultiIndex.from_tuples(sorted(best), names=["year", "q"])
    s = pd.Series({k: v[1] for k, v in best.items()})
    s.index = idx
    return s.sort_index().tail(limit)


def build_fundamentals(facts: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (annual_df last 10y, quarterly_df last 8q) with standard line items."""
    annual: Dict[str, pd.Series] = {}
    # (column, tags, is_balance_sheet)
    annual_specs = [
        ("Revenue", TAG_SETS["revenue"], False),
        ("Gross Profit", TAG_SETS["gross_profit"], False),
        ("Operating Income", TAG_SETS["operating_income"], False),
        ("Net Income", TAG_SETS["net_income"], False),
        ("EPS (diluted)", TAG_SETS["eps_diluted"], False),
        ("Shares (diluted)", TAG_SETS["shares_diluted"], False),
        ("Cash", TAG_SETS["cash"], True),
        ("LT Debt", TAG_SETS["debt_lt"], True),
        ("Current Debt", TAG_SETS["debt_current"], True),
        ("Equity", TAG_SETS["equity"], True),
    ]
    for col, tags, bs in annual_specs:
        s = _series_from_tag(facts, tags, ("10-K", "10-K/A"), quarterly=False,
                             limit=10, balance_sheet=bs)
        if not s.empty:
            s.index = [f"{y}" for y, _ in s.index]
            annual[col] = s

    ocf = _series_from_tag(facts, TAG_SETS["ocf"], ("10-K", "10-K/A"), quarterly=False, limit=10)
    capex = _series_from_tag(facts, TAG_SETS["capex"], ("10-K", "10-K/A"), quarterly=False, limit=10)
    if not ocf.empty and not capex.empty:
        ocf.index = [f"{y}" for y, _ in ocf.index]
        capex.index = [f"{y}" for y, _ in capex.index]
        annual["Free Cash Flow"] = (ocf - capex).reindex(
            sorted(set(ocf.index) | set(capex.index))
        )

    annual_df = pd.DataFrame(annual).sort_index() if annual else pd.DataFrame()

    quarterly: Dict[str, pd.Series] = {}
    for col, tags in [
        ("Revenue", TAG_SETS["revenue"]),
        ("Gross Profit", TAG_SETS["gross_profit"]),
        ("Operating Income", TAG_SETS["operating_income"]),
        ("Net Income", TAG_SETS["net_income"]),
        ("EPS (diluted)", TAG_SETS["eps_diluted"]),
    ]:
        s = _series_from_tag(facts, tags, ("10-Q", "10-Q/A"), quarterly=True, limit=8)
        if not s.empty:
            s.index = [f"{y}Q{q}" for y, q in s.index]
            quarterly[col] = s
    quarterly_df = pd.DataFrame(quarterly).sort_index() if quarterly else pd.DataFrame()
    return annual_df, quarterly_df


# ============================================================================
# yfinance data layer (prices, quotes, profile, holders, options)
# ============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def get_info(ticker: str) -> Dict[str, Any]:
    """Quote + profile + fundamentals snapshot. Empty dict on failure."""
    try:
        info = yf.Ticker(ticker).info
        return dict(info) if info else {}
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """OHLCV history. Empty DataFrame on failure."""
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_holders(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).institutional_holders
        if df is None or df.empty:
            return pd.DataFrame()
        return df.head(10).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_option_expiries(ticker: str) -> List[str]:
    try:
        return [str(x) for x in (yf.Ticker(ticker).options or [])]
    except Exception:
        return []


def get_yf_news(ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fallback news from yfinance when no Finnhub key is configured."""
    try:
        raw = yf.Ticker(ticker).get_news(count=limit) or []
    except Exception:
        try:
            raw = yf.Ticker(ticker).news or []
        except Exception:
            return []
    out = []
    for n in raw:
        content = n.get("content", n) if isinstance(n, dict) else {}
        ts = content.get("providerPublishTime") or n.get("providerPublishTime")
        out.append({
            "title": content.get("title") or n.get("title", ""),
            "publisher": (content.get("provider") or {}).get("displayName")
                         or n.get("publisher", ""),
            "link": (content.get("clickThroughUrl") or {}).get("url")
                    or content.get("canonicalUrl", {}).get("url")
                    or n.get("link", ""),
            "published": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                         if ts else "",
        })
    return [x for x in out if x["title"]][:limit]


# ============================================================================
# Finnhub (optional free-tier key)
# ============================================================================
def _finnhub(path: str, params: Dict[str, str]) -> Any:
    if not FINNHUB_KEY:
        raise RuntimeError("No Finnhub key configured")
    params = {**params, "token": FINNHUB_KEY}
    resp = requests.get(f"https://finnhub.io/api/v1/{path}", params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600, show_spinner=False)
def finnhub_news(ticker: str) -> List[Dict[str, Any]]:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=30)
    data = _finnhub("company-news", {
        "symbol": ticker.upper(),
        "from": start.isoformat(),
        "to": end.isoformat(),
    })
    out = []
    for n in (data or [])[:20]:
        ts = n.get("datetime")
        out.append({
            "title": n.get("headline", ""),
            "publisher": n.get("source", ""),
            "link": n.get("url", ""),
            "published": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                         if ts else "",
        })
    return out


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def finnhub_peers(ticker: str) -> List[str]:
    data = _finnhub("stock/peers", {"symbol": ticker.upper()})
    return [str(x).upper() for x in (data or []) if str(x).upper() != ticker.upper()][:10]


# ============================================================================
# FRED macro context (optional free key)
# ============================================================================
FRED_SERIES = {
    "Fed Funds Rate": "DFF",
    "10Y Treasury Yield": "DGS10",
    "CPI (Index)": "CPIAUCSL",
}


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def fred_latest(series_id: str) -> Optional[Tuple[str, float]]:
    if not FRED_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": FRED_KEY,
                    "file_type": "json", "sort_order": "desc", "limit": 5},
            timeout=20,
        )
        resp.raise_for_status()
        for obs in resp.json().get("observations", []):
            try:
                return obs["date"], float(obs["value"])
            except (ValueError, TypeError, KeyError):
                continue
    except Exception:
        pass
    return None


# Sector fallback peers when no Finnhub key (large, liquid names per sector).
SECTOR_PEERS: Dict[str, List[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "Healthcare": ["JNJ", "LLY", "UNH", "PFE", "MRK"],
    "Financial Services": ["JPM", "BAC", "V", "MA", "BRK-B"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB"],
    "Industrials": ["HON", "UNP", "UPS", "CAT", "GE"],
    "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "TMUS"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA"],
    "Basic Materials": ["LIN", "APD", "SHW", "ECL", "NEM"],
}


# ============================================================================
# Ratios, peers, DCF math
# ============================================================================
# ratio key -> (label, higher_is_better)
RATIO_DEFS: Dict[str, Tuple[str, bool]] = {
    "gross_margin": ("Gross Margin", True),
    "op_margin": ("Operating Margin", True),
    "net_margin": ("Net Margin", True),
    "roe": ("ROE", True),
    "roic": ("ROIC", True),
    "rev_cagr_3y": ("Revenue CAGR (3Y)", True),
    "rev_cagr_5y": ("Revenue CAGR (5Y)", True),
    "eps_cagr_3y": ("EPS CAGR (3Y)", True),
    "debt_equity": ("Debt / Equity", False),
    "current_ratio": ("Current Ratio", True),
    "pe": ("P/E (TTM)", False),
    "ev_ebitda": ("EV / EBITDA", False),
    "p_fcf": ("P / FCF", False),
    "div_yield": ("Dividend Yield", True),
}


def compute_ratios(ticker: str, info: Dict[str, Any],
                   annual: pd.DataFrame) -> Dict[str, float]:
    """Blend yfinance snapshot fields with SEC historical series into ratios."""
    r: Dict[str, float] = {}
    g = lambda k: info.get(k)

    r["gross_margin"] = g("grossMargins")
    r["op_margin"] = g("operatingMargins")
    r["net_margin"] = g("profitMargins")
    r["roe"] = g("returnOnEquity")
    r["pe"] = g("trailingPE")
    r["ev_ebitda"] = g("enterpriseToEbitda")
    r["div_yield"] = g("dividendYield")
    r["debt_equity"] = safe_div(g("debtToEquity"), 100)  # yfinance reports as %
    r["current_ratio"] = g("currentRatio")

    price = g("currentPrice") or g("regularMarketPrice")
    mkt_cap = g("marketCap")
    fcf_ttm = g("freeCashflow")
    r["p_fcf"] = safe_div(mkt_cap, fcf_ttm)

    # ROIC ≈ NOPAT / invested capital (21% statutory tax assumption)
    op_inc = g("operatingIncome") or g("ebit")
    total_debt = g("totalDebt") or 0
    equity = g("totalStockholderEquity")
    cash = g("totalCash") or 0
    if op_inc and equity:
        invested = (total_debt or 0) + equity - (cash or 0)
        r["roic"] = safe_div(op_inc * 0.79, invested)
    else:
        r["roic"] = float("nan")

    # CAGRs from SEC annual series (fall back to NaN when history is short)
    def _cagr_of(col: str, yrs: int) -> float:
        try:
            s = annual[col].dropna()
            if len(s) >= yrs + 1:
                return cagr(s.iloc[-(yrs + 1)], s.iloc[-1], yrs)
        except (KeyError, IndexError):
            pass
        return float("nan")

    r["rev_cagr_3y"] = _cagr_of("Revenue", 3)
    r["rev_cagr_5y"] = _cagr_of("Revenue", 5)
    r["eps_cagr_3y"] = _cagr_of("EPS (diluted)", 3)
    return {k: (float(v) if v is not None else float("nan")) for k, v in r.items()}


def peer_ratio_table(tickers: List[str]) -> pd.DataFrame:
    """One row per ticker, one column per ratio. Slow calls are cached."""
    rows = []
    for t in tickers:
        info = get_info(t)
        cik = ticker_to_cik(t)
        annual = pd.DataFrame()
        if cik:
            try:
                annual, _ = build_fundamentals(get_companyfacts(cik))
            except Exception:
                pass
        ratios = compute_ratios(t, info, annual)
        rows.append({"Ticker": t.upper(), **ratios})
    df = pd.DataFrame(rows).set_index("Ticker")
    return df


def dcf_value(fcf_base: float, growth: float, terminal_growth: float,
              discount: float, years: int, net_cash: float,
              shares: float) -> Tuple[float, pd.DataFrame]:
    """Simple two-stage DCF. Returns (value_per_share, projection_table)."""
    fcf, rows = fcf_base, []
    pv_sum = 0.0
    for yr in range(1, years + 1):
        fcf *= (1 + growth)
        pv = fcf / ((1 + discount) ** yr)
        pv_sum += pv
        rows.append({"Year": str(yr), "Projected FCF": fcf, "PV of FCF": pv})
    terminal = fcf * (1 + terminal_growth) / max(discount - terminal_growth, 1e-6)
    pv_terminal = terminal / ((1 + discount) ** years)
    rows.append({"Year": "Terminal", "Projected FCF": terminal, "PV of FCF": pv_terminal})
    equity_value = pv_sum + pv_terminal + net_cash
    value_per_share = safe_div(equity_value, shares)
    return value_per_share, pd.DataFrame(rows)


# ============================================================================
# Watchlist persistence
# ============================================================================
def load_watchlist() -> List[str]:
    try:
        data = json.loads(WATCHLIST_PATH.read_text())
        tickers = data.get("tickers", [])
        return [str(t).upper() for t in tickers if str(t).strip()]
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return []


def save_watchlist(tickers: List[str]) -> None:
    WATCHLIST_PATH.write_text(json.dumps(
        {"tickers": sorted(set(t.upper() for t in tickers))}, indent=2))


# ============================================================================
# UI: sidebar, overview, price chart
# ============================================================================
def render_sidebar() -> Tuple[str, bool]:
    st.sidebar.title("📈 Stock Research")
    st.sidebar.caption("Free & self-hosted")

    # --- ticker search with autocomplete from SEC list ---
    options: List[str] = []
    try:
        tmap = load_ticker_map()
        options = [f"{t} — {v['title'][:45]}" for t, v in sorted(tmap.items())]
    except Exception as e:
        st.sidebar.warning(f"SEC ticker list unavailable: {e}")

    default_idx = 0
    prev = st.session_state.get("ticker", "AAPL")
    if options:
        for i, o in enumerate(options):
            if o.startswith(prev + " "):
                default_idx = i
                break
    manual = st.sidebar.text_input("Ticker (or pick from the SEC list below)",
                                   value=prev, max_chars=10).strip().upper()
    ticker = manual or "AAPL"
    if options:
        choice = st.sidebar.selectbox("SEC company list (autocomplete — type to filter)",
                                      options, index=default_idx,
                                      key="sec_pick")
        if choice and st.sidebar.checkbox("Use selected SEC ticker", value=False):
            ticker = choice.split(" ")[0]
    st.session_state["ticker"] = ticker

    # --- watchlist add/remove ---
    wl = load_watchlist()
    c1, c2 = st.sidebar.columns(2)
    if c1.button("⭐ Watch", width="stretch", key="wl_add"):
        if ticker not in wl:
            wl.append(ticker)
            save_watchlist(wl)
            st.sidebar.success(f"{ticker} added to watchlist")
    if c2.button("✖ Unwatch", width="stretch", key="wl_rm"):
        if ticker in wl:
            wl.remove(ticker)
            save_watchlist(wl)
            st.sidebar.info(f"{ticker} removed")

    st.sidebar.divider()
    auto_refresh = st.sidebar.toggle("Auto-refresh quotes (60s)", value=False,
                                     help="Reloads price data every 60 seconds.")
    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=60_000, key="quote_refresh")
        except ImportError:
            st.sidebar.warning("Install streamlit-autorefresh for auto-refresh.")

    # --- macro context ---
    with st.sidebar.expander("🌍 Macro (FRED)", expanded=False):
        if not FRED_KEY:
            st.caption("Add a free FRED_API_KEY to show macro data.")
        else:
            for label, sid in FRED_SERIES.items():
                res = fred_latest(sid)
                if res:
                    d, v = res
                    suffix = "%" if "CPI" not in label else ""
                    st.metric(label, f"{v:.2f}{suffix}", help=f"As of {d}")
                else:
                    st.caption(f"{label}: unavailable")

    with st.sidebar.expander("🔑 Data sources", expanded=False):
        st.caption(f"SEC contact email: `{SEC_EMAIL}`")
        st.caption(f"Finnhub: {'✅ key set' if FINNHUB_KEY else '❌ not set (news/peers limited)'}")
        st.caption(f"FRED: {'✅ key set' if FRED_KEY else '❌ not set (macro hidden)'}")
        st.caption("Prices delayed ~15 min (Yahoo Finance).")

    return ticker, auto_refresh


def render_overview(ticker: str, info: Dict[str, Any]) -> Optional[float]:
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    chg = (price - prev_close) if price and prev_close else None
    chg_pct = safe_div(chg, prev_close) if chg is not None else float("nan")

    st.subheader(f"{info.get('longName') or info.get('shortName') or ticker} ({ticker.upper()})")
    st.caption(f"{info.get('sector', '—')} · {info.get('industry', '—')}")

    cols = st.columns(6)
    with cols[0]:
        st.metric("Price", fmt_money(price),
                  delta=fmt_pct(chg_pct) if chg_pct == chg_pct else None)
    cols[1].metric("Market Cap", fmt_money(info.get("marketCap")))
    cols[2].metric("52W Range", f"{fmt_money(info.get('fiftyTwoWeekLow'))} – "
                                f"{fmt_money(info.get('fiftyTwoWeekHigh'))}")
    cols[3].metric("P/E (TTM)", fmt_num(info.get("trailingPE")))
    cols[4].metric("Dividend Yield", fmt_pct(info.get("dividendYield")))
    cols[5].metric("Beta", fmt_num(info.get("beta")))
    st.caption("⏱ Prices delayed ~15 min (Yahoo Finance).")

    # ownership + options strip
    holders = get_holders(ticker)
    expiries = get_option_expiries(ticker)
    h1, h2 = st.columns(2)
    with h1:
        with st.expander(f"🏛 Top institutional holders ({len(holders)})"):
            if not holders.empty:
                show = holders[["Holder", "Shares", "Value", "% Out"]].copy() \
                    if set(["Holder", "Shares", "Value", "% Out"]).issubset(holders.columns) \
                    else holders
                st.dataframe(show, width="stretch", hide_index=True)
            else:
                st.caption("No holder data available.")
    with h2:
        with st.expander(f"📊 Options ({len(expiries)} expirations)"):
            if expiries:
                st.write(", ".join(expiries[:12]) + (" …" if len(expiries) > 12 else ""))
            else:
                st.caption("No options data (common for non-US or small listings).")
    return float(price) if price else None


RANGE_MAP = {
    "1D": ("1d", "5m"), "5D": ("5d", "15m"), "1M": ("1mo", "1d"),
    "6M": ("6mo", "1d"), "YTD": ("ytd", "1d"), "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"), "MAX": ("max", "1wk"),
}


def render_price_chart(ticker: str) -> None:
    st.subheader("Price chart")
    c1, c2, c3 = st.columns([2, 2, 3])
    rng = c1.radio("Range", list(RANGE_MAP.keys()), index=5, horizontal=True,
                   label_visibility="collapsed", key="rng")
    show_ma = c2.checkbox("50/200-day MAs", value=True, key="ma")
    overlay = c3.text_input("Overlay ticker (e.g. SPY) — normalized returns",
                            value="", max_chars=10, key="overlay").strip().upper()

    period, interval = RANGE_MAP[rng]
    df = get_history(ticker, period, interval)
    if df.empty:
        st.error(f"No price history for {ticker}. It may be delisted, foreign, or misspelled.")
        return

    close = df["Close"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, name=ticker.upper(),
                             line=dict(color="#22d3ee", width=2)))
    if show_ma and interval in ("1d", "1wk"):
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values, name="MA 50",
                                 line=dict(color="#f59e0b", width=1, dash="dash")))
        fig.add_trace(go.Scatter(x=ma200.index, y=ma200.values, name="MA 200",
                                 line=dict(color="#a78bfa", width=1, dash="dash")))

    if overlay:
        odf = get_history(overlay, period, interval)
        if odf.empty:
            st.warning(f"No data for overlay ticker {overlay}.")
        else:
            oclose = odf["Close"].dropna()
            # align on overlapping dates, normalize to 100 at start
            common = close.index.intersection(oclose.index)
            if len(common) > 1:
                n1 = close.loc[common] / close.loc[common].iloc[0] * 100
                n0 = oclose.loc[common] / oclose.loc[common].iloc[0] * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=n1.index, y=n1.values,
                                         name=f"{ticker.upper()} (norm.)",
                                         line=dict(color="#22d3ee", width=2)))
                fig.add_trace(go.Scatter(x=n0.index, y=n0.values,
                                         name=f"{overlay} (norm.)",
                                         line=dict(color="#4ade80", width=2)))
                st.caption("Normalized to 100 at range start.")
            else:
                st.warning("Not enough overlapping history for the overlay.")

    # volume bars on secondary axis (skip intraday clutter for 1D? keep simple)
    if "Volume" in df.columns and interval in ("1d", "1wk"):
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             yaxis="y2", opacity=0.25, marker_color="#64748b"))
        fig.update_layout(yaxis2=dict(title="Volume", overlaying="y",
                                      side="right", showgrid=False))

    fig.update_layout(
        template="plotly_dark", height=460, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.02), hovermode="x unified",
        title=f"{ticker.upper()} — {rng}",
    )
    st.plotly_chart(fig, width="stretch")
    csv_download_button(df.reset_index(), "⬇ Export price history (CSV)",
                        f"{ticker}_{rng}_prices.csv")


# ============================================================================
# UI: Fundamentals / Ratios / Peers tabs
# ============================================================================
def _bar_chart(df: pd.DataFrame, col: str, title: str) -> None:
    d = df[col].dropna()
    if d.empty:
        st.caption(f"No data for {col}.")
        return
    colors = ["#f87171" if v < 0 else "#22d3ee" for v in d.values]
    fig = go.Figure(go.Bar(x=d.index.astype(str), y=d.values, marker_color=colors,
                           name=col, hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=320, title=title,
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")


def render_fundamentals(ticker: str, cik: Optional[int]) -> pd.DataFrame:
    st.subheader("Fundamentals (SEC EDGAR)")
    if not cik:
        st.warning(f"{ticker} has no SEC XBRL filing data (typical for ETFs, foreign "
                   "listings, or funds). Fundamentals are unavailable.")
        return pd.DataFrame()
    try:
        facts = get_companyfacts(cik)
    except Exception as e:
        st.error(f"Could not load SEC filings data: {e}")
        return pd.DataFrame()

    annual, quarterly = build_fundamentals(facts)
    if annual.empty:
        st.warning("No annual XBRL facts found for this filer (recent IPO or sparse tagging).")
        return pd.DataFrame()

    freq = st.radio("Frequency", ["Annual (10Y)", "Quarterly (8Q)"], horizontal=True,
                    key="fund_freq")
    df = annual if freq.startswith("Annual") else quarterly
    st.caption("Values in USD. FCF = operating cash flow − capex.")

    metric = st.selectbox("Chart metric", list(df.columns), key="fund_metric")
    _bar_chart(df, metric, f"{ticker.upper()} — {metric} ({'annual' if freq.startswith('Annual') else 'quarterly'})")

    st.markdown("**Full table**")
    st.dataframe(df.style.format(lambda v: f"{v:,.0f}" if pd.notna(v) else "—"),
                 width="stretch")
    csv_download_button(df.reset_index().rename(columns={"index": "Period"}),
                        "⬇ Export fundamentals (CSV)", f"{ticker}_fundamentals.csv")
    return annual


def render_ratios(ticker: str, info: Dict[str, Any], annual: pd.DataFrame,
                  peer_df: pd.DataFrame) -> None:
    st.subheader("Ratios vs peer median")
    ratios = compute_ratios(ticker, info, annual)
    rows = []
    for key, (label, higher_better) in RATIO_DEFS.items():
        val = ratios.get(key, float("nan"))
        med = float("nan")
        if not peer_df.empty and key in peer_df.columns:
            med = peer_df[key].median(skipna=True)
        rows.append({"Ratio": label, "Company": val, "Peer median": med,
                     "higher_better": higher_better, "key": key})
    df = pd.DataFrame(rows)

    def _fmt(v: float, key: str) -> str:
        if pd.isna(v):
            return "—"
        if key in ("div_yield", "gross_margin", "op_margin", "net_margin", "roe", "roic",
                   "rev_cagr_3y", "rev_cagr_5y", "eps_cagr_3y"):
            return fmt_pct(v)
        return fmt_num(v)

    disp = pd.DataFrame({
        "Ratio": df["Ratio"],
        "Company": [ _fmt(v, k) for v, k in zip(df["Company"], df["key"])],
        "Peer median": [_fmt(v, k) for v, k in zip(df["Peer median"], df["key"])],
    })

    def _color(row: pd.Series) -> List[str]:
        i = row.name
        v, m, hb = df.loc[i, "Company"], df.loc[i, "Peer median"], df.loc[i, "higher_better"]
        if pd.isna(v) or pd.isna(m):
            return [""] * 3
        good = (v >= m) if hb else (v <= m)
        color = "color:#4ade80" if good else "color:#f87171"
        return ["", color, ""]

    st.dataframe(disp.style.apply(_color, axis=1), width="stretch",
                 hide_index=True)
    st.caption("Green = better than peer median · Red = worse. "
               "For multiples (P/E, EV/EBITDA, P/FCF, D/E) lower is better.")
    csv_download_button(disp, "⬇ Export ratios (CSV)", f"{ticker}_ratios.csv")


def render_peers(ticker: str, info: Dict[str, Any]) -> pd.DataFrame:
    st.subheader("Peer comparison")

    # auto-suggest peers
    suggested: List[str] = []
    try:
        if FINNHUB_KEY:
            suggested = finnhub_peers(ticker)
    except Exception:
        pass
    if not suggested:
        sector = info.get("sector", "")
        suggested = [p for p in SECTOR_PEERS.get(sector, []) if p != ticker.upper()][:5]
    if not suggested:
        suggested = ["SPY"]

    if "peer_list" not in st.session_state or st.session_state.get("peer_for") != ticker:
        st.session_state["peer_list"] = [ticker.upper()] + suggested[:4]
        st.session_state["peer_for"] = ticker

    peers: List[str] = st.session_state["peer_list"]
    add = st.text_input("Add peer ticker", max_chars=10, key="peer_add").strip().upper()
    c1, c2 = st.columns([1, 3])
    if c1.button("➕ Add", key="peer_add_btn") and add and add not in peers:
        peers.append(add)
        st.rerun()
    remove = c2.multiselect("Remove peers", [p for p in peers if p != ticker.upper()],
                            key="peer_rm")
    if remove:
        st.session_state["peer_list"] = [p for p in peers if p not in remove]
        st.rerun()

    if not suggested or suggested == ["SPY"]:
        st.caption("Peer suggestions unavailable — add tickers manually. "
                   "(Set FINNHUB_API_KEY for automatic peers.)")

    with st.spinner("Loading peer ratios…"):
        pdf = peer_ratio_table(peers)
    if pdf.empty:
        st.error("Could not load peer data.")
        return pdf

    # side-by-side table (valuation + growth subset, formatted)
    show_cols = ["pe", "ev_ebitda", "p_fcf", "net_margin", "roe",
                 "rev_cagr_3y", "debt_equity", "div_yield"]
    show_cols = [c for c in show_cols if c in pdf.columns]
    labels = {c: RATIO_DEFS[c][0] for c in show_cols}
    disp = pdf[show_cols].rename(columns=labels)
    pct_cols = {"Net Margin", "ROE", "Revenue CAGR (3Y)", "Dividend Yield"}
    styled = disp.style.format(
        lambda v, c=None: "—" if pd.isna(v)
        else (fmt_pct(v) if c in pct_cols else fmt_num(v)))
    st.dataframe(styled, width="stretch")
    csv_download_button(disp.reset_index(), "⬇ Export peer table (CSV)",
                        f"{ticker}_peers.csv")

    # scatter: growth vs valuation
    st.markdown("**Growth vs valuation**")
    sx = st.selectbox("X axis", ["rev_cagr_3y", "eps_cagr_3y", "net_margin"],
                      format_func=lambda k: RATIO_DEFS[k][0], key="sc_x")
    sy = st.selectbox("Y axis", ["pe", "ev_ebitda", "p_fcf"],
                      format_func=lambda k: RATIO_DEFS[k][0], key="sc_y")
    plot_df = pdf[[sx, sy]].dropna()
    if not plot_df.empty:
        fig = go.Figure()
        for tkr, row in plot_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row[sx]], y=[row[sy]], mode="markers+text", name=str(tkr),
                text=[str(tkr)], textposition="top center",
                marker=dict(size=14, color="#22d3ee" if str(tkr) == ticker.upper()
                            else "#64748b")))
        fig.update_layout(template="plotly_dark", height=420,
                          xaxis_title=RATIO_DEFS[sx][0], yaxis_title=RATIO_DEFS[sy][0],
                          margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Not enough data for the scatter plot.")
    return pdf


# ============================================================================
# UI: DCF / Filings / News / Watchlist tabs
# ============================================================================
def render_dcf(ticker: str, info: Dict[str, Any], annual: pd.DataFrame,
               price: Optional[float]) -> None:
    st.subheader("DCF calculator")
    st.caption("Simplified two-stage DCF for education — not investment advice.")

    # prefill from history
    fcf_hist = annual["Free Cash Flow"].dropna() if "Free Cash Flow" in annual else pd.Series(dtype=float)
    default_fcf = float(fcf_hist.iloc[-1]) if not fcf_hist.empty and fcf_hist.iloc[-1] > 0 else 0.0
    default_growth = 0.10
    if len(fcf_hist) >= 6:
        g = cagr(fcf_hist.iloc[-6], fcf_hist.iloc[-1], 5)
        if -0.2 < g < 0.5:
            default_growth = round(float(g), 3)

    cash = info.get("totalCash") or 0
    debt = info.get("totalDebt") or 0
    shares = info.get("sharesOutstanding") or info.get("floatShares") or 0

    c1, c2 = st.columns(2)
    fcf_base = c1.number_input("Base FCF (TTM, USD)", value=default_fcf, step=1e8,
                               format="%.0f", key="dcf_fcf",
                               help="Prefilled from latest SEC annual FCF when available.")
    growth = c1.slider("FCF growth rate (annual)", 0.0, 0.30, float(default_growth),
                       step=0.005, format="%.1f%%", key="dcf_g")
    tgrowth = c1.slider("Terminal growth rate", 0.0, 0.05, 0.025, step=0.0025,
                        format="%.2f%%", key="dcf_tg")
    discount = c2.slider("Discount rate (WACC)", 0.05, 0.20, 0.09, step=0.005,
                         format="%.1f%%", key="dcf_r")
    years = c2.slider("Projection years", 5, 10, 5, key="dcf_y")
    net_cash = c2.number_input("Net cash (cash − debt, USD)", value=float(cash - debt),
                               step=1e8, format="%.0f", key="dcf_nc")
    shares_in = c2.number_input("Shares outstanding", value=float(shares or 0),
                                step=1e6, format="%.0f", key="dcf_sh")

    if fcf_base <= 0 or shares_in <= 0:
        st.warning("Enter a positive base FCF and share count to run the DCF.")
        return
    if discount <= tgrowth:
        st.error("Discount rate must exceed terminal growth.")
        return

    value, proj = dcf_value(fcf_base, growth, tgrowth, discount, years,
                            net_cash, shares_in)
    m1, m2, m3 = st.columns(3)
    m1.metric("Implied value / share", fmt_money(value))
    m2.metric("Current price", fmt_money(price) if price else "—")
    if price and value:
        mos = (value - price) / value
        m3.metric("Margin of safety", fmt_pct(mos))
        if mos > 0.2:
            st.success(f"Trading ~{fmt_pct(mos)} below DCF value — a wide margin of safety.")
        elif mos > 0:
            st.info(f"Trading ~{fmt_pct(mos)} below DCF value.")
        else:
            st.warning(f"Trading ~{fmt_pct(-mos)} above DCF value at these assumptions.")
    with st.expander("Projection detail"):
        st.dataframe(proj.style.format({"Projected FCF": lambda v: fmt_money(v, 0),
                                        "PV of FCF": lambda v: fmt_money(v, 0)}),
                     width="stretch", hide_index=True)
    csv_download_button(proj, "⬇ Export DCF projection (CSV)", f"{ticker}_dcf.csv")


def render_filings(ticker: str, cik: Optional[int]) -> None:
    st.subheader("SEC filings")
    if not cik:
        st.warning(f"{ticker} has no SEC EDGAR filing index (ETFs/foreign listings).")
        return
    try:
        subs = get_submissions(cik)
    except Exception as e:
        st.error(f"Could not load filings: {e}")
        return

    sic = subs.get("sic", "")
    st.caption(f"SIC {sic} — {subs.get('sicDescription', '')} · FY ends {subs.get('fiscalYearEnd', '—')}")

    recent = subs.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    rows = []
    for f, d, a, doc in zip(forms, dates, accs, docs):
        if f in ("10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A", "4", "4/A"):
            url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                   f"{a.replace('-', '')}/{doc}")
            rows.append({"Date": d, "Form": f, "Document": doc, "Link": url})
    if not rows:
        st.caption("No recent 10-K / 10-Q / 8-K / Form 4 found.")
        return

    kind = st.radio("Filter", ["All", "10-K", "10-Q", "8-K", "Form 4"], horizontal=True,
                    key="fil_kind")
    filt = [r for r in rows if kind == "All" or r["Form"].startswith(kind.replace("Form ", ""))]
    for r in filt[:30]:
        st.markdown(f"**{r['Date']}** · `{r['Form']}` · "
                    f"[{r['Document']}]({r['Link']})")
    df = pd.DataFrame(filt)
    csv_download_button(df, "⬇ Export filings list (CSV)", f"{ticker}_filings.csv")


def render_news(ticker: str) -> None:
    st.subheader("News")
    items: List[Dict[str, Any]] = []
    source = ""
    if FINNHUB_KEY:
        try:
            items = finnhub_news(ticker)
            source = "Finnhub"
        except Exception as e:
            st.warning(f"Finnhub news failed ({e}); trying Yahoo fallback.")
    if not items:
        items = get_yf_news(ticker)
        source = "Yahoo Finance" if items else ""
    if not items:
        st.warning("No news available. Set FINNHUB_API_KEY for headlines.")
        if not FINNHUB_KEY:
            st.caption("Get a free key at finnhub.io → add as FINNHUB_API_KEY secret/env var.")
        return
    st.caption(f"Source: {source}")
    for n in items:
        title = n.get("title", "")
        link = n.get("link", "")
        label = f"[{title}]({link})" if link else title
        st.markdown(f"**{label}**  \n{n.get('publisher','')} · {n.get('published','')}")
    df = pd.DataFrame(items)
    csv_download_button(df, "⬇ Export news (CSV)", f"{ticker}_news.csv")


def render_watchlist() -> None:
    st.subheader("⭐ Watchlist")
    wl = load_watchlist()
    if not wl:
        st.info("Watchlist is empty. Add tickers from the sidebar ⭐ button.")
        return
    rows = []
    for t in wl:
        info = get_info(t)
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev = info.get("previousClose")
        rows.append({
            "Ticker": t,
            "Price": price,
            "1D %": safe_div((price - prev), prev) if price and prev else float("nan"),
            "Market Cap": info.get("marketCap"),
            "P/E": info.get("trailingPE"),
            "Sector": info.get("sector"),
        })
    df = pd.DataFrame(rows)
    disp = df.copy()
    disp["Price"] = disp["Price"].map(lambda v: fmt_money(v))
    disp["1D %"] = disp["1D %"].map(lambda v: fmt_pct(v))
    disp["Market Cap"] = disp["Market Cap"].map(lambda v: fmt_money(v))
    disp["P/E"] = disp["P/E"].map(lambda v: fmt_num(v))

    def _hl(row: pd.Series) -> List[str]:
        v = df.loc[row.name, "1D %"]
        if pd.isna(v):
            return [""] * len(row)
        c = "color:#4ade80" if v >= 0 else "color:#f87171"
        return [""] * 2 + [c] + [""] * (len(row) - 3)

    st.dataframe(disp.style.apply(_hl, axis=1), width="stretch", hide_index=True)
    csv_download_button(df, "⬇ Export watchlist (CSV)", "watchlist.csv")
    rm = st.multiselect("Remove from watchlist", wl, key="wl_tab_rm")
    if rm and st.button("Remove selected", key="wl_tab_rm_btn"):
        save_watchlist([t for t in wl if t not in rm])
        st.rerun()


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    ticker, _auto = render_sidebar()
    if not ticker:
        st.info("Enter a ticker to begin.")
        return

    with st.spinner(f"Loading {ticker}…"):
        info = get_info(ticker)
    if not info:
        st.error(f"Could not load data for **{ticker}**. Check the ticker symbol, "
                 "or it may be delisted / rate-limited — try again in a minute.")
        return

    cik = ticker_to_cik(ticker)
    price = render_overview(ticker, info)
    render_price_chart(ticker)

    tab_f, tab_r, tab_p, tab_d, tab_fl, tab_n, tab_w = st.tabs([
        "📑 Fundamentals", "⚖️ Ratios", "👥 Peers", "🧮 DCF",
        "🗂 Filings", "📰 News", "⭐ Watchlist",
    ])

    # Fundamentals first (also feeds ratios + DCF)
    with tab_f:
        annual = render_fundamentals(ticker, cik)
    # Peers before ratios so the ratio table can color-code vs peer medians
    with tab_p:
        peer_df = render_peers(ticker, info)
        st.session_state["peer_df_cache"] = peer_df
    with tab_r:
        cached = st.session_state.get("peer_df_cache")
        render_ratios(ticker, info, annual,
                      cached if cached is not None else pd.DataFrame())
    with tab_d:
        render_dcf(ticker, info, annual, price)
    with tab_fl:
        render_filings(ticker, cik)
    with tab_n:
        render_news(ticker)
    with tab_w:
        render_watchlist()

    st.divider()
    st.caption("Research & education only — not investment advice. Prices delayed ~15 min.")


if __name__ == "__main__":
    main()
