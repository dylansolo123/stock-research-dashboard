# 📈 Stock Research Dashboard

A free, self-hosted company research dashboard for evaluating stocks as investments.
Built with Python + Streamlit. No paid APIs required.

## Features

- **Ticker search** with autocomplete from the SEC's official company list
- **Overview**: price, daily change, market cap, sector/industry, 52-week range, auto-refresh
- **Interactive Plotly price chart**: 1D–MAX ranges, volume, 50/200-day MAs, SPY/comparison overlay
- **Fundamentals from SEC EDGAR filings**: 10 years annual + last 8 quarters
  (revenue, gross profit, operating income, net income, EPS, free cash flow, shares, debt, cash)
- **Ratios**: margins, ROE, ROIC, revenue/EPS CAGR, leverage, valuation multiples — color-coded vs peer median
- **Peer comparison**: auto-suggested peers (SIC-based or Finnhub), add/remove, ratio table + growth-vs-valuation scatter
- **DCF calculator** with editable growth / terminal growth / discount rate sliders
- **Filings**: latest 10-K, 10-Q, 8-K, Form 4 with direct EDGAR links
- **News** headlines with source, date, links
- **Watchlist** saved to local `watchlist.json`, with summary table
- **CSV export** on every data view
- **Macro context** (FRED): fed funds rate, 10Y yield, CPI

## Quick start (local)

```bash
cd stock-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → pick the repo, branch, and `app.py`.
3. (Optional) In the app's **Settings → Secrets**, add:
   ```toml
   FINNHUB_API_KEY = "your_key"
   FRED_API_KEY = "your_key"
   SEC_EMAIL = "you@example.com"
   ```
4. Deploy. The app works without any keys (news/peers/macro gracefully degrade).

## Free API keys (all optional)

| Key | Where to get it | Used for | Without it |
|---|---|---|---|
| `FINNHUB_API_KEY` | [finnhub.io](https://finnhub.io) free tier (60 calls/min) | News headlines, peer suggestions | Falls back to Yahoo news / sector peer list |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) free | Fed funds, 10Y yield, CPI | Macro panel hidden |
| `SEC_EMAIL` | — (your own email) | SEC EDGAR `User-Agent` header (SEC asks for contact info) | A placeholder is used; **set your real email** |

Keys are read from environment variables or Streamlit secrets (`st.secrets`).
On Streamlit Cloud, prefer the Secrets manager over env vars.

## Notes & known limitations

- **Prices are delayed** (~15 min) via Yahoo Finance; this is not a real-time trading tool.
- **SEC data**: fundamentals come from XBRL companyfacts. Tag names vary by filer
  (e.g. `Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax`); the app
  tries common variants but some companies/quarters may show gaps.
- **ETFs, mutual funds, foreign listings, and very recent IPOs** have no SEC XBRL data —
  the Fundamentals/Filings tabs will explain this instead of failing silently.
- Fiscal years not aligned to calendar years are mapped to calendar-year frames as reported.
- `yfinance` is unofficial and rate-limited; heavy use may briefly return empty data.
  Caching (5–15 min for prices, 1–7 days for filings) keeps usage low.
- Finnhub free tier: 60 API calls/minute.
- SEC EDGAR: max 10 requests/second — the app throttles and caches aggressively.
- Watchlist is stored in `watchlist.json` next to `app.py` (local file; on Streamlit Cloud
  it persists per deployment, not per user).
- DCF is a simplified educational model, not investment advice.

## Disclaimer

For research and educational purposes only. Not investment advice.
