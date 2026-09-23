"""Denný screening pre swing experiment na XTB (surfovací režim).

Beží na GitHub Actions po zatvorení amerického trhu. Stiahne celý americký trh
nad 2 mld. USD, spočíta pravidlá U, T a V pre každý titul a výsledok uloží
do results/. Ranný automat v Claude si potom stiahne jediný súbor.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

OUT = Path("results")
MIN_CAP = 2e9
MIN_AVG_VOL = 1_000_000
CHUNK = 100
FINALISTS_TO_CHECK = 15
BAD_NAME_WORDS = (" ETF", " Fund", " Notes", " Preferred", " Warrant", " Unit", " Right",
                  " Depositary Shares", " Debenture", " Trust Units")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- univerzum
def load_universe():
    """Všetky americké akcie nad 2 mld. USD z Nasdaq screenera. Pri chybe
    sa použije posledný uložený zoznam, aby beh nezlyhal úplne."""
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&download=true"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    }
    cache = OUT / "universe.csv"
    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        rows = r.json()["data"]["rows"]
        df = pd.DataFrame(rows)
        df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
        df = df[df["marketCap"] >= MIN_CAP]
        df["symbol"] = df["symbol"].astype(str).str.strip()
        df = df[~df["symbol"].str.contains(r"[\^ ]", regex=True, na=True)]
        mask = df["name"].fillna("").apply(lambda n: any(w.lower() in n.lower() for w in BAD_NAME_WORDS))
        df = df[~mask]
        df = df[["symbol", "name", "marketCap", "sector", "industry", "country"]].drop_duplicates("symbol")
        df.to_csv(cache, index=False)
        return df, "nasdaq"
    except Exception as e:  # noqa: BLE001
        log(f"Nasdaq screener zlyhal: {e}")
        if cache.exists():
            return pd.read_csv(cache), "cache"
        raise


def yf_symbol(s):
    return s.replace("/", "-").replace(".", "-")


# ---------------------------------------------------------------- dáta
def download(tickers):
    frames = {}
    for i in range(0, len(tickers), CHUNK):
        part = tickers[i:i + CHUNK]
        for attempt in range(3):
            try:
                data = yf.download(part, period="14mo", interval="1d", group_by="ticker",
                                   auto_adjust=False, threads=True, progress=False)
                break
            except Exception as e:  # noqa: BLE001
                log(f"chunk {i}: pokus {attempt + 1} zlyhal: {e}")
                time.sleep(10 * (attempt + 1))
        else:
            continue
        for t in part:
            try:
                d = data[t] if len(part) > 1 else data
                if isinstance(d.columns, pd.MultiIndex):
                    d = d.droplevel(0, axis=1)
                d = d.dropna(subset=["Close"])
                if len(d) >= 200:
                    frames[t] = d
            except Exception:  # noqa: BLE001
                pass
        time.sleep(1.5)
    return frames


# ---------------------------------------------------------------- výpočty
def metrics(d):
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    ma20 = c.rolling(20).mean()
    last = c.iloc[-1]
    hi252 = h.iloc[-252:].max()
    lo252 = l.iloc[-252:].min()
    return {
        "date": d.index[-1].strftime("%Y-%m-%d"),
        "close": round(float(last), 2),
        "chg_pct": round(float(last / c.iloc[-2] - 1) * 100, 2),
        "prev_high": round(float(h.iloc[-2]), 2),
        "ma20": round(float(ma20.iloc[-1]), 2),
        "ma20_5ago": round(float(ma20.iloc[-6]), 2),
        "ma50": round(float(c.rolling(50).mean().iloc[-1]), 2),
        "ma200": round(float(c.rolling(200).mean().iloc[-1]), 2),
        "atr14": round(float(tr.iloc[-14:].mean()), 2),
        "avg_vol63": int(v.iloc[-63:].mean()),
        "hi252": round(float(hi252), 2),
        "lo252": round(float(lo252), 2),
        "ret252_pct": round(float(last / c.iloc[-252] - 1) * 100, 2) if len(c) >= 252 else None,
        "ret63_pct": round(float(last / c.iloc[-64] - 1) * 100, 2),
    }


def regime(frames, sym):
    d = frames[sym]
    c = d["Close"]
    ma50 = c.rolling(50).mean()
    below = [bool(c.iloc[-k] < ma50.iloc[-k]) for k in (1, 2)]
    veto = (below[0] and below[1]) or bool(c.iloc[-1] < 0.99 * ma50.iloc[-1])
    return {
        "close": round(float(c.iloc[-1]), 2),
        "ma50": round(float(ma50.iloc[-1]), 2),
        "vs_ma50_pct": round(float(c.iloc[-1] / ma50.iloc[-1] - 1) * 100, 2),
        "below_last2": below,
        "veto": veto,
        "ret63_pct": round(float(c.iloc[-1] / c.iloc[-64] - 1) * 100, 2),
    }


def evaluate(m, spy63):
    fails = []
    if m["avg_vol63"] < MIN_AVG_VOL: fails.append("U:objem")
    if m["close"] < 0.85 * m["hi252"]: fails.append("U:>15% pod maximom")
    if m["chg_pct"] > 8: fails.append("U:deň >+8%")
    if m["lo252"] > 0 and m["hi252"] > 4 * m["lo252"]: fails.append("U:divoký")
    if m["ret252_pct"] is not None and m["ret252_pct"] <= 0: fails.append("U:rok v strate")
    sito = not fails
    if not (m["close"] > m["ma50"] and m["close"] > m["ma200"]): fails.append("T1")
    if not m["ma20"] > m["ma20_5ago"]: fails.append("T2")
    if m["close"] > 1.10 * m["ma20"]: fails.append("T3")
    if m["atr14"] / m["close"] > 0.05: fails.append("T4")
    if not m["ret63_pct"] > spy63: fails.append("T5")
    trigger = m["close"] > m["prev_high"]
    if not trigger: fails.append("V1")
    return sito, fails


def earnings_and_revenue(t):
    info = {"earnings_date": None, "bdays_to_earnings": None, "revenue_growth_pct": None}
    tk = yf.Ticker(t)
    try:
        cal = tk.calendar or {}
        ed = cal.get("Earnings Date")
        if ed:
            ed = sorted(pd.to_datetime(x).date() for x in ed)[0]
            info["earnings_date"] = ed.isoformat()
            info["bdays_to_earnings"] = int(np.busday_count(datetime.now(timezone.utc).date(), ed))
    except Exception as e:  # noqa: BLE001
        log(f"{t}: kalendár zlyhal: {e}")
    try:
        rg = tk.info.get("revenueGrowth")
        if rg is not None:
            info["revenue_growth_pct"] = round(float(rg) * 100, 1)
    except Exception as e:  # noqa: BLE001
        log(f"{t}: info zlyhalo: {e}")
    time.sleep(1)
    return info


# ---------------------------------------------------------------- hlavný beh
def main():
    OUT.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc)
    uni, uni_source = load_universe()
    tickers = [yf_symbol(s) for s in uni["symbol"].astype(str)]
    log(f"univerzum: {len(tickers)} titulov ({uni_source})")

    frames = download(["SPY", "QQQ"] + tickers)
    if "SPY" not in frames or "QQQ" not in frames:
        log("Chýbajú SPY alebo QQQ, končím.")
        sys.exit(1)
    last_session = frames["SPY"].index[-1].strftime("%Y-%m-%d")
    spy, qqq = regime(frames, "SPY"), regime(frames, "QQQ")

    meta = uni.assign(yf=[yf_symbol(s) for s in uni["symbol"].astype(str)]).set_index("yf")
    rows, stale = [], 0
    for t in tickers:
        if t not in frames:
            continue
        m = metrics(frames[t])
        if m["date"] != last_session:
            stale += 1
            continue
        sito, fails = evaluate(m, spy["ret63_pct"])
        rows.append({"ticker": t, "name": meta.at[t, "name"] if t in meta.index else "",
                     "market_cap_bn": round(float(meta.at[t, "marketCap"]) / 1e9, 2) if t in meta.index else None,
                     "sector": meta.at[t, "sector"] if t in meta.index else "",
                     **m, "rs63_pct": round(m["ret63_pct"] - spy["ret63_pct"], 2),
                     "sito": sito, "passed": not fails, "fails": ",".join(fails)})

    allm = pd.DataFrame(rows).sort_values("rs63_pct", ascending=False)
    allm.to_csv(OUT / "all.csv", index=False)

    passed = allm[allm["passed"]].head(FINALISTS_TO_CHECK)
    finalists = []
    for _, r in passed.iterrows():
        extra = earnings_and_revenue(r["ticker"])
        f1 = extra["bdays_to_earnings"] is not None and extra["bdays_to_earnings"] >= 5
        f2 = extra["revenue_growth_pct"] is not None and extra["revenue_growth_pct"] > 0
        finalists.append({
            "ticker": r["ticker"], "name": r["name"], "sector": r["sector"],
            "market_cap_bn": r["market_cap_bn"], "band": "2-5 mld." if (r["market_cap_bn"] or 0) < 5 else "nad 5 mld.",
            "close": r["close"], "entry_cap": round(r["close"] * 1.02, 2),
            "sl_hint": round(r["close"] * 1.02 * 0.93, 2), "tp_hint": round(r["close"] * 1.02 * 1.12, 2),
            "rs63_pct": r["rs63_pct"], "ret63_pct": r["ret63_pct"], "atr_pct": round(r["atr14"] / r["close"] * 100, 2),
            "vs_ma20_pct": round((r["close"] / r["ma20"] - 1) * 100, 2),
            **extra, "F1_earnings_ok": f1, "F2_revenue_ok": f2,
        })

    fail_counts = {}
    for f in allm.loc[allm["sito"], "fails"]:
        if f:
            first = f.split(",")[0]
            fail_counts[first] = fail_counts.get(first, 0) + 1

    result = {
        "generated_utc": started.strftime("%Y-%m-%d %H:%M"),
        "last_session": last_session,
        "universe_source": uni_source,
        "funnel": {
            "universe": len(tickers),
            "evaluated_fresh": int(len(allm)),
            "stale": stale,
            "missing": len(tickers) - sum(1 for t in tickers if t in frames),
            "sito": int(allm["sito"].sum()),
            "passed_T_and_V": int(allm["passed"].sum()),
            "finalists_checked": len(finalists),
        },
        "first_fail_after_sito": dict(sorted(fail_counts.items(), key=lambda x: -x[1])),
        "N1": {"SPY": spy, "QQQ": qqq, "veto": spy["veto"] or qqq["veto"]},
        "finalists": finalists,
        "near_misses": allm[allm["sito"] & (allm["fails"].str.count(",") == 0) & ~allm["passed"]]
            .head(10)[["ticker", "close", "rs63_pct", "fails"]].to_dict("records"),
    }
    (OUT / "latest.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    log(json.dumps(result["funnel"]))


if __name__ == "__main__":
    main()
