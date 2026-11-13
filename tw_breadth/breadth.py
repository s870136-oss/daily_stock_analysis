"""台股市場寬度：建檔 / 每日增量 / 評分推送

演算法沿用 HoMuChen/tw-market-breadth-skill（commit 9ef6caa, MIT）的 build_tw_breadth.py，
評分沿用 tradermonty/claude-trading-skills 的 market-breadth-analyzer（commit 4251eaa, MIT）。

狀態檔（存在 tw-breadth-data 分支的 state/ 底下）：
  closes_window.csv.gz   最近 WINDOW 個交易日、每檔上市普通股的收盤價（列=日期，欄=代號）
  breadth_history.csv    每日寬度原始值：Date, Raw, TAIEX, Eligible, Above
  market_breadth_history.json  原版評分程式的分數歷史（給趨勢判斷用）

子指令：
  bootstrap  用 FinMind 逐檔抓歷史（一次性，約 2 小時；免費額度有限速）
  update     用證交所 MI_INDEX 補上 state 之後的每個交易日（每天 1 次請求）
  score      產生原版相容 CSV → 執行原版評分 → 推送 Discord
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks

TZ = ZoneInfo("Asia/Taipei")
SMA_LEN = 200
WINDOW = 210  # 多留幾天緩衝；計算只用最後 200 個交易日
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TWSE_MI_INDEX = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
UA = {"User-Agent": "Mozilla/5.0 (tw-breadth; GitHub Actions; daily once)"}
COMMON_STOCK = re.compile(r"^[1-9]\d{3}$")  # 4 碼、非 0 開頭（排除 ETF）


# ---------------------------------------------------------------- 狀態讀寫

def load_state(state: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = pd.read_csv(state / "closes_window.csv.gz", index_col=0, parse_dates=True)
    closes.columns = closes.columns.astype(str)
    hist = pd.read_csv(state / "breadth_history.csv", parse_dates=["Date"])
    return closes, hist


def save_state(state: Path, closes: pd.DataFrame, hist: pd.DataFrame) -> None:
    state.mkdir(parents=True, exist_ok=True)
    closes = closes.sort_index().iloc[-WINDOW:]
    closes.index.name = "Date"
    closes.to_csv(state / "closes_window.csv.gz", float_format="%.4g", compression="gzip")
    hist = hist.sort_values("Date").drop_duplicates("Date", keep="last")
    hist.to_csv(state / "breadth_history.csv", index=False, date_format="%Y-%m-%d", float_format="%.6f")


def breadth_from_closes(closes: pd.DataFrame) -> pd.DataFrame:
    """每天『收盤價 > 200 日 SMA』的比例；分母只算當天有價且 200MA 算得出來的股票（同上游）。"""
    ma = closes.rolling(SMA_LEN).mean()
    eligible = ma.notna() & closes.notna()
    above = (closes > ma).astype(float).where(eligible)
    out = pd.DataFrame({
        "Raw": above.mean(axis=1),
        "Eligible": eligible.sum(axis=1).astype(int),
        "Above": above.fillna(0).sum(axis=1).astype(int),
    })
    return out.dropna(subset=["Raw"])


# ---------------------------------------------------------------- bootstrap（FinMind）

class FinMind:
    def __init__(self, token: str, interval: float):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}"})
        self.interval = interval
        self._last = 0.0

    def get(self, dataset: str, data_id: str | None = None, start: str | None = None) -> pd.DataFrame:
        params = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start:
            params["start_date"] = start
        for attempt in range(6):
            wait = self.interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.s.get(FINMIND_URL, params=params, timeout=60)
                payload = r.json()
            except Exception as e:  # noqa: BLE001
                print(f"  {dataset} {data_id}: 網路錯誤 {e}，重試", flush=True)
                time.sleep(10 * (attempt + 1))
                continue
            status = payload.get("status")
            if status == 200:
                return pd.DataFrame(payload.get("data", []))
            if status == 402 or r.status_code == 402:  # 超過每小時額度
                print(f"  FinMind 額度用完（{payload.get('msg')}），等 10 分鐘", flush=True)
                time.sleep(600)
                continue
            raise RuntimeError(f"FinMind {dataset} {data_id}: {status} {payload.get('msg')}")
        raise RuntimeError(f"FinMind {dataset} {data_id}: 重試多次仍失敗")


def cmd_bootstrap(a) -> None:
    token = os.environ.get("FINMIND_API_TOKEN")
    if not token:
        sys.exit("缺少 FINMIND_API_TOKEN")
    dataset = os.environ.get("FINMIND_PRICE_DATASET", "TaiwanStockPrice")
    fm = FinMind(token, a.interval)

    info = fm.get("TaiwanStockInfo")
    ids = sorted({s for s in info.loc[info["type"] == "twse", "stock_id"] if COMMON_STOCK.match(str(s))})
    if a.limit:
        ids = ids[: a.limit]
    print(f"universe: {len(ids)} 檔上市普通股；每次請求間隔 {a.interval}s", flush=True)

    closes = {}
    for i, sid in enumerate(ids, 1):
        df = fm.get(dataset, sid, a.fetch_start)
        if not df.empty:
            s = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
            closes[sid] = s.where(s > 0)
        if i % 50 == 0 or i == len(ids):
            print(f"  fetched {i}/{len(ids)}", flush=True)

    tx = fm.get("TaiwanStockPrice", "TAIEX", a.fetch_start)
    taiex = tx.set_index(pd.to_datetime(tx["date"]))["close"].astype(float).rename("TAIEX")

    closes = pd.DataFrame(closes).sort_index()
    b = breadth_from_closes(closes).join(taiex, how="inner")
    b = b[b.index >= pd.Timestamp(a.start)]
    hist = b.reset_index(names="Date")[["Date", "Raw", "TAIEX", "Eligible", "Above"]]
    save_state(Path(a.state), closes, hist)
    last = hist.iloc[-1]
    print(f"bootstrap 完成：{hist['Date'].iloc[0].date()} ~ {last['Date'].date()}，"
          f"{len(hist)} 個交易日，最新 raw={last['Raw']:.3f}", flush=True)


# ---------------------------------------------------------------- update（證交所 MI_INDEX）

def _num(x: str) -> float:
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return float("nan")


def fetch_twse_day(d: date) -> tuple[pd.Series, float] | None:
    """回傳（各檔收盤價, 加權指數）；非交易日或尚未公布回傳 None。"""
    r = requests.get(TWSE_MI_INDEX, params={"date": d.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"},
                     headers=UA, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("stat") != "OK" or not j.get("tables"):
        return None
    stocks = next((t for t in j["tables"] if "證券代號" in (t.get("fields") or [])), None)
    taiex = None
    for t in j["tables"]:
        for row in t.get("data") or []:
            if row and row[0] == "發行量加權股價指數":
                taiex = _num(row[1])
                break
        if taiex is not None:
            break
    if stocks is None or taiex is None or not stocks.get("data"):
        return None
    fi = stocks["fields"]
    ci, pi = fi.index("證券代號"), fi.index("收盤價")
    px = {row[ci]: _num(row[pi]) for row in stocks["data"] if COMMON_STOCK.match(row[ci])}
    s = pd.Series(px, dtype=float)
    return s.where(s > 0), taiex


def cmd_update(a) -> None:
    state = Path(a.state)
    if not (state / "breadth_history.csv").exists():
        sys.exit("還沒有建檔資料：請先手動執行一次 workflow，mode 選 bootstrap")
    closes, hist = load_state(state)
    last = hist["Date"].max().date()
    today = datetime.now(TZ).date()
    added = []
    d = last + timedelta(days=1)
    while d <= today:
        if d.weekday() < 5:
            res = fetch_twse_day(d)
            time.sleep(5)  # 對證交所保持低頻率
            if res is None:
                print(f"{d}: 無資料（休市或尚未公布）")
            else:
                px, taiex = res
                prev_n = int(closes.iloc[-1].notna().sum())
                if px.notna().sum() < 0.9 * prev_n:
                    raise SystemExit(f"{d}: 有價股票 {int(px.notna().sum())} 檔，少於前一日 {prev_n} 檔的 90%，判定資料異常")
                ts = pd.Timestamp(d)
                closes = pd.concat([closes, px.to_frame(ts).T]).sort_index()
                b = breadth_from_closes(closes.iloc[-SMA_LEN:]).iloc[-1]
                hist = pd.concat([hist, pd.DataFrame([{
                    "Date": ts, "Raw": b["Raw"], "TAIEX": taiex,
                    "Eligible": int(b["Eligible"]), "Above": int(b["Above"]),
                }])], ignore_index=True)
                added.append(d)
                print(f"{d}: raw={b['Raw']:.3f} ({int(b['Above'])}/{int(b['Eligible'])})，加權 {taiex:,.2f}")
        d += timedelta(days=1)
    if added:
        save_state(state, closes, hist)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"new_days={len(added)}\n")
    print(f"新增 {len(added)} 個交易日")


# ---------------------------------------------------------------- 產生上游相容 CSV

def trend_with_hysteresis(ma: pd.Series, threshold: float = 0.001) -> list[int]:
    trend, cur = [0] * len(ma), 0
    for i in range(1, len(ma)):
        diff = ma.iloc[i] - ma.iloc[i - 1]
        if cur <= 0 and diff > threshold:
            cur = 1
        elif cur >= 0 and diff < -threshold:
            cur = -1
        trend[i] = cur
    return trend


def build_csvs(hist: pd.DataFrame, out_dir: Path) -> pd.Series:
    h = hist.sort_values("Date").set_index("Date")
    raw = h["Raw"]
    ma200 = raw.ewm(span=200, adjust=False).mean()
    ma8 = raw.ewm(span=8, adjust=False).mean()
    trend = pd.Series(trend_with_hysteresis(ma200), index=ma200.index)
    peaks, _ = find_peaks(ma200, distance=50, prominence=0.015)
    troughs, _ = find_peaks(-ma200, distance=50, prominence=0.015)
    below = ma8[ma8 < 0.4]
    troughs_below = find_peaks(-below, prominence=0.02)[0] if len(below) >= 2 else np.array([], dtype=int)

    df = pd.DataFrame(index=raw.index)
    df["Date"] = df.index.strftime("%Y-%m-%d")
    df["S&P500_Price"] = h["TAIEX"].round(2)  # 欄名沿用以相容原版，內容是加權指數
    df["Breadth_Index_Raw"] = raw
    df["Breadth_Index_200MA"] = ma200
    df["Breadth_Index_8MA"] = ma8
    df["Breadth_200MA_Trend"] = trend
    df["Bearish_Signal"] = (trend == -1) & (ma8 < ma200)
    df["Is_Peak"] = False
    df["Is_Trough"] = False
    df["Is_Trough_8MA_Below_04"] = False
    df.iloc[peaks, df.columns.get_loc("Is_Peak")] = True
    df.iloc[troughs, df.columns.get_loc("Is_Trough")] = True
    for idx in troughs_below:
        df.loc[below.index[idx], "Is_Trough_8MA_Below_04"] = True
    df["Eligible_Count_200"] = h["Eligible"]
    df["Above_Count_200"] = h["Above"]

    summary = {
        "Average Peaks (200MA)": round(float(ma200.iloc[peaks].mean()), 3) if len(peaks) else 0.0,
        "Average Troughs (8MA < 0.4)": round(float(below.iloc[troughs_below].mean()), 3) if len(troughs_below) else 0.0,
        "Total Peaks Count": len(peaks),
        "Total Troughs Count": len(troughs),
        "Total Troughs (8MA < 0.4) Count": len(troughs_below),
        "Analysis Period Start": df["Date"].iloc[0],
        "Analysis Period End": df["Date"].iloc[-1],
        "Total Trading Days": len(df),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "market_breadth_data.csv", index=False)
    pd.DataFrame({"Metric": list(summary), "Value": list(summary.values())}).to_csv(
        out_dir / "market_breadth_summary.csv", index=False)
    return df.iloc[-1]


# ---------------------------------------------------------------- score + Discord

ZONE_ZH = {"Strong": "強勁", "Healthy": "健康", "Neutral": "中性", "Weakening": "轉弱", "Critical": "危險"}
ZONE_ICON = {"Strong": "🟢", "Healthy": "🟢", "Neutral": "🟡", "Weakening": "🟠", "Critical": "🔴"}
COMP_ZH = {
    "breadth_level_trend": "寬度水準與趨勢",
    "ma_crossover": "8MA／200MA 交叉",
    "cycle_position": "高低點循環位置",
    "bearish_signal": "空頭訊號",
    "historical_percentile": "歷史百分位",
    "divergence": "指數與寬度背離",
}


def run_upstream(analyzer_dir: Path, data_dir: Path, out_dir: Path) -> dict:
    handler = partial(SimpleHTTPRequestHandler, directory=str(data_dir))
    handler.log_message = lambda *args, **kw: None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            sys.executable, str(analyzer_dir / "scripts" / "market_breadth_analyzer.py"),
            "--detail-url", f"http://127.0.0.1:{port}/market_breadth_data.csv",
            "--summary-url", f"http://127.0.0.1:{port}/market_breadth_summary.csv",
            "--output-dir", str(out_dir),
        ], check=True)
    finally:
        srv.shutdown()
    reports = sorted(out_dir.glob("market_breadth_2*.json"))
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def format_message(rep: dict, last: pd.Series, hist: pd.DataFrame) -> str:
    c = rep["composite"]
    zone = c.get("zone", "?")
    raw, m8, m200 = last["Breadth_Index_Raw"], last["Breadth_Index_8MA"], last["Breadth_Index_200MA"]
    pink = bool(last["Breadth_200MA_Trend"] == -1 and m8 < m200)
    h = hist.sort_values("Date")
    prev_raw = h["Raw"].iloc[-2] if len(h) >= 2 else raw
    tx = h["TAIEX"].iloc[-1]
    tx5 = h["TAIEX"].iloc[-6] if len(h) >= 6 else tx
    raw5 = h["Raw"].iloc[-6] if len(h) >= 6 else raw

    trend = rep.get("trend_summary") or {}
    tr = ""
    if trend.get("direction") in ("improving", "deteriorating") and len(trend.get("entries", [])) >= 2:
        tr = f"（近期{'改善' if trend['direction'] == 'improving' else '惡化'} {trend.get('delta', 0):+.1f}）"

    lines = [
        f"## 📊 台股市場寬度　{last['Date']}",
        f"{ZONE_ICON.get(zone, '⚪')} **綜合分數 {c['composite_score']}/100　{ZONE_ZH.get(zone, zone)}（{zone}）**{tr}",
        "",
        f"- 站上 200 日線：**{raw:.1%}**（{int(last['Above_Count_200'])}/{int(last['Eligible_Count_200'])} 檔），前一日 {prev_raw:.1%}",
        f"- 8 日均：{m8:.1%}　200 日均：{m200:.1%}　{'8MA 在 200MA 之上' if m8 >= m200 else '8MA 在 200MA 之下'}",
        f"- 200 日均趨勢：{ {1: '上升', -1: '下降', 0: '持平'}.get(int(last['Breadth_200MA_Trend']), '?') }"
        + ("　⚠️ **Pink Zone**（200MA 下彎且 8MA < 200MA）" if pink else ""),
        f"- 近 5 日：加權指數 {(tx / tx5 - 1):+.2%}、寬度 {(raw - raw5) * 100:+.1f} 個百分點"
        + ("　⚠️ 指數漲、寬度跌（背離）" if tx > tx5 and raw < raw5 else ""),
        "",
        "**分項分數**",
    ]
    for k, comp in rep.get("components", {}).items():
        lines.append(f"- {COMP_ZH.get(k, k)}：{comp.get('score', '?')}")
    lines += [
        "",
        f"最強：{COMP_ZH.get(_key(c.get('strongest_health')), '?')}　最弱：{COMP_ZH.get(_key(c.get('weakest_health')), '?')}",
        f"-# 原版建議持股 {c.get('exposure_guidance', '?')} 是美股經驗值、門檻用 S&P 500 校準，台股僅供參考；"
        "未還原股價，除權息旺季寬度可能偏低。這是市場環境觀察，不是買賣訊號。",
    ]
    return "\n".join(lines)


def _key(item: dict | None) -> str | None:
    return (item or {}).get("component")


def send_discord(text: str) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        print("未設定 DISCORD_WEBHOOK_URL，只印出訊息：\n" + text)
        return
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 1900:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    chunks.append(cur)
    for ch in chunks:
        r = requests.post(url, json={"content": ch}, timeout=30)
        r.raise_for_status()
        time.sleep(1)


def cmd_score(a) -> None:
    state = Path(a.state)
    _, hist = load_state(state)
    work = Path(a.work)
    last = build_csvs(hist, work / "data")
    # 讓原版的分數歷史跨次保存
    hist_json = state / "market_breadth_history.json"
    out_dir = work / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    if hist_json.exists():
        (out_dir / "market_breadth_history.json").write_bytes(hist_json.read_bytes())
    rep = run_upstream(Path(a.analyzer), work / "data", out_dir)
    hist_json.write_bytes((out_dir / "market_breadth_history.json").read_bytes())
    msg = format_message(rep, last, hist)
    print(msg)
    if not a.dry_run:
        send_discord(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="state")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--fetch-start", default="2015-01-01")
    b.add_argument("--start", default="2016-11-01")
    b.add_argument("--interval", type=float, default=6.5, help="FinMind 每次請求最少間隔秒數（免費 600 次/小時）")
    b.add_argument("--limit", type=int, default=0, help="測試用：只抓前 N 檔")
    sub.add_parser("update")
    s = sub.add_parser("score")
    s.add_argument("--analyzer", required=True, help="tradermonty market-breadth-analyzer 資料夾")
    s.add_argument("--work", default="work")
    s.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"bootstrap": cmd_bootstrap, "update": cmd_update, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
