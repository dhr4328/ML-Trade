import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Ensure repo root is importable when running `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xgb_features import add_indicators, features as default_features


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v


def _env_float(name: str, default: float) -> float:
    v = _env(name, str(default))
    try:
        return float(v)
    except ValueError as e:
        raise ValueError(f"Invalid float for {name}={v!r}") from e


def _env_int(name: str, default: int) -> int:
    v = _env(name, str(default))
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"Invalid int for {name}={v!r}") from e


def fetch_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for symbol={symbol!r} period={period!r} interval={interval!r}")

    # yfinance returns columns like Open/High/Low/Close/Adj Close/Volume
    df = df.reset_index()
    def _norm_col(c) -> str:
        if isinstance(c, tuple):
            c = "_".join([str(x) for x in c if x is not None and str(x).strip() != ""])
        return str(c).lower().replace(" ", "_")

    # Flatten multiindex/tuple columns reliably.
    df.columns = [_norm_col(c) for c in df.columns]

    # If yfinance returns per-ticker columns (e.g. open_^nsei), normalize to open/high/low/close.
    def _first_col(prefix: str) -> Optional[str]:
        if prefix in df.columns:
            return prefix
        candidates = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix + "_")]
        return candidates[0] if candidates else None

    for base in ("open", "high", "low", "close", "adj_close"):
        c = _first_col(base)
        if c is not None and c != base:
            df[base] = df[c]

    # Coerce numeric columns (yfinance can yield object dtype in some environments).
    for c in ("open", "high", "low", "close", "adj_close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize expected columns for feature code.
    if "adj_close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj_close"]
    for req in ("open", "high", "low", "close"):
        if req not in df.columns:
            raise RuntimeError(f"Missing required column {req!r}. Got columns={list(df.columns)!r}")

    if "date" not in df.columns:
        # yfinance sometimes uses "datetime" column name.
        for candidate in ("datetime", "index"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "date"})
                break
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    return df


def build_latest_feature_row(df: pd.DataFrame, feature_names: list[str]) -> pd.Series:
    df2 = add_indicators(df)
    df2 = df2.dropna(subset=feature_names)
    if df2.empty:
        raise RuntimeError("Not enough data to compute indicators (all rows have NaNs).")
    latest = df2.iloc[-1]
    missing = [c for c in feature_names if c not in df2.columns]
    if missing:
        raise RuntimeError(f"Missing expected feature columns: {missing!r}")
    out = latest[feature_names]
    out = out.apply(lambda v: pd.to_numeric(v, errors="coerce"))
    return out.astype(float)


def predict_once(
    *,
    symbol: str,
    period: str,
    interval: str,
    model,
    feature_names: list[str],
    buy_th: float,
    sell_th: float,
) -> dict:
    df = fetch_ohlc(symbol, period, interval)
    x = build_latest_feature_row(df, feature_names).to_frame().T

    proba = model.predict_proba(x)[0]
    p0 = float(proba[0])
    p1 = float(proba[1])

    if p1 >= buy_th:
        signal = "BUY"
    elif p0 >= sell_th:
        signal = "SELL"
    else:
        signal = "NO_TRADE"

    last_row = df.iloc[-1]
    payload = {
        "symbol": symbol,
        "interval": interval,
        "period": period,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bar_time_utc": (
            pd.to_datetime(last_row.get("date"), utc=True).isoformat()
            if "date" in df.columns and pd.notna(last_row.get("date"))
            else None
        ),
        "close": float(last_row["close"]),
        "signal": signal,
        "proba_0": p0,
        "proba_1": p1,
        "thresholds": {"buy": buy_th, "sell": sell_th},
        "feature_names": feature_names,
        "features": {k: (None if pd.isna(v) else float(v)) for k, v in x.iloc[0].items()},
    }
    return payload


def _local_now(tz_name: str) -> datetime:
    # Uses zoneinfo when available (py3.9+). Falls back to UTC if tz isn't present.
    try:
        from zoneinfo import ZoneInfo  # py3.9+

        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.now(timezone.utc)


def _parse_hhmm(hhmm: str) -> tuple[int, int]:
    parts = hhmm.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM time {hhmm!r}")
    h = int(parts[0])
    m = int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid HH:MM time {hhmm!r}")
    return h, m


def _should_stop(now_local: datetime, close_h: int, close_m: int) -> bool:
    # Stop when local time is >= close time.
    return (now_local.hour, now_local.minute) >= (close_h, close_m)


def send_telegram_notification(payload: dict) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    signal = payload.get("signal")
    if signal not in ("BUY", "SELL"):
        return

    symbol = payload.get("symbol")
    close = payload.get("close")
    p0 = payload.get("proba_0")
    p1 = payload.get("proba_1")
    bar_time = payload.get("bar_time_utc")
    generated = payload.get("generated_at_utc")

    text = (
        f"NIFTY Signal: {signal}\n"
        f"Symbol: {symbol}\n"
        f"Price: {close:.2f}\n"
        f"P(target=0): {p0:.4f}\n"
        f"P(target=1): {p1:.4f}\n"
        f"Bar time (UTC): {bar_time}\n"
        f"Generated at (UTC): {generated}"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"Telegram send failed: {resp.status_code} {resp.text}")
    except Exception as exc:
        print(f"Telegram send error: {exc}")


def main() -> None:
    symbol = _env("SYMBOL", "^NSEI")
    period = _env("PERIOD", "5d")
    interval = _env("INTERVAL", "5m")
    model_path = _env("MODEL_PATH", "xgb_model.pkl")
    out_file = _env("OUT_FILE", "prediction.json")
    buy_th = _env_float("BUY_THRESHOLD", 0.70)
    sell_th = _env_float("SELL_THRESHOLD", 0.70)
    loop_mode = _env("LOOP_MODE", "0") in ("1", "true", "True", "yes", "YES")
    tz_name = _env("TZ_NAME", "Asia/Kolkata")
    market_close_hhmm = _env("MARKET_CLOSE_HHMM", "15:30")
    sleep_seconds = _env_int("SLEEP_SECONDS", 300)
    log_ndjson = _env("LOG_NDJSON", "predictions.ndjson")

    model = joblib.load(model_path)
    feature_names: list[str]
    if hasattr(model, "feature_names_in_"):
        feature_names = list(getattr(model, "feature_names_in_"))
    else:
        feature_names = list(default_features)

    close_h, close_m = _parse_hhmm(market_close_hhmm)

    while True:
        now_local = _local_now(tz_name)
        if loop_mode and _should_stop(now_local, close_h, close_m):
            print(
                json.dumps(
                    {
                        "event": "loop_stop",
                        "reason": "market_close_reached",
                        "tz": tz_name,
                        "market_close_hhmm": market_close_hhmm,
                        "now_local": now_local.isoformat(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return

        payload = predict_once(
            symbol=symbol,
            period=period,
            interval=interval,
            model=model,
            feature_names=feature_names,
            buy_th=buy_th,
            sell_th=sell_th,
        )

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        with open(log_ndjson, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

        # Optional Telegram notification for BUY/SELL signals.
        send_telegram_notification(payload)

        print(json.dumps(payload, indent=2, sort_keys=True))

        if not loop_mode:
            return

        time.sleep(max(1, sleep_seconds))


if __name__ == "__main__":
    # Ensures predictable numpy printing if anything leaks to logs.
    np.set_printoptions(suppress=True)
    main()

