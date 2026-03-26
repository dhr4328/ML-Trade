import pandas as pd

# Feature list used for both backtesting and live prediction.
features = [
    "RSI",
    "EMA20_dist",
    "MACD_pct",
    "MACD_signal_pct",
    "MACD_hist_pct",
    "momentum_pct",
    "BB_upper_dist",
    "BB_lower_dist",
    "return_3",
    "return_5",
]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    # RSI (14-period)
    window_rsi = 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window_rsi, min_periods=window_rsi).mean()
    avg_loss = loss.rolling(window=window_rsi, min_periods=window_rsi).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # EMA 20
    ema20 = close.ewm(span=20, adjust=False).mean()
    df["EMA20"] = ema20  # Legacy compatibility
    df["EMA20_dist"] = (close - ema20) / close * 100.0

    # MACD (12, 26, 9)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line_raw = ema_fast - ema_slow
    signal_line_raw = macd_line_raw.ewm(span=9, adjust=False).mean()
    hist_line_raw = macd_line_raw - signal_line_raw
    
    df["MACD"] = macd_line_raw              # Legacy
    df["MACD_signal"] = signal_line_raw     # Legacy
    df["MACD_hist"] = hist_line_raw         # Legacy

    # Normalized MACD for new model
    macd_pct = macd_line_raw / close * 100.0
    signal_pct = macd_pct.ewm(span=9, adjust=False).mean()
    df["MACD_pct"] = macd_pct
    df["MACD_signal_pct"] = signal_pct
    df["MACD_hist_pct"] = macd_pct - signal_pct

    df["momentum"] = close - ema20  # Legacy
    df["momentum_pct"] = (close - ema20) / close * 100.0

    # Bollinger Bands (20, 2.0)
    window_bb = 20
    rolling_mean = close.rolling(window=window_bb, min_periods=window_bb).mean()
    rolling_std = close.rolling(window=window_bb, min_periods=window_bb).std()
    bb_upper = rolling_mean + 2.0 * rolling_std
    bb_lower = rolling_mean - 2.0 * rolling_std
    
    df["BB_middle"] = rolling_mean  # Legacy
    df["BB_upper"] = bb_upper       # Legacy
    df["BB_lower"] = bb_lower       # Legacy
    
    df["BB_upper_dist"] = (close - bb_upper) / close * 100.0
    df["BB_lower_dist"] = (close - bb_lower) / close * 100.0

    df["return_3"] = df["close"].pct_change(periods=3, fill_method=None) * 100.0
    df["return_5"] = df["close"].pct_change(periods=5, fill_method=None) * 100.0
    
    return df


if __name__ == "__main__":
    # Backtest / training code
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    dt = pd.read_csv("nq.csv")
    if "volume" in dt.columns:
        dt = dt.drop(columns=["volume"])

    data = add_indicators(dt)

    lookahead = 8
    data["target"] = -1  # Default to NO_TRADE (which we will drop)

    for i in range(len(data) - lookahead):
        entry = data["close"].iloc[i]

        tp = entry * 1.004
        stoploss = entry * 0.997

        future_high = data["high"].iloc[i + 1 : i + lookahead].max()
        future_low = data["low"].iloc[i + 1 : i + lookahead].min()

        if future_high >= tp:
            data.loc[data.index[i], "target"] = 1
        elif future_low <= stoploss:
            data.loc[data.index[i], "target"] = 0

    # Drop the NO_TRADE scenarios (target == -1)
    data = data[data["target"] != -1].dropna()

    # Pass ONLY the scale-independent relative features to the model!
    X = data[features]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "xgb_model.pkl")

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    probability = model.predict_proba(X_test)
    signals = []
    for p in probability:
        if p[1] > 0.70:
            signals.append("BUY")
        elif p[0] > 0.70:
            signals.append("SELL")
        else:
            signals.append("No trade")

    capital = 1000
    profit = 0
    trade_profits = []
    equity_curve = []
    current_equity = capital

    risk_amount = capital * 0.1
    reward_amount = capital * 0.3

    for i, signal in enumerate(signals):
        if signal == "BUY":
            trade_pnl = reward_amount if y_test.iloc[i] == 1 else -risk_amount
        elif signal == "SELL":
            trade_pnl = reward_amount if y_test.iloc[i] == 0 else -risk_amount
        else:
            continue

        profit += trade_pnl
        current_equity += trade_pnl
        trade_profits.append(trade_pnl)
        equity_curve.append(current_equity)

    print(f"Total profit from strategy: {profit:.2f}")
    if trade_profits:
        win_trades = sum(1 for p in trade_profits if p > 0)
        win_rate = win_trades / len(trade_profits)
        print(f"Number of trades: {len(trade_profits)}")
        print(f"Win rate: {win_rate:.2%}")
    else:
        print("No trades were taken with the current signal threshold.")

    if equity_curve:
        plt.figure(figsize=(10, 5))
        plt.plot(equity_curve, label="Equity Curve")
        plt.xlabel("Trade number")
        plt.ylabel("Equity")
        plt.title("Strategy Equity Curve")
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(8, 4))
        clean_profits = [p for p in trade_profits if np.isfinite(p)]
        plt.hist(clean_profits, bins=20, edgecolor="black")
        plt.xlabel("Profit per trade")
        plt.ylabel("Frequency")
        plt.title("Distribution of Trade Profits")
        plt.grid(True)
