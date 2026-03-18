import pandas as pd
import pandas_ta as ta

# Feature list used for both backtesting and live prediction.
features = [
    "RSI",
    "EMA20",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "momentum",
    "BB_upper",
    "BB_middle",
    "BB_lower",
    "return_3",
    "return_5",
]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["EMA20"] = ta.ema(df["close"], length=20)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is None or macd.empty:
        df["MACD"] = pd.NA
        df["MACD_signal"] = pd.NA
        df["MACD_hist"] = pd.NA
    else:
        # pandas_ta names: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        df["MACD"] = macd.iloc[:, 0]
        df["MACD_hist"] = macd.iloc[:, 1]
        df["MACD_signal"] = macd.iloc[:, 2]

    df["momentum"] = df["close"] - df["EMA20"]

    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is None or bb.empty:
        df["BB_lower"] = pd.NA
        df["BB_middle"] = pd.NA
        df["BB_upper"] = pd.NA
    else:
        # pandas_ta names: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        df["BB_lower"] = bb.iloc[:, 0]
        df["BB_middle"] = bb.iloc[:, 1]
        df["BB_upper"] = bb.iloc[:, 2]

    df["return_3"] = df["close"].pct_change(periods=3)
    df["return_5"] = df["close"].pct_change(periods=5)
    return df


if __name__ == "__main__":
    # Backtest / training code (kept runnable, but won't execute on import)
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    dt = pd.read_csv("datasets/nq.csv")
    dt = dt.drop(columns="volume")

    data = add_indicators(dt)

    lookahead = 8
    data["target"] = 0

    for i in range(len(data) - lookahead):
        entry = data["close"][i]

        tp = entry * 1.004
        stoploss = entry * 0.997

        future_high = max(data["high"][i + 1 : i + lookahead])
        future_low = min(data["low"][i + 1 : i + lookahead])

        if future_high >= tp:
            data.loc[i, "target"] = 1
        elif future_low <= stoploss:
            data.loc[i, "target"] = 0

    X = data.drop(columns=["target", "date"])
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
    )
    model.fit(X_train, y_train)
    # joblib.dump(model, "xgb_model.pkl")

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
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
