import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# ==================== پارامترهای محافظه‌کار ====================
EMA_FAST = 20
EMA_SLOW = 60
EMA_TREND = 200
ATR_PERIOD = 14
ATR_MULTIPLIER = 0.1
RISK_REWARD_RATIO = 2.0
TRAILING_START = 0.05
TRAILING_STEP = 0.03
COMMISSION = 0.001
INITIAL_CAPITAL = 500.0

VOLUME_LOOKBACK = 20
VOLUME_THRESHOLD = 1.2
TIME_FILTER_ENABLED = True
TRADING_START_HOUR = 6
TRADING_END_HOUR = 18
ADX_PERIOD = 14
ADX_THRESHOLD = 30
RSI_PERIOD = 14
RSI_MAX = 65
BULLISH_CANDLE = True

SYMBOL = 'BTC-USDT'
TIMEFRAME = 60
KLINE_LIMIT = 300
LOG_FILE = "paper_trade_log.txt"

def fetch_klines(symbol, granularity=60, limit=300, end=None):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    params = {'granularity': granularity, 'limit': limit}
    if end:
        params['end'] = end
    resp = requests.get(url, params=params)
    data = resp.json()
    if not data or 'message' in data:
        return None
    df = pd.DataFrame(data, columns=['timestamp','low','high','open','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('timestamp').sort_index()
    return df[['open','high','low','close','volume']].astype(float)

def get_current_price(symbol):
    ticker = requests.get(f"https://api.coinbase.com/v2/prices/{symbol}/spot").json()
    return float(ticker['data']['amount'])

def calculate_indicators(df):
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df['ema_trend'] = df['close'].ewm(span=EMA_TREND, adjust=False).mean()
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([h-l, (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = h.diff()
    down_move = l.diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), -down_move, 0.0)
    plus_di = pd.Series(plus_dm, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean() / df['atr'] * 100
    minus_di = pd.Series(-minus_dm, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean() / df['atr'] * 100
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['avg_volume_20'] = df['volume'].rolling(window=VOLUME_LOOKBACK).mean()
    return df

def log_trade(msg):
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f"{timestamp} | {msg}\n")
    print(f"{timestamp} | {msg}")

def main():
    print("🚀 Paper Trading Bot (Coinbase) - محافظه‌کار")
    capital = INITIAL_CAPITAL
    in_position = False
    position_amount = 0.0
    entry_price = stop_loss = take_profit = target_distance = locked_pct = 0.0
    df = fetch_klines(SYMBOL, granularity=TIMEFRAME, limit=KLINE_LIMIT)
    if df is None:
        print("❌ دریافت داده اولیه ناموفق.")
        return
    last_index = df.index[-1]
    while True:
        try:
            new_df = fetch_klines(SYMBOL, granularity=TIMEFRAME, limit=5, end=last_index.isoformat())
            if new_df is not None and not new_df.empty:
                df = pd.concat([df, new_df])
                df = df[~df.index.duplicated(keep='last')].sort_index()
                last_index = df.index[-1]
            df = calculate_indicators(df)
            if len(df) < EMA_TREND + 2:
                time.sleep(30)
                continue
            i = -2
            cur_open = df['open'].iloc[i]
            cur_high = df['high'].iloc[i]
            cur_low = df['low'].iloc[i]
            cur_close = df['close'].iloc[i]
            cur_volume = df['volume'].iloc[i]
            cur_adx = df['adx'].iloc[i]
            cur_rsi = df['rsi'].iloc[i]
            if in_position:
                current_price = get_current_price(SYMBOL)
                progress = (current_price - entry_price) / target_distance
                if progress >= TRAILING_START:
                    stepped = np.floor(progress / TRAILING_STEP) * TRAILING_STEP
                    if stepped > locked_pct:
                        locked_pct = stepped
                        stop_loss = entry_price + (locked_pct * target_distance)
                        log_trade(f"🔒 تریلینگ فعال: %{locked_pct*100:.0f} | حد ضرر جدید {stop_loss:.2f}")
                exit_price = None
                if current_price <= stop_loss:
                    exit_price = stop_loss
                elif current_price >= take_profit:
                    exit_price = take_profit
                if exit_price is not None:
                    ret = (exit_price - entry_price) / entry_price - COMMISSION
                    capital *= (1 + ret)
                    log_trade(f"🚪 خروج | قیمت: {exit_price:.2f} | سود: %{ret*100:.2f} | سرمایه: {capital:.2f}")
                    in_position = False
            else:
                signal = (df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and
                          df['ema_fast'].iloc[i-1] <= df['ema_slow'].iloc[i-1] and
                          cur_close > df['ema_trend'].iloc[i])
                if not signal:
                    time.sleep(10)
                    continue
                avg_vol = df['avg_volume_20'].iloc[i]
                volume_ok = cur_volume > avg_vol * VOLUME_THRESHOLD if not np.isnan(avg_vol) else False
                hour = df.index[i].hour
                time_ok = (TRADING_START_HOUR <= hour < TRADING_END_HOUR) if TIME_FILTER_ENABLED else True
                adx_ok = cur_adx > ADX_THRESHOLD if not np.isnan(cur_adx) else False
                rsi_ok = cur_rsi < RSI_MAX if not np.isnan(cur_rsi) else False
                bullish_ok = (cur_close > cur_open) if BULLISH_CANDLE else True
                if volume_ok and time_ok and adx_ok and rsi_ok and bullish_ok:
                    entry_price = get_current_price(SYMBOL)
                    atr_val = df['atr'].iloc[i]
                    stop_dist = ATR_MULTIPLIER * atr_val
                    stop_loss = entry_price - stop_dist
                    take_profit = entry_price + RISK_REWARD_RATIO * stop_dist
                    target_distance = take_profit - entry_price
                    locked_pct = 0.0
                    position_amount = capital / entry_price
                    in_position = True
                    log_trade(f"🟢 ورود | قیمت: {entry_price:.2f} | SL:{stop_loss:.2f} TP:{take_profit:.2f}")
            time.sleep(10)
        except Exception as e:
            log_trade(f"❌ خطا: {str(e)}")
            time.sleep(30)

if __name__ == "__main__":
    main()
