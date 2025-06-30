import subprocess
import sys

# تثبيت الحزم المطلوبة قبل الاستيراد (طريقة مؤقتة لتجنب خطأ عدم وجود المكتبات)
required_packages = [
    "streamlit",
    "pandas",
    "requests",
    "plotly",
    "streamlit-autorefresh",
    "ta"
]

for package in required_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import streamlit as st
st.set_page_config(page_title="Forex Dashboard", layout="wide")  # يجب أن يكون أول شيء في السكريبت

import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import timedelta
from streamlit_autorefresh import st_autorefresh
import ta

API_KEY = "bf15a39017ca4f0ba029e29ce8334b76"
REFRESH_INTERVAL = 60

TP_PERCENT = 0.01
SL_PERCENT = 0.005
SAFE_RSI_MIN = 35
SAFE_RSI_MAX = 65
SUPPORT_RESISTANCE_LOOKBACK = 30

def fetch_data(symbol):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 500,
        "apikey": API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "values" not in data:
            st.error(f"API Error: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.astype(float)
        return df.sort_index()
    except Exception as e:
        st.error(f"Request failed: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    df['sma14'] = ta.trend.sma_indicator(df['close'], window=14)
    df['ema14'] = ta.trend.ema_indicator(df['close'], window=14)
    df['rsi14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    return df

def find_support_resistance(df):
    lookback_df = df.tail(SUPPORT_RESISTANCE_LOOKBACK)
    support = lookback_df['low'].min()
    resistance = lookback_df['high'].max()
    return support, resistance

def identify_signals(df, support, resistance):
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        price = row['close']
        near_support = abs(price - support) / support < 0.005
        near_resistance = abs(price - resistance) / resistance < 0.005

        buy_cond = (
            SAFE_RSI_MIN <= row['rsi14'] <= SAFE_RSI_MAX and
            near_support and
            row['macd'] > row['macd_signal'] and
            prev['macd'] <= prev['macd_signal'] and
            price > row['sma14']
        )
        sell_cond = (
            SAFE_RSI_MIN <= row['rsi14'] <= SAFE_RSI_MAX and
            near_resistance and
            row['macd'] < row['macd_signal'] and
            prev['macd'] >= prev['macd_signal'] and
            price < row['sma14']
        )

        if buy_cond:
            signals.append({'type': 'buy', 'date': row.name, 'price': price})
        elif sell_cond:
            signals.append({'type': 'sell', 'date': row.name, 'price': price})

    return signals

def display_chart(df, signals, support, resistance, symbol):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name="Close", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['sma14'], name="SMA 14", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema14'], name="EMA 14", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_high'], name="BB High", line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_low'], name="BB Low", line=dict(color='red', dash='dot')))

    fig.add_hline(y=support, line_dash="dash", line_color='green', annotation_text="Support", annotation_position="bottom left")
    fig.add_hline(y=resistance, line_dash="dash", line_color='red', annotation_text="Resistance", annotation_position="top left")

    for signal in signals:
        date = signal['date']
        price = signal['price']

        if signal['type'] == 'buy':
            tp = price * (1 + TP_PERCENT)
            sl = price * (1 - SL_PERCENT)
            marker_color = 'green'
            text = 'Buy'
        else:
            tp = price * (1 - TP_PERCENT)
            sl = price * (1 + SL_PERCENT)
            marker_color = 'red'
            text = 'Sell'

        fig.add_trace(go.Scatter(
            x=[date], y=[price],
            mode='markers+text',
            marker=dict(color=marker_color, size=12, symbol='triangle-up' if signal['type'] == 'buy' else 'triangle-down'),
            text=[text],
            textposition='top center',
            name=f"{text} Entry"
        ))

        fig.add_hline(y=tp, line_dash="dot", line_color='goldenrod', annotation_text="TP", annotation_position="top left")
        fig.add_hline(y=sl, line_dash="dot", line_color='firebrick', annotation_text="SL", annotation_position="bottom left")

    fig.update_layout(title=f"{symbol} Price Chart with Buy/Sell Signals & TP/SL",
                      height=700,
                      xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def show_previous_week_data(df):
    st.subheader("📅 بيانات الأسبوع السابق")
    if df.empty:
        st.info("لا توجد بيانات كافية للأسبوع السابق")
        return
    last_date = df.index.max()
    start_date = last_date - timedelta(days=7)
    last_week = df.loc[start_date:last_date]
    if last_week.empty:
        st.info("لا توجد بيانات كافية للأسبوع السابق")
        return
    st.dataframe(last_week[['open', 'high', 'low', 'close']].style.format("{:.4f}"), use_container_width=True)

def market_analysis(df, signals, support, resistance):
    st.subheader("📊 تحليل السوق والتوصيات")

    last = df.iloc[-1]
    price = last['close']
    reasons = []

    if not signals:
        st.info("لا توجد إشارات شراء أو بيع واضحة حاليا.")

        if last['rsi14'] > 70:
            reasons.append("مؤشر RSI مرتفع (أكثر من 70) يشير إلى تشبع شراء.")
        elif last['rsi14'] < 30:
            reasons.append("مؤشر RSI منخفض (أقل من 30) يشير إلى تشبع بيع.")

        if price > last['sma14']:
            reasons.append("السعر أعلى من المتوسط المتحرك البسيط (SMA14) مما يشير إلى اتجاه صاعد.")
        elif price < last['sma14']:
            reasons.append("السعر أقل من المتوسط المتحرك البسيط (SMA14) مما يشير إلى اتجاه هابط.")

        macd_trend = "صاعد" if last['macd'] > last['macd_signal'] else "هابط"
        reasons.append(f"مؤشر MACD يظهر اتجاه {macd_trend}.")

        dist_support = abs(price - support) / support
        dist_resistance = abs(price - resistance) / resistance
        reasons.append(f"السعر الحالي يبعد {dist_support*100:.2f}% عن الدعم و {dist_resistance*100:.2f}% عن المقاومة.")

        st.write("### الحالة الحالية للسوق:")
        for reason in reasons:
            st.write(f"- {reason}")

        st.write("### التوصية:")
        st.write("لا توجد إشارة واضحة حالياً. يُفضل انتظار تأكيدات أو إشارات أوضح قبل اتخاذ قرار.")
    else:
        last_signal = signals[-1]
        if last_signal['type'] == 'buy':
            st.success(f"✅ توصية شراء بتاريخ {last_signal['date'].date()} عند السعر {last_signal['price']:.4f}")
            st.write(f"جاءت إشارة الشراء بناءً على:\n- RSI بين {SAFE_RSI_MIN} و {SAFE_RSI_MAX}\n- السعر قريب من مستوى الدعم\n- تقاطع إيجابي لمؤشر MACD\n- السعر فوق SMA14")
            st.write(f"الهدف (TP) عند {last_signal['price'] * (1 + TP_PERCENT):.4f}، وقف الخسارة (SL) عند {last_signal['price'] * (1 - SL_PERCENT):.4f}")
        else:
            st.error(f"⛔ توصية بيع بتاريخ {last_signal['date'].date()} عند السعر {last_signal['price']:.4f}")
            st.write(f"جاءت إشارة البيع بناءً على:\n- RSI بين {SAFE_RSI_MIN} و {SAFE_RSI_MAX}\n- السعر قريب من مستوى المقاومة\n- تقاطع سلبي لمؤشر MACD\n- السعر تحت SMA14")
            st.write(f"الهدف (TP) عند {last_signal['price'] * (1 - TP_PERCENT):.4f}، وقف الخسارة (SL) عند {last_signal['price'] * (1 + SL_PERCENT):.4f}")

def main():
    symbols = ["XAU/USD", "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]
    symbol = st.selectbox("اختر رمز التداول:", symbols, index=0)

    st.title(f"📊 {symbol} Forex Dashboard with Buy/Sell Signals")

    st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="auto_refresh")

    df = fetch_data(symbol)
    if df.empty:
        st.stop()

    df = calculate_indicators(df)
    support, resistance = find_support_resistance(df)
    signals = identify_signals(df, support, resistance)

    st.subheader("📈 Price Chart & Indicators with Signals")
    display_chart(df, signals, support, resistance, symbol)

    show_previous_week_data(df)

    st.subheader("📊 Recent Data Snapshot")
    st.dataframe(df.tail(10).style.format("{:.4f}"), use_container_width=True)

    market_analysis(df, signals, support, resistance)

    st.button("🔁 تحديث يدوي", on_click=lambda: st.cache_data.clear())

if __name__ == "__main__":
    main()
