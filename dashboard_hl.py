import time
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st
import threading

TELEGRAM_TOKEN   = st.secrets.get("TELEGRAM_TOKEN", "")   # desde secrets de Streamlit Cloud
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")

def send_telegram(message: str):
    """Envía alerta a Telegram de forma asíncrona (no bloquea el dashboard)"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    def _send():
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})
    threading.Thread(target=_send).start()

def build_alert_message(coin: str, analysis: dict) -> str:
    direction_emoji = "🟢" if "LONG" in analysis["direction"] else "🔴"
    return (
        f"*🚨 SEÑAL FUERTE — Hyperliquid*\n"
        f"{direction_emoji} *{analysis['direction']}* en *{coin}/USDC*\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📊 Fuerza: *{analysis['strength']}/100*\n"
        f"💰 Precio: `${analysis.get('mark_price', 0):,.2f}`\n"
        f"📐 Leverage sugerido: *{analysis['leverage']}*\n"
        f"🛑 SL: `${analysis['sl_price']:,.2f}`\n"
        f"🎯 TP: `${analysis['tp_price']:,.2f}`\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📡 {' · '.join(analysis['signals'][:4])}"
    )

HL_URL = "https://api.hyperliquid.xyz/info"

# ─── CONFIG PÁGINA ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hyperliquid Panel",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DATA FETCHING ────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_candles(coin: str, interval: str, lookback_hours: int) -> pd.DataFrame:
    end   = int(datetime.now().timestamp() * 1000)
    start = int((datetime.now() - timedelta(hours=lookback_hours)).timestamp() * 1000)
    r = requests.post(HL_URL, json={"type": "candleSnapshot", "req": {
        "coin": coin, "interval": interval, "startTime": start, "endTime": end
    }})
    raw = r.json()
    df = pd.DataFrame(raw, columns=["t","T","s","i","o","c","h","l","v","n"])
    df = df[["t","o","h","l","c","v"]].rename(columns={"t":"time","o":"open","h":"high","l":"low","c":"close","v":"volume"})
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.set_index("time")

@st.cache_data(ttl=60)
def get_market_context(coin: str) -> dict:
    r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"})
    meta, ctxs = r.json()
    coins = [m["name"] for m in meta["universe"]]
    if coin not in coins:
        return {}
    idx = coins.index(coin)
    ctx = ctxs[idx]
    return {
        "funding_rate":       float(ctx["funding"]) * 100,
        "open_interest_usd":  float(ctx["openInterest"]) * float(ctx["markPx"]),
        "mark_price":         float(ctx["markPx"]),
        "oracle_price":       float(ctx["oraclePx"]),
        "premium":            (float(ctx["markPx"]) / float(ctx["oraclePx"]) - 1) * 100,
    }

# ─── INDICADORES ─────────────────────────────────────────────────────────────

def calc_rsi(s: pd.Series, p=14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).rolling(p).mean()
    loss = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + gain / loss)

def calc_macd(s: pd.Series):
    ml  = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sig = ml.ewm(span=9).mean()
    return ml, sig, ml - sig

def calc_bb(s: pd.Series, p=20):
    sma   = s.rolling(p).mean()
    std   = s.rolling(p).std()
    return sma + 2*std, sma, sma - 2*std

def calc_atr(df: pd.DataFrame, p=14) -> float:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return round(tr.rolling(p).mean().iloc[-1], 2)

def calc_emas(s: pd.Series):
    return (s.ewm(span=20).mean(), s.ewm(span=50).mean(), s.ewm(span=200).mean())

def calc_volume(df: pd.DataFrame):
    cutoff = df.index[-1] - pd.Timedelta(hours=24)
    vol_24h = df.loc[df.index >= cutoff, "volume"].sum()
    avg_7d  = df["volume"].resample("1D").sum().mean()
    return round(vol_24h, 2), round(avg_7d, 2), round(vol_24h / avg_7d if avg_7d > 0 else 1, 2)

# ─── ANÁLISIS ─────────────────────────────────────────────────────────────

def analyze_signals(df: pd.DataFrame, ctx: dict, close: pd.Series, account_size: float = 10000) -> dict:
    """Analizador con gestión de riesgo completa"""
    price = close.iloc[-1]
    rsi_now = calc_rsi(close).iloc[-1]
    _, _, macd_h = calc_macd(close)
    macd_now = macd_h.iloc[-1]
    bb_u, bb_m, bb_l = calc_bb(close)
    pct_b = (price - bb_l.iloc[-1]) / (bb_u.iloc[-1] - bb_l.iloc[-1]) * 100
    ema20, ema50, ema200 = calc_emas(close)
    vol_24h, avg_7d, vol_ratio = calc_volume(df)
    funding = ctx.get('funding_rate', 0)
    atr_val = calc_atr(df)
    
    score_long, score_short = 0, 0
    signals = []
    
    # [MISMAS REGLAS DE ANTES PARA RSI, MACD, EMAs, BB, Volumen, Funding]
    if rsi_now < 30:
        score_long += 20; signals.append("🟢 RSI sobrevendido")
    elif rsi_now > 70:
        score_short += 20; signals.append("🔴 RSI sobrecomprado")
    elif rsi_now > 50:
        score_long += 10; signals.append("✅ RSI alcista")
    else:
        score_short += 10; signals.append("✅ RSI bajista")
    
    if macd_now > 0:
        score_long += 15; signals.append("📈 MACD alcista")
    else:
        score_short += 15; signals.append("📉 MACD bajista")
    
    if price > ema20.iloc[-1]: score_long += 10; signals.append("EMA20 ✅")
    else: score_short += 10; signals.append("EMA20 ❌")
    if price > ema50.iloc[-1]: score_long += 15; signals.append("EMA50 ✅")
    else: score_short += 15; signals.append("EMA50 ❌")
    
    if pct_b < 20: 
        score_long += 15; signals.append("BB: zona rebote")
    elif pct_b > 80:
        score_short += 15; signals.append("BB: zona corrección")
    
    if vol_ratio > 1.5:
        score_long += 5; score_short += 5; signals.append(f"🔥 Vol {vol_ratio:.1f}x")
    
    if funding < -0.01:
        score_long += 15; signals.append("Funding longs")
    elif funding > 0.01:
        score_short += 15; signals.append("Funding shorts")
    
    # GESTIÓN DE RIESGO
    direction = "🟢 LONG" if score_long > score_short else ("🔴 SHORT" if score_short > score_long else "🟡 NEUTRAL")
    strength = max(score_long, score_short)
    
    # Leverage y riesgo según puntuación
    if strength >= 80:
        leverage = "10x", 0.02  # 2% riesgo
        sl_mult  = 1.5
        rr_ratio = 3.0
        emoji, label = "🚀", "MUY FUERTE"
    elif strength >= 60:
        leverage = "5x", 0.01   # 1% riesgo
        sl_mult  = 2.0
        rr_ratio = 2.0
        emoji, label = "✅", "BUENA"
    elif strength >= 40:
        leverage = "3x", 0.005  # 0.5% riesgo
        sl_mult  = 2.5
        rr_ratio = 1.5
        emoji, label = "⚠️", "DÉBIL"
    else:
        leverage = "-", "-"
        sl_mult = rr_ratio = 0
        emoji, label = "❌", "EVITAR"
    
    # Cálculos de posición
    if sl_mult > 0:
        sl_distance  = atr_val * sl_mult
        risk_amount  = account_size * leverage[1]
        position_usd = risk_amount / sl_mult * price / 100  # ajustado por leverage implícito
        tp_distance  = sl_distance * rr_ratio
        
        if "LONG" in direction:
            sl_price = price - sl_distance
            tp_price = price + tp_distance
        else:
            sl_price = price + sl_distance
            tp_price = price - tp_distance
        
        position_size = position_usd / price  # contratos
    else:
        sl_price = tp_price = position_size = risk_amount = 0
    
    return {
        "direction": direction,
        "strength": strength,
        "emoji": emoji,
        "label": label,
        "leverage": leverage[0],
        "risk_pct": leverage[1]*100,
        "sl_price": round(sl_price, 2),
        "tp_price": round(tp_price, 2),
        "position_size": round(position_size, 4),
        "risk_amount": round(risk_amount, 0),
        "atr": round(atr_val, 2),
        "rr_ratio": rr_ratio,  
        "signals": signals[:8],
        "account_size": account_size
    }

# ─── GRÁFICO PRINCIPAL ────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame, coin: str) -> go.Figure:
    close = df["close"]
    bb_u, bb_m, bb_l     = calc_bb(close)
    ema20, ema50, ema200 = calc_emas(close)
    rsi_series           = calc_rsi(close)
    macd_l, macd_s, macd_h = calc_macd(close)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=(f"{coin}/USDC — Velas 1h", "RSI (14)", "MACD")
    )

    # Velas
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=close, name="Precio",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=bb_u, line=dict(color="rgba(100,149,237,0.4)", width=1), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_m, line=dict(color="rgba(100,149,237,0.6)", width=1, dash="dot"), name="BB Mid"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_l, line=dict(color="rgba(100,149,237,0.4)", width=1), name="BB Lower",
                             fill="tonexty", fillcolor="rgba(100,149,237,0.05)"), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=ema20,  line=dict(color="#FFA500", width=1.2), name="EMA20"),  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema50,  line=dict(color="#FF6347", width=1.2), name="EMA50"),  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema200, line=dict(color="#DA70D6", width=1.5), name="EMA200"), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=rsi_series, line=dict(color="#00BFFF", width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", row=2, col=1)

    # MACD
    colors_hist = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_h]
    fig.add_trace(go.Bar(x=df.index, y=macd_h, marker_color=colors_hist, name="MACD Hist", opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_l, line=dict(color="#00BFFF", width=1.2), name="MACD"),   row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_s, line=dict(color="#FF6347", width=1.2), name="Señal"),  row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=750,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig

# ─── LAYOUT STREAMLIT ─────────────────────────────────────────────────────────

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuración")
    COIN     = st.selectbox("Activo", ["BTC","ETH","SOL","HYPE","WIF","PEPE","ARB","AVAX"], index=0)
    INTERVAL = st.selectbox("Intervalo velas", ["15m","1h","4h","1d"], index=1)
    LOOKBACK = st.slider("Historial (horas)", min_value=48, max_value=720, value=240, step=24)
    st.divider()
    auto_refresh = st.toggle("Auto-refresh 60s", value=False)
    st.divider()
    account_size = st.number_input("Tamaño cuenta (USDC)", min_value=1000, max_value=1000000, value=10000, step=1000)
    refresh_btn  = st.button("🔄 Actualizar ahora", use_container_width=True)
    st.caption(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")

# Auto-refresh
if auto_refresh:
    time.sleep(60)
    st.rerun()

# Header
st.title(f"📊 Panel Hyperliquid — {COIN}/USDC")

# Datos
with st.spinner("Cargando datos..."):
    df  = get_candles(COIN, INTERVAL, LOOKBACK)
    ctx = get_market_context(COIN)
    close = df["close"]

# ─── MÉTRICAS SUPERIORES ──────────────────────────────────────────────────────
price    = close.iloc[-1]
chg_24h  = (price / close.iloc[-24] - 1) * 100 if len(close) >= 24 else 0
rsi_now  = round(calc_rsi(close).iloc[-1], 2)
_, _, hist_series = calc_macd(close)
macd_now = round(hist_series.iloc[-1], 4)
atr_now  = calc_atr(df)
vol_24h, avg_7d, vol_ratio = calc_volume(df)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Mark Price",     f"${ctx.get('mark_price', price):,.2f}",  f"{chg_24h:+.2f}% 24h")
col2.metric("RSI (14)",       rsi_now, "Sobrecomprado" if rsi_now > 70 else ("Sobrevendido" if rsi_now < 30 else "Neutral"))
col3.metric("MACD Hist",      macd_now, "Alcista ▲" if macd_now > 0 else "Bajista ▼")
col4.metric("ATR (14)",       atr_now,  "Volatilidad por vela")
col5.metric("Funding Rate",   f"{ctx.get('funding_rate', 0):+.4f}%", "Longs pagan" if ctx.get('funding_rate', 0) > 0 else "Shorts pagan")
col6.metric("Open Interest",  f"${ctx.get('open_interest_usd', 0)/1e6:.1f}M")

st.divider()

# ─── GRÁFICO ──────────────────────────────────────────────────────────────────
st.plotly_chart(build_chart(df, COIN), use_container_width=True)

# ─── ANALIZADOR CON GESTIÓN DE RIESGO ─────────────────────────────────────────
analysis = analyze_signals(df, ctx, close, account_size)

# ─── ALERTA TELEGRAM ──────────────────────────────────────────────────────────
ALERT_THRESHOLD = 70

alert_key = f"last_alert_{COIN}"
last_strength = st.session_state.get(alert_key, 0)

if analysis["strength"] >= ALERT_THRESHOLD and analysis["strength"] != last_strength:
    msg = build_alert_message(COIN, analysis)
    send_telegram(msg)
    st.session_state[alert_key] = analysis["strength"]
    st.toast(f"📬 Alerta Telegram enviada — {analysis['direction']}", icon="🚨")

# ─── ANALIZADOR CON GESTIÓN DE RIESGO ─────────────────────────────────────────
st.subheader("🤖 ANALIZADOR + GESTIÓN DE RIESGO")
col1, col2, col3 = st.columns([1,1,2])

with col1:
    st.metric("Dirección", analysis["direction"])
with col2:
    st.metric("Fuerza", f"{analysis['emoji']} {analysis['strength']}", analysis["label"])

with col3:
    st.success(analysis["signals"][:3])  # top 3 señales

st.divider()

# Tabla de trade plan
if analysis["leverage"] != "-":
    st.subheader("📋 PLAN DE TRADE AUTOMÁTICO")
    trade_data = {
        "Parámetro": ["Leverage", "Riesgo cuenta", "Posición (USDC)", "Tamaño (contratos)", "SL", "TP", "RR"],
        "Valor": [analysis["leverage"], f"{analysis['risk_pct']}% (${analysis['risk_amount']})", 
                 f"${analysis['position_size']*price:.0f}", f"{analysis['position_size']:.4f}",
                 f"${analysis['sl_price']:.2f}", f"${analysis['tp_price']:.2f}", f"1:{analysis['rr_ratio']:.1f}"]
    }
    st.dataframe(pd.DataFrame(trade_data), hide_index=True)
    
    st.caption("**Todas las señales:** " + " · ".join(analysis["signals"]))
else:
    st.warning("❌ Sin confluencia suficiente — NO OPERAR")


# ─── TABLAS INFERIORES ────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📦 Volumen")
    st.dataframe(pd.DataFrame({
        "Métrica": ["Volumen 24h", "Media diaria 7d", "Ratio vs media"],
        "Valor":   [f"{vol_24h:,.2f}", f"{avg_7d:,.2f}", f"{vol_ratio:.2f}x"]
    }), hide_index=True, use_container_width=True)

with col_b:
    st.subheader("📐 EMAs actuales")
    ema20, ema50, ema200 = calc_emas(close)
    ema_data = {
        "EMA": ["EMA20", "EMA50", "EMA200"],
        "Valor": [f"{ema20.iloc[-1]:,.2f}", f"{ema50.iloc[-1]:,.2f}", f"{ema200.iloc[-1]:,.2f}"],
        "Precio vs EMA": [
            "▲ Por encima" if price > ema20.iloc[-1] else "▼ Por debajo",
            "▲ Por encima" if price > ema50.iloc[-1] else "▼ Por debajo",
            "▲ Por encima" if price > ema200.iloc[-1] else "▼ Por debajo",
        ]
    }
    st.dataframe(pd.DataFrame(ema_data), hide_index=True, use_container_width=True)
