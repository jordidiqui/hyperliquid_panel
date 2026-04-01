import os, json, time
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st

HL_URL = "https://api.hyperliquid.xyz/info"

st.set_page_config(page_title="Hyperliquid Panel", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

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
    df = df[["t","o","h","l","c","v"]].rename(
        columns={"t":"time","o":"open","h":"high","l":"low","c":"close","v":"volume"})
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
        "funding_rate":      float(ctx["funding"]) * 100,
        "open_interest_usd": float(ctx["openInterest"]) * float(ctx["markPx"]),
        "mark_price":        float(ctx["markPx"]),
        "oracle_price":      float(ctx["oraclePx"]),
        "premium":           (float(ctx["markPx"]) / float(ctx["oraclePx"]) - 1) * 100,
    }

# ─── INDICADORES ──────────────────────────────────────────────────────────────

def calc_rsi(s: pd.Series, p=14) -> pd.Series:
    d    = s.diff()
    gain = d.clip(lower=0).rolling(p).mean()
    loss = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + gain / loss)

def calc_macd(s: pd.Series):
    ml  = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sig = ml.ewm(span=9).mean()
    return ml, sig, ml - sig

def calc_bb(s: pd.Series, p=20):
    sma = s.rolling(p).mean()
    std = s.rolling(p).std()
    return sma + 2*std, sma, sma - 2*std

def calc_atr(df: pd.DataFrame, p=14) -> float:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return round(tr.rolling(p).mean().iloc[-1], 2)

def calc_emas(s: pd.Series):
    return s.ewm(span=20).mean(), s.ewm(span=50).mean(), s.ewm(span=200).mean()

def calc_volume(df: pd.DataFrame):
    cutoff  = df.index[-1] - pd.Timedelta(hours=24)
    vol_24h = df.loc[df.index >= cutoff, "volume"].sum()
    avg_7d  = df["volume"].resample("1D").sum().mean()
    return round(vol_24h, 2), round(avg_7d, 2), round(vol_24h / avg_7d if avg_7d > 0 else 1, 2)

# ─── RÉGIMEN DE MERCADO ───────────────────────────────────────────────────────

def detect_regime(df: pd.DataFrame, close: pd.Series) -> dict:
    """
    Detecta si el mercado está en TREND o RANGE.

    Criterios:
    - Cruce de EMA20: muchos cruces en últimas 48 velas → RANGE
    - ADX sintético (pendiente de EMA50): pendiente fuerte → TREND
    - Volatilidad relativa: ATR actual vs media 30d
    """
    ema20, ema50, ema200 = calc_emas(close)
    atr_now  = calc_atr(df, 14)
    atr_30d  = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1).rolling(720).mean().iloc[-1]  # 30d en 1h

    # 1. Cruces de precio sobre EMA20 en las últimas 48 velas
    window  = min(48, len(close))
    above   = (close.iloc[-window:] > ema20.iloc[-window:]).astype(int)
    crossings = above.diff().abs().sum()

    # 2. Pendiente normalizada de EMA50 (últimas 20 velas)
    ema50_slope = (ema50.iloc[-1] - ema50.iloc[-20]) / ema50.iloc[-20] * 100

    # 3. ATR relativo
    atr_ratio = atr_now / atr_30d if atr_30d > 0 else 1.0

    # Puntuación de rango: más cruces + poca pendiente + ATR normal = RANGE
    range_score = 0
    if crossings >= 6:   range_score += 2
    elif crossings >= 4: range_score += 1
    if abs(ema50_slope) < 0.5:  range_score += 2
    elif abs(ema50_slope) < 1.5: range_score += 1
    if atr_ratio < 1.2: range_score += 1

    regime = "RANGE" if range_score >= 3 else "TREND"

    # Dirección de tendencia si aplica
    trend_dir = "NEUTRAL"
    if regime == "TREND":
        trend_dir = "ALCISTA" if ema50_slope > 0 else "BAJISTA"

    return {
        "regime":      regime,
        "trend_dir":   trend_dir,
        "crossings":   int(crossings),
        "ema50_slope": round(ema50_slope, 3),
        "atr_ratio":   round(atr_ratio, 2),
        "range_score": range_score,
    }

# ─── SOPORTES Y RESISTENCIAS AUTOMÁTICOS ─────────────────────────────────────

def calc_support_resistance(df: pd.DataFrame, atr: float, n_levels=5) -> dict:
    """
    Detecta niveles S/R usando:
    1. Swing highs/lows (máximos/mínimos locales con ventana de 5 velas)
    2. Clustering de niveles cercanos (dentro de 0.5 * ATR)
    3. Ranking por número de toques + recencia

    Devuelve listas de soportes y resistencias ordenadas por precio.
    """
    price  = df["close"].iloc[-1]
    window = 5  # velas a cada lado para considerar swing
    cluster_dist = atr * 0.5

    # Swing highs
    highs = []
    for i in range(window, len(df) - window):
        if df["high"].iloc[i] == df["high"].iloc[i-window:i+window+1].max():
            highs.append((df["high"].iloc[i], i))

    # Swing lows
    lows = []
    for i in range(window, len(df) - window):
        if df["low"].iloc[i] == df["low"].iloc[i-window:i+window+1].min():
            lows.append((df["low"].iloc[i], i))

    def cluster_levels(raw_levels, cluster_dist):
        """Agrupa niveles cercanos y los pondera por recencia y frecuencia."""
        if not raw_levels:
            return []
        levels = sorted(raw_levels, key=lambda x: x[0])
        clustered = []
        group = [levels[0]]
        for lvl, idx in levels[1:]:
            if abs(lvl - group[-1][0]) <= cluster_dist:
                group.append((lvl, idx))
            else:
                avg_price = np.mean([g[0] for g in group])
                recency   = max([g[1] for g in group])  # cuanto más alto, más reciente
                touches   = len(group)
                clustered.append({
                    "price":   round(avg_price, 2),
                    "touches": touches,
                    "recency": recency,
                    "score":   touches * 2 + recency / len(df) * 10
                })
                group = [(lvl, idx)]
        # último grupo
        if group:
            avg_price = np.mean([g[0] for g in group])
            recency   = max([g[1] for g in group])
            touches   = len(group)
            clustered.append({
                "price":   round(avg_price, 2),
                "touches": touches,
                "recency": recency,
                "score":   touches * 2 + recency / len(df) * 10
            })
        return sorted(clustered, key=lambda x: x["score"], reverse=True)

    all_highs = cluster_levels(highs, cluster_dist)
    all_lows  = cluster_levels(lows,  cluster_dist)

    # Separar: por encima del precio = resistencias, por debajo = soportes
    resistances = sorted([l for l in all_highs if l["price"] > price],
                         key=lambda x: x["price"])[:n_levels]
    supports    = sorted([l for l in all_lows  if l["price"] < price],
                         key=lambda x: x["price"], reverse=True)[:n_levels]

    # Nivel más cercano de cada tipo
    nearest_res = resistances[0]["price"] if resistances else None
    nearest_sup = supports[0]["price"]    if supports    else None

    # Distancia en % y ATRs al nivel más cercano
    dist_res_pct = (nearest_res / price - 1) * 100 if nearest_res else None
    dist_sup_pct = (price / nearest_sup - 1) * 100 if nearest_sup else None
    dist_res_atr = (nearest_res - price) / atr      if nearest_res else None
    dist_sup_atr = (price - nearest_sup) / atr      if nearest_sup else None

    return {
        "resistances":    resistances,
        "supports":       supports,
        "nearest_res":    nearest_res,
        "nearest_sup":    nearest_sup,
        "dist_res_pct":   round(dist_res_pct, 2) if dist_res_pct else None,
        "dist_sup_pct":   round(dist_sup_pct, 2) if dist_sup_pct else None,
        "dist_res_atr":   round(dist_res_atr, 2) if dist_res_atr else None,
        "dist_sup_atr":   round(dist_sup_atr, 2) if dist_sup_atr else None,
    }

# ─── ANÁLISIS PRINCIPAL (PESOS DINÁMICOS) ────────────────────────────────────

def analyze_signals(df: pd.DataFrame, ctx: dict, close: pd.Series,
                    account_size: float = 10000) -> dict:
    price    = close.iloc[-1]
    rsi_now  = calc_rsi(close).iloc[-1]
    _, _, macd_h = calc_macd(close)
    macd_now = macd_h.iloc[-1]
    bb_u, bb_m, bb_l = calc_bb(close)
    pct_b    = (price - bb_l.iloc[-1]) / (bb_u.iloc[-1] - bb_l.iloc[-1]) * 100
    ema20, ema50, ema200 = calc_emas(close)
    vol_24h, avg_7d, vol_ratio = calc_volume(df)
    funding  = ctx.get("funding_rate", 0)
    atr_val  = calc_atr(df)

    # ── Régimen y S/R ────────────────────────────────────────────────────────
    regime  = detect_regime(df, close)
    sr      = calc_support_resistance(df, atr_val)

    is_range = regime["regime"] == "RANGE"

    # ── Pesos dinámicos según régimen ────────────────────────────────────────
    # RANGE: RSI y BB lideran (reversión a media)
    # TREND: EMAs y MACD lideran (continuación)
    if is_range:
        W_RSI_EXTREME = 25   # sobreventa/sobrecompra en rango → señal muy fiable
        W_RSI_MID     = 5    # RSI en zona media → menos relevante
        W_MACD        = 10   # MACD genera whipsaws en rango
        W_EMA20       = 8
        W_EMA50       = 8
        W_EMA200      = 5
        W_BB          = 20   # BB extremo en rango → señal fuerte
        W_CONFLUENCE  = 8
    else:  # TREND
        W_RSI_EXTREME = 12   # RSI extremo en tendencia = señal falsa frecuente
        W_RSI_MID     = 8
        W_MACD        = 18
        W_EMA20       = 12
        W_EMA50       = 15
        W_EMA200      = 10
        W_BB          = 8
        W_CONFLUENCE  = 12

    score_long, score_short = 0, 0
    signals = []

    # RSI
    if rsi_now < 30:
        score_long  += W_RSI_EXTREME
        signals.append(f"🟢 RSI sobrevendido ({rsi_now:.0f})")
    elif rsi_now > 70:
        score_short += W_RSI_EXTREME
        signals.append(f"🔴 RSI sobrecomprado ({rsi_now:.0f})")
    elif rsi_now > 50:
        score_long  += W_RSI_MID
        signals.append(f"✅ RSI alcista ({rsi_now:.0f})")
    else:
        score_short += W_RSI_MID
        signals.append(f"✅ RSI bajista ({rsi_now:.0f})")

    # MACD
    if macd_now > 0:
        score_long  += W_MACD
        signals.append("📈 MACD alcista")
    else:
        score_short += W_MACD
        signals.append("📉 MACD bajista")

    # EMAs
    if price > ema20.iloc[-1]:   score_long  += W_EMA20;  signals.append("EMA20 ✅")
    else:                         score_short += W_EMA20;  signals.append("EMA20 ❌")
    if price > ema50.iloc[-1]:   score_long  += W_EMA50;  signals.append("EMA50 ✅")
    else:                         score_short += W_EMA50;  signals.append("EMA50 ❌")
    if price > ema200.iloc[-1]:  score_long  += W_EMA200; signals.append("EMA200 ✅")
    else:                         score_short += W_EMA200; signals.append("EMA200 ❌")

    # Bollinger Bands
    if pct_b < 20:
        score_long  += W_BB
        signals.append(f"BB: zona rebote ({pct_b:.0f}%)")
    elif pct_b > 80:
        score_short += W_BB
        signals.append(f"BB: zona corrección ({pct_b:.0f}%)")

    # Volumen
    if vol_ratio > 1.5:
        score_long  += 5; score_short += 5
        signals.append(f"🔥 Vol {vol_ratio:.1f}x")

    # Funding
    if funding < -0.01:
        score_long  += 15; signals.append(f"Funding bajista {funding:.3f}%")
    elif funding > 0.01:
        score_short += 15; signals.append(f"Funding alcista {funding:.3f}%")

    # Confluencia RSI + MACD
    if rsi_now > 50 and macd_now > 0:
        score_long  += W_CONFLUENCE
        signals.append("⚡ Confluencia alcista RSI+MACD")
    elif rsi_now < 50 and macd_now < 0:
        score_short += W_CONFLUENCE
        signals.append("⚡ Confluencia bajista RSI+MACD")

    # ── Bonus S/R: precio cerca de nivel clave ────────────────────────────────
    sr_signal = None
    if sr["dist_sup_atr"] is not None and sr["dist_sup_atr"] <= 1.5:
        # Muy cerca de soporte → refuerza long si otros indicadores apuntan arriba
        score_long  += 15
        sr_signal    = f"📍 Soporte en ${sr['nearest_sup']:,.0f} ({sr['dist_sup_atr']:.1f} ATR)"
        signals.append(sr_signal)
    if sr["dist_res_atr"] is not None and sr["dist_res_atr"] <= 1.5:
        # Muy cerca de resistencia → refuerza short
        score_short += 15
        sr_signal    = f"🚧 Resist. en ${sr['nearest_res']:,.0f} ({sr['dist_res_atr']:.1f} ATR)"
        signals.append(sr_signal)

    # ── Resultado ─────────────────────────────────────────────────────────────
    direction = ("🟢 LONG" if score_long > score_short
                 else "🔴 SHORT" if score_short > score_long
                 else "🟡 NEUTRAL")
    strength  = max(score_long, score_short)

    if strength >= 80:   leverage,risk = "40x",0.30; sl_m,rr_r = 1.5,3.0; emoji,label = "🚀","MUY FUERTE"
    elif strength >= 60: leverage,risk = "10x",0.10; sl_m,rr_r = 2.0,2.0; emoji,label = "✅","BUENA"
    elif strength >= 40: leverage,risk = "3x", 0.05; sl_m,rr_r = 2.5,1.5; emoji,label = "⚠️","DÉBIL"
    else:                leverage,risk = "-",  0;    sl_m,rr_r = 0,0;      emoji,label = "❌","EVITAR"

    if sl_m > 0:
        sl_dist      = atr_val * sl_m
        risk_amount  = account_size * risk
        position_usd = risk_amount / sl_m * price / 100
        tp_dist      = sl_dist * rr_r
        sl_price     = price - sl_dist if "LONG" in direction else price + sl_dist
        tp_price     = price + tp_dist if "LONG" in direction else price - tp_dist
        pos_size     = position_usd / price
    else:
        sl_price = tp_price = pos_size = risk_amount = 0

    return {
        "direction": direction, "strength": strength,
        "emoji": emoji, "label": label,
        "leverage": leverage, "risk_pct": risk * 100,
        "sl_price": round(sl_price, 2), "tp_price": round(tp_price, 2),
        "position_size": round(pos_size, 4), "risk_amount": round(risk_amount, 0),
        "atr": round(atr_val, 2), "rr_ratio": rr_r,
        "signals": signals[:10],
        "account_size": account_size,
        "regime": regime,
        "sr": sr,
        "score_long": score_long, "score_short": score_short,
    }

# ─── GRÁFICO PRINCIPAL ────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame, coin: str, sr: dict) -> go.Figure:
    close        = df["close"]
    bb_u, bb_m, bb_l = calc_bb(close)
    ema20, ema50, ema200 = calc_emas(close)
    rsi_s        = calc_rsi(close)
    macd_l, macd_s, macd_h = calc_macd(close)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.60, 0.20, 0.20],
                        subplot_titles=(f"{coin}/USDC — 1h", "RSI (14)", "MACD"))

    # Velas
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=close,
        name="Precio", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    # Bollinger
    fig.add_trace(go.Scatter(x=df.index, y=bb_u, line=dict(color="rgba(100,149,237,0.4)",width=1), name="BB↑"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_m, line=dict(color="rgba(100,149,237,0.5)",width=1,dash="dot"), name="BB mid"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_l, line=dict(color="rgba(100,149,237,0.4)",width=1), name="BB↓",
                             fill="tonexty", fillcolor="rgba(100,149,237,0.05)"), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=ema20,  line=dict(color="#FFA500",width=1.2), name="EMA20"),  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema50,  line=dict(color="#FF6347",width=1.2), name="EMA50"),  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema200, line=dict(color="#DA70D6",width=1.5), name="EMA200"), row=1, col=1)

    # ── S/R como líneas horizontales ─────────────────────────────────────────
    for lvl in sr.get("supports", [])[:3]:
        fig.add_hline(y=lvl["price"], line=dict(color="#26a69a", width=1.2, dash="dot"),
                      annotation_text=f"S {lvl['price']:,.0f} ({lvl['touches']}t)",
                      annotation_font_color="#26a69a", annotation_position="right",
                      row=1, col=1)
    for lvl in sr.get("resistances", [])[:3]:
        fig.add_hline(y=lvl["price"], line=dict(color="#ef5350", width=1.2, dash="dot"),
                      annotation_text=f"R {lvl['price']:,.0f} ({lvl['touches']}t)",
                      annotation_font_color="#ef5350", annotation_position="right",
                      row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=rsi_s, line=dict(color="#00BFFF",width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # MACD
    colors_h = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_h]
    fig.add_trace(go.Bar(x=df.index, y=macd_h, marker_color=colors_h, name="Hist", opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_l, line=dict(color="#00BFFF",width=1.2), name="MACD"),  row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_s, line=dict(color="#FF6347",width=1.2), name="Señal"), row=3, col=1)

    fig.update_layout(template="plotly_dark", height=750,
                      margin=dict(l=0,r=0,t=30,b=0),
                      xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h",y=1.02,x=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig

# ─── LAYOUT STREAMLIT ─────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuración")
    COIN        = st.selectbox("Activo", ["BTC","ETH","SOL","HYPE","WIF","PEPE","ARB","AVAX"], index=0)
    INTERVAL    = st.selectbox("Intervalo", ["15m","1h","4h","1d"], index=1)
    LOOKBACK    = st.slider("Historial (horas)", 48, 720, 240, 24)
    st.divider()
    auto_refresh   = st.toggle("Auto-refresh 60s", value=False)
    account_size   = st.number_input("Cuenta (USDC)", 1000, 1_000_000, 10000, 1000)
    refresh_btn    = st.button("🔄 Actualizar", use_container_width=True)
    st.caption(f"Última act.: {datetime.now().strftime('%H:%M:%S')}")

if auto_refresh:
    time.sleep(60)
    st.rerun()

st.title(f"📊 Hyperliquid Panel — {COIN}/USDC")

with st.spinner("Cargando..."):
    df   = get_candles(COIN, INTERVAL, LOOKBACK)
    ctx  = get_market_context(COIN)
    close = df["close"]

price    = close.iloc[-1]
chg_24h  = (price / close.iloc[-24] - 1) * 100 if len(close) >= 24 else 0
rsi_now  = round(calc_rsi(close).iloc[-1], 2)
_, _, hist_s = calc_macd(close)
macd_now = round(hist_s.iloc[-1], 4)
atr_now  = calc_atr(df)
vol_24h, avg_7d, vol_ratio = calc_volume(df)

# ── Métricas superiores ───────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Mark Price",   f"${ctx.get('mark_price',price):,.2f}",  f"{chg_24h:+.2f}% 24h")
c2.metric("RSI (14)",     rsi_now, "Sobrecomprado" if rsi_now>70 else ("Sobrevendido" if rsi_now<30 else "Neutral"))
c3.metric("MACD Hist",    macd_now, "Alcista ▲" if macd_now>0 else "Bajista ▼")
c4.metric("ATR (14)",     atr_now, "Vol/vela")
c5.metric("Funding",      f"{ctx.get('funding_rate',0):+.4f}%", "Longs pagan" if ctx.get('funding_rate',0)>0 else "Shorts pagan")
c6.metric("Open Interest",f"${ctx.get('open_interest_usd',0)/1e6:.1f}M")

st.divider()
analysis = analyze_signals(df, ctx, close, account_size)
regime   = analysis["regime"]
sr       = analysis["sr"]

# ── Régimen + S/R en banner ───────────────────────────────────────────────────
reg_col = "#26a69a" if regime["regime"]=="RANGE" else "#FFA500"
st.markdown(
    f"<div style='background:{reg_col}22;border-left:4px solid {reg_col};"
    f"padding:10px 16px;border-radius:6px;margin-bottom:12px'>"
    f"<b>Régimen:</b> {regime['regime']} "
    f"{'↕️ (rebotes y rechazos)' if regime['regime']=='RANGE' else '→ (tendencia ' + regime['trend_dir'] + ')'}"
    f" &nbsp;|&nbsp; <b>Cruces EMA20:</b> {regime['crossings']} en 48 velas"
    f" &nbsp;|&nbsp; <b>Pendiente EMA50:</b> {regime['ema50_slope']:+.2f}%"
    f" &nbsp;|&nbsp; <b>ATR ratio:</b> {regime['atr_ratio']:.2f}x"
    f"</div>", unsafe_allow_html=True
)

# ── Gráfico ───────────────────────────────────────────────────────────────────
st.plotly_chart(build_chart(df, COIN, sr), use_container_width=True)

# ── Análisis + señal ──────────────────────────────────────────────────────────
st.subheader("🤖 Señal + Gestión de Riesgo")
col1, col2, col3, col4 = st.columns([1,1,1,2])
col1.metric("Dirección",  analysis["direction"])
col2.metric("Fuerza",     f"{analysis['emoji']} {analysis['strength']}", analysis["label"])
col3.metric("Régimen",    regime["regime"], f"Peso RSI {'alto' if regime['regime']=='RANGE' else 'normal'}")
with col4:
    st.write("**Señales activas:**")
    for s in analysis["signals"][:6]:
        st.write(f"· {s}")

# ── S/R tabla ─────────────────────────────────────────────────────────────────
st.divider()
col_sr1, col_sr2 = st.columns(2)

with col_sr1:
    st.subheader("🛡️ Soportes detectados")
    if sr["supports"]:
        sup_df = pd.DataFrame(sr["supports"])[["price","touches"]].rename(
            columns={"price":"Precio","touches":"Toques"})
        sup_df["Dist %"] = ((price / sup_df["Precio"] - 1) * 100).round(2).astype(str) + "%"
        sup_df["Dist ATR"] = ((price - sup_df["Precio"]) / atr_now).round(1)
        st.dataframe(sup_df, hide_index=True, use_container_width=True)
    else:
        st.info("Sin soportes detectados en el rango visible.")

with col_sr2:
    st.subheader("🚧 Resistencias detectadas")
    if sr["resistances"]:
        res_df = pd.DataFrame(sr["resistances"])[["price","touches"]].rename(
            columns={"price":"Precio","touches":"Toques"})
        res_df["Dist %"] = ((res_df["Precio"] / price - 1) * 100).round(2).astype(str) + "%"
        res_df["Dist ATR"] = ((res_df["Precio"] - price) / atr_now).round(1)
        st.dataframe(res_df, hide_index=True, use_container_width=True)
    else:
        st.info("Sin resistencias detectadas en el rango visible.")

# ── Plan de trade ─────────────────────────────────────────────────────────────
st.divider()
if analysis["leverage"] != "-":
    st.subheader("📋 Plan de Trade")
    trade_data = {
        "Parámetro": ["Leverage","Riesgo cuenta","SL","TP","RR","ATR"],
        "Valor": [
            analysis["leverage"],
            f"{analysis['risk_pct']}% (${analysis['risk_amount']})",
            f"${analysis['sl_price']:,.2f}",
            f"${analysis['tp_price']:,.2f}",
            f"1:{analysis['rr_ratio']:.1f}",
            f"${analysis['atr']:,.2f}",
        ]
    }
    st.dataframe(pd.DataFrame(trade_data), hide_index=True)
else:
    st.warning("❌ Señal insuficiente — sin operación recomendada")

# ── Volumen + EMAs ────────────────────────────────────────────────────────────
st.divider()
ca, cb = st.columns(2)
with ca:
    st.subheader("📦 Volumen")
    st.dataframe(pd.DataFrame({
        "Métrica": ["Volumen 24h","Media diaria 7d","Ratio vs media"],
        "Valor":   [f"{vol_24h:,.2f}", f"{avg_7d:,.2f}", f"{vol_ratio:.2f}x"]
    }), hide_index=True, use_container_width=True)
with cb:
    st.subheader("📐 EMAs")
    ema20v, ema50v, ema200v = calc_emas(close)
    st.dataframe(pd.DataFrame({
        "EMA": ["EMA20","EMA50","EMA200"],
        "Valor": [f"{ema20v.iloc[-1]:,.2f}", f"{ema50v.iloc[-1]:,.2f}", f"{ema200v.iloc[-1]:,.2f}"],
        "Posición": [
            "▲ Encima" if price>ema20v.iloc[-1] else "▼ Debajo",
            "▲ Encima" if price>ema50v.iloc[-1] else "▼ Debajo",
            "▲ Encima" if price>ema200v.iloc[-1] else "▼ Debajo",
        ]
    }), hide_index=True, use_container_width=True)
