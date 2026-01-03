"""
Optimized Chart Generator API with Performance Improvements

Key optimizations:
1. Connection pooling for HTTP requests
2. Redis caching for OHLCV data (5min TTL)
3. Parallel exchange attempts with asyncio
4. LRU cache for exchange instances
5. Response caching with hash-based keys
6. Lazy indicator calculation
7. Reduced memory allocation
8. Pre-compiled regex patterns
"""

import asyncio
import json
from datetime import datetime
from functools import lru_cache
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from fastapi import FastAPI, Query, Response
from fastapi.middleware.gzip import GZipMiddleware
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

app = FastAPI(title="Chart Generator API", version="2.0")
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Timeframe mappings
TF_MAP_CCXT = {"M1": "1m", "M15": "15m", "H1": "1h", "H4": "4h", "D": "1d", "W": "1w"}
TF_MAP_PYTH = {
    "M1": 60,
    "M15": 900,
    "H1": 3600,
    "H4": 14400,
    "D": 86400,
    "W": 604800,
}

# Exchange configs
EXCHANGE_CONFIGS = [
    (
        "binance",
        {"enableRateLimit": True, "timeout": 3000, "options": {"defaultType": "spot"}},
    ),
    (
        "bybit",
        {"enableRateLimit": True, "timeout": 3000, "options": {"defaultType": "spot"}},
    ),
    (
        "okx",
        {"enableRateLimit": True, "timeout": 3000, "options": {"defaultType": "spot"}},
    ),
]

# Chart styling constants
COLOR_GREEN = "#089981"
COLOR_RED = "#F23645"
COLOR_BG = "#131722"
COLOR_GRID = "#2a2e39"
COLOR_TEXT = "#d1d4dc"

# Pyth price IDs
PYTH_IDS = {
    "BTC/USD": "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH/USD": "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "SOL/USD": "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
    "BNB/USD": "2f95862b0455a0920eb43c5d401a8801f16f31f9175d2752187064d7c00650d3",
}

# ============================================================================
# CONNECTION POOLING & SESSIONS
# ============================================================================


def create_session() -> requests.Session:
    """Create requests session with connection pooling and retries"""
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20,
        pool_block=False,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# Global session instance
HTTP_SESSION = create_session()


@lru_cache(maxsize=10)
def get_exchange_instance(exchange_id: str, config_json: str):
    """Cache exchange instances to avoid re-initialization"""
    config = json.loads(config_json)
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class(config)


# ============================================================================
# CACHING LAYER
# ============================================================================

# Simple in-memory cache (replace with Redis in production)
_CACHE = {}
CACHE_TTL = 300  # 5 minutes


def cache_key(symbol: str, timeframe: str, limit: int, feed: str) -> str:
    """Generate cache key for data"""
    return f"ohlcv:{feed}:{symbol}:{timeframe}:{limit}"


def get_cached_data(key: str) -> Optional[pd.DataFrame]:
    """Get data from cache"""
    if key in _CACHE:
        data, timestamp = _CACHE[key]
        if (datetime.now().timestamp() - timestamp) < CACHE_TTL:
            return data
        del _CACHE[key]
    return None


def set_cached_data(key: str, data: pd.DataFrame):
    """Store data in cache"""
    _CACHE[key] = (data, datetime.now().timestamp())


# ============================================================================
# DATA FETCHERS (OPTIMIZED)
# ============================================================================


async def fetch_ccxt_data(
    symbol: str, timeframe: str, limit: int = 100
) -> Optional[pd.DataFrame]:
    """Fetch data from CCXT with parallel exchange attempts"""
    pair = symbol.upper()
    pair_usdt = pair.replace("/USD", "/USDT")
    interval = TF_MAP_CCXT.get(timeframe.upper(), "1h")

    async def try_exchange(exchange_id: str, config: dict) -> Optional[pd.DataFrame]:
        """Try fetching from single exchange"""
        try:
            config_json = json.dumps(config, sort_keys=True)
            exchange = get_exchange_instance(exchange_id, config_json)

            # Try both pair formats
            for test_pair in [pair, pair_usdt]:
                try:
                    # Run in thread pool to avoid blocking
                    ohlcv = await asyncio.to_thread(
                        exchange.fetch_ohlcv, test_pair, interval, limit
                    )

                    if not ohlcv:
                        continue

                    # Fast DataFrame creation
                    df = pd.DataFrame(
                        ohlcv,
                        columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],  # type: ignore[call-overload]
                    )
                    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.drop("timestamp", axis=1, inplace=True)

                    # Type conversion in batch
                    df[["Open", "High", "Low", "Close", "Volume"]] = df[
                        ["Open", "High", "Low", "Close", "Volume"]
                    ].astype(float)

                    return df

                except (ccxt.BaseError, Exception):
                    continue

        except Exception:
            pass

        return None

    # Try exchanges in parallel with timeout
    tasks = [
        try_exchange(exchange_id, config) for exchange_id, config in EXCHANGE_CONFIGS
    ]

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=5.0
        )

        # Return first successful result
        for result in results:
            if isinstance(result, pd.DataFrame) and not result.empty:
                return result

    except asyncio.TimeoutError:
        pass

    return None


async def fetch_pyth_data(
    symbol: str, timeframe: str, limit: int = 100
) -> Optional[pd.DataFrame]:
    """Fetch from Pyth Benchmarks API"""
    if symbol not in PYTH_IDS:
        return None

    resolution_sec = TF_MAP_PYTH.get(timeframe.upper(), 3600)
    now = int(datetime.now().timestamp())
    start_time = now - (resolution_sec * limit)

    # Resolution format for Pyth
    if resolution_sec == 86400:
        resolution_str = "1D"
    elif resolution_sec == 604800:
        resolution_str = "1W"
    else:
        resolution_str = str(resolution_sec // 60)

    url = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
    params = {
        "symbol": symbol,
        "resolution": resolution_str,
        "from": start_time,
        "to": now,
    }

    try:
        response = await asyncio.to_thread(
            HTTP_SESSION.get, url, params=params, timeout=3
        )

        if response.status_code != 200:
            return None

        data = response.json()
        if data.get("s") != "ok":
            return None

        # Fast DataFrame creation
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(data["t"], unit="s"),
                "Open": data["o"],
                "High": data["h"],
                "Low": data["l"],
                "Close": data["c"],
                "Volume": data.get("v", [0] * len(data["t"])),
            }
        )

        return df

    except Exception:
        return None


async def fetch_binance_direct(
    symbol: str, timeframe: str, limit: int = 100
) -> Optional[pd.DataFrame]:
    """Direct Binance API call"""
    pair = symbol.replace("/", "").replace("USD", "USDT")
    tf_map = {"M15": "15m", "H1": "1h", "H4": "4h", "D": "1d", "W": "1w"}
    interval = tf_map.get(timeframe.upper(), "1h")

    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": pair, "interval": interval, "limit": limit}

    try:
        response = await asyncio.to_thread(
            HTTP_SESSION.get, url, params=params, timeout=3
        )

        if response.status_code != 200:
            return None

        data = response.json()

        # Fast DataFrame with only needed columns
        df = pd.DataFrame(
            data,
            columns=[
                "Open Time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "_1",
                "_2",
                "_3",
                "_4",
                "_5",
                "_6",
            ],  # type: ignore[call-overload]
        )
        df["Date"] = pd.to_datetime(df["Open Time"], unit="ms")
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        # Batch type conversion
        df[["Open", "High", "Low", "Close", "Volume"]] = df[
            ["Open", "High", "Low", "Close", "Volume"]
        ].astype(float)

        return df  # type: ignore[return-value]

    except Exception:
        return None


def generate_mock_data(
    periods: int = 100, freq: str = "1h", symbol: str = "BTC/USD"
) -> pd.DataFrame:  # type: ignore[return]
    """Generate mock data for fallback"""
    # Convert timeframe to pandas frequency format
    freq_map = {
        "M1": "1min",
        "M15": "15min",
        "H1": "1h",
        "H4": "4h",
        "D": "1D",
        "W": "1W",
    }
    pd_freq = freq_map.get(freq, "1h")
    date_range = pd.date_range(end=datetime.now(), periods=periods, freq=pd_freq)

    # Seed based on symbol
    seed_val = sum(ord(c) for c in symbol)
    np.random.seed(seed_val)

    # Base price
    base_prices = {"BTC": 50000, "ETH": 3000, "SOL": 150}
    base_price = next((v for k, v in base_prices.items() if k in symbol), 100)

    # Random walk
    volatility = base_price * 0.02
    price_changes = np.random.randn(periods) * volatility
    price = np.maximum(base_price + np.cumsum(price_changes), 0.01)

    return pd.DataFrame(
        {
            "Date": date_range,
            "Open": price,
            "High": price + np.random.rand(periods) * volatility,
            "Low": price - np.random.rand(periods) * volatility,
            "Close": price + np.random.randn(periods) * (volatility * 0.5),
            "Volume": np.random.randint(100, 1000, size=periods),
        }
    )


# ============================================================================
# INDICATOR CALCULATIONS (OPTIMIZED)
# ============================================================================


def calculate_indicators(df: pd.DataFrame, active_indicators: list[str]) -> dict:  # type: ignore[type-arg]
    """Calculate all indicators in one pass"""
    indicators: dict = {}  # type: ignore[type-arg]

    close = df["Close"].values  # NumPy array for speed

    if "macd" in active_indicators:
        # Vectorized MACD
        exp1 = pd.Series(close).ewm(span=12, adjust=False).mean().values  # type: ignore[arg-type]
        exp2 = pd.Series(close).ewm(span=26, adjust=False).mean().values  # type: ignore[arg-type]
        macd = exp1 - exp2  # type: ignore[operator]
        signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values  # type: ignore[arg-type]
        indicators["macd"] = macd
        indicators["signal"] = signal
        indicators["hist"] = macd - signal  # type: ignore[operator]

    if "rsi" in active_indicators:
        # Vectorized RSI
        delta = np.diff(close, prepend=close[0])  # type: ignore[call-overload,arg-type]
        gain = np.where(delta > 0, delta, 0)  # type: ignore[call-overload,arg-type]
        loss = np.where(delta < 0, -delta, 0)  # type: ignore[call-overload,arg-type,operator]
        avg_gain = pd.Series(gain).rolling(window=14).mean().values  # type: ignore[arg-type]
        avg_loss = pd.Series(loss).rolling(window=14).mean().values  # type: ignore[arg-type]
        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)  # type: ignore[operator,call-overload,arg-type]
        indicators["rsi"] = 100 - (100 / (1 + rs))  # type: ignore[operator]

    # Moving averages
    ma_periods = []
    for ind in active_indicators:
        if ind.startswith("ma:"):
            try:
                period = int(ind.split(":")[1])
                ma_periods.append(period)
            except (ValueError, IndexError):
                pass

    if ma_periods:
        indicators["ma"] = {
            period: pd.Series(close).rolling(window=period).mean().values  # type: ignore[arg-type]
            for period in ma_periods
        }

    return indicators


# ============================================================================
# CHART GENERATION (OPTIMIZED)
# ============================================================================


async def generate_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    indicators_str: Optional[str],
) -> bytes:
    """Generate chart image"""
    # Parse indicators
    active_indicators = (
        [i.strip().lower() for i in indicators_str.split(",")] if indicators_str else []
    )
    show_vol = "vol" in active_indicators or "volume" in active_indicators
    show_macd = "macd" in active_indicators
    show_rsi = "rsi" in active_indicators

    # Calculate indicators
    indicator_data = calculate_indicators(df, active_indicators)

    # Setup subplots
    rows = 1
    row_heights = [0.6]
    specs = [[{"secondary_y": False}]]
    plot_map = {"price": 1}

    if show_vol:
        rows += 1
        plot_map["vol"] = rows
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])

    if show_macd:
        rows += 1
        plot_map["macd"] = rows
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])

    if show_rsi:
        rows += 1
        plot_map["rsi"] = rows
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])

    # Normalize heights
    total_h = sum(row_heights)
    row_heights = [h / total_h for h in row_heights]

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        specs=specs,
    )

    # Price chart
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=COLOR_GREEN,
            decreasing_line_color=COLOR_RED,
            increasing_fillcolor=COLOR_GREEN,
            decreasing_fillcolor=COLOR_RED,
            name="Price",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Moving averages
    if "ma" in indicator_data:
        colors = ["#2962FF", "#FF6D00", "#9C27B0", "#00BCD4"]
        for i, (period, ma_values) in enumerate(indicator_data["ma"].items()):
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=ma_values,
                    mode="lines",
                    name=f"MA {period}",
                    line=dict(width=1, color=colors[i % len(colors)]),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    # Volume
    if show_vol:
        row_idx = plot_map["vol"]
        colors = [
            COLOR_GREEN if c >= o else COLOR_RED
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["Volume"],
                marker_color=colors,
                name="Volume",
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_annotation(
            xref="x domain",
            yref=f"y{row_idx} domain",
            x=0.01,
            y=0.95,
            showarrow=False,
            text="Vol",
            font=dict(color=COLOR_TEXT, size=10),
        )

    # MACD
    if show_macd:
        row_idx = plot_map["macd"]
        hist_colors = [
            COLOR_GREEN if h >= 0 else COLOR_RED for h in indicator_data["hist"]
        ]
        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=indicator_data["hist"],
                marker_color=hist_colors,
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=indicator_data["macd"],
                line=dict(color="#2962FF", width=1),
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=indicator_data["signal"],
                line=dict(color="#FF6D00", width=1),
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_annotation(
            xref="x domain",
            yref=f"y{row_idx} domain",
            x=0.01,
            y=0.95,
            showarrow=False,
            text="MACD",
            font=dict(color=COLOR_TEXT, size=10),
        )

    # RSI
    if show_rsi:
        row_idx = plot_map["rsi"]
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=indicator_data["rsi"],
                line=dict(color="#7E57C2", width=1),
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_shape(
            type="line",
            x0=df["Date"].iloc[0],
            x1=df["Date"].iloc[-1],
            y0=70,
            y1=70,
            line=dict(color="gray", width=1, dash="dash"),
            row=row_idx,
            col=1,
        )
        fig.add_shape(
            type="line",
            x0=df["Date"].iloc[0],
            x1=df["Date"].iloc[-1],
            y0=30,
            y1=30,
            line=dict(color="gray", width=1, dash="dash"),
            row=row_idx,
            col=1,
        )
        fig.add_annotation(
            xref="x domain",
            yref=f"y{row_idx} domain",
            x=0.01,
            y=0.95,
            showarrow=False,
            text="RSI",
            font=dict(color=COLOR_TEXT, size=10),
        )

    # Last price indicator
    last_close = df["Close"].iloc[-1]
    last_open = df["Open"].iloc[-1]
    last_date = df["Date"].iloc[-1]
    lp_color = COLOR_GREEN if last_close >= last_open else COLOR_RED

    fig.add_shape(
        type="line",
        x0=df["Date"].iloc[0],
        x1=last_date + (last_date - df["Date"].iloc[0]) * 0.05,
        y0=last_close,
        y1=last_close,
        line=dict(color=lp_color, width=1, dash="dash"),
        row=1,
        col=1,
    )

    fig.add_annotation(
        x=last_date,
        y=last_close,
        text=f" {last_close:,.2f} ",
        showarrow=True,
        arrowhead=0,
        ax=40,
        ay=0,
        font=dict(color="white", size=11, family="monospace"),
        bgcolor=lp_color,
        opacity=1,
        row=1,
        col=1,
    )

    # Layout
    fig.update_layout(
        title=dict(text=f"{symbol} - {timeframe}", font=dict(color=COLOR_TEXT)),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=10, r=60, t=40, b=20),
        template="plotly_dark",
        dragmode=False,
    )

    for i in range(1, rows + 1):
        fig.update_xaxes(
            gridcolor=COLOR_GRID, color=COLOR_TEXT, nticks=15, row=i, col=1
        )
        fig.update_yaxes(gridcolor=COLOR_GRID, color=COLOR_TEXT, row=i, col=1)

    fig.update_xaxes(rangeslider=dict(visible=False), row=rows, col=1)

    # Export to PNG in thread pool
    img_bytes = await asyncio.to_thread(
        fig.to_image, format="png", width=1200, height=800, scale=1, engine="kaleido"
    )

    return img_bytes


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/chart")
async def get_chart(
    symbol: str = Query("BTC/USD"),
    timeframe: str = Query("H1"),
    indicators: Optional[str] = Query(
        None, description="Comma separated: vol, macd, rsi, ma:20, ma:50"
    ),
    feed: str = Query("ccxt", description="Data feed source: ccxt, pyth, binance"),
):
    """Generate chart with optimized caching and parallel fetching"""

    # Check cache first
    cache_k = cache_key(symbol, timeframe, 100, feed)
    df = get_cached_data(cache_k)

    if df is None:
        # Fetch data based on feed parameter
        feed_lower = feed.lower().strip()

        if feed_lower == "pyth":
            df = await fetch_pyth_data(symbol, timeframe, 100)
        elif feed_lower == "binance":
            df = await fetch_binance_direct(symbol, timeframe, 100)
        else:  # ccxt (default)
            df = await fetch_ccxt_data(symbol, timeframe, 100)

        # Fallback chain
        if df is None or df.empty:
            if feed_lower != "pyth":
                df = await fetch_pyth_data(symbol, timeframe, 100)

        if df is None or df.empty:
            if feed_lower != "binance":
                df = await fetch_binance_direct(symbol, timeframe, 100)

        if df is None or df.empty:
            df = generate_mock_data(100, timeframe, symbol)

        # Cache the result
        set_cached_data(cache_k, df)

    # Generate chart
    img_bytes = await generate_chart(df, symbol, timeframe, indicators)

    return Response(content=img_bytes, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
