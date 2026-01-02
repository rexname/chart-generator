from fastapi import FastAPI, Response, Query
from fastapi.concurrency import run_in_threadpool
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta

app = FastAPI()

# Cache for Pyth Price IDs (Symbol -> ID)
PYTH_IDS = {
    "BTC/USD": "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH/USD": "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "SOL/USD": "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
    "BNB/USD": "2f95862b0455a0920eb43c5d401a8801f16f31f9175d2752187064d7c00650d3",
    "DOGE/USD": "dcef50dd0a4cd2dcc17e45df1676dcb336a11a61c69df7a0299b0150c672d25c",
    "ADA/USD": "2a01deaec9e51a579277b34b1223a9c8479b500127e9f3b31571bf1396d00f6b",
    "XRP/USD": "ec5d399846a9209f3fe5881d70aae9268c94339ff9817e8d18ff19fa05eea1c8",
    "MATIC/USD": "5de33a9112c2b700b8d30b8a3402c103575938be665eb29d95502a9d83e15b01",
    "LINK/USD": "8ac0c70fff57e9a0ffd5ee5638ebc728027d99991eb87af8147d32f4b5f48035",
    "LTC/USD": "6e3f2fa7185c7c00f68d6d56d10c55452f36b8567226d9c66e22f2f11187425f"
}

def get_pyth_benchmarks(symbol="BTC/USD", timeframe="H1", limit=100):
    """
    Fetch Historical Data from Pyth Benchmarks API (TradingView History compatible)
    """
    price_id = PYTH_IDS.get(symbol.upper())
    if not price_id:
        return get_binance_data(symbol, timeframe, limit)

    # Mapping timeframe to resolution (seconds)
    # Pyth Benchmarks API uses resolution in seconds
    # Supported: 1, 60, 3600, 86400
    tf_seconds = {
        "M1": 60,
        "M15": 900, # Pyth might fallback to 60s and we aggregate, or nearest
        "H1": 3600,
        "H4": 14400,
        "D": 86400,
        "W": 604800
    }
    resolution = tf_seconds.get(timeframe.upper(), 3600)
    
    # Calculate start/end time
    now = int(datetime.now().timestamp())
    start_time = now - (resolution * limit)
    
    # Pyth Benchmarks Endpoint (TradingView UDF compatible)
    # https://benchmarks.pyth.network/v1/shims/tradingview/history
    url = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
    params = {
        "symbol": symbol, # Pyth TV shim uses symbol name directly usually or we might need ID
        "resolution": str(resolution // 60) if resolution >= 60 else "1", # TV format (minutes) or D
        "from": start_time,
        "to": now
    }
    
    # Correction for resolution format for Pyth TV Shim
    if resolution == 86400: params['resolution'] = '1D'
    elif resolution == 604800: params['resolution'] = '1W'
    else: params['resolution'] = str(resolution // 60)

    try:
        r = requests.get(url, params=params, timeout=10)
        
        # If Pyth TV Shim fails (symbol mapping issues), try raw benchmarks or fallback
        if r.status_code != 200:
            # Fallback to direct ID usage if symbol name fails
            # Actually Pyth Benchmarks raw API is /v1/updates... 
            # Let's stick to the shim if possible, or fallback to Binance
            print(f"Pyth Shim failed: {r.status_code}, falling back to Binance")
            return get_binance_data(symbol, timeframe, limit)
            
        data = r.json()
        
        if data.get('s') != 'ok':
            return get_binance_data(symbol, timeframe, limit)
            
        # Parse TradingView UDF response
        # {s: "ok", t: [time], o: [open], h: [high], l: [low], c: [close], v: [volume]}
        df = pd.DataFrame({
            'Date': pd.to_datetime(data['t'], unit='s'),
            'Open': data['o'],
            'High': data['h'],
            'Low': data['l'],
            'Close': data['c'],
            'Volume': data.get('v', [0]*len(data['t']))
        })
        
        return df

    except Exception as e:
        print(f"Error fetching Pyth data: {e}")
        return get_binance_data(symbol, timeframe, limit)

def get_binance_data(symbol="BTC/USD", timeframe="H1", limit=100):
    # Convert symbol to Binance format (BTC/USD -> BTCUSDT)
    pair = symbol.replace("/", "").replace("USD", "USDT")
    
    # Map timeframe
    tf_map = {
        "M15": "15m",
        "H1": "1h",
        "H4": "4h",
        "D": "1d",
        "W": "1w"
    }
    interval = tf_map.get(timeframe.upper(), "1h")
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": pair,
        "interval": interval,
        "limit": limit
    }
    
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        
        # [Open Time, Open, High, Low, Close, Volume, ...]
        df = pd.DataFrame(data, columns=[
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "Number of Trades",
            "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
        ])
        
        df['Date'] = pd.to_datetime(df['Open Time'], unit='ms')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
            
        return df
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        # Fallback to Mock if Binance fails or symbol invalid
        return generate_mock_data(limit, '1h', symbol)

def generate_mock_data(periods=100, freq='1h', symbol='BTC/USD'):
    date_range = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
    
    # Generate seed based on symbol string to make chart consistent per symbol
    # but different across symbols
    seed_val = sum(ord(c) for c in symbol) 
    np.random.seed(seed_val)
    
    # Base price variation
    base_price = 100 if 'USD' in symbol else 0.001
    if 'BTC' in symbol: base_price = 50000
    if 'SOL' in symbol: base_price = 150
    if 'ETH' in symbol: base_price = 3000
    
    # Random walk
    volatility = base_price * 0.02
    price_changes = np.random.randn(periods) * volatility
    price = base_price + np.cumsum(price_changes)
    
    # Ensure no negative prices
    price = np.maximum(price, 0.01)
    
    data = {
        'Date': date_range,
        'Open': price,
        'High': price + np.random.rand(periods) * volatility,
        'Low': price - np.random.rand(periods) * volatility,
        'Close': price + np.random.randn(periods) * (volatility * 0.5),
        'Volume': np.random.randint(100, 1000, size=periods)
    }
    df = pd.DataFrame(data)
    return df

@app.get("/api/chart")
async def get_chart(
    symbol: str = Query("BTC/USD"),
    timeframe: str = Query("H1"),
    ma: str | None = Query(None)
):
    # 1. Data (Try Pyth Benchmarks first, then Binance)
    df = await run_in_threadpool(get_pyth_benchmarks, symbol, timeframe, 100)

    # 2. Create Figure
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color='#089981', 
        decreasing_line_color='#F23645',
        increasing_fillcolor='#089981',
        decreasing_fillcolor='#F23645'
    )])

    # 3. Add MA
    if ma:
        for m in ma.split(','):
            period = int(m)
            ma_val = df['Close'].rolling(window=period).mean()
            fig.add_trace(go.Scatter(
                x=df['Date'], y=ma_val, 
                mode='lines', 
                name=f'MA {period}',
                line=dict(width=1)
            ))

    # 4. TradingView Dark Layout
    fig.update_layout(
        title=dict(text=f"{symbol} - {timeframe}", font=dict(color='#d1d4dc')),
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        xaxis=dict(
            gridcolor='#2a2e39', 
            color='#d1d4dc',
            rangeslider=dict(visible=False),
            nticks=20  # Optimize rendering ticks
        ),
        yaxis=dict(
            gridcolor='#2a2e39', 
            color='#d1d4dc'
        ),
        margin=dict(l=10, r=40, t=40, b=20), # Tight margin
        showlegend=False,
        # Performance optimizations
        template="plotly_dark"
    )

    # 5. Export to PNG via Kaleido (Async Wrapper)
    # run_in_threadpool prevents blocking the main event loop
    img_bytes = await run_in_threadpool(
        fig.to_image, 
        format="png", 
        width=1200, 
        height=800, 
        scale=1,
        engine="kaleido"
    )
    
    return Response(content=img_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
