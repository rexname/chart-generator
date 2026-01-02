from fastapi import FastAPI, Response, Query
from fastapi.concurrency import run_in_threadpool
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

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
    "LTC/USD": "6e3f2fa7185c7c00f68d6d56d10c55452f36b8567226d9c66e22f2f11187425f",
    "HYPE/USD": "0x4279e31cc369bbcc2faf022b382b080e32a8e689ff20fbc530d2a603eb6cd98b"
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
        # Reduced timeout for faster fallback
        r = requests.get(url, params=params, timeout=3)
        
        # If Pyth TV Shim fails (symbol mapping issues), try raw benchmarks or fallback
        if r.status_code != 200:
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

    except Exception:
        # Silent fallback to Binance
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
    
    # Try multiple public APIs for redundancy
    apis = [
        # Binance (Primary)
        ("https://api.binance.com/api/v3/klines", "binance"),
        # CoinGecko (Backup - limited to OHLC usually, but lets try a public endpoint if needed)
        # Actually CoinGecko is rate limited heavily.
        # Let's use Bybit or OKX public APIs as robust alternatives.
        
        # Bybit Public API (V5)
        ("https://api.bybit.com/v5/market/kline", "bybit"),
    ]
    
    for base_url, source in apis:
        try:
            if source == "binance":
                params = {"symbol": pair, "interval": interval, "limit": limit}
                r = requests.get(base_url, params=params, timeout=3)
                r.raise_for_status()
                data = r.json()
                # Binance Parsing
                df = pd.DataFrame(data, columns=[
                    "Open Time", "Open", "High", "Low", "Close", "Volume",
                    "Close Time", "Quote Asset Volume", "Number of Trades",
                    "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
                ])
                df['Date'] = pd.to_datetime(df['Open Time'], unit='ms')
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                return df
                
            elif source == "bybit":
                # Bybit Interval Map: 15, 60, 240, D, W
                bybit_tf = interval.replace("m", "").replace("h", "60").replace("460", "240").replace("1d", "D").replace("1w", "W")
                if bybit_tf == "160": bybit_tf = "60" # Fix 1h -> 60
                
                params = {"category": "spot", "symbol": pair, "interval": bybit_tf, "limit": limit}
                r = requests.get(base_url, params=params, timeout=3)
                r.raise_for_status()
                data = r.json()
                
                if data['retCode'] != 0: continue
                
                # Bybit: [startTime, open, high, low, close, volume, turnover]
                # Reverse needed as Bybit returns newest first usually? Check API.
                # Bybit returns list.
                
                raw_list = data['result']['list']
                # Bybit returns descending (newest first), we need ascending
                raw_list.reverse()
                
                df = pd.DataFrame(raw_list, columns=["startTime", "open", "high", "low", "close", "volume", "turnover"])
                df['Date'] = pd.to_datetime(pd.to_numeric(df['startTime']), unit='ms')
                df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                return df

        except Exception:
            continue
            
    # Fallback to Mock if ALL APIs fail (Network Unreachable)
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
    indicators: str | None = Query(None, description="Comma separated: vol, macd, rsi, ma:20, ma:50")
):
    # 1. Data (Try Pyth Benchmarks first, then Binance)
    df = await run_in_threadpool(get_pyth_benchmarks, symbol, timeframe, 100)
    
    # Data Cleaning: Filter out invalid dates (e.g., 1970 Epoch issues)
    # This prevents the chart from being squeezed if there's a stray 0 timestamp
    # if not df.empty and 'Date' in df.columns:
    #     # Ensure we only keep recent data (e.g. post 2020)
    #     df = df[df['Date'] > '2020-01-01']
    #     df = df.sort_values('Date').reset_index(drop=True)
    
    if df.empty:
        # Fallback to Mock if cleaning resulted in empty data
        df = generate_mock_data(100, timeframe, symbol)

    # Determine X-Axis Range explicitly to prevent auto-scaling to 1970
    # min_date = df['Date'].min()
    # max_date = df['Date'].max()
    # Add a small buffer to the right for the price label
    # max_date_buffer = max_date + (max_date - min_date) * 0.05

    # Parse indicators
    active_indicators = [i.strip().lower() for i in indicators.split(',')] if indicators else []
    show_vol = 'vol' in active_indicators or 'volume' in active_indicators
    show_macd = 'macd' in active_indicators
    show_rsi = 'rsi' in active_indicators
    
    # Extract MAs (e.g., "ma:20", "ma:50")
    ma_periods = []
    for ind in active_indicators:
        if ind.startswith('ma:'):
            try:
                period = int(ind.split(':')[1])
                ma_periods.append(period)
            except:
                pass

    # Calculate Indicators if needed
    if show_macd:
        # Standard MACD (12, 26, 9)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['hist'] = df['macd'] - df['signal']
    
    if show_rsi:
        # Standard RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

    # 2. Setup Subplots
    # Row 1: Price (Always)
    # Row 2: Volume (Optional)
    # Row 3: MACD/RSI (Optional - stacking them if multiple selected)
    
    rows = 1
    row_heights = [0.6] # Base height for Price
    specs = [[{"secondary_y": False}]]
    
    # Logic to stack subplots
    plot_map = {'price': 1}
    
    if show_vol:
        rows += 1
        plot_map['vol'] = rows
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])
        
    if show_macd:
        rows += 1
        plot_map['macd'] = rows
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])
        
    if show_rsi:
        rows += 1
        plot_map['rsi'] = rows
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])

    # Normalize heights
    total_h = sum(row_heights)
    row_heights = [h/total_h for h in row_heights]

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=rows, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        row_heights=row_heights,
        specs=specs
    )

    # --- ROW 1: PRICE ---
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color='#089981', 
        decreasing_line_color='#F23645',
        increasing_fillcolor='#089981',
        decreasing_fillcolor='#F23645',
        name="Price",
        showlegend=False
    ), row=1, col=1)

    # Add MA to Price (Parsed from indicators)
    for period in ma_periods:
        ma_val = df['Close'].rolling(window=period).mean()
        fig.add_trace(go.Scatter(
            x=df['Date'], y=ma_val, 
            mode='lines', 
            name=f'MA {period}',
            line=dict(width=1),
            showlegend=False
        ), row=1, col=1)

    # --- OPTIONAL ROWS ---
    
    # Volume
    if show_vol:
        row_idx = plot_map['vol']
        colors = ['#089981' if c >= o else '#F23645' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(
            x=df['Date'], y=df['Volume'],
            marker_color=colors,
            name="Volume",
            showlegend=False
        ), row=row_idx, col=1)
        # Add Volume label
        fig.add_annotation(xref="x domain", yref=f"y{row_idx} domain", x=0.01, y=0.95, showarrow=False, text="Vol", font=dict(color='#d1d4dc', size=10))

    # MACD
    if show_macd:
        row_idx = plot_map['macd']
        # Histogram
        hist_colors = ['#089981' if h >= 0 else '#F23645' for h in df['hist']]
        fig.add_trace(go.Bar(
            x=df['Date'], y=df['hist'],
            marker_color=hist_colors,
            name="Histogram",
            showlegend=False
        ), row=row_idx, col=1)
        # MACD Line
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['macd'],
            line=dict(color='#2962FF', width=1),
            name="MACD",
            showlegend=False
        ), row=row_idx, col=1)
        # Signal Line
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['signal'],
            line=dict(color='#FF6D00', width=1),
            name="Signal",
            showlegend=False
        ), row=row_idx, col=1)
        fig.add_annotation(xref="x domain", yref=f"y{row_idx} domain", x=0.01, y=0.95, showarrow=False, text="MACD", font=dict(color='#d1d4dc', size=10))

    # RSI
    if show_rsi:
        row_idx = plot_map['rsi']
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['rsi'],
            line=dict(color='#7E57C2', width=1),
            name="RSI",
            showlegend=False
        ), row=row_idx, col=1)
        # RSI Bands (70/30)
        fig.add_shape(type="line", x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=70, y1=70, line=dict(color="gray", width=1, dash="dash"), row=row_idx, col=1)
        fig.add_shape(type="line", x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=30, y1=30, line=dict(color="gray", width=1, dash="dash"), row=row_idx, col=1)
        fig.add_annotation(xref="x domain", yref=f"y{row_idx} domain", x=0.01, y=0.95, showarrow=False, text="RSI", font=dict(color='#d1d4dc', size=10))

    # --- LAST PRICE INDICATOR ---
    last_close = df['Close'].iloc[-1]
    last_open = df['Open'].iloc[-1]
    last_date = df['Date'].iloc[-1]
    lp_color = '#089981' if last_close >= last_open else '#F23645'
    
    # Horizontal Line (Full Width like TradingView)
    fig.add_shape(
        type="line",
        x0=df['Date'].iloc[0], 
        x1=last_date + (last_date - df['Date'].iloc[0]) * 0.05, # Extend slightly to right
        y0=last_close, 
        y1=last_close,
        line=dict(color=lp_color, width=1, dash="dash"),
        row=1, col=1
    )
    
    # Price Label (Badge)
    fig.add_annotation(
        x=last_date, # Anchor to the last date
        y=last_close,
        text=f" {last_close:,.2f} ",
        showarrow=True,
        arrowhead=0,
        ax=40, # Shift right by 40px
        ay=0,
        font=dict(color='white', size=11, family="monospace"),
        bgcolor=lp_color,
        opacity=1,
        row=1, col=1
    )

    # 4. TradingView Dark Layout
    fig.update_layout(
        title=dict(text=f"{symbol} - {timeframe}", font=dict(color='#d1d4dc')),
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        margin=dict(l=10, r=60, t=40, b=20),
        template="plotly_dark",
        dragmode=False
    )
    
    # Update Grid for all subplots
    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor='#2a2e39', color='#d1d4dc', nticks=15, row=i, col=1)
        fig.update_yaxes(gridcolor='#2a2e39', color='#d1d4dc', row=i, col=1)
    
    # Hide rangeslider on bottom-most plot
    fig.update_xaxes(rangeslider=dict(visible=False), row=rows, col=1)


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
