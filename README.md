# Chart Generator API

High-performance, headless financial chart generator API built with **FastAPI** and **Plotly + Kaleido**.  
Generates "TradingView-like" charts server-side and returns them as PNG images.

## Features

*   **⚡ Fast & Async**: Built on FastAPI with async threadpool for concurrent rendering.
*   **🎨 TradingView Style**: Dark mode, professional candle colors, grid systems.
*   **📊 Multi-Source Data Feeds**:
    *   **Primary (Default)**: CCXT - Unified access to 100+ exchanges (Binance, Bybit, OKX, etc).
    *   **Fallback 1**: Pyth Network Benchmarks (Real Historical Data).
    *   **Fallback 2**: Binance Public API (Direct).
    *   **Fail-safe**: Mathematical Random Walk (if all APIs down).
*   **📈 Technical Indicators**: Volume, MACD, RSI, Moving Averages (MA:20, MA:50, etc).
*   **🛠 Configurable**: Support for multiple timeframes (M1, M15, H1, H4, D, W) and 100+ symbols.
*   **🚀 Headless**: No browser required (uses Kaleido C++ engine).

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/rexname/chart-generator.git
    cd chart-generator
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Verify Installation**
    ```bash
    python -c "import ccxt, plotly, kaleido; print('All dependencies OK')"
    ```

## Usage

### Run Server
```bash
# Development
python main.py

# Production (Multi-worker)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-03T12:00:00.000000"
}
```

#### `GET /api/chart`

Generates a chart image.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | string | `BTC/USD` | Asset symbol (e.g., `ETH/USD`, `SOL/USD`, `BNB/USD`). |
| `timeframe` | string | `H1` | Timeframe: `M1`, `M15`, `H1`, `H4`, `D`, `W`. |
| `indicators` | string | `None` | Comma-separated indicators: `vol`, `macd`, `rsi`, `ma:20`, `ma:50`, etc. |
| `feed` | string | `ccxt` | Data feed source: `ccxt` (default), `pyth`, `binance`. |

**Examples:**

*   **Standard Bitcoin Chart (H1, CCXT feed)**
    ```
    http://localhost:8000/api/chart?symbol=BTC/USD&timeframe=H1
    ```

*   **Ethereum with Full Indicators**
    ```
    http://localhost:8000/api/chart?symbol=ETH/USD&timeframe=H4&indicators=vol,macd,rsi,ma:20,ma:50
    ```

*   **Solana Daily with Pyth Feed**
    ```
    http://localhost:8000/api/chart?symbol=SOL/USD&timeframe=D&indicators=vol,rsi&feed=pyth
    ```

*   **BNB 15-Minute with Binance Direct**
    ```
    http://localhost:8000/api/chart?symbol=BNB/USD&timeframe=M15&feed=binance
    ```

### Data Feed Options

#### 1. CCXT (Default - Recommended)
```
?feed=ccxt
```
- Unified API for 100+ exchanges
- Automatic fallback: Binance → Bybit → OKX
- Best reliability and coverage
- Handles both `/USD` and `/USDT` pairs

#### 2. Pyth Network
```
?feed=pyth
```
- Real-time oracle data
- Lower latency for supported pairs
- Falls back to Binance if symbol not found

#### 3. Binance Direct
```
?feed=binance
```
- Direct Binance API access
- Fast for USDT pairs
- Limited to Binance-listed assets

### Supported Indicators

| Indicator | Parameter | Description |
| :--- | :--- | :--- |
| **Volume** | `vol` or `volume` | Trading volume bars below price chart |
| **MACD** | `macd` | MACD histogram + signal lines (12,26,9) |
| **RSI** | `rsi` | Relative Strength Index (14) with 30/70 bands |
| **Moving Average** | `ma:20`, `ma:50`, etc | Simple Moving Average with custom period |

**Example with all indicators:**
```
?indicators=vol,macd,rsi,ma:20,ma:50,ma:200
```

## Testing

Run the comprehensive test suite:

```bash
# Start server first
python main.py &

# Run tests
python test_api.py
```

Test coverage includes:
- All 3 data feed types
- Multiple symbols (BTC, ETH, SOL, BNB)
- All timeframes (M15, H1, H4, D)
- Indicator combinations
- Fallback mechanisms

## Architecture

```
┌─────────────┐
│   Request   │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────┐
│  FastAPI Route Handler              │
│  - Parse params (symbol, TF, feed)  │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│  Data Feed Layer (Async)            │
│  ┌─────────────┐                    │
│  │ CCXT (pri)  │ → Binance/Bybit/OKX│
│  └─────┬───────┘                    │
│        │ (fallback)                 │
│  ┌─────v───────┐                    │
│  │ Pyth        │ → Benchmarks API   │
│  └─────┬───────┘                    │
│        │ (fallback)                 │
│  ┌─────v───────┐                    │
│  │ Mock Data   │ → Random Walk      │
│  └─────────────┘                    │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│  Indicator Calculation              │
│  - MA, MACD, RSI, Volume            │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│  Plotly Chart Rendering             │
│  - Candlesticks + Subplots          │
│  - TradingView Dark Theme           │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│  Kaleido Export (Threadpool)        │
│  - PNG conversion (1200x800)        │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────┐
│ PNG Response│
└─────────────┘
```

## Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```bash
docker build -t chart-gen .
docker run -p 8000:8000 chart-gen
```

### Docker Compose

```bash
docker-compose up -d
```

### VPS Deployment

Requirements:
- Ubuntu 20.04+ / Debian 11+
- Python 3.10+
- 512MB RAM minimum (2GB recommended for production)

```bash
# Install dependencies
apt update && apt install python3-pip python3-venv -y

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with systemd (production)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Performance

- **Response Time**: 10-20s per chart (includes data fetch + render)
- **Throughput**: ~10-20 concurrent requests (4 workers)
- **Memory**: ~100MB per worker
- **Cache**: Consider adding Redis for OHLCV data caching

## Troubleshooting

### Network/API Issues

If CCXT fails to connect:
1. Check internet connectivity
2. Verify exchange APIs are not blocked
3. System will automatically fallback to Pyth → Mock data

### Kaleido Issues

If PNG rendering fails:
```bash
pip install --upgrade kaleido
```

### Symbol Not Found

Try different feed sources:
- `?feed=ccxt` - Best coverage
- `?feed=pyth` - Limited to Pyth-supported pairs
- Use `/USDT` instead of `/USD` for broader support

## License

MIT

## Support

For issues or feature requests, open an issue on GitHub.