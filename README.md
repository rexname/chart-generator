# Pyth Chart Generator API

High-performance, headless financial chart generator API built with **FastAPI** and **Plotly + Kaleido**.  
Generates "TradingView-like" charts server-side and returns them as PNG images.

## Features

*   **⚡ Fast & Async**: Built on FastAPI with async threadpool for concurrent rendering.
*   **🎨 TradingView Style**: Dark mode, professional candle colors, grid systems.
*   **📊 Multi-Source Data**:
    *   **Primary**: Pyth Network Benchmarks (Real Historical Data).
    *   **Fallback**: Binance Public API.
    *   **Fail-safe**: Mathematical Random Walk (if all APIs down).
*   **🛠 Configurable**: Support for Timeframes (M15, H1, H4, D), Moving Averages (MA), and Symbols.
*   **🚀 Headless**: No browser required (uses Kaleido C++ engine).

## Installation

1.  **Clone the repository**
    ```bash
    git clone <repo_url>
    cd chart-gen
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Chrome/Chromium Engine (for Kaleido)**
    ```bash
    kaleido_get_chrome
    ```

## Usage

### Run Server
```bash
# Development
uvicorn main:app --reload

# Production (Multi-worker)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

#### `GET /api/chart`

Generates a chart image.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | string | `BTC/USD` | Asset symbol (e.g., `ETH/USD`, `SOL/USD`). |
| `timeframe` | string | `H1` | Timeframe: `M15`, `H1`, `H4`, `D`, `W`. |
| `ma` | string | `None` | Comma-separated Moving Averages (e.g., `20,50`). |

**Examples:**

*   **Standard Bitcoin Chart (H1)**
    `http://localhost:8000/api/chart?symbol=BTC/USD`

*   **Solana Daily with MA 20 & 50**
    `http://localhost:8000/api/chart?symbol=SOL/USD&timeframe=D&ma=20,50`

*   **Ethereum 15-Minute**
    `http://localhost:8000/api/chart?symbol=ETH/USD&timeframe=M15`

## Architecture

1.  **Request Handling**: FastAPI receives the request.
2.  **Data Fetching**:
    *   Attempts to fetch OHLCV from **Pyth Benchmarks**.
    *   If unavailable, falls back to **Binance API**.
    *   If offline, generates **Mock Data** (seeded by symbol name).
3.  **Rendering**: Plotly creates a figure with a custom Dark Theme.
4.  **Export**: Kaleido engine converts the figure to PNG bytes (in a threadpool).
5.  **Response**: Returns `image/png` directly.

## Deployment

Deployable on any VPS (Ubuntu/Debian recommended).
Requires ~512MB RAM for comfortable concurrency.

```bash
# Dockerfile example (Optional)
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN kaleido_get_chrome
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```
