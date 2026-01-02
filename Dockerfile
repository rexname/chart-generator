FROM python:3.11-slim

WORKDIR /app
# Install minimal dependencies for Headless Chrome (Kaleido)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libatk-bridge2.0-0 libgtk-3-0 libasound2 libx11-xcb1 \
    libxcomposite1 libxcursor1 libxdamage1 libxi6 libxtst6 libcups2 libxss1 libxrandr2 \
    libxfixes3 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*
# Install Python deps & Chrome engine
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && kaleido_get_chrome
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
