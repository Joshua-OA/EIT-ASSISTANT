FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Create a 512MB swap file to help with memory pressure
RUN fallocate -l 512M /swapfile && chmod 600 /swapfile && mkswap /swapfile

CMD ["sh", "-c", "swapon /swapfile 2>/dev/null; streamlit run main.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true"]
