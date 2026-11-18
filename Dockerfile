FROM solo33-colorcart:1.0
USER root
RUN rm -rf /app && mkdir -p /app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV TRACE_DATA_DIR=/data PORT=18936 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
HEALTHCHECK --interval=20s --timeout=3s --start-period=5s --retries=3 CMD python3 -B -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18936/tracedesk/state')"
EXPOSE 18936
CMD ["python3", "-B", "app.py"]
