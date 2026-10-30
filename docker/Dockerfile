# MRI Brain Tumor — Streamlit inference container
FROM python:3.12-slim

WORKDIR /app

# system deps for OpenCV + TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# python deps (inference only — small)
COPY requirements/requirements_inference.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements_inference.txt

# app + models + reports
COPY app/ ./app/
COPY models/ ./models/
COPY reports/ ./reports/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]