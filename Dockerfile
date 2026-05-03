# Hugging Face Spaces — Docker SDK build for the CIFAR-10 Streamlit app.
# HF Spaces runs the container with port 7860 exposed; everything else
# is up to us.

FROM python:3.11-slim

# System-level deps that TensorFlow / Pillow occasionally need.
# libgl1 is required for OpenCV-style image ops; libglib2.0-0 for some
# Pillow plugins. Both are tiny.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces wants the container to run as a non-root user with UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install Python deps first so Docker can cache the layer when only
# app code changes.
COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy the rest of the build context (app code, models, figures).
COPY --chown=user . .

EXPOSE 7860

# Streamlit on HF Spaces:
#   - port 7860 (HF convention)
#   - bind 0.0.0.0 so the container is reachable
#   - headless=true skips opening a browser
#   - enableCORS=false avoids the embed warning page
CMD ["streamlit", "run", "app/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]
