FROM python:3.9-slim

# Create non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code and model
COPY --chown=user app.py .
COPY --chown=user inference_engine.py .
COPY --chown=user config.yaml .
COPY --chown=user model/best.pth model/best.pth

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
