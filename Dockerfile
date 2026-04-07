FROM python:3.10-slim

# Hugging Face Space requirements and general best practices
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user specifying UID 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies first (better layer caching)
COPY --chown=user requirements.txt pyproject.toml ./

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy codebase
COPY --chown=user . .

USER user

# Default entry point
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
