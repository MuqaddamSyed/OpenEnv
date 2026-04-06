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

COPY --chown=user requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY --chown=user . .

USER user

# Default entry point running validation baseline inference upon Docker run
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
