# ===== base image =====
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (add only what you need)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# install deps
COPY requirements.txt .
# Copy app code
COPY . .

RUN pip install -r requirements.txt



# Expose app port
EXPOSE 8000


# Select which app to run at build (or override at run)
# apps.belgenavi.main:app  OR  apps.trialmatch.main:app
ARG MODULE_PATH=trial_matcher_fastapi:app
ENV MODULE_PATH=${MODULE_PATH}

# Start Uvicorn
CMD ["uvicorn", "trial_matcher_fastapi:app", "--reload","--host", "127.0.0.1", "--port", "8000"]