# Use standard Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and tests
COPY src/ ./src/
COPY tests/ ./tests/
COPY README.md .

# Create directories for models and data
RUN mkdir -p data models

# Generate synthetic dataset and train models before startup
RUN python -c "from src.sustainable_campus.ai.dataset_generator import generate_campus_dataset; generate_campus_dataset('data/campus_energy_demand.csv')" && \
    python -c "from src.sustainable_campus.ai.demand_prediction import train_and_save_models; train_and_save_models('data/campus_energy_demand.csv', 'models/')"

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run Streamlit
CMD ["streamlit", "run", "src/sustainable_campus/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
