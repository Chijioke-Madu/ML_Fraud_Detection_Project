# ---- Base Image ----
FROM python:3.10-slim

# ---- Working Directory ----
WORKDIR /app

# ---- Copy project files ----
COPY . /app

# ---- Install dependencies ----
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---- Expose port ----
EXPOSE 8000

# ---- Start FastAPI server ----
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
