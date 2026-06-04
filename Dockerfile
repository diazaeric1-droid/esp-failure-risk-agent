FROM python:3.11-slim

# libgomp1 is needed by XGBoost; harmless elsewhere.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=user . ./

EXPOSE 7860
CMD ["streamlit", "run", "demo/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
