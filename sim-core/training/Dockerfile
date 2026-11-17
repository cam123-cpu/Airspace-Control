# Training image for the dronesim camera detector.
#   docker build -f training/Dockerfile -t dronesim-train .          (run from sim-core/)
#   docker run --gpus all --shm-size=16g -v /data:/data dronesim-train training/run_pipeline.sh
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged user: the container never needs root to train.
RUN useradd -m -u 1000 trainer && mkdir -p /data && chown trainer:trainer /data
WORKDIR /app
COPY --chown=trainer:trainer training/requirements.txt training/requirements.txt
RUN pip install -r training/requirements.txt
COPY --chown=trainer:trainer . /app
USER trainer

# Fail the build if the pipeline's own tests do not pass in this environment.
RUN python -m pytest -q training/tests

ENTRYPOINT ["bash"]
CMD ["training/run_pipeline.sh"]
