# Lambda Cloud

Lambda's on-demand instances have no launch CLI in this repo; use the dashboard.

1. Launch a 1x A100 (40/80 GB) or 1x H100 PCIe instance with the default Lambda Stack image.
   Add your SSH key. Attach a persistent filesystem if you want the raw datasets to survive
   instance termination (AOT slices and Anti-UAV are hundreds of GB).
2. Firewall: in the dashboard, restrict port 22 to your own IP.
3. Copy the code and run the same container as on AWS/GCP:

```bash
rsync -az --exclude data --exclude results ./ ubuntu@<ip>:~/sim-core/
ssh ubuntu@<ip>
sudo mkdir -p /data && sudo chown ubuntu /data      # or use the persistent filesystem mount
cd sim-core && docker build -f training/Dockerfile -t dronesim-train .
tmux new -s train 'docker run --gpus all --shm-size=32g -v /data:/data dronesim-train training/run_pipeline.sh'
```

If Docker with GPU support is not present on the image, install the NVIDIA Container Toolkit
or run without Docker: `pip install -r training/requirements.txt && bash training/run_pipeline.sh`.

4. Copy `/data/models` back and **terminate the instance**: Lambda bills until you do.
