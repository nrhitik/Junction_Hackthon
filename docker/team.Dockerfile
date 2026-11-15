# Team submission v2: classical pipeline + team safety + rule-based instruction reader (no LLM, CPU only).
# One image for all tasks: the runner's reset carries task + instruction; the policy switches on task.
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir numpy==2.2.6 scipy==1.15.2 opencv-python-headless==4.10.0.84 pyyaml \
        msgpack "websockets>=12" pin==3.3.1

WORKDIR /app
COPY app/ /app/
# real lid (task 3) + real head-camera colour profile
ENV AALTO_LID_LAYOUT=real AALTO_LID_HEIGHT=floor AALTO_PROFILE=real
# load-time check at build: config, URDF + collision meshes (team safety), calibration, parser, shape model
RUN python3 -c "from aaltonomous.baselines.classical.policy_server import ClassicalServerPolicy; ClassicalServerPolicy(); from aaltonomous.baselines.classical.rules_parser import parse_rules; assert parse_rules('Put the yellow objects in the left bin, the purple objects in the center bin and the orange objects in the right bin.'); import aaltonomous.brain.shapes; print('policy loads')"
EXPOSE 8000
CMD ["python3", "-m", "aaltonomous.baselines.classical.policy_server"]
