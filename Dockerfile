# CUDA base image with development tools (compiler, profiling support)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install essentials + Nsight tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget curl vim \
    && rm -rf /var/lib/apt/lists/*

# 2️⃣ Upgrade pip/setuptools/wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# 3️⃣ Install CUDA-compatible PyTorch + audio stack
RUN pip install --no-cache-dir --root-user-action=ignore \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Symlink python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch (CUDA 12.1 wheels)
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install common ML + debugging tools
# 4️⃣ Install ML/STT + audio + compatibility deps
RUN pip install --no-cache-dir --root-user-action=ignore \
    transformers \
    datasets \
    peft \
    accelerate \
    torchaudio \
    bitsandbytes \
    openai-whisper \
    librosa \
    soundfile \
    huggingface_hub \
    requests==2.28.2 \
    tqdm==4.65.0 \
    openxlab==0.1.2

# 5️⃣ Add JupyterLab and data science stack
RUN pip install --no-cache-dir --root-user-action=ignore \
    pandas==1.5.3 \
    numpy==1.24.4 \
    matplotlib==3.7.2 \
    packaging>=22.0 \
    pytz==2023.3 \
    rich==13.4.2 \
    networkx==3.1 \
    jupyterlab \
    notebook \
    ipykernel

# 6️⃣ Install Explanation Tools (LIME + SHAP + Dependents)
RUN pip install --no-cache-dir --root-user-action=ignore \
    scipy==1.11.4  \
    scikit-learn==1.3.2

WORKDIR /workspace

# 7️⃣ Copy example script and sample
COPY train_cnn.py train_rnn.py  /workspace/

EXPOSE 8888

CMD ["/bin/bash"]
