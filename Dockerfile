# 베이스 이미지: CUDA 12.1 및 cuDNN 8 포함
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    wget \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 설치
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda clean -afy

# 작업 디렉토리 설정
WORKDIR /workspace/motion-diffusion-model

# MDM 저장소 복제
RUN git clone https://huggingface.co/datasets/NamYeongCho/mdm-gentest . && \
    git lfs install

# Conda 환경 생성 및 활성화
COPY environment.yml .
RUN conda env create -f environment.yml && \
    echo "conda activate mdm" >> ~/.bashrc

# 환경 활성화 후 추가 패키지 설치
SHELL ["conda", "run", "--no-capture-output", "-n", "mdm", "/bin/bash", "-c"]
RUN python -m spacy download en_core_web_sm && \
    pip install git+https://github.com/openai/CLIP.git

# 의존성 복제 및 설정
RUN git clone https://huggingface.co/datasets/NamYeongCho/mdm-dependency /workspace/motion-diffusion-model/mdm-dependency && \
    mkdir -p /workspace/motion-diffusion-model/body_models && \
    mv /workspace/motion-diffusion-model/mdm-dependency/smpl.zip /workspace/motion-diffusion-model/body_models/ && \
    unzip /workspace/motion-diffusion-model/body_models/smpl.zip -d /workspace/motion-diffusion-model/body_models/

# 모델 다운로드 및 save 디렉토리로 이동
RUN mkdir -p /workspace/motion-diffusion-model/save && \
    git clone https://huggingface.co/datasets/NamYeongCho/mdm-test-model /workspace/mdm-test-model && \
    mv /workspace/mdm-test-model/* /workspace/motion-diffusion-model/save/

# HumanML3D 텍스트 압축 해제
RUN unzip /workspace/motion-diffusion-model/datasets/HumanML3D/text.zip -d /workspace/motion-diffusion-model/datasets/HumanML3D/

# 기본 명령어 설정
CMD [ "conda", "run", "--no-capture-output", "-n", "mdm", "python", "mdm_generate.py" ]
