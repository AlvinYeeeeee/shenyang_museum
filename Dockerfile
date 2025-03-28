FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN mkdir /workspace/Models
WORKDIR /workspace/

# Install llama-index
RUN pip install llama-index

# Install MeloTTS
RUN git clone https://github.com/myshell-ai/MeloTTS.git && cd MeloTTS && pip install -e . && python -m unidic download

# Install huggingface
RUN pip install transformers

# Install embedding
RUN git clone https://huggingface.co/BAAI/bge-large-zh-v1.5

RUN mv bge-large-zh-v1.5 /workspace/Models

# Install speech to text

RUN cd /workspace/Models && git clone https://huggingface.co/openai/whisper-large-v3-turbo

# Install ollama

RUN curl -fsSL https://ollama.com/install.sh | sh

# RUN ollama start && ollama pull qwen2.5:0.5b

COPY requirements.txt /workspace/

RUN pip install --no-cache-dir -r requirements.txt

RUN pip uninstall -y torchvision && pip install torchvision llama-index-llms-yi

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs

RUN cd /workspace/Models/bge-large-zh-v1.5 && git lfs install && git lfs pull

RUN cd /workspace/Models/whisper-large-v3-turbo && git lfs install && git lfs pull

RUN pip install 'accelerate>=0.26.0'