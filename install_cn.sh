set -xe

echo "export HF_ENDPOINT='https://hf-mirror.com'" >> ~/.bashrc && source ~/.bashrc

apt install git vim curl

# Install llama-index
pip uninstall -y torchvision && pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --no-cache-dir -r requirements.txt

# Install git lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs

# mkdir /workspace/Models


# Install embedding
cd /workspace/Models && git clone https://hf-mirror.com/BAAI/bge-large-zh-v1.5


# Install speech to text
cd /workspace/Models && git clone https://hf-mirror.com/openai/whisper-large-v3-turbo


# Install MeloTTS
git clone https://gitee.com/xiachunwei/MeloTTS.git && cd MeloTTS && pip install -e . && python -m unidic download


# Install ollama
curl -fsSL https://ollama.com/install.sh | sh


# cd /workspace/Models/bge-large-zh-v1.5 && git lfs install && git lfs pull

# cd /workspace/Models/whisper-large-v3-turbo && git lfs install && git lfs pull
pip install https://download.pytorch.org/whl/cpu/torchaudio-2.5.1%2Bcpu-cp310-cp310-linux_x86_64.whl#sha256=c64af43548713e5abc3e9e3b5f33a2a47c57122ee953e0b0cb102f7855f8d017


