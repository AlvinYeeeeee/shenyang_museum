
mkdir /workspace/Models
cd /workspace/

# Install llama-index
pip install llama-index

# Install MeloTTS
git clone https://github.com/myshell-ai/MeloTTS.git && cd MeloTTS && pip install -e . && python -m unidic download

# Install huggingface
pip install transformers

# Install embedding
git clone https://huggingface.co/BAAI/bge-large-zh-v1.5

mv bge-large-zh-v1.5 /workspace/Models

# Install speech to text

cd /workspace/Models && git clone https://huggingface.co/openai/whisper-large-v3-turbo

# Install ollama

curl -fsSL https://ollama.com/install.sh | sh

pip install --no-cache-dir -r requirements.txt

pip uninstall -y torchvision && pip install torchvision llama-index-llms-yi

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs

cd /workspace/Models/bge-large-zh-v1.5 && git lfs install && git lfs pull

cd /workspace/Models/whisper-large-v3-turbo && git lfs install && git lfs pull

pip install 'accelerate>=0.26.0'