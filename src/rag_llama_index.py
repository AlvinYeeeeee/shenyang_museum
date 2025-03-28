import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.vllm import Vllm
from llama_index.core.prompts import SelectorPromptTemplate, PromptTemplate, PromptType
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


class RagLlamaIndex:
    def __init__(self,
                 documents_path: str,
                 mode="openai",
                 model="qwen2.5:3b",
                 embedding_model="BAAI/bge-large-zh-v1.5",
                 ollama_url = 'http://localhost:11434'):
        self.documents = SimpleDirectoryReader(documents_path).load_data()
        if mode == "openai":
            os.environ[
                "OPENAI_API_KEY"] = "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"
        elif mode == "ollama":
            Settings.llm = Ollama(model=model, base_url=ollama_url,  request_timeout=60.0)
        elif mode == "vllm":
            Settings.llm = Vllm(
                model="Qwen/Qwen2.5-3B",
                max_new_tokens=1024,
                vllm_kwargs={
                    "swap_space": 1,
                    "gpu_memory_utilization": 0.5,
                    "max_model_len": 4096,
                },
            )
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        self.index = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = self.index.as_query_engine()
        self.prompt_str = ("下面是上下文信息\n"
                           "---------------------\n"
                           "{context_str}\n"
                           "---------------------\n"
                           "基于前面给定的信息，不要参考先验知识，"
                           "直接回答下面的问题，不要输出'根据提供的信息'这一类的开场白语句。\n"
                           "如果没有相关信息，请回答'I don't know'.\n"
                           "问题: {query_str}\n"
                           "答案: ")
        self.prompter = SelectorPromptTemplate(default_template=PromptTemplate(
            self.prompt_str, prompt_type=PromptType.QUESTION_ANSWER))

    def query(self, query_str):
        self.query_engine._get_prompt_modules(
        )["response_synthesizer"].update_prompts(
            {"text_qa_template": self.prompter})
        response = self.query_engine.query(query_str)
        return response
