import torch
import os
from transformers import AutoModel

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_OFF"] = "True"
os.environ["POSTHOG_DISABLED"] = "1"

class JinaReranker:
    """
    Resident Jina Reranker v3.

    The process is expected to be launched with:
        CUDA_VISIBLE_DEVICES=2

    Therefore logical cuda:0 == physical GPU2.
    """

    def __init__(
        self,
        model_name="jinaai/jina-reranker-v3",
        device="cuda",
        max_query_length=512,
        max_doc_length=2048,
    ):
        self.model_name = model_name
        self.device = device
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

        print("[Jina] Loading reranker...")

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.model.to(self.device)
        self.model.eval()

        print(
            f"[Jina] Loaded on {self.device} "
            f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
        )
        
    @torch.inference_mode()
    def rerank(self, query, documents, top_n=None):
        if not documents:
            return []

        results = self.model.rerank(
            query,
            documents,
            top_n=top_n or len(documents),
            max_query_length=self.max_query_length,
            max_doc_length=self.max_doc_length,
        )

        return results