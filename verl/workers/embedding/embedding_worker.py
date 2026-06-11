from __future__ import annotations
import torch
import numpy as np
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.protocol import DataProto

class EmbeddingWorker(Worker):
    def __init__(self, config):
        super().__init__()
        self.model_name = config.get("model_name", "BAAI/bge-m3")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"EmbeddingWorker initialized on {self.device}")
        
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers")
            
        print(f"Loading model {self.model_name} to {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.to(dtype=torch.bfloat16)
        return {"status": "success"}

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_embeddings(self, data: DataProto) -> DataProto:
        if 'text' in data.non_tensor_batch:
            text_batch = data.non_tensor_batch['text']
        elif 'text' in data.batch:
            text_batch = data.batch['text'].tolist()
        else:
            raise ValueError("DataProto must contain 'text' field")

        embeddings = self.model.encode(
            text_batch,
            batch_size=1024,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        output = DataProto.from_dict(tensors={
            "embeddings": torch.from_numpy(embeddings)
        })
        
        # 按照规范，将数据转回 CPU 避免通信时的 CUDA 序列化问题（虽然 verl 也能处理 GPU Tensor）
        output = output.to('cpu')
        
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def offload_model(self):
        self.model.to('cpu')
        torch.cuda.empty_cache()