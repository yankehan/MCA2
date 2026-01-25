import json
from .model import CVDDNet, FrozenBertEncoder, TokenEmbedding
from .trainer import CVDDTrainer
class CVDD:
    def __init__(self, ad_score: str = "context_dist_mean"):
        self.ad_score = ad_score
        self.net_name = None
        self.net = None
        self.trainer = None
        self.optimizer_name = None
        self.train_dists = None
        self.test_dists = None
        self.results = {
            "context_vectors": None,
            "train_time": None,
            "test_time": None,
            "test_auc": None,
            "test_auprc": None,
            "test_scores": None,
        }
    def set_network(self, dataset, embedding_size: int = 100, attention_size: int = 150, n_attention_heads: int = 3,
                    freeze_embedding: bool = False, bert_name: str = "bert-base-uncased", bert_cache_dir: str = None):
        if getattr(dataset, "use_bert_tokenizer", False):
            name = getattr(dataset, "bert_name", None) or bert_name
            cache_dir = getattr(dataset, "bert_cache_dir", None) or bert_cache_dir
            encoder = FrozenBertEncoder(pretrained_model_name=name, cache_dir=cache_dir)
        else:
            encoder = TokenEmbedding(dataset.encoder.vocab_size, embedding_size, freeze=freeze_embedding)
        self.net = CVDDNet(encoder, attention_size=attention_size, n_attention_heads=n_attention_heads)
    def train(self, dataset, optimizer_name: str = "adam", lr: float = 1e-3, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 64, lambda_p: float = 1.0,
              alpha_scheduler: str = "logarithmic", weight_decay: float = 0.5e-6, device: str = "cuda",
              n_jobs_dataloader: int = 0, show_progress: bool = True, desc_prefix: str = ""):
        self.optimizer_name = optimizer_name
        self.trainer = CVDDTrainer(
            optimizer_name=optimizer_name,
            lr=lr,
            n_epochs=n_epochs,
            lr_milestones=lr_milestones,
            batch_size=batch_size,
            lambda_p=lambda_p,
            alpha_scheduler=alpha_scheduler,
            weight_decay=weight_decay,
            device=device,
            n_jobs_dataloader=n_jobs_dataloader,
            show_progress=show_progress,
            desc_prefix=desc_prefix,
        )
        self.net = self.trainer.train(dataset, self.net)
        self.train_dists = self.trainer.train_dists
        self.results["context_vectors"] = self.trainer.c
        self.results["train_time"] = self.trainer.train_time
    def test(self, dataset, device: str = "cuda", n_jobs_dataloader: int = 0, show_progress: bool = True,
             desc_prefix: str = ""):
        if self.trainer is None:
            self.trainer = CVDDTrainer(device=device, n_jobs_dataloader=n_jobs_dataloader)
        self.trainer.device = device
        self.trainer.n_jobs_dataloader = n_jobs_dataloader
        self.trainer.show_progress = show_progress
        self.trainer.desc_prefix = desc_prefix
        self.trainer.test(dataset, self.net, ad_score=self.ad_score)
        self.test_dists = self.trainer.test_dists
        self.results["test_time"] = self.trainer.test_time
        self.results["test_auc"] = self.trainer.test_auc
        self.results["test_auprc"] = self.trainer.test_auprc
        self.results["test_scores"] = self.trainer.test_scores
    def save_results(self, export_json: str):
        with open(export_json, "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)
