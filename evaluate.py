import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.lit_eval import LitEvaluator
from src.lit_model import LitGPT


@torch.no_grad()
def _collect_features(dataloader, gpt: LitGPT, device="cuda"):
    """
    Returns
    -------
    X : (N, n_embd)  float32
    y : (N,)         int64
    """
    gpt.model.eval()
    feats, labels = [], []

    for imgs, y in tqdm(dataloader, desc="collect"):
        # ---- tokens ------------------------------------------------------
        idx = (
            imgs.view(imgs.size(0), -1)  # raw pixels 0‥1
            .mul(255)
            .round()
            .long()
            .to(device)
        )

        # ---- frozen GPT features (mean-pool all tokens) ------------------
        f = gpt.model.get_features_long(idx)  # (B, S, 768)
        f = f.mean(dim=1).cpu()  # (B, 768)

        feats.append(f)
        labels.append(y)

    X = torch.cat(feats).numpy()
    y = torch.cat(labels).numpy()
    return X.astype(np.float32), y.astype(np.int64)


def linear_probe_accuracy(
    model: LitGPT,
    datamodule,
    device: str = "cuda",
    test_size: float = 0.2,
    max_iter: int = 2000,
):
    """
    Prints a one-shot linear-probe accuracy on frozen GPT features.
    """
    loader = datamodule.train_dataloader()  # one full epoch is enough
    X, y = _collect_features(loader, model.to(device), device=device)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    clf = LogisticRegression(
        max_iter=max_iter,
        multi_class="multinomial",
        verbose=0,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"[linear probe]  accuracy = {acc*100:.2f} %")
    return acc


@hydra.main(config_path="configs", config_name="evaluate", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    """
    The main evaluation script.

    This script is orchestrated by Hydra. It will instantiate the datamodule,
    and the LitEvaluator, which loads a pretrained model from a checkpoint
    to run evaluations.
    """
    # --- Set seed for reproducibility --- #
    pl.seed_everything(cfg.seed)

    # --- Instantiate DataModule --- #
    datamodule = hydra.utils.instantiate(cfg.data)

    # --- Instantiate the Evaluator --- #
    model_config = hydra.utils.instantiate(cfg.model)
    model = LitGPT(
        model_config,
        use_vqvae=cfg.lit_model.use_vqvae,
        vqvae_path=cfg.lit_model.vqvae_path,
        lr=cfg.lit_model.lr,
    )
    model.setup()
    checkpoint_path = cfg.get("checkpoint_path", None)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    model.load_state_dict(state_dict, strict=True)

    evaluator = hydra.utils.instantiate(cfg.evaluator, gpt_model=model)

    # datamodule.setup("fit")  # Prepare the datamodule for training
    # linear_probe_accuracy(model, datamodule, device="cuda")

    # --- Instantiate Trainer --- #
    trainer = hydra.utils.instantiate(cfg.trainer)

    # --- Start Training the linear probe --- #
    trainer.fit(evaluator, datamodule=datamodule)

    # --- Test the trained linear probe --- #
    trainer.test(evaluator, datamodule=datamodule)


if __name__ == "__main__":
    evaluate()
