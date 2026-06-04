# molpal/models/molformer.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from molpal.models.base import Model
from sklearn.ensemble import RandomForestRegressor

class MolFormerAl(Model):
    provides = {"means", "means_and_vars"}

    def __init__(self, device="cuda", batch_size=128, **kwargs):
        super().__init__()
        self.device = device
        self.batch_size = batch_size

        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct"
        )
        self.encoder = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct"
        ).to(device)
        self.encoder.eval()

        from sklearn.ensemble import RandomForestRegressor
        self.regressor = RandomForestRegressor(n_estimators=100, n_jobs=-1)

    # -----------------------------
    # REQUIRED: model type
    # -----------------------------
    @property
    def type_(self):
        return "molformer"

    # -----------------------------
    # REQUIRED: training entrypoint
    # -----------------------------
    def train(self, X, y):
        X_emb = self._encode(X)
        self.regressor.fit(X_emb, y)

    # -----------------------------
    # REQUIRED: prediction
    # -----------------------------
    def predict(self, X):
        X_emb = self._encode(X)
        return self.regressor.predict(X_emb)

    def get_means(self, X):
        return self.predict(X)

    def get_means_and_vars(self, X):
        X_emb = self._encode(X)

        preds = []
        for tree in self.regressor.estimators_:
            preds.append(tree.predict(X_emb))

        preds = np.vstack(preds)
        means = preds.mean(axis=0)
        vars_ = preds.var(axis=0)

        return means, vars_

    # -----------------------------
    # REQUIRED: save / load
    # -----------------------------
    def save(self, path):
        import joblib
        joblib.dump(self.regressor, path + "/rf.pkl")

    def load(self, path):
        import joblib
        self.regressor = joblib.load(path + "/rf.pkl")

    # -----------------------------
    # INTERNAL: encoder
    # -----------------------------
    def _encode(self, smiles_list):
        import torch
        import numpy as np

        all_embs = []

        for i in range(0, len(smiles_list), self.batch_size):
            batch = smiles_list[i:i+self.batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.encoder(**inputs)

            emb = outputs.last_hidden_state.mean(dim=1)
            all_embs.append(emb.cpu().numpy())

        return np.vstack(all_embs)

