from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class ChemBERTa:
    provides = {"means", "means_and_vars"}

    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MTR", device="cuda", **kwargs):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)

    def type_(self):
        return "chemberta"

    def train(self, X, y, **kwargs):
        return self  # pretrained

    def get_means(self, X):
        return self.predict(X)

    def get_means_and_vars(self, X):
        means = self.predict(X)
        vars_ = np.zeros_like(means)
        return means, vars_
    
    def apply(self, X, *args, **kwargs):
        # MolPAL passes featurizer, acq, batch_size, etc.
        # We ignore all of them for ChemBERTa.
        return self.get_means_and_vars(X)


    def save(self, path):
        pass

    def load(self, path):
        pass

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, smiles_list):
        import numpy as np

        # 1. Handle NumPy arrays
        if isinstance(smiles_list, np.ndarray):
            # Flatten in case it's 2D, then to list
            smiles_list = smiles_list.ravel().tolist()

        # 2. Wrap scalars
        if not isinstance(smiles_list, (list, tuple)):
            smiles_list = [smiles_list]

        # 3. Force everything to string (what the tokenizer expects)
        smiles_list = [str(s) for s in smiles_list]

        inputs = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze()

        return logits.cpu().numpy()


