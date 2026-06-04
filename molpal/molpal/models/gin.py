import torch
import torch.nn as nn
import numpy as np
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import GINConv, global_add_pool
from rdkit import Chem

class GINVirtual(nn.Module):
    provides = {"means", "means_and_vars"}

    def __init__(self, hidden_dim=300, device="cuda", **kwargs):
        super().__init__()
        self.device = device

        # Atom encoder from OGB
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)

        # 5-layer GIN (standard OGB architecture)
        self.convs = nn.ModuleList()
        for _ in range(5):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_layer))

        # Virtual node
        self.virtualnode_embedding = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # Small surrogate head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.to(device)

    def type_(self):
        return "gin"

    def train(self, X, y, **kwargs):
        # No training — pretrained encoder + small head
        return self

    def get_means(self, X):
        means = self.predict(X)
        return np.atleast_1d(means)

    def get_means_and_vars(self, X):
        means = self.predict(X)
        means = np.atleast_1d(means)
        vars_ = np.zeros_like(means)
        return means, vars_




    def apply(self, X, *args, **kwargs):
        return self.get_means_and_vars(X)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def fit(self, X, y, **kwargs):
        return self

    # ---------------------------
    # Core prediction
    # ---------------------------
    def predict(self, smiles_list):
        if isinstance(smiles_list, np.ndarray):
            smiles_list = smiles_list.ravel().tolist()
        if not isinstance(smiles_list, (list, tuple)):
            smiles_list = [smiles_list]
        smiles_list = [str(s) for s in smiles_list]

        graphs = [self.smiles_to_graph(s) for s in smiles_list]
        batch = self.collate(graphs).to(self.device)

        x = self.atom_encoder(batch.x)
        h = x + self.virtualnode_embedding.weight

        for conv in self.convs:
            h = conv(h, batch.edge_index)

        g = global_add_pool(h, batch.batch)
        out = self.head(g).squeeze()

        return out.detach().cpu().numpy()

    # ---------------------------
    # Graph utilities
    # ---------------------------
    def smiles_to_graph(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # 1 node, 1 feature (dummy), 2D tensor
            x = torch.zeros((1, 1), dtype=torch.long)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return {"x": x, "edge_index": edge_index}

        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        x = torch.tensor(atoms, dtype=torch.long)

        # Ensure x is 2D: (N, 1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # shape (N, 1)

        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])

        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return {"x": x, "edge_index": edge_index}


    def collate(self, graphs):
        xs, eis, batches = [], [], []
        offset = 0
        for i, g in enumerate(graphs):
            xs.append(g["x"])
            eis.append(g["edge_index"] + offset)
            batches.append(torch.full((g["x"].shape[0],), i, dtype=torch.long))
            offset += g["x"].shape[0]

        x = torch.cat(xs)
        edge_index = torch.cat(eis, dim=1)
        batch = torch.cat(batches)

        class Batch:
            def __init__(self, x, edge_index, batch):
                self.x = x
                self.edge_index = edge_index
                self.batch = batch

            def to(self, device):
                self.x = self.x.to(device)
                self.edge_index = self.edge_index.to(device)
                self.batch = self.batch.to(device)
                return self

        return Batch(x, edge_index, batch)

