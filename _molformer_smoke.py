"""Smoke test: load DeepChem/MoLFormer-c3-100M and embed SMILES in .venv."""
import time
t0 = time.time()
import torch
from transformers import AutoTokenizer, AutoModel

MODEL = "ibm-research/MoLFormer-XL-both-10pct"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True)
model.eval()
print(f"loaded in {time.time()-t0:.1f}s, hidden={model.config.hidden_size}", flush=True)

smiles = ["CC(=O)Oc1ccccc1C(=O)O", "C1=CC=CN=C1", "CCNCC"]
enc = tok(smiles, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    out = model(**enc)
cls = out.last_hidden_state[:, 0, :]
mean = out.last_hidden_state.mean(dim=1)
print("CLS shape:", tuple(cls.shape), "mean shape:", tuple(mean.shape))
print("CLS[0][:5]:", cls[0, :5].tolist())
print("mean[0][:5]:", mean[0, :5].tolist())

# throughput estimate
smiles_batch = [smiles[0]] * 256
enc = tok(smiles_batch, return_tensors="pt", padding=True, truncation=True)
torch.cuda.synchronize() if torch.cuda.is_available() else None
t1 = time.time()
with torch.no_grad():
    _ = model(**enc)
t2 = time.time()
n = len(smiles_batch)
print(f"batch of {n}: {t2-t1:.2f}s -> {n/(t2-t1):.1f} SMILES/s (CPU)")
print("OK")
