# train_lejepa_bge_m3.py
# pip install torch transformers datasets wandb

import os, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import wandb
from torch.amp import autocast, GradScaler


# --- LeJEPA: SIGReg (ported 1:1 in spirit from the minimal LeJEPA snippet) ---
class SIGReg(nn.Module):
    def __init__(self, knots=17, t_max=3.0, n_proj=256):
        super().__init__()
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.n_proj = n_proj

    def forward(self, proj_vbn):  # [V, B, D]
        V, B, D = proj_vbn.shape
        proj = proj_vbn.reshape(V * B, D)  # [VB, D]

        A = torch.randn(D, self.n_proj, device=proj.device, dtype=proj.dtype)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-12)  # unit columns

        x_t = (proj @ A).unsqueeze(-1) * self.t.to(proj.dtype)  # [VB, M, K]
        # symmetric CF trick: integrate [0, t_max] and double via weights above
        err = (x_t.cos().mean(0) - self.phi.to(proj.dtype)).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights.to(proj.dtype)) * proj.size(0)
        return statistic.mean()


# --- Backbone + projector (document-level vector) ---
def cls_pool(last_hidden_state):
    return last_hidden_state[:, 0]  # CLS pooling (bge-style)

class BGEEncoder(nn.Module):
    def __init__(self, model_name="BAAI/bge-m3", proj_dim=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hid = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hid, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim),
        )
        # optional predictor head g(.) for asymmetry; you can set g=Identity() to mimic the simplest invariance form
        self.pred = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, input_ids_vbl, attention_mask_vbl):  # [V,B,L]
        V, B, L = input_ids_vbl.shape
        input_ids = input_ids_vbl.reshape(V * B, L)
        attention_mask = attention_mask_vbl.reshape(V * B, L)

        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        emb = cls_pool(out.last_hidden_state)              # [VB, H]
        proj = self.proj(emb)                              # [VB, D]
        proj = proj.reshape(V, B, -1)                      # [V,B,D]
        return emb.reshape(V, B, -1), proj                 # (for debugging/logging)


# --- Text view generation (two corruptions of the same doc) ---
def corrupt_ids(input_ids, tokenizer, p_token_drop=0.10, p_span_mask=0.10, span_len=8):
    # token dropout -> [MASK]
    ids = input_ids.clone()
    special = torch.zeros_like(ids, dtype=torch.bool)
    for sid in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
        if sid is not None:
            special |= (ids == sid)

    # token-level mask
    m = (torch.rand_like(ids.float()) < p_token_drop) & (~special)
    ids[m] = tokenizer.mask_token_id

    # span masking
    if p_span_mask > 0:
        L = ids.size(0)
        n_spans = int((L * p_span_mask) / max(span_len, 1))
        for _ in range(n_spans):
            start = random.randint(1, max(1, L - span_len - 1))
            end = min(L - 1, start + span_len)
            span = torch.arange(start, end, device=ids.device)
            span = span[~special[span]]
            ids[span] = tokenizer.mask_token_id

    return ids


class LocalTextDataset(torch.utils.data.Dataset):
    """
    Expects a plain-text file with 1 document per line (already segmented).
    """
    def __init__(self, path, tokenizer, max_len=2048, V=2):
        self.lines = open(path, "r", encoding="utf-8").read().splitlines()
        self.tok = tokenizer
        self.max_len = max_len
        self.V = V

    def __len__(self): return len(self.lines)

    def __getitem__(self, i):
        text = self.lines[i]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attn = torch.ones_like(ids)

        views_ids = []
        views_attn = []
        for _ in range(self.V):
            v_ids = corrupt_ids(ids, self.tok)
            views_ids.append(v_ids)
            views_attn.append(attn)

        return views_ids, views_attn


def collate(batch, pad_id):
    V = len(batch[0][0])
    # pad per view
    out_ids, out_attn = [], []
    for v in range(V):
        ids_v = [b[0][v] for b in batch]
        attn_v = [b[1][v] for b in batch]
        ids_v = nn.utils.rnn.pad_sequence(ids_v, batch_first=True, padding_value=pad_id)
        attn_v = nn.utils.rnn.pad_sequence(attn_v, batch_first=True, padding_value=0)
        out_ids.append(ids_v)
        out_attn.append(attn_v)
    return torch.stack(out_ids, 0), torch.stack(out_attn, 0)  # [V,B,L]


def main():
    # ---- config (edit as needed) ----
    cfg = dict(
        model_name="BAAI/bge-m3",
        text_path="corpus.txt",     # 1 doc per line
        max_len=2048,               # bge-m3 supports long docs; raise if you can afford it
        V=2,                        # two views
        proj_dim=256,
        lamb=0.02,                  # LeJEPA trade-off
        lr=2e-5,
        wd=0.05,
        bs=16,
        epochs=1,
        log_every=20,
        fp16=True,
    )

    wandb.init(project="lejepa-bge-m3", config=cfg)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    ds = LocalTextDataset(cfg["text_path"], tok, max_len=cfg["max_len"], V=cfg["V"])
    dl = DataLoader(
        ds,
        batch_size=cfg["bs"],
        shuffle=True,
        num_workers=2,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        drop_last=True,
    )

    net = BGEEncoder(cfg["model_name"], proj_dim=cfg["proj_dim"]).to(device)
    sigreg = SIGReg().to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scaler = GradScaler(enabled=(cfg["fp16"] and device == "cuda"))

    step = 0
    net.train()
    for epoch in range(cfg["epochs"]):
        for input_ids_vbl, attn_vbl in dl:
            input_ids_vbl = input_ids_vbl.to(device, non_blocking=True)
            attn_vbl = attn_vbl.to(device, non_blocking=True)

            with autocast(device_type=device, enabled=(cfg["fp16"] and device == "cuda")):
                _, proj_vbd = net(input_ids_vbl, attn_vbl)   # [V,B,D]
                mu = proj_vbd.mean(0, keepdim=True)          # [1,B,D]
                pred_loss = (proj_vbd - mu).square().mean()  # LeJEPA minimal example-style invariance/prediction
                reg_loss = sigreg(proj_vbd)
                lejepa_loss = (1 - cfg["lamb"]) * pred_loss + cfg["lamb"] * reg_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(lejepa_loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % cfg["log_every"] == 0:
                wandb.log(
                    {
                        "train/lejepa": lejepa_loss.item(),
                        "train/pred": pred_loss.item(),
                        "train/sigreg": reg_loss.item(),
                        "train/epoch": epoch,
                        "train/step": step,
                    }
                )
            step += 1

    wandb.finish()


if __name__ == "__main__":
    main()
