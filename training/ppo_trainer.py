"""
ppo_trainer.py — Xybernetex Engine Policy Trainer

V3: Pure behavioral cloning (cross-entropy on expert actions).
The old A2C-style policy gradient with offline TD-target advantages was
fundamentally broken — the value function bootstraps from static data,
producing noisy advantages that cause the DONE logit to dominate globally.

Behavioral cloning is the correct approach for learning from a heuristic
expert demonstrator:  minimize  CE(π(s), a_expert)  directly.
The network learns P(action|state) without unstable advantage weighting.
"""
import os
import glob
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from rl_chassis import ActorCritic, STATE_DIM, ACTION_DIM

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR    = "data"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "policy_v3.pt")  # V3: Behavioral Cloning

BATCH_SIZE  = 512     # Smaller batches = more gradient steps per epoch
EPOCHS      = 120     # More epochs — BC converges cleanly, no instability risk
LR          = 1e-3    # Higher LR: cross-entropy has clean gradients
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05   # Light label smoothing to prevent overconfidence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset Loading ───────────────────────────────────────────────────────────

class RolloutDataset(Dataset):
    def __init__(self, data_dir: str):
        self.states, self.actions = [], []
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        for f in files:
            data = np.load(f)
            self.states.append(data['states'])
            self.actions.append(data['actions'])

        self.states  = torch.tensor(np.concatenate(self.states),  dtype=torch.float32)
        self.actions = torch.tensor(np.concatenate(self.actions), dtype=torch.long)

        # ── Dataset diagnostics ──────────────────────────────────────────────
        action_names = ["PLAN", "ANALYZE", "INVOKE_TOOL", "EXECUTE_CODE", "WRITE_ARTIFACT", "DONE"]
        counts = torch.bincount(self.actions, minlength=ACTION_DIM)
        print(f"\n[Dataset] Action distribution:")
        for i, name in enumerate(action_names):
            pct = 100.0 * counts[i].item() / len(self.actions)
            print(f"  {name:20s}: {counts[i].item():6d}  ({pct:5.1f}%)")
        print()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# ── Training Loop ─────────────────────────────────────────────────────────────

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    dataset = RolloutDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    policy = ActorCritic(STATE_DIM, ACTION_DIM).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine annealing — LR decays smoothly to near-zero, preventing late overfitting
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print(f"--- Xybernetex Training Session: Policy V3 (Behavioral Cloning) ---")
    print(f"[Device] {device} | [Transitions] {len(dataset)} | [Epochs] {EPOCHS}")
    print(f"[LR] {LR} -> cosine -> 1e-5 | [Label Smoothing] {LABEL_SMOOTH}")
    print()

    best_loss = float('inf')
    best_acc  = 0.0

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch in dataloader:
            states, expert_actions = [x.to(device) for x in batch]

            # Forward pass — only need logits (critic unused in BC)
            logits, _ = policy(states)

            # Pure cross-entropy: learn P(action|state) = expert's action
            loss = F.cross_entropy(logits, expert_actions, label_smoothing=LABEL_SMOOTH)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * states.size(0)

            # Track accuracy — what % of states produce the expert's action via argmax
            predicted = logits.argmax(dim=-1)
            epoch_correct += (predicted == expert_actions).sum().item()
            epoch_total += states.size(0)

        scheduler.step()

        avg_loss = epoch_loss / epoch_total
        accuracy = 100.0 * epoch_correct / epoch_total
        lr_now   = scheduler.get_last_lr()[0]

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_acc  = accuracy
            torch.save(policy.state_dict(), MODEL_PATH)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                f"Acc: {accuracy:5.1f}% | LR: {lr_now:.2e} | "
                f"Best: {best_loss:.4f} ({best_acc:.1f}%)"
            )

    # Final save (best was already saved, but save final too for comparison)
    final_path = os.path.join(MODEL_DIR, "policy_v3_final.pt")
    torch.save(policy.state_dict(), final_path)

    print(f"\n[Success] Best policy saved to {MODEL_PATH}")
    print(f"  Best Loss: {best_loss:.4f} | Best Accuracy: {best_acc:.1f}%")
    print(f"  Final weights also saved to {final_path}")

if __name__ == "__main__":
    train()
