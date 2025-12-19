import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
from tqdm import tqdm
import math

# Import the modified dataset
from pretrain.data_preparation import OsuManiaDataset, collate_stage2, make_collate_fn

# ==========================================
# Environment-driven defaults (can be overridden by CLI args)
# ==========================================
def _parse_bool_env(val, default=False):
    if val is None:
        return default
    val = str(val).strip().lower()
    return val in ("1", "true", "yes", "on")

def _parse_list_env(val):
    if val is None:
        return None
    s = str(val).strip()
    # Accept JSON list, os.pathsep separated, or comma separated
    if s.startswith('['):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    if os.pathsep in s:
        return [p for p in s.split(os.pathsep) if p]
    if ',' in s:
        return [p.strip() for p in s.split(',') if p.strip()]
    return [s]

# Defaults (from your provided list)
DEFAULT_DATA_DIR = [
    #"/mnt/p/dodosu/mania/2015/", # for incredibly low load testing only
    "/mnt/p/dodosu/mania/2024/",
    "/mnt/p/dodosu/mania/2023/","/mnt/p/dodosu/mania/2022/","/mnt/p/dodosu/mania/2021/",
    "/mnt/p/dodosu/mania/2020/","/mnt/p/dodosu/mania/2019/","/mnt/p/dodosu/mania/2018/",
    # "/mnt/p/dodosu/mania/2017/","/mnt/p/dodosu/mania/2016/","/mnt/p/dodosu/mania/2015/",
    # "/mnt/p/dodosu/mania/2014/","/mnt/p/dodosu/mania/2013/",
    "/mnt/p/dodosu/mania/2025nov/"
]
TAIKO_DATA_DIR = [
    "/mnt/p/dodosu/taiko/2024/",
    #"/mnt/p/dodosu/taiko/2023/","/mnt/p/dodosu/taiko/2022/","/mnt/p/dodosu/taiko/2021/",
    #"/mnt/p/dodosu/taiko/2020/","/mnt/p/dodosu/taiko/2019/","/mnt/p/dodosu/taiko/2018/",
    #"/mnt/p/dodosu/taiko/2017/","/mnt/p/dodosu/taiko/2016/","/mnt/p/dodosu/taiko/2015/",
    #"/mnt/p/dodosu/taiko/2014/","/mnt/p/dodosu/taiko/2013/",
    #"/mnt/p/dodosu/taiko/2025nov/"
]
DEFAULT_MODEL_SAVE_DIR = "/mnt/code/elec/osu/model/"
DEFAULT_KEYS = 4
DEFAULT_EPOCHS_ONSET = 10
DEFAULT_EPOCHS_PATTERN = 20
DEFAULT_BATCH_SIZE = 16 # at 8 before, idk which is best yet
DEFAULT_MAX_FRAMES = 2560 # 2560 # best at 0
DEFAULT_LR = 0.0 # 0.0 is None, a non 0.0 value would override lr-o, lr-pg, lr-pd
DEFAULT_CACHE_PROCESSED = True
DEFAULT_NUM_CPU_THREADS = 24
DEFAULT_CUDA_AVAILABLE = True
DEFAULT_USE_AMP = True

IS_NAN_TEST = True
LR_ONSET = 1e-4
LR_GENERATOR = 5e-5
LR_DISCRIMINATOR = 5e-4
#all 3e-4 not work at GAN
#5e-5/1e-4 works fine but seems to have a constant d and fluctuating g

# Load from environment if present
env_DATA_DIR = _parse_list_env(os.environ.get('DATA_DIR'))
DATA_DIR = env_DATA_DIR if env_DATA_DIR is not None else DEFAULT_DATA_DIR
MODEL_SAVE_DIR = os.environ.get('MODEL_SAVE_DIR', DEFAULT_MODEL_SAVE_DIR)
KEYS = int(os.environ.get('KEYS', DEFAULT_KEYS))
EPOCHS_ONSET = int(os.environ.get('EPOCHS_ONSET', DEFAULT_EPOCHS_ONSET))
EPOCHS_PATTERN = int(os.environ.get('EPOCHS_PATTERN', DEFAULT_EPOCHS_PATTERN))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', DEFAULT_BATCH_SIZE))
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', DEFAULT_MAX_FRAMES))
LR = float(os.environ.get('LR', DEFAULT_LR))
CACHE_PROCESSED = _parse_bool_env(os.environ.get('CACHE_PROCESSED'), DEFAULT_CACHE_PROCESSED)
NUM_CPU_THREADS = int(os.environ.get('NUM_CPU_THREADS', DEFAULT_NUM_CPU_THREADS))
CUDA_AVAILABLE = _parse_bool_env(os.environ.get('CUDA_AVAILABLE'), DEFAULT_CUDA_AVAILABLE)
USE_AMP = _parse_bool_env(os.environ.get('USE_AMP'), DEFAULT_USE_AMP)

# Additional pragmatic defaults (may be useful for train_v2)
DEFAULT_ONSET_SAVE = os.path.join(MODEL_SAVE_DIR, 'onset_latest.pt')
DEFAULT_PATTERN_SAVE = os.path.join(MODEL_SAVE_DIR, 'pattern_gen_latest.pt')

# ==========================================
# MODELS
# ==========================================
class OnsetDetector(nn.Module):
    def __init__(self, n_mels=100, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.extra_conv = nn.Conv1d(128, 128, 3, padding=1)  #test:
        self.extra_bn = nn.BatchNorm1d(128)  #test:
        self.extra_relu = nn.ReLU()  #test:

        self.lstm = nn.LSTM(128, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.extra_lstm = nn.LSTM(hidden_size*2, hidden_size, num_layers=1, bidirectional=True, batch_first=True)  #test:

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.extra_conv(x)  #test:
        x = self.extra_bn(x)  #test:
        x = self.extra_relu(x)  #test:

        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)
        x, _ = self.extra_lstm(x)  #test:
        logits = self.classifier(x)
        return logits


class PatternGenerator(nn.Module):
    def __init__(self, input_dim=100, hidden_size=256, keys=4, num_layers=3):
        super().__init__()

        # 新增 CNN（输入为 mel_events 的特征维度）
        self.pre_cnn1 = nn.Conv1d(input_dim, input_dim, 3, padding=1)  #test:
        self.pre_relu1 = nn.ReLU()  #test:
        self.pre_cnn2 = nn.Conv1d(input_dim, input_dim, 3, padding=1)  #test:
        self.pre_relu2 = nn.ReLU()  #test:

        self.audio_proj = nn.Linear(input_dim, hidden_size)
        self.delta_proj = nn.Linear(1, hidden_size)
        self.bpm_proj = nn.Linear(1, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, keys)

    def forward(self, mel_events, time_deltas, bpm, mask=None):

        # Apply new CNN layers（必须 permute）
        x_cnn = mel_events.permute(0, 2, 1).contiguous()  #test:
        x_cnn = self.pre_cnn1(x_cnn)  #test:
        x_cnn = self.pre_relu1(x_cnn)  #test:
        x_cnn = self.pre_cnn2(x_cnn)  #test:
        x_cnn = self.pre_relu2(x_cnn)  #test:
        mel_events = x_cnn.permute(0, 2, 1).contiguous()  #test:

        x = self.audio_proj(mel_events) + \
            self.delta_proj(time_deltas) + \
            self.bpm_proj(bpm).unsqueeze(1)

        src_key_padding_mask = ~mask if mask is not None else None
        x = x.contiguous()
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        return self.head(x) # [B, Seq, Keys]

class PatternDiscriminator(nn.Module):
    def __init__(self, keys=4, audio_dim=100, hidden_size=128):
        super().__init__()
        self.chart_conv = nn.Conv1d(keys, hidden_size, 3, padding=1)
        self.audio_conv = nn.Conv1d(audio_dim, hidden_size, 3, padding=1)

        # 新增的 conv 层
        self.extra_conv1 = nn.Conv1d(hidden_size*2, hidden_size*2, 3, padding=1)  #test:
        self.extra_relu1 = nn.LeakyReLU(0.2)  #test:
        self.extra_conv2 = nn.Conv1d(hidden_size*2, hidden_size*2, 3, padding=1)  #test:
        self.extra_relu2 = nn.LeakyReLU(0.2)  #test:

        self.net = nn.Sequential(
            nn.Conv1d(hidden_size*2, 256, 3, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 3, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )
        
    def forward(self, chart_logits, mel_events):
        c = self.chart_conv(chart_logits.transpose(1, 2))
        a = self.audio_conv(mel_events.transpose(1, 2))
        x = torch.cat([c, a], dim=1)

        # Apply new convs
        x = self.extra_conv1(x)  #test:
        x = self.extra_relu1(x)  #test:
        x = self.extra_conv2(x)  #test:
        x = self.extra_relu2(x)  #test:

        return self.net(x)


# ==========================================
# TRAINING LOOPS
# ==========================================
def train_onset(args, device):
    """Train Phase 1: Onset Detection"""
    dataset = OsuManiaDataset(args.data_dir, stage='onset', cache_processed=args.cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                        collate_fn=make_collate_fn(max_frames=args.max_frames), num_workers=args.workers)
    
    model = OnsetDetector().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([10.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Use new torch.amp API
    scaler = torch.amp.GradScaler(enabled=(args.amp and args.cuda and torch.cuda.is_available()))
    
    print("--- Starting Stage 1: Onset Detection ---")
    for epoch in range(args.epochs_onset):
        model.train()
        total_loss = 0.0
        
        # ADD: Loop counter to avoid division by zero
        num_batches = 0 
        
        for batch in tqdm(loader, desc=f"Onset Epoch {epoch+1}/{args.epochs_onset}"):
            mel = batch['mel'].to(device).contiguous()
            labels = batch['labels'].to(device).contiguous()
            
            # Pool labels to match model output size
            labels_pooled = nn.functional.max_pool1d(labels.transpose(1, 2), 2).transpose(1, 2).contiguous()
            
            with torch.cuda.amp.autocast(enabled=(args.amp and args.cuda)):
                preds = model(mel)
                loss = criterion(preds, labels_pooled)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 2. ADD: Gradient Clipping (Crucial for LSTM)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 3. ADD: NaN Check
            if IS_NAN_TEST:
                if torch.isnan(loss):
                    print("Warning: NaN loss detected in Onset training. Skipping step.")
                    optimizer.zero_grad() # Clear gradients
                    continue

            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), args.save_dir_onset)

def train_pattern(args, device):
    """Train Phase 2: Pattern Generation (GAN)"""
    dataset = OsuManiaDataset(args.data_dir, stage='pattern', keys=args.keys, cache_processed=args.cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                        collate_fn=collate_stage2, num_workers=args.workers)
    
    gen = PatternGenerator(keys=args.keys).to(device)
    disc = PatternDiscriminator(keys=args.keys).to(device)
    
    opt_g = optim.AdamW(gen.parameters(), lr=args.lr)
    opt_d = optim.AdamW(disc.parameters(), lr=args.lr)
    
    ce_criterion = nn.BCEWithLogitsLoss()
    gan_criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(enabled=(args.amp and args.cuda and torch.cuda.is_available()))
    
    print("--- Starting Stage 2: Pattern GAN ---")
    for epoch in range(args.epochs_pattern):
        gen.train(); disc.train()
        
        # 1. ADD: Tracking stats
        total_d_loss = 0
        total_g_loss = 0
        num_batches = 0
        
        for batch in tqdm(loader, desc=f"Pattern Epoch {epoch+1}/{args.epochs_pattern}"):
            mel = batch['mel_events'].to(device).contiguous()
            delta = batch['time_deltas'].to(device).contiguous()
            bpm = batch['bpm'].to(device).contiguous()
            real_labels = batch['labels'].to(device).contiguous()
            mask = batch['mask'].to(device).contiguous()
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_d.zero_grad()
            with torch.cuda.amp.autocast(enabled=(args.amp and args.cuda)):
                # Real
                real_pred = disc(real_labels.float(), mel)
                d_loss_real = gan_criterion(real_pred, torch.ones_like(real_pred))
                
                # Fake
                fake_logits = gen(mel, delta, bpm, mask)
                fake_probs = torch.sigmoid(fake_logits)
                fake_pred = disc(fake_probs.detach(), mel)
                d_loss_fake = gan_criterion(fake_pred, torch.zeros_like(fake_pred))
                
                # Random (Pairs)
                idx = torch.randperm(args.keys, device=device)
                shuffled_labels = real_labels[:, :, idx]
                rand_pred = disc(shuffled_labels.float(), mel)
                d_loss_rand = gan_criterion(rand_pred, torch.zeros_like(rand_pred))
                
                d_loss = d_loss_real + (d_loss_fake + d_loss_rand) * 0.5

            scaler.scale(d_loss).backward()
            
            # 2. ADD: Clip Discriminator
            scaler.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            scaler.step(opt_d)
            
            # ---------------------
            # Train Generator
            # ---------------------
            opt_g.zero_grad()
            with torch.cuda.amp.autocast(enabled=(args.amp and args.cuda)):
                # Re-run forward pass to keep graph for G
                fake_logits = gen(mel, delta, bpm, mask)
                fake_probs = torch.sigmoid(fake_logits)
                
                g_pred = disc(fake_probs, mel)
                adv_loss = gan_criterion(g_pred, torch.ones_like(g_pred))
                
                # Reconstruction Loss (Safe masking)
                mask_flat = mask.view(-1)
                # 3. ADD: Safety check for empty masks
                if mask_flat.sum() > 0:
                    recon_loss = ce_criterion(
                        fake_logits.view(-1, args.keys)[mask_flat], 
                        real_labels.view(-1, args.keys)[mask_flat].float()
                    )
                else:
                    recon_loss = torch.tensor(0.0, device=device)

                # Weighting: Prioritize reconstruction early on
                g_loss = recon_loss * 10.0 + adv_loss * 0.1  # Reduced adv weight slightly for stability

            scaler.scale(g_loss).backward()
            
            # 4. ADD: Clip Generator
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
            
            # 5. ADD: NaN checks
            if args.is_nan_test:
                if torch.isnan(g_loss):
                    print("Warning: NaN detected in GAN Generator step. Resetting grads.")
                    opt_g.zero_grad()
                    continue
                if torch.isnan(d_loss):
                    print("Warning: NaN detected in GAN Discriminator step. Resetting grads.")
                    opt_d.zero_grad()
                    continue
                
            scaler.step(opt_g)
            scaler.update()
            
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            num_batches += 1

        print(f"Epoch {epoch+1} | D_Loss: {total_d_loss/max(1,num_batches):.4f} | G_Loss: {total_g_loss/max(1,num_batches):.4f}")
        torch.save(gen.state_dict(), args.save_dir_pattern)
        torch.save(disc.state_dict(), os.path.join(args.save_dir, 'pattern_disc_latest.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, nargs='+', default=DATA_DIR,
                        help="Paths to data directories (can pass multiple). Can also be set via DATA_DIR env var.")
    parser.add_argument("--save_dir", type=str, default=MODEL_SAVE_DIR)
    parser.add_argument("--stage", type=str, default="both", choices=["onset", "pattern", "both"]) 
    parser.add_argument("--epochs_onset", type=int, default=EPOCHS_ONSET)
    parser.add_argument("--epochs_pattern", type=int, default=EPOCHS_PATTERN)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--keys", type=int, default=KEYS)
    parser.add_argument("--cache", action="store_true", default=CACHE_PROCESSED)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    # parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--workers", type=int, default=NUM_CPU_THREADS)
    parser.add_argument("--cuda", action="store_true", default=CUDA_AVAILABLE)
    parser.add_argument("--amp", action="store_true", default=USE_AMP)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--save_dir_onset", type=str, default=DEFAULT_ONSET_SAVE)
    parser.add_argument("--save_dir_pattern", type=str, default=DEFAULT_PATTERN_SAVE)
    parser.add_argument("--is_nan_test", action="store_true", default=IS_NAN_TEST)
    parser.add_argument("--lr", type=float, default=LR, help="Note: Overrides lr-o, lr-pg, lr-pd if provided.")
    parser.add_argument("--lr-o", type=float, default=LR_ONSET)
    parser.add_argument("--lr-pg", type=float, default=LR_GENERATOR)
    parser.add_argument("--lr-pd", type=float, default=LR_DISCRIMINATOR)
    args = parser.parse_args()

    print("DoDoDual - TRAINING")
    if args.lr != 0.0: 
        args.lr_o = args.lr
        args.lr_pg = args.lr
        args.lr_pd = args.lr
    # Normalize data_dir: argparse with nargs='+' yields list, but env may have supplied a single string
    if isinstance(args.data_dir, str):
        args.data_dir = _parse_list_env(args.data_dir) or [args.data_dir]

    # Check data dirs exist (dataset class will also verify)
    valid_dirs = []
    for d in args.data_dir:
        if os.path.exists(d):
            valid_dirs.append(d)
        else:
            print(f"Warning: data directory does not exist: {d}")
    if not valid_dirs:
        print("Warning: no existing data directories found from provided DATA_DIR list.")
    args.data_dir = valid_dirs if valid_dirs else args.data_dir

    os.makedirs(args.save_dir, exist_ok=True)

    # Device selection
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    print(f"Data Directories: ",end='')
    for dir in args.data_dir:
        print(dir,end='; ')

    # Run requested stages
    if args.stage in ["onset", "both"]:
        train_onset(args, device)
        print("Onset trained successfully. ")

    if args.stage in ["pattern", "both"]:
        train_pattern(args, device)
        print("Pattern trained successfully. ")
    print("Normal termination of train.py. ")
