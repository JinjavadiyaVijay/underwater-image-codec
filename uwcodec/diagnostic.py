import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob

from uwcodec.codecs.v2_codec import UWCodecV2
from uwcodec.models.v2_decoder import Upsample2x
import uwcodec.models.v2_decoder as v2_decoder

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# 1. Load the model
checkpoint_path = "outputs/v2/budget_128_ema_final/best.pt"
codec = UWCodecV2.load(checkpoint_path, device=device)
codec.eval()

# 2. Get 10 representative EUVP validation images
dataset_dir = "S:\\IMG_compressors\\datasets\\EUVP"
# Try to find images
image_paths = glob.glob(os.path.join(dataset_dir, "**", "*.jpg"), recursive=True)
if len(image_paths) == 0:
    image_paths = glob.glob(os.path.join(dataset_dir, "**", "*.png"), recursive=True)

if len(image_paths) < 10:
    raise ValueError(f"Could not find enough images in {dataset_dir}")

image_paths = sorted(image_paths)[:10]

print(f"Found {len(image_paths)} images, taking 10.")

def process_image(img_path):
    img = Image.open(img_path).convert("RGB")
    # Resize to 128x128 for the network
    img = img.resize((128, 128), Image.LANCZOS)
    img_np = np.array(img)
    return img_np

fig, axes = plt.subplots(10, 6, figsize=(18, 30))
col_titles = ["Original", "Enc->Quant->Dec", "Serialized Recon", "Sem Map", "Det Map L1", "Dec Before Final"]

for ax, col in zip(axes[0], col_titles):
    ax.set_title(col)

max_diff_stats = []
sem_unique_list = []
det_unique_list = []

for i, path in enumerate(image_paths):
    img_np = process_image(path)
    
    # a. Original
    axes[i, 0].imshow(img_np)
    axes[i, 0].axis('off')
    
    # b. Enc->Quant->Dec (Forward Pass)
    x = codec._preprocess(img_np).to(device)
    with torch.no_grad():
        out = codec(x)
        recon_direct = codec._postprocess(out["reconstruction"])
        
    axes[i, 1].imshow(recon_direct)
    axes[i, 1].axis('off')
    
    # c. Exact encode -> payload -> decode
    payload = codec.encode(img_np, budget=128)
    recon_serial = codec.decode(payload)
    
    axes[i, 2].imshow(recon_serial)
    axes[i, 2].axis('off')
    
    # Calculate max pixel difference
    diff = np.abs(recon_direct.astype(np.int32) - recon_serial.astype(np.int32))
    max_diff = np.max(diff)
    max_diff_stats.append(max_diff)
    
    # Extact indices
    # We have to unpack payload to get exact indices
    vq_data, _ = codec._payload_fmt.unpack(payload)
    
    # Sem Map (Level 1)
    sem_l1_bytes = vq_data[:16]
    sem_idx = np.array(list(sem_l1_bytes)).reshape(4, 4)
    sem_unique = len(np.unique(sem_idx))
    sem_unique_list.append(sem_unique)
    
    axes[i, 3].imshow(sem_idx, cmap='nipy_spectral')
    axes[i, 3].axis('off')
    
    # Det Map (Level 1)
    offset = 16 * 3 # 3 semantic levels
    det_l1_bytes = vq_data[offset : offset + 64]
    det_idx = np.array(list(det_l1_bytes)).reshape(8, 8)
    det_unique = len(np.unique(det_idx))
    det_unique_list.append(det_unique)
    
    axes[i, 4].imshow(det_idx, cmap='nipy_spectral')
    axes[i, 4].axis('off')
    
    # d. Decoder output BEFORE final output layer
    # Let's extract the feature map just before the final conv
    with torch.no_grad():
        sem_z = codec.sem_encoder(x)
        det_z = codec.det_encoder(x)
        sem_q, _ = codec.sem_rvq(sem_z)
        det_q1, _ = codec.det_vq1(det_z)
        det_mask = codec._make_det_mask(codec.det_l1_tokens, device)
        det_q_masked = det_q1 * det_mask
        if codec.det_vq2 is not None:
            det_residual = (det_z - det_q1)
            det_q2, _ = codec.det_vq2(det_residual)
            det_mask_l2 = codec._make_det_mask(codec.det_l2_tokens, device)
            det_q_masked = det_q_masked + det_q2 * det_mask_l2
            
        # Manually run decoder up to the layer before final
        h = codec.decoder.input_proj(sem_q)
        h = codec.decoder.bottom_blocks(h)
        h = codec.decoder.up1(h)
        h = codec.decoder.film_8x8(h, det_q_masked)
        h = codec.decoder.mid_8x8(h)
        h = codec.decoder.up2(h)
        h = codec.decoder.mid_16x16(h)
        h = codec.decoder.up3(h)
        h = codec.decoder.mid_32x32(h)
        h = codec.decoder.up4(h)
        h = codec.decoder.up5(h)
        
        # In output_head: nn.Conv2d -> GroupNorm -> SiLU -> Conv2d -> Sigmoid
        # Let's visualize after up5 (128x128, 8 channels)
        h_vis = h[0].mean(dim=0).cpu().numpy()
        
    axes[i, 5].imshow(h_vis, cmap='viridis')
    axes[i, 5].axis('off')

plt.tight_layout()
plt.savefig("diagnostic_grid.png")
print("Saved diagnostic_grid.png")

print(f"Max Pixel Difference (Direct vs Serialized): {max(max_diff_stats)}")
print(f"Unique Semantic Codes (L1) (Avg): {np.mean(sem_unique_list):.1f} / 16")
print(f"Unique Detail Codes (L1) (Avg): {np.mean(det_unique_list):.1f} / 64")


# 3. TEMPORARY diagnostic decoder experiments
# Original Up5 mode
print("Running modified decoders...")

def get_modified_codec(mode="nearest"):
    # Create a fresh codec with loaded weights
    modified_codec = UWCodecV2.load(checkpoint_path, device=device)
    
    # Patch Upsample2x
    class PatchedUpsample2x(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
            self.norm = v2_decoder._gn(out_ch)
            self.mode = mode
            
        def forward(self, x):
            if self.mode == "nearest":
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            return F.silu(self.norm(self.conv(x)))
            
    # Replace upsamples in decoder
    # We can just iterate modules and replace, or directly override forward in existing instances
    # We can just iterate modules and replace, or directly override forward in existing instances
    
    for name, module in modified_codec.decoder.named_modules():
        if isinstance(module, Upsample2x):
            # We will just monkey patch its forward method on the existing object
            # to preserve its weights!
            def make_forward(mod, mode=mode):
                def forward(x):
                    if mode == "nearest":
                        x = F.interpolate(x, scale_factor=2, mode="nearest")
                    else:
                        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
                    return F.silu(mod.norm(mod.conv(x)))
                return forward
            
            module.forward = make_forward(module)
            
    return modified_codec

codec_nearest = get_modified_codec("nearest")
codec_bilinear = get_modified_codec("bilinear") # This should be identical to the original!

# Test on one image
img_np = process_image(image_paths[0])
x = codec._preprocess(img_np).to(device)

with torch.no_grad():
    out_orig = codec(x)
    recon_orig = codec._postprocess(out_orig["reconstruction"])
    
    out_nearest = codec_nearest(x)
    recon_nearest = codec_nearest._postprocess(out_nearest["reconstruction"])
    
    out_bilinear = codec_bilinear(x)
    recon_bilinear = codec_bilinear._postprocess(out_bilinear["reconstruction"])

fig2, ax2 = plt.subplots(1, 4, figsize=(16, 4))
ax2[0].imshow(img_np)
ax2[0].set_title("Original")
ax2[1].imshow(recon_orig)
ax2[1].set_title("Recon (Original)")
ax2[2].imshow(recon_nearest)
ax2[2].set_title("Recon (Nearest)")
ax2[3].imshow(recon_bilinear)
ax2[3].set_title("Recon (Bilinear)")

for a in ax2:
    a.axis('off')
    
plt.savefig("diagnostic_comparison.png")
print("Saved diagnostic_comparison.png")

print("Diagnostic script complete.")
