import torch

from config.config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(cfg.seg_checkpoint_path, map_location=device)
encoder = model.encoder
encoder.eval()

x = torch.randn(1, 3, 1024,1024).to(device)

with torch.no_grad():
    features = encoder(x)

print(f"Number of encoder outputs: {len(features)}")

for i, feature in enumerate(features):
    print(f"Stage {i}: {feature.shape}")

print(encoder)
print("out_channels:", encoder.out_channels)
print("depth:", encoder._depth)