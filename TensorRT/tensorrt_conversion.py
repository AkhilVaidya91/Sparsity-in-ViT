import torch
import torch.nn as nn
import math

IMAGE_SIZE      = 32
NUM_CHANNELS    = 3
NUM_CLASSES     = 43
PATCH_SIZE      = 4
HIDDEN_SIZE     = 384
TRANSFORMER_BLOCKS = 10
ATTENTION_HEADS = 8
MLP_DIMENSION   = 4 * HIDDEN_SIZE
NUM_PATCHES     = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 64
DROPOUT         = 0.15

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(NUM_CHANNELS, HIDDEN_SIZE,
                                     kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(HIDDEN_SIZE)
        self.layer_norm2 = nn.LayerNorm(HIDDEN_SIZE)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE, num_heads=ATTENTION_HEADS,
            batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, MLP_DIMENSION),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(MLP_DIMENSION, HIDDEN_SIZE),
            nn.Dropout(dropout))
    def forward(self, x):
        x_norm = self.layer_norm1(x)
        attn_out, _ = self.multihead_attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.layer_norm2(x))
        return x

class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
    def forward(self, x):
        return self.fc(self.layer_norm(x))

class VisionTransformer(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.patch_embedding  = PatchEmbedding()
        self.cls_token        = nn.Parameter(torch.randn(1, 1, HIDDEN_SIZE))
        self.position_embeddings = nn.Parameter(torch.randn(1, NUM_PATCHES + 1, HIDDEN_SIZE))
        self.dropout          = nn.Dropout(dropout)
        self.transformer_blocks = nn.Sequential(
            *[TransformerEncoder(dropout) for _ in range(TRANSFORMER_BLOCKS)])
        self.mlp_head = MLPHead()
    def forward(self, x):
        x = self.patch_embedding(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1) + self.position_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        return self.mlp_head(x[:, 0, :])

device = torch.device('cuda')
model  = VisionTransformer().to(device)
model.load_state_dict(torch.load('/kaggle/input/models/akhilvaidya/gtsrb-vit/pytorch/default/1/vit_gtsrb_clahe.pth', map_location=device))
model.eval()

dummy = torch.randn(1, 3, 32, 32, device=device)

torch.onnx.export(
    model, dummy, '/kaggle/working/vit_gtsrb.onnx',
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
    do_constant_folding=True,
    verbose=False
)

## TensorRT Engine

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_trt_engine(onnx_path: str, engine_path: str):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, TRT_LOGGER)
    config  = builder.create_builder_config()

    # FP16 + 2 GB workspace
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    print(f"Network: {network.num_layers} layers, "
          f"input {network.get_input(0).shape}, output {network.get_output(0).shape}")

    engine_bytes = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)
    print(f"TensorRT FP16 engine saved → {engine_path}")

build_trt_engine('/kaggle/working/vit_gtsrb.onnx', '/kaggle/working/vit_gtsrb_fp16.trt')