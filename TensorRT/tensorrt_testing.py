import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time
from glob import glob

MEAN = np.array([0.3337, 0.3064, 0.3171], dtype=np.float32)
STD  = np.array([0.2672, 0.2564, 0.2629], dtype=np.float32)
INPUT_SHAPE  = (1, 3, 32, 32)
OUTPUT_SHAPE = (1, 43)
TEST_DIR     = '/kaggle/input/datasets/akhilvaidya/inferernce'
WARMUP_RUNS  = 20

def preprocess(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis]

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime    = trt.Runtime(TRT_LOGGER)
with open('/kaggle/working/vit_gtsrb_fp16.trt', 'rb') as f:
    engine  = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# Discover input/output tensor names from the engine
input_name  = None
output_name = None
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.INPUT:
        input_name = name
    else:
        output_name = name

print(f"Input tensor : '{input_name}'  shape={engine.get_tensor_shape(input_name)}")
print(f"Output tensor: '{output_name}' shape={engine.get_tensor_shape(output_name)}")

h_input  = cuda.pagelocked_empty(int(np.prod(INPUT_SHAPE)),  dtype=np.float32)
h_output = cuda.pagelocked_empty(int(np.prod(OUTPUT_SHAPE)), dtype=np.float32)
d_input  = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream   = cuda.Stream()

context.set_tensor_address(input_name,  int(d_input))
context.set_tensor_address(output_name, int(d_output))

def infer(img_array: np.ndarray) -> np.ndarray:
    np.copyto(h_input, img_array.ravel())
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream.handle)      # replaces execute_async_v2
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output.copy().reshape(OUTPUT_SHAPE)

# warmup

print(f"\nWarming up for {WARMUP_RUNS} runs...")
dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
for _ in range(WARMUP_RUNS):
    infer(dummy)
print("✓ Warmup complete\n")

image_paths = (glob(os.path.join(TEST_DIR, '*.png')) +
               glob(os.path.join(TEST_DIR, '*.jpg')) +
               glob(os.path.join(TEST_DIR, '*.ppm')))

assert len(image_paths) > 0, f"No images found in {TEST_DIR}"
print(f"Running inference on {len(image_paths)} images...\n")

latencies, predictions = [], []

for path in image_paths:
    inp = preprocess(path)
    t0  = time.perf_counter()
    out = infer(inp)
    t1  = time.perf_counter()
    latencies.append((t1 - t0) * 1000)
    predictions.append(np.argmax(out))

latencies = np.array(latencies)

print("=" * 40)
print("      TensorRT FP16 Inference Metrics")
print("=" * 40)
print(f"  Images tested   : {len(latencies)}")
print(f"  Mean latency    : {latencies.mean():.2f} ms")
print(f"  Median (p50)    : {np.percentile(latencies, 50):.2f} ms")
print(f"  p95 latency     : {np.percentile(latencies, 95):.2f} ms")
print(f"  p99 latency     : {np.percentile(latencies, 99):.2f} ms")
print(f"  Min / Max       : {latencies.min():.2f} / {latencies.max():.2f} ms")
print(f"  Throughput      : {1000 / latencies.mean():.1f} img/sec")
print("=" * 40)
print(f"\nSample predictions (first 10): {predictions[:10]}")