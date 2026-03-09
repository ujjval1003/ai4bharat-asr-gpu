#!/bin/bash
set -e

echo "🧹 Cleaning old environment..."
deactivate 2>/dev/null || true
rm -rf nemo ai4bharat-nemo _ai4bharat_nemo_src ~/.cache/torch_extensions ~/.cache/huggingface
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "📦 Creating fresh venv..."
python3.10 -m venv nemo
source nemo/bin/activate   # Works in Git Bash (Windows) or bin/ in Linux/WSL

echo "⬆️ Upgrading tools..."
pip install --upgrade pip wheel setuptools

echo "🔥 Installing GPU PyTorch (CUDA 12.6) for RTX 3050..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "📌 Pinning critical dependencies..."
pip install "numpy==1.26.4" "pyarrow==14.0.2" "pandas==2.2.3" "datasets==2.15.0" "numba==0.59.1" "scipy<1.15"

echo "📌 ONNXRuntime and torchcodec..."
pip install onnxruntime torchcodec

echo "📥 Cloning AI4Bharat NeMo..."
git clone https://github.com/AI4Bharat/NeMo.git ai4bharat-nemo
cd ai4bharat-nemo
git checkout nemo-v2

echo "🛠️ Installing build deps..."
pip install Cython pybind11

echo "🔄 Running NeMo reinstall..."
bash reinstall.sh

echo "🔧 Applying compatibility patches..."
cd ..

pip install "huggingface-hub==0.23.2" --force-reinstall --no-deps
pip install "transformers==4.36.2" --force-reinstall --no-deps
pip install "tokenizers==0.15.2" --force-reinstall --no-deps
pip install "numpy==1.26.4" --force-reinstall

cd ~/AI4Bharat/ai4bharat-nemo

echo "🔄 Resetting and patching hf_io_mixin.py..."
git checkout -- nemo/core/classes/mixins/hf_io_mixin.py

sed -i '/from abc import ABC/a from typing import Any' nemo/core/classes/mixins/hf_io_mixin.py
sed -i 's/from huggingface_hub import HfApi, ModelCard, ModelCardData, ModelFilter, hf_hub_download/from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download/' nemo/core/classes/mixins/hf_io_mixin.py
sed -i 's/Optional\[Union\[ModelFilter, List\[ModelFilter\]\]\]/Any/g' nemo/core/classes/mixins/hf_io_mixin.py
sed -i 's/Union\[ModelFilter, List\[ModelFilter\]\]/Any/g' nemo/core/classes/mixins/hf_io_mixin.py
sed -i 's/: ModelFilter/: Any/g' nemo/core/classes/mixins/hf_io_mixin.py
sed -i 's/ -> ModelFilter/ -> Any/g' nemo/core/classes/mixins/hf_io_mixin.py
sed -i '/ModelFilter/d' nemo/core/classes/mixins/hf_io_mixin.py

echo "🔄 Rebuilding NeMo..."
pip install -e . --no-deps

echo "🔧 Auto-adding ai4bharat-nemo to PYTHONPATH..."
echo "export PYTHONPATH=\"\$PYTHONPATH:$(pwd)\"" >> ../nemo/bin/activate

echo "🧪 Final test (will use RTX 3050 GPU)..."
python -c '
import torch
print("✅ CUDA available:", torch.cuda.is_available())
print("✅ GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
import nemo
print("✅ NeMo version:", nemo.__version__)
import nemo.collections.asr as nemo_asr
print("✅ ASR imported successfully")
'

cd ..

echo "✅ Setup complete for your RTX 3050 laptop!"
echo ""
echo "=== HOW TO RUN FROM NOW ON ==="
echo "1. Activate (only once per terminal):"
echo "   source nemo/bin/activate"
echo ""
echo "2. Run your script normally (uses GPU automatically):"
echo "   python ai4bharat-gu.py"
echo ""
echo "3. Login once for the gated model:"
echo "   huggingface-cli login"