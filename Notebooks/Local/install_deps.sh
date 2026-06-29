#!/bin/bash
# ==============================================================================
# Script de Instalación de Dependencias para FarmifAI (Linux / VPS / GPU Rental)
# ==============================================================================
# Este script crea un entorno virtual de Python, instala PyTorch con soporte CUDA
# y compila llama-cpp-python con aceleración GPU (CUDA). También instala Unsloth.

set -e

echo "=== 1. Creando entorno virtual de Python ==="
python3 -m venv venv
source venv/bin/activate

echo "=== 2. Actualizando pip, setuptools y wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== 3. Detectando hardware y CUDA ==="
HAS_NVIDIA=false
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    HAS_NVIDIA=true
    echo "✅ GPU NVIDIA detectada."
else
    echo "⚠️ No se detectó GPU NVIDIA. Se instalará para ejecución en CPU."
fi

# Detectar versión de CUDA
CUDA_VERSION=""
if [ "$HAS_NVIDIA" = true ]; then
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
        echo "✅ Versión de CUDA detectada (nvcc): $CUDA_VERSION"
    else
        # Intentar extraer de nvidia-smi
        CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
        echo "✅ Versión de CUDA detectada (nvidia-smi): $CUDA_VERSION"
    fi
fi

echo "=== 4. Instalando PyTorch ==="
if [ "$HAS_NVIDIA" = true ]; then
    # Por defecto instalamos PyTorch compatible con CUDA 12.1/12.4
    echo "Instalando PyTorch con soporte CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Instalando PyTorch para CPU..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "=== 5. Instalando requerimientos generales ==="
pip install -r requirements.txt

echo "=== 6. Instalando llama-cpp-python ==="
if [ "$HAS_NVIDIA" = true ]; then
    echo "Compilando llama-cpp-python con soporte CUDA..."
    # Limpiar caché de compilación previa
    pip cache remove llama_cpp_python || true
    # Compilar con soporte CUDA
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
else
    echo "Instalando llama-cpp-python para CPU..."
    pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
fi

echo "=== 7. Instalando Unsloth (solo si hay GPU NVIDIA) ==="
if [ "$HAS_NVIDIA" = true ]; then
    echo "Instalando Unsloth y dependencias de optimización..."
    # Instalar bitsandbytes y dependencias requeridas
    pip install bitsandbytes xformers --no-cache-dir
    
    # Intentar instalar la versión correspondiente de Unsloth
    # Si la versión de CUDA es 12.4 o superior
    if [[ "$CUDA_VERSION" == "12.4"* ]] || [[ "$CUDA_VERSION" == "12.5"* ]]; then
        echo "Instalando Unsloth para CUDA 12.4..."
        pip install "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"
    else
        echo "Instalando Unsloth para CUDA 12.1 (Default)..."
        pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
    fi
    
    # Dependencias adicionales para métricas de evaluación
    pip install trl transformers
else
    echo "⚠️ Omitiendo Unsloth ya que requiere GPU NVIDIA para entrenamiento."
fi

echo "=============================================================================="
echo "🎉 ¡Instalación completada con éxito!"
echo "=============================================================================="
echo "Para activar el entorno virtual y arrancar Jupyter, ejecuta:"
echo "  source venv/bin/activate"
echo "  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser"
echo "=============================================================================="
