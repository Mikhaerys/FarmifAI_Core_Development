# ==============================================================================
# Script de Instalación de Dependencias para FarmifAI (Windows PowerShell)
# ==============================================================================
# Este script crea un entorno virtual de Python, instala PyTorch con soporte CUDA
# e intenta configurar llama-cpp-python.

$ErrorActionPreference = "Stop"

Write-Host "=== 1. Creando entorno virtual de Python ===" -ForegroundColor Green
python -m venv venv
Write-Host "Activando entorno virtual..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host "=== 2. Actualizando pip, setuptools y wheel ===" -ForegroundColor Green
python -m pip install --upgrade pip setuptools wheel

Write-Host "=== 3. Detectando GPU NVIDIA ===" -ForegroundColor Green
$hasNvidia = $false
try {
    & nvidia-smi | Out-Null
    $hasNvidia = $true
    Write-Host "✅ GPU NVIDIA detectada." -ForegroundColor Green
} catch {
    Write-Host "⚠️ No se detectó GPU NVIDIA o nvidia-smi no está en el PATH. Se instalará para CPU." -ForegroundColor Yellow
}

Write-Host "=== 4. Instalando PyTorch ===" -ForegroundColor Green
if ($hasNvidia) {
    Write-Host "Instalando PyTorch con soporte CUDA 12.1..." -ForegroundColor Green
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "Instalando PyTorch para CPU..." -ForegroundColor Green
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

Write-Host "=== 5. Instalando requerimientos generales ===" -ForegroundColor Green
pip install -r requirements.txt

Write-Host "=== 6. Instalando llama-cpp-python ===" -ForegroundColor Green
if ($hasNvidia) {
    Write-Host "Intentando compilar llama-cpp-python con soporte CUDA..." -ForegroundColor Green
    Write-Host "Nota: Requiere Visual Studio Build Tools con C++ y CUDA Toolkit instalado." -ForegroundColor Yellow
    
    $env:CMAKE_ARGS = "-DGGML_CUDA=on"
    try {
        pip cache remove llama_cpp_python
        pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
        Write-Host "✅ llama-cpp-python compilado con CUDA correctamente." -ForegroundColor Green
    } catch {
        Write-Host "❌ Error compilando con CUDA. Instalando versión CPU por defecto..." -ForegroundColor Red
        $env:CMAKE_ARGS = ""
        pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
    }
} else {
    Write-Host "Instalando llama-cpp-python para CPU..." -ForegroundColor Green
    pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
}

Write-Host "=== 7. Nota sobre Unsloth en Windows ===" -ForegroundColor Green
Write-Host "⚠️ Unsloth (usado para entrenamiento) no se soporta de forma nativa en Windows fuera de WSL2." -ForegroundColor Yellow
Write-Host "Si planeas entrenar el modelo (FarmifAI_LoRA_Training.ipynb), se recomienda usar una máquina Linux" -ForegroundColor Yellow
Write-Host "en la nube (VPS con GPU) o configurar Windows Subsystem for Linux (WSL2)." -ForegroundColor Yellow

Write-Host "`n==============================================================================" -ForegroundColor Green
Write-Host "🎉 ¡Instalación completada!" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green
Write-Host "Para activar el entorno virtual y arrancar Jupyter, ejecuta:" -ForegroundColor Green
Write-Host "  .\\venv\\Scripts\\Activate.ps1" -ForegroundColor Green
Write-Host "  jupyter notebook" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green
