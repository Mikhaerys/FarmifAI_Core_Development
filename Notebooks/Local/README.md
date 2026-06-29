# 🌱 FarmifAI - Notebooks Preparados para Ejecución Local y VPS

Esta carpeta contiene la colección de notebooks de **FarmifAI** adaptados para funcionar de manera agnóstica tanto en **Google Colab** como en tu **computadora local** o un **servidor en la nube** (VPS o GPU rental como RunPod, Vast.ai, Lambda Labs, etc.) ejecutando Linux.

---

## 📂 Contenido del Directorio

*   **`dataset_generator.ipynb`**: Generador automático de datasets agrícolas usando la API de DeepSeek.
*   **`evaluacion_dataset_SLM.ipynb`**: Evaluador estadístico y semántico de la calidad de datasets en formato ChatML.
*   **`FarmifAI_LoRA_Training.ipynb`**: Fine-tuning optimizado de Qwen 3.5 con LoRA usando la librería Unsloth.
*   **`rag_hybrid_reranker.ipynb`**: Pipeline de RAG híbrido (BM25 + Semántica) y reranking con Cross-Encoder.
*   **`rag_slm_colab.ipynb`**: Asistente conversacional con interfaz de Gradio que integra el RAG y el modelo SLM a través de `llama.cpp`.

---

## ⚡ Instalación de Dependencias

### Opción A: Servidores en la Nube / VPS (Linux) - Recomendado para GPUs
Si estás rentando una GPU en la nube (ej: RunPod con una T4, L4, RTX 3090, etc.) o corriendo en un servidor Linux:

1.  Dale permisos de ejecución al script:
    ```bash
    chmod +x install_deps.sh
    ```
2.  Ejecuta el script para configurar el entorno virtual e instalar todas las dependencias:
    ```bash
    ./install_deps.sh
    ```
3.  Activa el entorno virtual:
    ```bash
    source venv/bin/activate
    ```
4.  Inicia Jupyter Server para que sea accesible desde tu navegador externo:
    ```bash
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
    ```

### Opción B: Computadora Local (Windows)
1.  Abre PowerShell en esta carpeta.
2.  Ejecuta el script de PowerShell (asegúrate de tener Python 3.10+ instalado):
    ```powershell
    .\install_deps.ps1
    ```
3.  Activa el entorno virtual:
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
4.  Inicia Jupyter:
    ```powershell
    jupyter notebook
    ```

---

## 🔧 Detalles y Adaptaciones Técnicas Realizadas

1.  **Detección Automática de Entornos**: Todos los notebooks detectan si se están ejecutando en Colab (`IN_COLAB`). Si están en Colab, siguen permitiendo descargas al navegador y cargas mediante widgets web. Si están en local, leen archivos directamente del disco y guardan los reportes localmente.
2.  **Parámetros de Interfaz**: En `rag_slm_colab.ipynb`, el método `demo.launch()` está configurado con `server_name="0.0.0.0"` al ejecutarse fuera de Colab. Esto te permitirá acceder a la UI de Gradio desde tu navegador web a través de la dirección IP pública del VPS (ej: `http://<IP-DE-TU-VPS>:7860`).
3.  **Prevención de Conflictos de Memoria CUDA**: El modelo de embeddings RAG ha sido optimizado para ejecutarse en CPU para consultas individuales. Esto previene el error crítico de GPU `cudaErrorIllegalAddress` que ocurría al compartir el contexto CUDA simultáneamente entre PyTorch (`sentence-transformers`) y C++ (`llama-cpp-python`).
4.  **No más cargas innecesarias**: Si los archivos necesarios (`dataset_agricola.jsonl`, `knowledge_base.json`, etc.) ya se encuentran en el directorio actual, los notebooks los cargarán automáticamente sin pedirte que los subas.

---

## ⚙️ Preparación de Datos Antes de Ejecutar

Para un funcionamiento fluido local:
*   Coloca el archivo de base de conocimientos (`knowledge_base.json` o `chunks_agricolas.json`) en esta misma carpeta.
*   En `dataset_generator.ipynb`, asegúrate de ingresar tu API key de DeepSeek en la sección de configuración.
