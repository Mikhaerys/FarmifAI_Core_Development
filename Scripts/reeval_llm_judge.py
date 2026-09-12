#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚜 FarmifAI: Script de Reevaluación y Reparación de Métricas de LLM Judge
(Faithfulness, Answer Relevancy y G-Eval)

Este script permite reparar o completar de manera eficiente las métricas evaluadas
por un LLM Judge en archivos JSONL (ej. `eval_progress.jsonl`, `eval_results.jsonl`),
solucionando fallos de red, timeouts o rate limits de OpenRouter/DeepSeek.

Características clave:
1. Reevaluación Granular e Independiente:
   - Si solo falló la Llamada 1 (Faithfulness / Relevancy), solo reevalúa la Llamada 1.
   - Si solo falló la Llamada 2 (G-Eval), solo reevalúa la Llamada 2.
   - Preserva intactas las llamadas que ya tuvieron éxito y todas las métricas locales
     (generación, ROUGE, BLEU, Flesch, Coseno, BERTScore, NLI, SelfCheckGPT).
2. Detección Inteligente de Fallos:
   - Detecta banderas explícitas ('FAILED_API' o valores None).
   - Permite usar la bandera `--fix_zeros` para detectar y reparar archivos previos
     donde los fallos de la API fueron grabados erróneamente con valor 0.0.
3. Resiliencia de Red:
   - Backoff exponencial con jitter (hasta 5 reintentos automáticos).
   - Inspección defensiva de respuestas nulas o vacías de la API de OpenRouter.
   - Extracción robusta de JSON tolerante a markdown (```json ... ```).

Uso:
    python Scripts/reeval_llm_judge.py --input eval_progress.jsonl --provider openrouter --fix_zeros
"""

import os
import sys
import json
import argparse
import time
import random
import re
from typing import Optional, Dict, Any, Tuple

# Importaciones condicionales para permitir tests unitarios ligeros
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", **kwargs):
        print(f"[INFO] {desc}...")
        return iterable


def log_message(msg: str):
    """Escribe un mensaje en la consola respetando barras de progreso de tqdm si están presentes."""
    if hasattr(tqdm, "write"):
        tqdm.write(msg)
    else:
        print(msg)


def extract_json_from_response(content: str) -> Optional[Dict[str, Any]]:
    """Extrae y parsea un objeto JSON de una cadena de texto, tolerando markdown."""
    if not content:
        return None
    content_clean = content.strip()
    # Eliminar bloques de código markdown ```json ... ``` si existen
    if "```" in content_clean:
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", content_clean, re.DOTALL)
        if match:
            content_clean = match.group(1).strip()

    # Si aún no es JSON directo, buscar el bloque exterior { ... }
    if not (content_clean.startswith("{") and content_clean.endswith("}")):
        match = re.search(r"(\{.*\})", content_clean, re.DOTALL)
        if match:
            content_clean = match.group(1).strip()

    try:
        return json.loads(content_clean)
    except json.JSONDecodeError:
        return None


def init_judge_client(provider: str, api_key: str):
    """Inicializa el cliente compatible con OpenAI para DeepSeek u OpenRouter."""
    if OpenAI is None:
        print("[ERROR] Falta el paquete 'openai'. Instálalo con: pip install openai")
        sys.exit(1)

    if provider.lower() == "deepseek":
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    elif provider.lower() == "openrouter":
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/FarmifAI",
                "X-Title": "FarmifAI LLM Judge Evaluator",
            }
        )
    else:
        raise ValueError(
            f"Proveedor no soportado: '{provider}'. Usa 'openrouter' o 'deepseek'.")


def call_llm_judge(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 30.0
) -> Optional[Dict[str, Any]]:
    """Ejecuta una llamada al LLM Judge con validación defensiva y reintentos exponenciales."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            # Validación defensiva ante respuestas truncadas o vacías de OpenRouter
            if not response or not hasattr(response, "choices") or response.choices is None or len(response.choices) == 0:
                err_msg = getattr(response, "error",
                                  None) or "response.choices es None o vacía"
                raise ValueError(f"API retornó respuesta inválida ({err_msg})")

            choice = response.choices[0]
            if not hasattr(choice, "message") or choice.message is None:
                raise ValueError("Respuesta de la API no contiene 'message'")

            content = choice.message.content
            if not content:
                raise ValueError(
                    "Contenido devuelto por el LLM Judge está vacío (None)")

            parsed = extract_json_from_response(content)
            if not parsed:
                raise ValueError(
                    f"No se pudo decodificar JSON válido del texto: {content[:100]}...")

            return parsed

        except Exception as e:
            delay = min(base_delay * (2 ** attempt) +
                        random.uniform(0.5, 1.5), max_delay)
            if attempt < max_retries - 1:
                log_message(
                    f"  [RETRY {attempt+1}/{max_retries}] Error en LLM Judge ({type(e).__name__}: {e}). Reintentando en {delay:.1f}s...")
                time.sleep(delay)
            else:
                log_message(
                    f"  [ERROR] LLM Judge falló tras {max_retries} intentos: {e}")
                return None


def evaluate_fr(client, model: str, question: str, knowledge: str, gen_answer: str, max_retries: int = 5) -> Optional[Dict[str, Any]]:
    """Ejecuta Llamada 1: Faithfulness & Answer Relevancy."""
    prompt_sys = (
        "Eres un evaluador experto en IA agrícola. Analiza la respuesta generada en base al contexto y la pregunta.\n"
        "Debes responder ESTRICTAMENTE en formato JSON con dos campos:\n"
        "1. 'faithfulness': puntaje de 0 a 5 indicando qué tanto de lo afirmado está respaldado por el contexto (5=100% respaldado, 0=contradicción total o invención).\n"
        "2. 'relevancy': puntaje de 0 a 5 indicando qué tan directa y completa es la respuesta a la pregunta del usuario (5=excelente y directa, 0=evasiva/irrelevante).\n"
        "Formato JSON esperado: {\"faithfulness\": <int 0-5>, \"relevancy\": <int 0-5>, \"reasoning\": \"<breve explicación>\"}"
    )
    prompt_usr = f"Pregunta: {question}\n\nContexto (<knowledge>):\n{knowledge}\n\nRespuesta Generada:\n{gen_answer}"
    res = call_llm_judge(client, model, prompt_sys,
                         prompt_usr, max_retries=max_retries)
    if res and ("faithfulness" in res or "relevancy" in res):
        return {
            "faithfulness": max(0.0, min(5.0, float(res.get("faithfulness", 0.0)))),
            "answer_relevancy": max(0.0, min(5.0, float(res.get("relevancy", 0.0)))),
            "reasoning": str(res.get("reasoning", ""))
        }
    return None


def evaluate_geval(client, model: str, question: str, gen_answer: str, max_retries: int = 5) -> Optional[Dict[str, Any]]:
    """Ejecuta Llamada 2: G-Eval (Coherence, Consistency, Fluency)."""
    prompt_sys = (
        "Eres un juez evaluador riguroso utilizando el framework G-Eval. Evalúa la respuesta en 3 dimensiones en escala de 1 a 5:\n"
        "1. 'coherence': Flujo lógico y organización estructural de la respuesta (1=desorganizada, 5=perfectamente estructurada).\n"
        "2. 'consistency': Ausencia de contradicciones lógicas internas o con los hechos agrícolas (1=contradictoria, 5=altamente consistente).\n"
        "3. 'fluency': Calidad gramatical en español, claridad y tono adaptado para un agricultor (1=incomprensible/robótico, 5=español natural, claro y excelente).\n"
        "Realiza un análisis paso a paso (CoT) y responde ESTRICTAMENTE en JSON con esta estructura:\n"
        "{\"coherence\": {\"score\": <int 1-5>, \"steps\": \"<análisis>\"}, \"consistency\": {\"score\": <int 1-5>, \"steps\": \"<análisis>\"}, \"fluency\": {\"score\": <int 1-5>, \"steps\": \"<análisis>\"}}"
    )
    prompt_usr = f"Pregunta: {question}\n\nRespuesta Generada:\n{gen_answer}"
    res = call_llm_judge(client, model, prompt_sys,
                         prompt_usr, max_retries=max_retries)
    if res and ("coherence" in res or "consistency" in res or "fluency" in res):
        def _get_score(data, key):
            val = data.get(key)
            if isinstance(val, dict):
                return float(val.get("score", 0.0))
            elif isinstance(val, (int, float)):
                return float(val)
            return 0.0

        return {
            "geval_coherence": max(1.0, min(5.0, _get_score(res, "coherence"))),
            "geval_consistency": max(1.0, min(5.0, _get_score(res, "consistency"))),
            "geval_fluency": max(1.0, min(5.0, _get_score(res, "fluency"))),
        }
    return None


def should_reevaluate_fr(metrics: Dict[str, Any], fix_zeros: bool, force_all: bool) -> bool:
    """Determina si la Llamada 1 (Faithfulness & Relevancy) necesita reevaluación."""
    if force_all:
        return True
    if metrics.get("judge_fr_status") == "FAILED_API":
        return True
    faith = metrics.get("faithfulness")
    rel = metrics.get("answer_relevancy")
    if faith is None or rel is None:
        return True
    if fix_zeros and (faith == 0.0 and rel == 0.0):
        return True
    return False


def should_reevaluate_geval(metrics: Dict[str, Any], fix_zeros: bool, force_all: bool) -> bool:
    """Determina si la Llamada 2 (G-Eval) necesita reevaluación."""
    if force_all:
        return True
    if metrics.get("judge_geval_status") == "FAILED_API":
        return True
    coh = metrics.get("geval_coherence")
    cons = metrics.get("geval_consistency")
    flue = metrics.get("geval_fluency")
    if coh is None or cons is None or flue is None:
        return True
    # En G-Eval la escala válida es 1 a 5. Cualquier valor de 0.0 indica fallo de evaluación
    if fix_zeros and (coh == 0.0 or cons == 0.0 or flue == 0.0):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="FarmifAI: Reevaluación y Reparación Resiliente de LLM Judge"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Ruta al archivo JSONL de resultados a evaluar/reparar.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Ruta del archivo de salida. Por defecto se sobreescribe o crea sufijo _judge_fixed.jsonl")
    parser.add_argument("--provider", "-p", type=str, default="openrouter",
                        choices=["openrouter", "deepseek"],
                        help="Proveedor del LLM Judge (por defecto 'openrouter').")
    parser.add_argument("--api_key", "-k", type=str, default=None,
                        help="API Key del proveedor. Si no se pasa, se lee de variables de entorno o se solicita.")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Modelo del juez (por defecto 'nvidia/nemotron-3-ultra-550b-a55b:free' para openrouter o 'deepseek-v4-flash' para deepseek).")
    parser.add_argument("--fix_zeros", action="store_true",
                        help="Si se activa, trata los valores 0.0 de LLM Judge en respuestas no vacías como fallos de API a reparar.")
    parser.add_argument("--force_all", action="store_true",
                        help="Fuerza la reevaluación de todas las muestras sin importar su estado previo.")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Número máximo de reintentos con backoff exponencial por llamada (por defecto 5).")
    parser.add_argument("--delay_between_samples", type=float, default=0.5,
                        help="Pausa de cortesía en segundos entre muestras para evitar saturar rate limits (por defecto 0.5s).")

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[ERROR] No existe el archivo de entrada: '{input_path}'")
        sys.exit(1)

    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_judge_fixed{ext}"

    # Resolver API Key
    api_key = args.api_key
    if not api_key:
        env_var = "OPENROUTER_API_KEY" if args.provider == "openrouter" else "DEEPSEEK_API_KEY"
        api_key = os.getenv(env_var) or os.getenv("JUDGE_API_KEY")
    if not api_key:
        import getpass
        print(
            f"\n🔑 Por favor ingresa tu API Key para {args.provider.upper()}:")
        api_key = getpass.getpass()

    # Resolver Modelo
    model = args.model
    if not model:
        if args.provider == "openrouter":
            model = "nvidia/nemotron-3-ultra-550b-a55b:free"
        else:
            model = "deepseek-v4-flash"

    print("=" * 75)
    print(" 🚜 FARMIF-AI: REEVALUADOR RESILIENTE DE LLM JUDGE")
    print("=" * 75)
    print(f"📂 Archivo Entrada: {input_path}")
    print(f"💾 Archivo Salida : {output_path}")
    print(f"🤖 Proveedor      : {args.provider.upper()} (Modelo: {model})")
    print(f"🔁 Máx Reintentos : {args.max_retries} con Backoff Exponencial")
    print(
        f"🩹 Reparar 0.0s   : {'SÍ (Activado)' if args.fix_zeros else 'NO (Solo FAILED_API o None)'}")
    print(f"⚡ Forzar Todos   : {'SÍ' if args.force_all else 'NO'}")
    print("=" * 75)

    client = init_judge_client(args.provider, api_key)

    # 1. Cargar registros
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                print(
                    f"[WARNING] Línea {line_idx} inválida en {input_path}, omitida. ({e})")

    print(
        f"\n[INFO] {len(records)} registros cargados. Analizando necesidades de reevaluación...")

    # Identificar casos que requieren acción
    pending_fr = []
    pending_geval = []
    for i, r in enumerate(records):
        metrics = r.get("metrics", {})
        gen_ans = r.get("generated_answer", "")
        # Si la respuesta generada está vacía, no tiene sentido reevaluar con LLM Judge
        if not gen_ans.strip():
            continue
        if should_reevaluate_fr(metrics, args.fix_zeros, args.force_all):
            pending_fr.append(i)
        if should_reevaluate_geval(metrics, args.fix_zeros, args.force_all):
            pending_geval.append(i)

    print(
        f"  • Muestras pendientes de Llamada 1 (Faithfulness / Relevancy): {len(pending_fr)}")
    print(
        f"  • Muestras pendientes de Llamada 2 (G-Eval Coherence/Cons/Flue): {len(pending_geval)}")

    all_pending_indices = sorted(list(set(pending_fr + pending_geval)))
    print(
        f"  • Total de muestras únicas que requieren atención: {len(all_pending_indices)} / {len(records)}")

    if not all_pending_indices:
        print("\n✨ ¡Todas las muestras ya cuentan con métricas válidas de LLM Judge! No se requiere reevaluación.")
        return

    # 2. Bucle de Reevaluación
    fixed_fr_count = 0
    fixed_geval_count = 0
    failed_fr_count = 0
    failed_geval_count = 0

    print("\n🚀 Iniciando reevaluación selectiva...")
    for idx in tqdm(all_pending_indices, desc="Reevaluando LLM Judge"):
        rec = records[idx]
        metrics = rec.setdefault("metrics", {})
        question = rec.get("question", "")
        knowledge = rec.get("knowledge", "")
        gen_ans = rec.get("generated_answer", "")

        # Ejecutar Llamada 1 si corresponde
        if idx in pending_fr:
            res_fr = evaluate_fr(
                client, model, question, knowledge, gen_ans, max_retries=args.max_retries)
            if res_fr:
                metrics["faithfulness"] = res_fr["faithfulness"]
                metrics["answer_relevancy"] = res_fr["answer_relevancy"]
                metrics["judge_fr_status"] = "SUCCESS"
                fixed_fr_count += 1
            else:
                metrics["judge_fr_status"] = "FAILED_API"
                if "faithfulness" not in metrics or metrics["faithfulness"] == 0.0:
                    metrics["faithfulness"] = None
                if "answer_relevancy" not in metrics or metrics["answer_relevancy"] == 0.0:
                    metrics["answer_relevancy"] = None
                failed_fr_count += 1

        # Ejecutar Llamada 2 si corresponde
        if idx in pending_geval:
            res_geval = evaluate_geval(
                client, model, question, gen_ans, max_retries=args.max_retries)
            if res_geval:
                metrics["geval_coherence"] = res_geval["geval_coherence"]
                metrics["geval_consistency"] = res_geval["geval_consistency"]
                metrics["geval_fluency"] = res_geval["geval_fluency"]
                metrics["judge_geval_status"] = "SUCCESS"
                fixed_geval_count += 1
            else:
                metrics["judge_geval_status"] = "FAILED_API"
                if "geval_coherence" not in metrics or metrics["geval_coherence"] == 0.0:
                    metrics["geval_coherence"] = None
                if "geval_consistency" not in metrics or metrics["geval_consistency"] == 0.0:
                    metrics["geval_consistency"] = None
                if "geval_fluency" not in metrics or metrics["geval_fluency"] == 0.0:
                    metrics["geval_fluency"] = None
                failed_geval_count += 1

        # Pausa leve para proteger rate limits
        if args.delay_between_samples > 0:
            time.sleep(args.delay_between_samples)

    # 3. Guardar registros actualizados
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 4. Resumen final
    print("\n" + "=" * 75)
    print(" 🏁 ¡PROCESO DE REEVALUACIÓN Y REPARACIÓN FINALIZADO!")
    print("=" * 75)
    print(f"📄 Archivo actualizado: {output_path}")
    print(
        f"✅ Llamada 1 (Faithfulness / Relevancy) reparadas con éxito: {fixed_fr_count} (Fallidas: {failed_fr_count})")
    print(
        f"✅ Llamada 2 (G-Eval) reparadas con éxito                 : {fixed_geval_count} (Fallidas: {failed_geval_count})")

    judge_cols = ["faithfulness", "answer_relevancy",
                  "geval_coherence", "geval_consistency", "geval_fluency"]
    print("\n📊 PROMEDIOS Y COBERTURA DE MÉTRICAS LLM JUDGE (Sin Contaminación de Ceros):")
    print(f"{'Métrica':<25} | {'Válidos':<8} | {'Nulos/Fallos':<12} | {'Promedio':<10}")
    print("-" * 65)
    for col in judge_cols:
        vals = [r.get("metrics", {}).get(col)
                for r in records if r.get("metrics", {}).get(col) is not None]
        null_count = len(records) - len(vals)
        mean_val = (sum(vals) / len(vals)) if vals else float("nan")
        print(f"{col:<25} | {len(vals):<8} | {null_count:<12} | {mean_val:.2f}")

    print("=" * 75)


if __name__ == "__main__":
    main()
