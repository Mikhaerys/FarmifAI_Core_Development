#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚜 FarmifAI: Script de Reevaluación Específica para BERTScore (Precision, Recall, F1)

Este script permite añadir las métricas de BERTScore (Precision, Recall y F1 Score)
a un archivo de resultados previo en formato JSONL (ej. `eval_results.jsonl` o `eval_results_Qwen3.5.jsonl`).

Características clave:
1. Re-calcula únicamente las submétricas de BERTScore (Precision, Recall, F1 Score).
2. Resiliente ante respuestas vacías, nulas o versiones recientes de HuggingFace transformers (`use_fast_tokenizer=True`).
3. PRESERVA INTACTAS todas las demás métricas existentes.

Uso en Google Colab o Terminal Local:
    python Scripts/reeval_bertscore.py --input output/JSON/eval_results_Qwen3.5.jsonl
"""

import os
import sys
import json
import argparse
import time

# Intentar importar librerías necesarias
try:
    import torch
    import bert_score
    from bert_score import BERTScorer
except ImportError as e:
    print(f"[ERROR] Falta una dependencia necesaria para BERTScore: {e}")
    print("Por favor instala las dependencias ejecutando:")
    print("pip install bert-score torch tqdm pandas")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", **kwargs):
        print(f"[INFO] {desc}...")
        return iterable


def extract_generated_answer(text):
    if not text:
        return ""
    import re
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def main():
    parser = argparse.ArgumentParser(description="FarmifAI: Reevaluación e Inclusión de BERTScore (Precision, Recall, F1)")
    parser.add_argument("--input", "-i", type=str, default="eval_results.jsonl",
                        help="Ruta al archivo JSONL que contiene los resultados ya evaluados.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Ruta del archivo JSONL de salida. Si no se especifica, se creará uno con sufijo '_with_bertscore.jsonl'.")
    parser.add_argument("--lang", type=str, default="es",
                        help="Código de idioma para BERTScore (por defecto 'es').")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Nombre o ruta del modelo BERT HuggingFace (opcional, ej. 'dccuchile/bert-base-spanish-wwm-cased').")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[ERROR] El archivo de entrada '{input_path}' no existe.")
        sys.exit(1)

    if args.output is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_with_bertscore{ext}"
    else:
        output_path = os.path.abspath(args.output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 75)
    print(" 🚜 FarmifAI: REEVALUACIÓN Y ADICIÓN DE BERTSCORE (Precision, Recall, F1)")
    print("=" * 75)
    print(f"📂 Archivo de entrada : {input_path}")
    print(f"📁 Archivo de salida  : {output_path}")
    print(f"🌐 Idioma BERTScore   : {args.lang}")
    if args.model_type:
        print(f"🤖 Modelo BERT        : {args.model_type}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Dispositivo GPU/CPU: {device.upper()}")
    print("-" * 75)

    # Cargar registros
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception as e:
                    print(f"[WARNING] Línea {line_num} ignorada por error JSON: {e}")

    if not records:
        print("[ERROR] No se encontraron registros válidos en el archivo de entrada.")
        sys.exit(1)

    print(f"[INFO] Total de muestras a procesar: {len(records)}")

    # Cargar BERTScorer garantizando uso de fast tokenizer para compatibilidad con transformers >= 4.40
    print("[INFO] Cargando modelo de BERTScore...")
    start_time = time.time()
    try:
        kwargs = {"lang": args.lang, "device": device, "rescale_with_baseline": False, "use_fast_tokenizer": True}
        if args.model_type:
            kwargs["model_type"] = args.model_type
        scorer = BERTScorer(**kwargs)
        print(f"✅ BERTScorer cargado en {time.time() - start_time:.2f} segundos.")
    except Exception as e:
        print(f"[WARNING] No se pudo inicializar con use_fast_tokenizer=True ({e}). Reintentando con configuración por defecto...")
        try:
            kwargs.pop("use_fast_tokenizer", None)
            scorer = BERTScorer(**kwargs)
            print(f"✅ BERTScorer cargado en {time.time() - start_time:.2f} segundos.")
        except Exception as e2:
            print(f"[ERROR] No se pudo inicializar BERTScorer: {e2}")
            sys.exit(1)

    print("-" * 75)

    # Preparar pares de candidato y referencia garantizando textos no vacíos
    cands = []
    refs = []
    valid_mask = []

    for rec in records:
        generated_answer = rec.get("generated_answer", "")
        if not generated_answer and rec.get("generated_full", ""):
            generated_answer = extract_generated_answer(rec["generated_full"])
        
        reference_answer = rec.get("reference_answer", "")

        g_str = str(generated_answer).strip() if generated_answer else ""
        r_str = str(reference_answer).strip() if reference_answer else ""

        if g_str and r_str:
            cands.append(g_str)
            refs.append(r_str)
            valid_mask.append(True)
        else:
            # Texto sustituto válido para evitar errores de tokenización en bert_score
            cands.append("Sin respuesta")
            refs.append("Sin respuesta")
            valid_mask.append(False)

    # Calcular BERTScore en lote para alta eficiencia
    print("[INFO] Calculando BERTScore (Precision, Recall, F1) en lote...")
    b_start = time.time()
    P, R, F1 = scorer.score(cands, refs)
    b_duration = time.time() - b_start
    print(f"✅ Cálculo completado en {b_duration:.2f} segundos ({len(records)/max(0.001, b_duration):.1f} muestras/seg).")

    # Consolidar nuevos registros
    reevaluated_records = []
    p_vals, r_vals, f1_vals = [], [], []

    for i, rec in enumerate(records):
        if valid_mask[i]:
            p_score = float(P[i].item())
            r_score = float(R[i].item())
            f1_score = float(F1[i].item())
        else:
            p_score, r_score, f1_score = 0.0, 0.0, 0.0

        p_vals.append(p_score)
        r_vals.append(r_score)
        f1_vals.append(f1_score)

        # Actualizar diccionario de métricas preservando todas las anteriores
        new_rec = dict(rec)
        metrics = dict(new_rec.get("metrics", {}))

        metrics["bertscore_precision"] = round(p_score, 4)
        metrics["bertscore_recall"] = round(r_score, 4)
        metrics["bertscore_f1"] = round(f1_score, 4)

        new_rec["metrics"] = metrics
        reevaluated_records.append(new_rec)

    # Escribir archivo de salida
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in reevaluated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # =====================================================================
    # REPORTE DE RESULTADOS DE BERTSCORE
    # =====================================================================
    avg_p = sum(p_vals) / max(1, len(p_vals))
    avg_r = sum(r_vals) / max(1, len(r_vals))
    avg_f1 = sum(f1_vals) / max(1, len(f1_vals))

    print("\n" + "=" * 75)
    print(" 🏁 ¡REEVALUACIÓN DE BERTSCORE COMPLETADA CON ÉXITO!")
    print("=" * 75)
    print(f"📄 Registros guardados en: {output_path}")
    print("\n📊 PROMEDIOS DE SUBMÉTRICAS BERTSCORE OBTENIDOS:")
    print(f"  • BERTScore Precision : {avg_p:.4f} ({avg_p*100:.2f}%)")
    print(f"  • BERTScore Recall    : {avg_r:.4f} ({avg_r*100:.2f}%)")
    print(f"  • BERTScore F1 Score  : {avg_f1:.4f} ({avg_f1*100:.2f}%)")
    print("=" * 75)


if __name__ == "__main__":
    main()
