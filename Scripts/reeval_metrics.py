#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚜 FarmifAI: Script de Reevaluación Rápida de Métricas (Lexicales, Semánticas y Cross-Encoder NLI)

Este script permite reevaluar de manera eficiente las respuestas que ya fueron generadas y evaluadas en un archivo JSONL previo
(ej. `eval_progress.jsonl` o `Eval_results_example.jsonl`).

Características clave:
1. Re-calcula únicamente las métricas que dependen del texto y de modelos locales (Format Adherence, ROUGE, BLEU, Flesch Español, Similitud Coseno y el nuevo NLI por oraciones multilingüe con `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`).
2. PRESERVA INTACTAS las métricas costosas/lentas obtenidas por LLM as a Judge (`faithfulness`, `answer_relevancy`, `geval_coherence`, `geval_consistency`, `geval_fluency`) y `selfcheck_consistency`.
3. Muestra un reporte estadístico comparativo (Antes vs. Después) para analizar el impacto de la corrección del NLI.

Uso en Google Colab o Terminal Local:
    python Scripts/reeval_metrics.py --input output/JSON/Eval_results_example.jsonl --output output/JSON/Eval_results_example_reevaluated.jsonl
"""

import os
import sys
import json
import argparse
import re
import time
from collections import Counter

# Intentar importar librerías necesarias con mensajes amigables si faltan
try:
    import numpy as np
    import torch
    from rouge_score import rouge_scorer
    import sacrebleu
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import nltk
except ImportError as e:
    print(f"[ERROR] Falta una dependencia necesaria para la evaluación: {e}")
    print("Por favor instala las dependencias ejecutando:")
    print("pip install numpy torch rouge-score sacrebleu scikit-learn sentence-transformers nltk tqdm pandas")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", **kwargs):
        print(f"[INFO] {desc}...")
        return iterable

# Descargar diccionarios NLTK de forma silenciosa
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# =====================================================================
# DEFINICIÓN DE MÉTRICAS Y UTILIDADES
# =====================================================================

def eval_format_adherence(text):
    if not text:
        return 0.0
    reasoning_matches = re.findall(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if len(reasoning_matches) == 1 and len(answer_matches) == 1:
        return 1.0 if text.find('<reasoning>') < text.find('<answer>') else 0.0
    return 0.0


def extract_generated_answer(text):
    if not text:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def count_syllables_es(word):
    word = word.lower()
    vowels = 'aeiouáéíóúü'
    return max(1, sum(1 for char in word if char in vowels))


def eval_flesch_reading_ease_es(text):
    """
    Índice de Legibilidad de Flesch adaptado al Español (Fórmula de Fernández Huerta).
    > 60: Fácil / Claramente legible por agricultores
    < 50: Complejo / Técnico
    """
    if not text:
        return 0.0
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\w+', text)
    if not words or not sentences:
        return 0.0

    syllables = sum(count_syllables_es(w) for w in words)
    score = 206.84 - 60.0 * (syllables / len(words)) - 1.02 * (len(words) / len(sentences))
    return round(max(0.0, min(100.0, score)), 2)


def eval_rouge_bleu(rouge_evaluator, ref_answer, gen_answer):
    if not gen_answer or not ref_answer:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

    scores = rouge_evaluator.score(ref_answer, gen_answer)
    try:
        bleu = sacrebleu.sentence_bleu(gen_answer, [ref_answer]).score
    except Exception:
        bleu = 0.0

    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure,
        "bleu": bleu
    }


def eval_cosine_similarity(embed_model, ref_answer, gen_answer):
    if not gen_answer or not ref_answer:
        return 0.0
    embs = embed_model.encode([ref_answer, gen_answer], show_progress_bar=False)
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])


def eval_nli_relation_claim_level(nli_model, knowledge, gen_answer):
    """
    Evaluación de Natural Language Inference (Cross-Encoder) a nivel de oración (Claim-Level NLI).
    Soluciona el problema de los modelos de documento que marcan como 'neutral' el texto completo
    por culpa de saludos conversacionales ("Claro que sí, mire...") o explicaciones empíricas adicionales.
    """
    if not knowledge or not gen_answer:
        return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0, "label": "neutral"}

    # Mapeo dinámico id2label de la configuración del modelo
    id2label = getattr(nli_model.model.config, "id2label", {0: "entailment", 1: "neutral", 2: "contradiction"})
    label_map = {str(v).lower(): int(k) for k, v in id2label.items()}

    # Descomponer gen_answer en oraciones independientes
    try:
        sentences = [s.strip() for s in nltk.tokenize.sent_tokenize(gen_answer, language="spanish") if len(s.strip()) > 10]
    except Exception:
        sentences = [s.strip() for s in re.split(r'[.!?]+', gen_answer) if len(s.strip()) > 10]
    
    if not sentences:
        sentences = [gen_answer.strip()]

    pairs = [(knowledge, s) for s in sentences]
    all_logits = nli_model.predict(pairs, show_progress_bar=False)
    
    if len(sentences) == 1 and all_logits.ndim == 1:
        all_logits = np.expand_dims(all_logits, axis=0)

    # Softmax para obtener probabilidades
    exp_logits = np.exp(all_logits - np.max(all_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    ent_idx = label_map.get("entailment", 0)
    con_idx = label_map.get("contradiction", 2)
    neu_idx = label_map.get("neutral", 1)

    ent_probs = probs[:, ent_idx]
    con_probs = probs[:, con_idx]
    neu_probs = probs[:, neu_idx]

    max_con = float(np.max(con_probs))
    mean_ent = float(np.mean(ent_probs))
    max_ent = float(np.max(ent_probs))
    mean_neu = float(np.mean(neu_probs))

    # Lógica de clasificación factual informada para RAG:
    # Si al menos una afirmación entra en contradicción grave, lo marcamos como contradiction.
    # Si hay alto soporte en el promedio o máximo sin contradicción, marcamos entailment.
    if max_con > 0.5:
        top_label = "contradiction"
    elif mean_ent > 0.35 or max_ent > 0.65:
        top_label = "entailment"
    else:
        top_label = "neutral"

    return {
        "contradiction": max_con,
        "entailment": mean_ent,
        "neutral": mean_neu,
        "label": top_label
    }


# =====================================================================
# FUNCIÓN PRINCIPAL DE REEVALUACIÓN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="FarmifAI: Reevaluación de Métricas Lexicales y NLI Multilingüe")
    parser.add_argument("--input", "-i", type=str, default="output/JSON/Eval_results_example.jsonl",
                        help="Ruta al archivo JSONL que contiene los resultados ya evaluados.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Ruta del archivo JSONL de salida. Si no se especifica, se creará uno con sufijo '_reevaluated.jsonl'.")
    parser.add_argument("--nli_model", type=str, default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                        help="Nombre o ruta del modelo Cross-Encoder NLI multilingüe en Hugging Face.")
    parser.add_argument("--embed_model", type=str, default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="Nombre del modelo SentenceTransformer para embeddings y similitud coseno.")
    
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[ERROR] El archivo de entrada '{input_path}' no existe.")
        sys.exit(1)

    if args.output is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_reevaluated{ext}"
    else:
        output_path = os.path.abspath(args.output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 75)
    print(" 🚜 FarmifAI: REEVALUACIÓN DE MÉTRICAS (Sin re-ejecutar LLM Judges)")
    print("=" * 75)
    print(f"📂 Archivo de entrada : {input_path}")
    print(f"📁 Archivo de salida  : {output_path}")
    print(f"🌐 Modelo NLI         : {args.nli_model}")
    print(f"📐 Modelo Embeddings  : {args.embed_model}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Dispositivo GPU/CPU: {device.upper()}")
    print("-" * 75)

    print("[INFO] Cargando modelos locales...")
    start_time = time.time()
    embed_model = SentenceTransformer(args.embed_model, device=device)
    nli_model = CrossEncoder(args.nli_model, device=device)
    rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    print(f"✅ Modelos cargados en {time.time() - start_time:.2f} segundos.")
    print("-" * 75)

    # Cargar registros existentes
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception as e:
                    print(f"[WARNING] Línea {line_num} ignorada por error JSON: {e}")

    print(f"[INFO] Total de muestras a reevaluar: {len(records)}")

    # Seguimiento de estadísticas Antes vs Después
    old_nli_counts = Counter()
    new_nli_counts = Counter()
    nli_transitions = Counter()
    old_flesch = []
    new_flesch = []

    reevaluated_records = []

    for rec in tqdm(records, desc="Reevaluando muestras"):
        # 1. Preservar campos base del registro
        idx = rec.get("sample_index", 0)
        question = rec.get("question", "")
        knowledge = rec.get("knowledge", "")
        reference_answer = rec.get("reference_answer", "")
        generated_full = rec.get("generated_full", "")
        generated_answer = rec.get("generated_answer", "")
        if not generated_answer and generated_full:
            generated_answer = extract_generated_answer(generated_full)

        old_metrics = rec.get("metrics", {})

        # Registrar estadísticas del estado anterior
        old_label = old_metrics.get("nli_label", "unknown")
        old_nli_counts[old_label] += 1
        if "flesch_reading_ease" in old_metrics:
            old_flesch.append(old_metrics["flesch_reading_ease"])

        # 2. Recalcular métricas rápidas (Lexicales / Semánticas / NLI)
        fmt_score = eval_format_adherence(generated_full)
        rb_scores = eval_rouge_bleu(rouge_evaluator, reference_answer, generated_answer)
        flesch_score = eval_flesch_reading_ease_es(generated_answer)
        cos_score = eval_cosine_similarity(embed_model, reference_answer, generated_answer)
        nli_res = eval_nli_relation_claim_level(nli_model, knowledge, generated_answer)

        new_label = nli_res["label"]
        new_nli_counts[new_label] += 1
        nli_transitions[(old_label, new_label)] += 1
        new_flesch.append(flesch_score)

        # 3. Consolidar diccionario de métricas
        # SE PRESERVAN exactos los valores de selfcheck y LLM judges si existen
        updated_metrics = {
            "format_adherence": fmt_score,
            "rouge1": rb_scores["rouge1"],
            "rouge2": rb_scores["rouge2"],
            "rougeL": rb_scores["rougeL"],
            "bleu": rb_scores["bleu"],
            "flesch_reading_ease": flesch_score,
            "cosine_similarity": cos_score,
            "nli_entailment": nli_res["entailment"],
            "nli_contradiction": nli_res["contradiction"],
            "nli_neutral": nli_res["neutral"],
            "nli_label": new_label,
            # Métricas preservadas sin gastar API ni re-muestreo
            "selfcheck_consistency": old_metrics.get("selfcheck_consistency", 0.0),
            "faithfulness": old_metrics.get("faithfulness", 0.0),
            "answer_relevancy": old_metrics.get("answer_relevancy", 0.0),
            "geval_coherence": old_metrics.get("geval_coherence", 0.0),
            "geval_consistency": old_metrics.get("geval_consistency", 0.0),
            "geval_fluency": old_metrics.get("geval_fluency", 0.0)
        }

        # Actualizar registro
        new_rec = dict(rec)
        new_rec["generated_answer"] = generated_answer
        new_rec["metrics"] = updated_metrics
        reevaluated_records.append(new_rec)

    # Escribir archivo de salida en tiempo real
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in reevaluated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # =====================================================================
    # REPORTE EJECUTIVO Y COMPARATIVO
    # =====================================================================
    print("\n" + "=" * 75)
    print(" 🏁 ¡REEVALUACIÓN COMPLETADA EXITOSAMENTE!")
    print("=" * 75)
    print(f"📄 Registros guardados en: {output_path}")
    print("\n📊 COMPARATIVA DE ETIQUETAS NLI (ANTES vs. DESPUÉS):")
    all_labels = ["entailment", "neutral", "contradiction"]
    print(f"{'Etiqueta':<15} | {'Antes (Monolingüe)':<20} | {'Después (Multilingüe Claim-Level)':<30}")
    print("-" * 70)
    for lbl in all_labels:
        old_c = old_nli_counts[lbl]
        new_c = new_nli_counts[lbl]
        old_pct = (old_c / max(1, len(records))) * 100
        new_pct = (new_c / max(1, len(records))) * 100
        print(f"{lbl.capitalize():<15} | {old_c:<6} ({old_pct:>5.1f}%)      | {new_c:<6} ({new_pct:>5.1f}%)")

    print("\n🔄 MATRIZ DE TRANSICIÓN DE ETIQUETAS:")
    for (old_l, new_l), count in nli_transitions.most_common():
        if old_l != new_l:
            print(f"  • {count} muestra(s) pasaron de [{old_l.upper()}]  ──>  [{new_l.upper()}]")
        else:
            print(f"  • {count} muestra(s) se mantuvieron como [{old_l.upper()}]")

    if old_flesch and new_flesch:
        print("\n📖 CAMBIO EN ÍNDICE DE LEGIBILIDAD FLESCH:")
        print(f"  • Promedio anterior (Fórmula en Inglés) : {np.mean(old_flesch):.2f}")
        print(f"  • Promedio nuevo (Fernández Huerta ES)  : {np.mean(new_flesch):.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
