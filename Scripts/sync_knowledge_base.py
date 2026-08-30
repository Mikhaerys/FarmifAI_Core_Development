#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚜 FarmifAI: Sincronización de Chunks de Conocimiento (Dataset -> Knowledge Base)

Este script:
1. Lee uno a uno los registros de un dataset JSONL (ej. `dataset_agricola.jsonl`).
2. Extrae el fragmento de conocimiento contenido dentro de las etiquetas `<knowledge>...</knowledge>`.
3. Busca dicho texto en la base de conocimientos (`knowledge_base.json`) mediante comparación robusta
   (coincidencia exacta, normalización de espacios en blanco y normalización Unicode).
4. Si el chunk no se encuentra en la base de conocimientos, crea una nueva entrada con el siguiente
   `chunk_number` secuencial y lo agrega a la lista de chunks.
5. Permite realizar simulaciones (--dry-run), crear respaldos automáticos (--backup), y definir
   metadatos predeterminados para los chunks incorporados.

Uso:
    python Scripts/sync_knowledge_base.py
    python Scripts/sync_knowledge_base.py --dry-run
    python Scripts/sync_knowledge_base.py --dataset output/JSON/dataset_agricola.jsonl --kb output/JSON/knowledge_base.json
"""

import os
import sys
import json
import re
import shutil
import argparse
import unicodedata
from datetime import datetime
from typing import Dict, List, Tuple, Set, Any, Optional

# Asegurar codificación UTF-8 en consola de Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", **kwargs):
        print(f"[INFO] {desc}...")
        return iterable


def normalize_text_for_comparison(text: str) -> str:
    """
    Normaliza un texto para comparaciones robustas:
    - Normaliza caracteres Unicode a forma NFC.
    - Elimina espacios al inicio y final.
    - Colapsa secuencias de espacios en blanco / saltos de línea a un único espacio.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def extract_knowledge_chunks_from_dataset(dataset_path: str) -> List[Tuple[int, str]]:
    """
    Extrae los chunks contenidos dentro de las etiquetas <knowledge>...</knowledge>
    de cada línea del archivo dataset JSONL.

    Retorna una lista de tuplas: (número_línea, texto_chunk_limpio)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No se encontró el archivo del dataset: {dataset_path}")

    knowledge_pattern = re.compile(r"<knowledge>\s*(.*?)\s*</knowledge>", re.DOTALL)
    extracted_chunks = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                print(f"⚠️ [Línea {line_idx}] Error decodificando JSON: {e}")
                continue

            # Buscar en mensajes (formato OpenAI / ShareGPT / Anthropic)
            knowledge_text = None
            messages = data.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    match = knowledge_pattern.search(content)
                    if match:
                        knowledge_text = match.group(1).strip()
                        break

            # Si no está en 'messages', buscar directamente en la raíz o en campos comunes
            if knowledge_text is None:
                for field in ["user", "prompt", "input", "text", "content", "knowledge"]:
                    val = data.get(field)
                    if isinstance(val, str):
                        match = knowledge_pattern.search(val)
                        if match:
                            knowledge_text = match.group(1).strip()
                            break

            if knowledge_text:
                extracted_chunks.append((line_idx, knowledge_text))
            else:
                print(f"⚠️ [Línea {line_idx}] No se encontró etiqueta <knowledge> en el registro.")

    return extracted_chunks


def load_knowledge_base(kb_path: str) -> Dict[str, Any]:
    """
    Carga el archivo JSON de la base de conocimientos.
    """
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"No se encontró el archivo de la base de conocimientos: {kb_path}")

    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    if not isinstance(kb_data, dict) or "chunks" not in kb_data:
        raise ValueError(f"El archivo '{kb_path}' no tiene el formato esperado (debe contener la clave 'chunks').")

    return kb_data


def sync_chunks(
    kb_data: Dict[str, Any],
    dataset_chunks: List[Tuple[int, str]],
    default_doc_id: str = "dataset_agricola_extra",
    default_citation: str = "Dataset Agrícola - Documento complementario",
    default_date: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Compara uno a uno los chunks del dataset contra la base de conocimientos.
    Si un chunk no existe, lo crea y lo añade a la base de conocimientos.

    Retorna:
    - Lista de nuevos chunks agregados.
    - Diccionario con estadísticas de la operación.
    """
    chunks_list = kb_data["chunks"]

    # Determinar el chunk_number inicial más alto
    max_chunk_number = 0
    for c in chunks_list:
        c_num = c.get("chunk_number", 0)
        if isinstance(c_num, int) and c_num > max_chunk_number:
            max_chunk_number = c_num

    if max_chunk_number == 0 and len(chunks_list) > 0:
        max_chunk_number = len(chunks_list)

    # Construir conjunto de búsqueda rápida indexado por texto normalizado
    known_exact_texts: Set[str] = set()
    known_norm_texts: Set[str] = set()

    for c in chunks_list:
        txt = c.get("text", "")
        if txt:
            known_exact_texts.add(txt.strip())
            known_norm_texts.add(normalize_text_for_comparison(txt))

    if not default_date:
        default_date = datetime.now().strftime("%m %Y")

    new_chunks_added: List[Dict[str, Any]] = []
    found_count = 0
    added_count = 0

    print("🔍 Procesando y comparando chunks...")
    for line_idx, raw_chunk_text in tqdm(dataset_chunks, desc="Buscando en KB"):
        clean_text = raw_chunk_text.strip()
        norm_text = normalize_text_for_comparison(clean_text)

        # 1. Comprobar si ya existe en la KB original o fue agregado previamente en esta misma ejecución
        if clean_text in known_exact_texts or norm_text in known_norm_texts:
            found_count += 1
            continue

        # 2. Si es un chunk nuevo, construir el nuevo objeto chunk
        max_chunk_number += 1
        new_chunk = {
            "document_id": default_doc_id,
            "citation": default_citation,
            "date": default_date,
            "text": clean_text,
            "chunk_number": max_chunk_number
        }

        # Registrar en la lista de chunks de la base de conocimientos
        chunks_list.append(new_chunk)
        new_chunks_added.append(new_chunk)

        # Actualizar los índices en memoria para evitar duplicados si el mismo chunk
        # vuelve a aparecer más adelante en el dataset
        known_exact_texts.add(clean_text)
        known_norm_texts.add(norm_text)
        added_count += 1

    # Actualizar metadata si existe
    if "metadata_filtrado" in kb_data and isinstance(kb_data["metadata_filtrado"], dict):
        kb_data["metadata_filtrado"]["total_chunks_limpios"] = len(chunks_list)
        kb_data["metadata_filtrado"]["total_chunks_agregados_desde_dataset"] = added_count
        kb_data["metadata_filtrado"]["fecha_ultima_sincronizacion"] = datetime.now().isoformat()

    stats = {
        "total_dataset_items": len(dataset_chunks),
        "initial_kb_chunks": len(chunks_list) - added_count,
        "already_found_in_kb": found_count,
        "new_chunks_added": added_count,
        "final_kb_chunks": len(chunks_list),
        "final_max_chunk_number": max_chunk_number
    }

    return new_chunks_added, stats


def save_knowledge_base(kb_data: Dict[str, Any], output_path: str, backup: bool = True) -> None:
    """
    Guarda la base de conocimientos actualizada en formato JSON con indentación.
    Crea un respaldo .bak si el archivo ya existe y backup es True.
    """
    # Crear carpeta destino si no existe
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if backup and os.path.exists(output_path):
        backup_path = f"{output_path}.bak"
        print(f"💾 Creando respaldo en '{backup_path}'...")
        shutil.copy2(output_path, backup_path)

    print(f"💾 Guardando base de conocimientos actualizada en '{output_path}'...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)
    print("✅ Archivo guardado exitosamente.")


def main():
    parser = argparse.ArgumentParser(
        description="🚜 FarmifAI: Sincroniza chunks entre dataset_agricola.jsonl y knowledge_base.json"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="output/JSON/dataset_agricola.jsonl",
        help="Ruta al dataset JSONL con las etiquetas <knowledge> (por defecto: output/JSON/dataset_agricola.jsonl)"
    )
    parser.add_argument(
        "--kb",
        type=str,
        default="output/JSON/knowledge_base.json",
        help="Ruta al archivo knowledge_base.json (por defecto: output/JSON/knowledge_base.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta donde guardar el archivo JSON resultante (si se omite, sobreescribe --kb)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Desactiva la creación de un archivo de respaldo (.bak) antes de sobreescribir"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta la búsqueda y muestra estadísticas sin modificar ningún archivo"
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default="dataset_agricola_extra",
        help="Identificador de documento para los nuevos chunks (por defecto: 'dataset_agricola_extra')"
    )
    parser.add_argument(
        "--citation",
        type=str,
        default="Dataset Agrícola - Documento complementario",
        help="Cita bibliográfica para los nuevos chunks agregados"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Fecha para los nuevos chunks (por defecto: mes y año actual, ej. '06 2026')"
    )

    args = parser.parse_args()

    output_path = args.output if args.output is not None else args.kb

    print("=" * 70)
    print("🚜 FarmifAI - SINCRONIZADOR DE CHUNKS: DATASET -> KNOWLEDGE BASE")
    print("=" * 70)
    print(f"📄 Dataset entrada : {args.dataset}")
    print(f"📚 Knowledge Base  : {args.kb}")
    print(f"🎯 Archivo salida  : {output_path} {'[MODO DRY-RUN]' if args.dry_run else ''}")
    print("=" * 70)

    # 1. Extraer chunks del dataset
    print(f"📖 Leyendo dataset desde '{args.dataset}'...")
    extracted_chunks = extract_knowledge_chunks_from_dataset(args.dataset)
    print(f"✅ Se extrajeron {len(extracted_chunks)} etiquetas <knowledge> del dataset.")

    # 2. Cargar base de conocimientos
    print(f"\n📖 Leyendo base de conocimientos desde '{args.kb}'...")
    kb_data = load_knowledge_base(args.kb)
    initial_count = len(kb_data.get("chunks", []))
    print(f"✅ Base de conocimientos cargada con {initial_count} chunks iniciales.")

    # 3. Sincronizar
    new_chunks, stats = sync_chunks(
        kb_data=kb_data,
        dataset_chunks=extracted_chunks,
        default_doc_id=args.doc_id,
        default_citation=args.citation,
        default_date=args.date
    )

    # 4. Mostrar resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE LA SINCRONIZACIÓN")
    print("=" * 70)
    print(f"• Chunks leídos del dataset      : {stats['total_dataset_items']:,}")
    print(f"• Chunks iniciales en KB         : {stats['initial_kb_chunks']:,}")
    print(f"• Chunks encontrados ya en KB    : {stats['already_found_in_kb']:,}")
    print(f"• Nuevos chunks agregados a KB   : {stats['new_chunks_added']:,}")
    print(f"• Total chunks en KB resultante  : {stats['final_kb_chunks']:,}")
    print(f"• Último chunk_number asignado   : {stats['final_max_chunk_number']:,}")
    print("=" * 70)

    if new_chunks:
        print("\n🔍 Vista previa del primer chunk nuevo agregado:")
        sample = new_chunks[0]
        print(f"  [Chunk #{sample['chunk_number']}] Doc ID: {sample['document_id']}")
        preview_text = sample['text'][:180] + "..." if len(sample['text']) > 180 else sample['text']
        print(f"  Texto: {preview_text}")

        if len(new_chunks) > 1:
            print("\n🔍 Vista previa del último chunk nuevo agregado:")
            sample_last = new_chunks[-1]
            print(f"  [Chunk #{sample_last['chunk_number']}] Doc ID: {sample_last['document_id']}")
            preview_last = sample_last['text'][:180] + "..." if len(sample_last['text']) > 180 else sample_last['text']
            print(f"  Texto: {preview_last}")

    # 5. Guardar cambios si no es dry-run
    if args.dry_run:
        print("\nℹ️ [MODO DRY-RUN ACTIVADO] No se realizaron modificaciones en los archivos.")
    elif stats["new_chunks_added"] > 0:
        print("\n💾 Procediendo a guardar cambios...")
        save_knowledge_base(kb_data, output_path, backup=not args.no_backup)
        print("🎉 Proceso de sincronización completado con éxito.")
    else:
        print("\n✨ Todos los chunks del dataset ya existían en la base de conocimientos. No se requirió actualización.")


if __name__ == "__main__":
    main()
