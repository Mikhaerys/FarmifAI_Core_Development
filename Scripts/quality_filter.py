import json
import re
import hashlib
from pathlib import Path
from datetime import datetime

from dateutil import parser as dateparser
from dateutil.parser import ParserError

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Archivo de entrada: el resultado de merge.py ({"chunks": [...]})
input_file = Path("...")

# Carpeta de salida (se crea si no existe)
output_folder = Path("...")

# Archivos de salida
clean_output_file = output_folder / "chunks_limpios.json"          # aceptados, listos para usar
revision_output_file = output_folder / "chunks_para_revision.json"  # necesitan ojo humano
discarded_output_file = output_folder / "chunks_descartados.json"  # duplicados / estructura irreparable
report_file = output_folder / "reporte_clasificacion.json"         # resumen + detalle de motivos

# Formato estándar de fecha del dataset (ISO 8601)
DATE_FORMAT_FULL = "%Y-%m-%d"     # 2020-05-14
DATE_FORMAT_YEAR_MONTH = "%Y-%m"  # 2020-05
DATE_FORMAT_YEAR = "%Y"           # 2020

required_fields = ["document_id", "citation", "date", "text"]


# ============================================================
# 1. FECHAS: normalizar a formato estándar
# ============================================================

MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def normalize_date(raw_date):
    """
    Intenta normalizar una fecha a formato estándar del dataset.
    Devuelve (fecha_normalizada, granularidad, ok) donde ok indica si
    se pudo interpretar con confianza.
    granularidad: 'day' | 'month' | 'year' | None
    """
    if raw_date is None:
        return None, None, False

    text = str(raw_date).strip()
    if not text or text.lower() in ("s.f.", "sin fecha", "n/a", "na", "desconocida"):
        return None, None, False

    # Caso: solo año (4 dígitos)
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        if 1500 <= year <= datetime.now().year + 1:
            return f"{year:04d}", "year", True
        return text, None, False

    # Caso: "15 de enero de 2020" / "enero de 2020" (formatos en español)
    text_lower = text.lower()
    match_full_es = re.search(
        r"(\d{1,2})\s+de\s+([a-záéíóúñ]+)\s+de\s+(\d{4})", text_lower
    )
    if match_full_es:
        day, month_name, year = match_full_es.groups()
        month = MESES_ES.get(month_name)
        if month:
            try:
                dt = datetime(int(year), month, int(day))
                return dt.strftime(DATE_FORMAT_FULL), "day", True
            except ValueError:
                pass

    match_month_es = re.search(r"([a-záéíóúñ]+)\s+de\s+(\d{4})", text_lower)
    if match_month_es:
        month_name, year = match_month_es.groups()
        month = MESES_ES.get(month_name)
        if month:
            return f"{int(year):04d}-{month:02d}", "month", True

    # Caso: usar dateutil para formatos numéricos (dd/mm/yyyy, yyyy-mm-dd, etc.)
    for dayfirst in (True, False):
        try:
            dt = dateparser.parse(text, dayfirst=dayfirst, fuzzy=False)
            # Validar que el resultado sea razonable
            if dt and 1500 <= dt.year <= datetime.now().year + 1:
                return dt.strftime(DATE_FORMAT_FULL), "day", True
        except (ParserError, ValueError, OverflowError):
            continue

    # No se pudo interpretar con confianza
    return text, None, False


# ============================================================
# 2. DUPLICADOS EXACTOS
# ============================================================

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


# ============================================================
# 3. TABLAS IRRECONSTRUIBLES O CON ERRORES DE FORMATO
# ============================================================

def detect_broken_table(text: str):
    """
    Heurística: detecta señales de tablas mal extraídas / mal formateadas.
    No corrige nada, solo marca para revisión con el motivo detectado.
    """
    issues = []
    if not text:
        return issues

    lines = text.split("\n")

    # a) Filas con "|" (markdown-style) con número de columnas inconsistente
    pipe_lines = [l for l in lines if l.count("|") >= 2]
    if len(pipe_lines) >= 2:
        col_counts = {l.count("|") for l in pipe_lines}
        if len(col_counts) > 1:
            issues.append("Filas tipo tabla con número de columnas inconsistente (posible tabla rota).")

    # b) Se menciona "tabla"/"cuadro" pero el contenido parece texto corrido sin estructura
    if re.search(r"\b(tabla|cuadro)\b", text, flags=re.IGNORECASE):
        # muchas cifras sueltas separadas por espacios múltiples sugieren columnas colapsadas
        collapsed_columns = re.findall(r"\d+(?:[.,]\d+)?\s{2,}\d+(?:[.,]\d+)?", text)
        if len(collapsed_columns) >= 2:
            issues.append("Se menciona tabla/cuadro y hay patrones de columnas colapsadas (espacios múltiples entre cifras).")

    # c) Muchas líneas cortas consecutivas con solo números/símbolos (extracción rota de tabla)
    short_numeric_lines = sum(
        1 for l in lines if l.strip() and len(l.strip()) < 15 and re.search(r"\d", l)
    )
    if short_numeric_lines >= 5:
        issues.append("Múltiples líneas cortas con números sueltos, típico de una tabla mal extraída.")

    return issues


# ============================================================
# 4. REFERENCIAS A FIGURAS/TABLAS SIN CONTEXTO
# ============================================================

def detect_unlinked_references(text: str):
    """
    Heurística: detecta menciones a "Figura X" / "Tabla X" / "Cuadro X" / "Gráfico X"
    sin suficiente contenido descriptivo alrededor (sugiere que al chunk le falta contexto).
    """
    issues = []
    if not text:
        return issues

    pattern = re.compile(
        r"\b(figura|fig\.?|tabla|cuadro|gr[aá]fic[oa])\s*\.?\s*n?[°º]?\s*\d+",
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return issues

    for m in matches:
        start, end = m.span()
        window_after = text[end:end + 150].strip()
        window_before = text[max(0, start - 150):start].strip()
        # Si casi no hay texto descriptivo antes/después de la referencia, se marca
        if len(window_after) < 20 and len(window_before) < 20:
            issues.append(f"Referencia '{m.group(0).strip()}' aparece sin contexto descriptivo suficiente alrededor.")

    return issues


# ============================================================
# 5. PALABRAS MALFORMADAS
# ============================================================

MOJIBAKE_MARKERS = ["Ã©", "Ã¡", "Ã³", "Ã±", "Â", "�", "Ã\x81", "Ã\x93"]


def detect_malformed_words(text: str):
    """
    Heurística: detecta artefactos de encoding roto, palabras cortadas por guion
    de salto de línea, y palabras con mezclas anómalas de letras/dígitos.
    """
    issues = []
    if not text:
        return issues

    # a) Artefactos de encoding (mojibake)
    for marker in MOJIBAKE_MARKERS:
        if marker in text:
            issues.append(f"Posible artefacto de encoding roto detectado ('{marker}').")
            break

    # b) Palabras cortadas por guion de salto de línea, ej: "docu-\nmento"
    if re.search(r"[a-záéíóúñ]-\s*\n\s*[a-záéíóúñ]", text, flags=re.IGNORECASE):
        issues.append("Se detectaron palabras posiblemente cortadas por guion de salto de línea (ej: 'docu-\\nmento').")

    # c) Palabras con dígitos incrustados en medio de letras (ej: 'pala8ra')
    if re.search(r"\b[a-záéíóúñ]+\d+[a-záéíóúñ]+\b", text, flags=re.IGNORECASE):
        issues.append("Se detectaron palabras con dígitos incrustados entre letras (posible OCR erróneo).")

    # d) Secuencias de consonantes muy largas sin vocales (poco probable en español)
    if re.search(r"\b[bcdfghjklmnñpqrstvwxyz]{6,}\b", text, flags=re.IGNORECASE):
        issues.append("Se detectaron palabras con secuencias largas de consonantes sin vocales (posible palabra malformada).")

    return issues


# ============================================================
# 6. VALIDACIÓN / REPARACIÓN DE ESTRUCTURA JSON DEL CHUNK
# ============================================================

def validate_and_fix_structure(chunk, index):
    """
    Revisa que el chunk tenga los campos requeridos con el tipo esperado.
    Intenta reparar problemas menores (campos faltantes opcionales, tipos
    simples). Devuelve (chunk_reparado, es_reparable, motivos).
    """
    motivos = []

    if not isinstance(chunk, dict):
        return chunk, False, ["El chunk no es un objeto JSON."]

    missing = [f for f in required_fields if f not in chunk]
    if missing:
        return chunk, False, [f"Campos faltantes irreparables: {missing}"]

    # 'text' debe ser string no vacío
    if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
        return chunk, False, ["El campo 'text' está vacío o no es texto."]

    # 'document_id' y 'citation' deben poder representarse como string
    for field in ("document_id", "citation"):
        if not isinstance(chunk.get(field), (str, int, float)):
            return chunk, False, [f"El campo '{field}' tiene un tipo no soportado ({type(chunk.get(field))})."]
        if isinstance(chunk[field], (int, float)):
            chunk[field] = str(chunk[field])
            motivos.append(f"Campo '{field}' convertido a texto.")

    return chunk, True, motivos


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def load_chunks(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "chunks" not in data:
        raise ValueError("El archivo de entrada no tiene la clave 'chunks'.")
    return data["chunks"]


def main():
    output_folder.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(input_file)
    print(f"Chunks de entrada: {len(chunks)}")

    accepted = []
    revision = []
    discarded = []

    seen_hashes = set()

    for idx, chunk in enumerate(chunks, start=1):
        record_motivos = []
        status = "aceptado"

        # 1) Estructura
        chunk, es_reparable, motivos_estructura = validate_and_fix_structure(chunk, idx)
        record_motivos.extend(motivos_estructura)
        if not es_reparable:
            discarded.append({
                "chunk_index": idx,
                "motivo": "estructura_irreparable",
                "detalle": motivos_estructura,
                "chunk": chunk,
            })
            continue

        # 2) Duplicados exactos (dentro de este archivo)
        h = text_hash(chunk["text"])
        if h in seen_hashes:
            discarded.append({
                "chunk_index": idx,
                "motivo": "duplicado_exacto",
                "detalle": ["El texto normalizado ya existe en un chunk anterior de este archivo."],
                "chunk": chunk,
            })
            continue
        seen_hashes.add(h)

        # 3) Fechas: normalizar
        normalized_date, granularity, date_ok = normalize_date(chunk.get("date"))
        if date_ok:
            if normalized_date != chunk.get("date"):
                record_motivos.append(f"Fecha normalizada de '{chunk.get('date')}' a '{normalized_date}' (granularidad: {granularity}).")
            chunk["date"] = normalized_date
        else:
            status = "revision"
            record_motivos.append(f"No fue posible normalizar la fecha con confianza: '{chunk.get('date')}'.")

        # 4) Tablas irreconstruibles
        table_issues = detect_broken_table(chunk["text"])
        if table_issues:
            status = "revision"
            record_motivos.extend(table_issues)

        # 5) Referencias a figuras/tablas sin contexto
        ref_issues = detect_unlinked_references(chunk["text"])
        if ref_issues:
            status = "revision"
            record_motivos.extend(ref_issues)

        # 6) Palabras malformadas
        word_issues = detect_malformed_words(chunk["text"])
        if word_issues:
            status = "revision"
            record_motivos.extend(word_issues)

        entry = {
            "chunk_index": idx,
            "document_id": chunk.get("document_id"),
            "motivos": record_motivos,
            "chunk": chunk,
        }

        if status == "aceptado":
            accepted.append(chunk)
        else:
            entry["motivo"] = "requiere_revision"
            revision.append(entry)

    # Renumerar chunk_number consecutivo solo en los aceptados (dataset limpio final)
    for i, chunk in enumerate(accepted, start=1):
        chunk["chunk_number"] = i

    # ============================================================
    # GUARDAR SALIDAS
    # ============================================================

    with open(clean_output_file, "w", encoding="utf-8") as f:
        json.dump({"chunks": accepted}, f, ensure_ascii=False, indent=2)

    with open(revision_output_file, "w", encoding="utf-8") as f:
        json.dump({"chunks_para_revision": revision}, f, ensure_ascii=False, indent=2)

    with open(discarded_output_file, "w", encoding="utf-8") as f:
        json.dump({"chunks_descartados": discarded}, f, ensure_ascii=False, indent=2)

    reporte = {
        "resumen": {
            "chunks_entrada": len(chunks),
            "aceptados": len(accepted),
            "para_revision": len(revision),
            "descartados": len(discarded),
        },
        "criterios_aplicados": [
            "Fechas inconsistentes o mal codificadas -> normalizadas a formato estándar (ISO 8601) cuando fue posible; si no, marcado para revisión.",
            "Duplicados exactos -> se conserva el primero, el resto se descarta.",
            "Tablas irreconstruibles o con errores de formato -> marcado para revisión (heurística).",
            "Referencias a figuras/tablas sin contexto -> marcado para revisión (heurística).",
            "Palabras malformadas -> marcado para revisión (heurística).",
            "Estructura JSON inválida (campos faltantes/tipos no soportados) -> descartado si es irreparable.",
        ],
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    print("\n===================================")
    print("FILTRO DE CALIDAD COMPLETADO")
    print("===================================")
    print("Chunks de entrada:", len(chunks))
    print("Aceptados (limpios):", len(accepted))
    print("Para revisión:", len(revision))
    print("Descartados:", len(discarded))
    print("\nArchivo limpio final:", clean_output_file)
    print("Archivo para revisión:", revision_output_file)
    print("Archivo descartados:", discarded_output_file)
    print("Reporte:", report_file)


if __name__ == "__main__":
    main()
