"""
Presidio demo for the compliance-ticket-classifier pipeline.

Кроки:
  1. Custom recognizer   — те, чого немає з коробки (internal user ID)
  2. Analyzer            — що саме знайдено в тексті тікета (з score)
  3. Anonymizer          — чотири оператори на ОДНОМУ тексті
  4. Sanitized logging   — Presidio перед записом у predictions.jsonl
  5. Scan gold.parquet   — наївний прохід vs строгий; false positives

Запуск (у venv-presidio, з будь-якої директорії):
    python research/presidio_demo.py

Крок 5 потребує data/labeled/gold.parquet + pandas/pyarrow:
    dvc pull data/labeled/gold.parquet   (в основному venv)
    pip install pandas pyarrow           (у venv-presidio)
"""
import json
from pathlib import Path

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Шлях рахуємо від файлу, а не від cwd — тож cd не потрібен.
REPO_ROOT = Path(__file__).resolve().parent.parent   # research/ -> корінь репо
GOLD = REPO_ROOT / "data" / "labeled" / "gold.parquet"

# Тікети у схемі gold.parquet (issue_id, source, text, ...)
TICKETS = [
    {
        "issue_id": "COMP-142",
        "source": "jira",
        "text": (
            "Store user email and IP address in the audit log table. "
            "Reported by Kateryna Dubas (k.dubas@setuniversity.edu.ua), "
            "affected user UID-88231 from 192.168.14.77."
        ),
    },
    {
        "issue_id": "COMP-155",
        "source": "jira",
        "text": (
            "Add user data deletion endpoint for GDPR compliance. "
            "Contact the DPO at dpo@example.org before purging UID-40119."
        ),
    },
    {
        "issue_id": "COMP-160",
        "source": "github",
        "text": "Fix memory leak in the etcd client connection pool",  # PII тут НЕМАЄ
    },
]

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Усі сутності, які шукаємо "наївно" (як у README Presidio).
ENTITIES = [
    "EMAIL_ADDRESS", "IP_ADDRESS", "PERSON",
    "PHONE_NUMBER", "INTERNAL_USER_ID",
]

# Строгий набір: лише те, що має жорсткий regex/валідатор,
# а не NER-здогадку. Плюс поріг упевненості.
STRICT_ENTITIES = ["EMAIL_ADDRESS", "INTERNAL_USER_ID"]
STRICT_THRESHOLD = 0.8

# ======================================================================
print("=" * 70)
print("КРОК 1, CUSTOM RECOGNIZER: до і після")
print("=" * 70)

probe = TICKETS[0]["text"]

# ДО: Presidio нічого не знає про наші внутрішні UID.
# Увага: він не повертає порожній список, а кидає ValueError.
try:
    before = analyzer.analyze(text=probe, entities=["INTERNAL_USER_ID"], language="en")
    print(f"\nДО реєстрації    -> знайдено INTERNAL_USER_ID: {len(before)}")
except ValueError as e:
    print(f"\nДО реєстрації    -> ValueError: {e}")

# Реєструємо власний розпізнавач.
# score=0.9 у патерні, але context enhancer підніме його до 1.00,
# бо поруч зі знахідкою стоїть слово "user".
uid_recognizer = PatternRecognizer(
    supported_entity="INTERNAL_USER_ID",
    patterns=[Pattern(name="uid", regex=r"\bUID-\d{5}\b", score=0.9)],
    context=["user", "uid", "account"],
)
analyzer.registry.add_recognizer(uid_recognizer)

after = analyzer.analyze(text=probe, entities=["INTERNAL_USER_ID"], language="en")
print(f"ПІСЛЯ реєстрації -> знайдено INTERNAL_USER_ID: {len(after)}")
for f in after:
    print(f"   {f.entity_type:<18} score={f.score:.2f}  -> {probe[f.start:f.end]!r}")


def find_pii(text: str, entities=None):
    """Знайти PII. Порожній список, якщо нічого (або якщо recognizer відсутній)."""
    try:
        return analyzer.analyze(
            text=text, entities=entities or ENTITIES, language="en"
        )
    except ValueError:
        # fail-loud за замовчуванням -> у проді гасимо, щоб не покласти /predict
        return []


# ======================================================================
print("\n" + "=" * 70)
print("КРОК 2, ANALYZER: що знайдено")
print("=" * 70)

for t in TICKETS:
    findings = find_pii(t["text"])
    print(f"\n[{t['issue_id']}] {t['text'][:60]}...")
    if not findings:
        print("   (PII не знайдено)")
    for f in sorted(findings, key=lambda x: x.start):
        value = t["text"][f.start:f.end]
        print(f"   {f.entity_type:<18} score={f.score:.2f}  -> {value!r}")

# ======================================================================
print("\n" + "=" * 70)
print("КРОК 3, ANONYMIZER: чотири оператори на одному тексті")
print("=" * 70)

sample = TICKETS[0]["text"]
findings = find_pii(sample)

strategies = {
    "replace (плейсхолдер)": {
        "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
    },
    "mask (часткове)": {
        "DEFAULT": OperatorConfig(
            "mask", {"masking_char": "*", "chars_to_mask": 12, "from_end": False}
        ),
    },
    "hash (стабільний, незворотний)": {
        "DEFAULT": OperatorConfig("hash", {"hash_type": "sha256"}),
    },
    "per-entity (по-різному для кожного типу)": {
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
        "IP_ADDRESS": OperatorConfig("replace", {"new_value": "<IP>"}),
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "INTERNAL_USER_ID": OperatorConfig("hash", {"hash_type": "sha256"}),
    },
}

print(f"\nОРИГІНАЛ:\n   {sample}\n")
for name, ops in strategies.items():
    out = anonymizer.anonymize(text=sample, analyzer_results=findings, operators=ops)
    print(f"{name}:\n   {out.text}\n")

# ======================================================================
print("=" * 70)
print("КРОК 4, САНІТИЗОВАНЕ ЛОГУВАННЯ (те, що йде у predictions.jsonl)")
print("=" * 70)

SAFE_OPS = {
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
    "IP_ADDRESS": OperatorConfig("replace", {"new_value": "<IP>"}),
    "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
    "INTERNAL_USER_ID": OperatorConfig("hash", {"hash_type": "sha256"}),
}


def sanitize(text: str) -> str:
    """Прибрати PII перед тим, як текст покине сервіс."""
    findings = find_pii(text)
    if not findings:
        return text
    return anonymizer.anonymize(
        text=text, analyzer_results=findings, operators=SAFE_OPS
    ).text


# Імітація log_prediction() з src/serve/app.py
print()
for t in TICKETS[:2]:
    record = {
        "issue_id": t["issue_id"],
        "text": sanitize(t["text"]),   # <-- єдиний доданий рядок
        "label": "relevant",
        "confidence": 0.8398,
    }
    print(json.dumps(record, ensure_ascii=False)[:150] + " ...")

# ======================================================================
print("\n" + "=" * 70)
print("КРОК 5, СКАН РЕАЛЬНОГО ДАТАСЕТУ (gold.parquet)")
print("=" * 70)

if not GOLD.exists():
    print(f"\n{GOLD} не знайдено. Спочатку: dvc pull data/labeled/gold.parquet")
    raise SystemExit(0)

import pandas as pd  # noqa: E402  (імпорт тут, щоб крок 5 був опційним)

df = pd.read_parquet(GOLD)
n = len(df)
print(f"\nЗавантажено {n} тікетів\n")

# --- Прохід А: наївний (усі сутності, будь-який score) ---
naive_hits = 0
entity_counts = {}
examples = []

for row in df.itertuples():
    findings = find_pii(row.text)
    if not findings:
        continue
    naive_hits += 1
    for f in findings:
        entity_counts[f.entity_type] = entity_counts.get(f.entity_type, 0) + 1
    if len(examples) < 3:
        examples.append((row.issue_id, row.text, findings))

# --- Прохід Б: строгий (тільки regex-сутності + поріг score) ---
strict_hits = 0
for row in df.itertuples():
    findings = find_pii(row.text, entities=STRICT_ENTITIES)
    if any(f.score >= STRICT_THRESHOLD for f in findings):
        strict_hits += 1

naive_share = naive_hits / n * 100
strict_share = strict_hits / n * 100

print("ПРОХІД А: наївний (усі сутності, будь-який score):")
print(f"   тікетів з 'PII': {naive_hits} з {n}  ({naive_share:.1f}%)\n")
print("   Знайдені сутності (всього входжень):")
for ent, cnt in sorted(entity_counts.items(), key=lambda x: -x[1]):
    print(f"      {ent:<18} {cnt}")

print("\n   Приклади (перші 3) — ДИВІТЬСЯ, ЩО САМЕ ЗНАЙДЕНО:")
for issue_id, text, findings in examples:
    print(f"\n   [{issue_id}] {text[:60].replace(chr(10), ' ')}...")
    for f in sorted(findings, key=lambda x: x.start)[:6]:
        print(f"      {f.entity_type:<18} score={f.score:.2f}  -> {text[f.start:f.end]!r}")

print("\n" + "-" * 70)
print("ПРОХІД Б: строгий (тільки EMAIL_ADDRESS + INTERNAL_USER_ID, score >= 0.8):")
print(f"   тікетів з PII: {strict_hits} з {n}  ({strict_share:.1f}%)")

print("\n" + "=" * 70)
print("ВИСНОВОК")
print("=" * 70)
print(f"""
Наївно:  {naive_share:.1f}%  ({naive_hits}/{n})
Строго:  {strict_share:.1f}%  ({strict_hits}/{n})
""")