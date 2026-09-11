import io
import urllib.request
import unicodedata
from datasets import load_dataset

def normalize_text(text):
    """Normalize Yorùbá text to keep diacritics combined cleanly."""
    if text is None: return ""
    return unicodedata.normalize('NFC', text.strip())

def download_raw_tsv(url):
    """Downloads raw TSV/TXT lines directly from GitHub/HF via HTTP."""
    print(f"📡 Downloading raw data from: {url}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        return response.read().decode('utf-8').splitlines()

# Setup containers
train_pairs = []
val_pairs = []
test_pairs = []

# ==========================================
# 1. DOWNLOAD MENYO-20k (Via Raw Git Data)
# ==========================================
print("⏳ Fetching MENYO-20k...")
menyo_base_url = "https://githubusercontent.com"

# MENYO Train Split
for line in download_raw_tsv(f"{menyo_base_url}/train.tsv"):
    parts = line.split('\t')
    if len(parts) >= 2:
        train_pairs.append((normalize_text(parts[0]), normalize_text(parts[1])))

# MENYO Dev Split
for line in download_raw_tsv(f"{menyo_base_url}/dev.tsv"):
    parts = line.split('\t')
    if len(parts) >= 2:
        val_pairs.append((normalize_text(parts[0]), normalize_text(parts[1])))

# MENYO Test Split
for line in download_raw_tsv(f"{menyo_base_url}/test.tsv"):
    parts = line.split('\t')
    if len(parts) >= 2:
        test_pairs.append((normalize_text(parts[0]), normalize_text(parts[1])))


# ==========================================
# 2. DOWNLOAD MAFAND-MT (Via Raw Git Data)
# ==========================================
print("\n⏳ Fetching MAFAND-MT (Masakhane)...")
mafand_base_url = "https://githubusercontent.com"

# Process MAFAND Train
en_lines = download_raw_tsv(f"{mafand_base_url}/train.en")
yo_lines = download_raw_tsv(f"{mafand_base_url}/train.yo")
for en, yo in zip(en_lines, yo_lines):
    train_pairs.append((normalize_text(en), normalize_text(yo)))

# Process MAFAND Dev
en_lines = download_raw_tsv(f"{mafand_base_url}/dev.en")
yo_lines = download_raw_tsv(f"{mafand_base_url}/dev.yo")
for en, yo in zip(en_lines, yo_lines):
    val_pairs.append((normalize_text(en), normalize_text(yo)))

# Process MAFAND Test
en_lines = download_raw_tsv(f"{mafand_base_url}/test.en")
yo_lines = download_raw_tsv(f"{mafand_base_url}/test.yo")
for en, yo in zip(en_lines, yo_lines):
    test_pairs.append((normalize_text(en), normalize_text(yo)))


# ==========================================
# 3. LOAD FLORES-200 (Stable JSON Native)
# ==========================================
print("\n⏳ Fetching FLORES-200 (Native Loading)...")
# FLORES does not use python scripts to load, so it never triggers the runtime error
flores = load_dataset("facebook/flores", "eng_Latn-yor_Latn")

for item in flores['dev']:
    val_pairs.append((normalize_text(item['sentence_eng_Latn']), normalize_text(item['sentence_yor_Latn'])))
for item in flores['devtest']:
    test_pairs.append((normalize_text(item['sentence_eng_Latn']), normalize_text(item['sentence_yor_Latn'])))


# ==========================================
# SUMMARY REPORT
# ==========================================
print("\n🚀 Pipeline Loading Complete!")
print(f"Total Aggregated Training Pairs:   {len(train_pairs):,}")
print(f"Total Aggregated Validation Pairs: {len(val_pairs):,}")
print(f"Total Aggregated Test Pairs:       {len(test_pairs):,}")

# Sample Verification printout
if train_pairs:
    print("\nSample Training Entry:")
    print(f"  [EN]: {train_pairs[150][0]}")
    print(f"  [YO]: {train_pairs[150][1]}")
