# export_laion_tsv.py
from datasets import load_dataset
import csv

DATASET = "laion/laion2B-en-aesthetic"  # example; replace with what you use
SPLIT = "train"
OUT_TSV = "laion2B/laion_urls_captions.tsv"
N = 1000000  # how many samples you want to download

ds = load_dataset(DATASET, split=SPLIT, streaming=True)

with open(OUT_TSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["url", "caption"])
    for i, ex in enumerate(ds):
        # Column names vary a bit across LAION subsets; try these common ones:
        url = ex.get("url") or ex.get("image_url") or ex.get("URL") or ex.get("imageURL")
        cap = ex.get("text") or ex.get("caption") or ex.get("TEXT") or ex.get("caption_en")
        if not url or not cap:
            continue
        writer.writerow([url, cap])
        if (i + 1) >= N:
            break

print(f"Wrote {OUT_TSV}")
