import json
import os
import glob

PAPERS_DIR = "research2/literature/papers"
META_DIR = "research2/literature/metadata"

with open(os.path.join(META_DIR, "downloaded_15_papers_metadata.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

# Deduplicate CBM and ensure exactly 5 per journal
final_papers = {}
for j_name in ["CBM", "BSPC", "JBHI"]:
    seen_titles = set()
    cleaned_list = []
    for item in data.get(j_name, []):
        t = item['title'].strip().lower()
        if t not in seen_titles:
            seen_titles.add(t)
            cleaned_list.append(item)
    final_papers[j_name] = cleaned_list[:5]

# Remove extra files from directory
valid_filenames = set()
for j_name, papers in final_papers.items():
    for idx, p in enumerate(papers, 1):
        p['index'] = idx
        valid_filenames.add(p['filename'])

for f in glob.glob(os.path.join(PAPERS_DIR, "*.pdf")):
    base = os.path.basename(f)
    if base not in valid_filenames:
        print(f"Removing duplicate/untracked file: {base}")
        os.remove(f)

with open(os.path.join(META_DIR, "clean_15_papers_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(final_papers, f, indent=2)

print("\nFinal 15 papers retained:")
for j_name, papers in final_papers.items():
    print(f"\n--- {j_name} ({len(papers)} papers) ---")
    for p in papers:
        filepath = os.path.join(PAPERS_DIR, p['filename'])
        exists = os.path.exists(filepath)
        size_mb = os.path.getsize(filepath) / 1024 / 1024 if exists else 0
        print(f"  [{p['index']}] ({p['year']}) {p['title'][:65]}... ({size_mb:.2f} MB)")
