import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('research2/literature/metadata/recent_papers_structural_analysis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for fname, d in data.items():
    print(f"\n==================== {fname} ({d['pages']} pages) ====================")
    print("Found Metrics:", d['found_metrics'])
    print("\nSample Figure Captions:")
    for fig in d['figures'][:8]:
        clean = fig.encode('ascii', errors='replace').decode('ascii')
        print("  *", clean[:140])
    print("\nSample Table Captions:")
    for tbl in d['tables'][:6]:
        clean = tbl.encode('ascii', errors='replace').decode('ascii')
        print("  *", clean[:140])
    print("\nSection Headings detected:")
    for h in d['headings'][:10]:
        clean = h.encode('ascii', errors='replace').decode('ascii')
        print("  -", clean)
