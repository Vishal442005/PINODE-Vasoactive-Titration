import urllib.request
import urllib.parse
import json
import os
import re
import time

PAPERS_DIR = "research2/literature/papers"
META_DIR = "research2/literature/metadata"
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
}

def clean_filename(title, max_len=50):
    clean = re.sub(r'[^\w\s-]', '', title)
    clean = re.sub(r'[-\s]+', '_', clean).strip('_')
    return clean[:max_len]

def download_pdf(url, output_path):
    if 'sciencedirect.com/science/article' in url and not url.endswith('.pdf'):
        return False, "Skipping ScienceDirect landing page"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            content = resp.read()
            if content.startswith(b'%PDF-') and len(content) > 30000:
                with open(output_path, 'wb') as f:
                    f.write(content)
                return True, len(content)
            else:
                return False, f"Not PDF (size: {len(content)}, header: {content[:15]})"
    except Exception as e:
        return False, str(e)

# Load existing candidates
with open(os.path.join(META_DIR, "candidates.json"), "r", encoding="utf-8") as f:
    candidates_by_journal = json.load(f)

downloaded_papers = {k: [] for k in ["CBM", "BSPC", "JBHI"]}

for j_name in ["CBM", "BSPC", "JBHI"]:
    print(f"\n==================== PROCESSING {j_name} ====================", flush=True)
    candidates = candidates_by_journal.get(j_name, [])
    
    def relevance_score(item):
        t = (item.get('title') or '').lower()
        score = 0
        if 'arterial' in t or 'abp' in t: score += 6
        if 'blood pressure' in t: score += 5
        if 'photoplethysmography' in t or 'ppg' in t: score += 5
        if 'hypotension' in t: score += 6
        if 'hemodynamic' in t: score += 5
        if 'sepsis' in t or 'icu' in t: score += 4
        if 'waveform' in t or 'pulse' in t: score += 4
        if 'infusion' in t or 'control' in t: score += 5
        score += min(item.get('citations', 0) / 10.0, 5)
        return score

    candidates.sort(key=relevance_score, reverse=True)
    
    success_count = 0
    for cand in candidates:
        if success_count >= 5:
            break
        
        title = cand.get('title') or 'Untitled'
        year = cand.get('year') or 2023
        clean_tit = clean_filename(title, max_len=45)
        
        target_filename = f"{j_name}_{success_count+1:02d}_{year}_{clean_tit}.pdf"
        target_filepath = os.path.join(PAPERS_DIR, target_filename)
        
        pdf_urls = cand.get('pdf_urls', [])
        pdf_urls.sort(key=lambda x: (
            'arxiv.org' in x[1] or 'medrxiv.org' in x[1] or 'techrxiv.org' in x[1] or 'hal.science' in x[1],
            x[1].endswith('.pdf'),
            'sciencedirect.com' not in x[1]
        ), reverse=True)

        download_success = False
        for repo_name, p_url in pdf_urls:
            print(f"[{j_name} #{success_count+1}] Trying {repo_name} -> {p_url[:80]}...", flush=True)
            ok, res = download_pdf(p_url, target_filepath)
            if ok:
                size_mb = res / 1024 / 1024
                print(f"  -> SUCCESS! Saved {target_filename} ({size_mb:.2f} MB)", flush=True)
                download_success = True
                downloaded_papers[j_name].append({
                    'venue': j_name,
                    'index': success_count + 1,
                    'filename': target_filename,
                    'file_size_mb': round(size_mb, 2),
                    'title': title,
                    'year': year,
                    'doi': cand.get('doi'),
                    'openalex_id': cand.get('id'),
                    'download_url': p_url,
                    'citations': cand.get('citations', 0)
                })
                success_count += 1
                break
            else:
                print(f"     Failed: {res[:60]}", flush=True)
                
        time.sleep(0.3)

    print(f"Finished {j_name}: Downloaded {success_count} / 5 papers.", flush=True)

with open(os.path.join(META_DIR, "downloaded_15_papers_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(downloaded_papers, f, indent=2)

print("\nAll downloads completed. Results saved to metadata.", flush=True)
