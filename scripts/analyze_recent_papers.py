import pypdfium2 as pdfium
import glob
import os
import re
import json

recent_files = sorted(glob.glob('research2/literature/papers/*2025*.pdf') + glob.glob('research2/literature/papers/*2026*.pdf'))

analysis = {}

for f in recent_files:
    fname = os.path.basename(f)
    print(f"Analyzing {fname}...")
    pdf = pdfium.PdfDocument(f)
    n_pages = len(pdf)
    
    full_text = []
    for i in range(n_pages):
        page = pdf[i]
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        full_text.append(text)
    
    joined_text = "\n".join(full_text)
    
    # Extract headings (lines like "1. Introduction", "2. Methods", "3. Results")
    headings = re.findall(r'(?m)^(?:\d+\.?\s+[A-Z][A-Za-z\s,-]{3,50}|[A-Z\s]{4,30})$', joined_text)
    headings = [h.strip() for h in headings if len(h.strip()) > 3][:25]
    
    # Extract Figure captions
    fig_captions = re.findall(r'(?i)(Fig(?:ure)?\.?\s*\d+[:\.\s][^\n\r]+(?:(?:\n[^\n\r]+){1,3})?)', joined_text)
    clean_figs = [re.sub(r'\s+', ' ', fc).strip() for fc in fig_captions][:15]
    
    # Extract Table captions
    tbl_captions = re.findall(r'(?i)(Table\s*\d+[:\.\s][^\n\r]+(?:(?:\n[^\n\r]+){1,2})?)', joined_text)
    clean_tbls = [re.sub(r'\s+', ' ', tc).strip() for tc in tbl_captions][:10]
    
    # Look for metrics mentioned
    metrics_keywords = ['RMSE', 'MAPE', 'AUC', 'AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1', 'Brier', 'Calibration', 'ECE', 'Lead time', 'TITR', 'WIS', 'Delay', 'Runtime', 'Latency', 'VRAM', 'FLOPs']
    found_metrics = [m for m in metrics_keywords if re.search(r'\b' + m + r'\b', joined_text, re.IGNORECASE)]
    
    analysis[fname] = {
        'pages': n_pages,
        'headings': headings,
        'figures': clean_figs,
        'tables': clean_tbls,
        'found_metrics': list(set(found_metrics)),
        'sample_abstract': joined_text[:1200].replace('\n', ' ')
    }

with open('research2/literature/metadata/recent_papers_structural_analysis.json', 'w', encoding='utf-8') as out:
    json.dump(analysis, out, indent=2, ensure_ascii=False)

print("Analysis complete! Saved to metadata/recent_papers_structural_analysis.json")
