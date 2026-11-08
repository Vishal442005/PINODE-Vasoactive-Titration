import urllib.request
import urllib.parse
import json

sources = {
    "CBM": "S44278595",
    "BSPC": "S8427965",
    "JBHI": "S2495854775"
}

searches = [
    "blood pressure waveform",
    "hypotension prediction",
    "arterial pressure photoplethysmography",
    "hemodynamic sepsis ICU",
    "vasopressor infusion"
]

results_by_journal = {k: [] for k in sources}

for j_name, s_id in sources.items():
    print(f"\n=================== SEARCHING {j_name} ({s_id}) ===================")
    seen_ids = set()
    for q in searches:
        url = f"https://api.openalex.org/works?filter=primary_location.source.id:{s_id},publication_year:2020-2026,open_access.is_oa:true&search={urllib.parse.quote(q)}&per_page=15&select=id,doi,title,publication_year,cited_by_count,locations,open_access"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Research Assistant)'})
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            for w in data.get('results', []):
                wid = w['id']
                if wid in seen_ids:
                    continue
                seen_ids.add(wid)
                
                # Check for direct pdf_url in locations
                valid_pdf_urls = []
                for loc in w.get('locations', []):
                    p_url = loc.get('pdf_url')
                    if p_url:
                        valid_pdf_urls.append((loc.get('source', {}).get('display_name', 'Repo'), p_url))
                
                if valid_pdf_urls:
                    results_by_journal[j_name].append({
                        'id': wid,
                        'doi': w.get('doi'),
                        'title': w.get('title'),
                        'year': w.get('publication_year'),
                        'citations': w.get('cited_by_count', 0),
                        'pdf_urls': valid_pdf_urls
                    })
        except Exception as e:
            print(f"Error on query {q}: {e}")
            
    print(f"Total candidates with direct PDF URLs for {j_name}: {len(results_by_journal[j_name])}")
    for item in results_by_journal[j_name][:5]:
        print(f"  [{item['year']}] {item['title'][:70]}...")
        print(f"       PDF: {item['pdf_urls'][0][1]}")

with open("research2/literature/metadata/candidates.json", "w", encoding="utf-8") as f:
    json.dump(results_by_journal, f, indent=2)
