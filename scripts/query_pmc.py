import urllib.request
import urllib.parse
import json
import os

journals = {
    "CBM": 'JOURNAL:"Computers in biology and medicine"',
    "BSPC": 'JOURNAL:"Biomedical signal processing and control"',
    "JBHI": 'JOURNAL:"IEEE journal of biomedical and health informatics"'
}

keywords = "(hypotension OR vasopressor OR blood pressure OR waveform OR sepsis OR hemodynamics OR PPG)"

for name, j_query in journals.items():
    query = f'{j_query} AND PUB_YEAR:[2020 TO 2026] AND {keywords} AND OPEN_ACCESS:y'
    encoded = urllib.parse.quote(query)
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={encoded}&format=json&resultType=core&pageSize=15"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        results = data.get('resultList', {}).get('result', [])
        print(f"\n=== {name} (Hits: {data.get('hitCount', 0)}) ===")
        valid_pmc = [r for r in results if r.get('pmcid')]
        print(f"Found {len(valid_pmc)} papers with PMCID:")
        for r in valid_pmc[:6]:
            print(f"  [{r.get('pmcid')}] ({r.get('pubYear')}) {r.get('title')[:75]}...")
    except Exception as e:
        print(f"Error querying {name}: {e}")
