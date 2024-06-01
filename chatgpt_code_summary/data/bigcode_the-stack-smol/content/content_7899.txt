import requests
import json

from pybliometrics.scopus import AbstractRetrieval

arr_authors = [
    '55949131000', #EG
    '56344636600', #MF
    '6602888121', #MG
    '7005314544' #SR
    ]
MY_API_KEY = 'afd5bb57359cd0e85670e92a9a282d48'

from pybliometrics.scopus.utils import config
#config['Authentication']['APIKey'] = 'afd5bb57359cd0e85670e92a9a282d48'

bib = set()

def get_scopus_info(SCOPUS_ID):
    url = ("http://api.elsevier.com/content/abstract/scopus_id/"
          + SCOPUS_ID
          + "?field=authors,title,publicationName,volume,issueIdentifier,"
          + "prism:pageRange,coverDate,article-number,doi,citedby-count,prism:aggregationType")
    resp = requests.get(url,
                    headers={'Accept':'application/json',
                             'X-ELS-APIKey': MY_API_KEY})

    return json.loads(resp.text.encode('utf-8'))

for author in arr_authors:

    resp = requests.get("http://api.elsevier.com/content/search/scopus?query=AU-ID(" + author + ")&field=dc:identifier",
                        headers={'Accept':'application/json',
                                'X-ELS-APIKey': MY_API_KEY})

    results = resp.json()
    #print(results)
    
    i = 0
    for r in results['search-results']["entry"]:
    
        sid = [str(r['dc:identifier'])]
        # some entries seem to have json parse errors, so we catch those
        print(sid[0].replace('SCOPUS_ID:',''))
        ab = AbstractRetrieval(sid[0].replace('SCOPUS_ID:',''))

        bib.add(str(ab.get_html()))
        break
    break

with open('bib.bib', 'w') as file:
    for bibitem in bib:
        file.write(bibitem)
        file.write('\n')