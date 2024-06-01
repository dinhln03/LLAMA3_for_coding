# From http://rodp.me/2015/how-to-extract-data-from-the-web.html
import time
import sys
import uuid
import json
import markdown
from collections import Counter
from requests import get
from lxml import html
from unidecode import unidecode
import urllib
import lxml.html
from readability.readability import Document



def getDoc(url):
    t = time.time()
    t2 = time.time()
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1'}
    r = get(url,headers=headers)
    print("*"*30)
    print("Getting url took " + str(time.time()-t2))
    print("*"*30)
    redirectUrl = str(uuid.uuid3(uuid.NAMESPACE_DNS, str(r.url)))[0:5]


    newContent = r.content
    parsed_doc = html.fromstring(newContent)
    with open('doc.html','w') as f:
        f.write(newContent)


    parents_with_children_counts = []
    parent_elements = parsed_doc.xpath('//body//*/..')
    for parent in parent_elements:
        children_counts = Counter([child.tag for child in parent.iterchildren()])
        parents_with_children_counts.append((parent, children_counts))

    parents_with_children_counts.sort(key=lambda x: x[1].most_common(1)[0][1], reverse=True)

    docStrings = {}
    last = len(parents_with_children_counts)
    if last > 20:
        last = 20

    t2 = time.time()
    for i in range(last):
        docString = ""
        numLines = 0
        for child in parents_with_children_counts[i][0]: # Possibly [1][0]
            tag = str(child.tag)

            #print(tag)
            if tag == 'style' or tag == 'iframe':
                continue
            if tag == 'font' or tag == 'div' or tag == 'script':
                tag = 'p'
            try:
                startTag = "<" + tag + ">"
                endTag = "</" + tag + ">"
            except:
                startTag = '<p>'
                endTag = '</p>'
            try:
                str_text = child.text_content().encode('utf-8')
                #str_text = " ".join(str_text.split())
                str_text = json.dumps(str_text)
                str_text = str_text.replace('\"','').replace('\\n','\n')
                str_text = str_text.replace('\\t','').replace('\\r','')
                str_text = str_text.replace('\u0092',"'").replace('\\u00e2\\u0080\\u0099',"'").replace('\u2019',"'")
                str_text = str_text.replace('\u0093','"').replace('\u00e2\u0080\u009c','"').replace('\u00e2\u0080\u009d','"').replace('\u201c','"').replace('\u201d','"')
                str_text = str_text.replace('\u0094','"').replace('\u00e2\u0080" ','')
                for foo in range(5):
                    str_text = str_text.replace('<br> <br>','<br>')
                str_text = str_text.replace('\u0096','-').replace('\u2014','-').replace('\\u00a0',' ')
                str_text = str_text.replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
                str_text = str_text.replace('\\','').replace('u2026 ','').replace('u00c2','')
                newString = startTag + str_text + endTag + "\n"
                newString = str_text + "\n\n"
                if (len(newString) > 50000 or 
                        len(newString)<14 or 
                        '{ "' in newString or 
                        '{"' in newString or 
                        "function()" in newString or 
                        'else {' in newString or 
                        '.js' in newString or 
                        'pic.twitter' in newString or 
                        '("' in newString or 
                        'ajax' in newString or 
                        'var ' in newString or 
                        ('Advertisement' in newString and len(newString)<200) or 
                        'Continue reading' in newString or 
                        ('Photo' in newString and 'Credit' in newString) or 
                        'window.' in newString or 
                        ');' in newString or 
                        '; }' in newString or
                        'CDATA' in newString or
                        '()' in newString):
                    continue
                #print(len(newString))
                if len(newString) > 50 and ':' not in newString:
                    numLines += 1
                docString += newString
            except:
                #print('error')
                pass
        docStrings[i] = {}
        docStrings[i]['docString'] = markdown.markdown(docString)
        docStrings[i]['word_per_p'] = float(len(docString.split())) / float(len(docStrings[i]['docString'].split('<p>')))
        docStrings[i]['numLines'] = numLines
        docStrings[i]['docString_length'] = len(docString)
        try:
            docStrings[i]['score']=numLines*docStrings[i]['word_per_p']
            #docStrings[i]['score']=1000*numLines / sum(1 for c in docString if c.isupper())
        except:
            docStrings[i]['score'] = 0

    print("*"*30)
    print("Looping took " + str(time.time()-t2))
    print("*"*30)
    
    with open('test.json','w') as f:
        f.write(json.dumps(docStrings,indent=2))

    bestI = 0
    bestNumLines = 0
    for i in range(len(docStrings)):
        if (docStrings[i]['word_per_p']>12 and
                docStrings[i]['score'] > bestNumLines and 
                docStrings[i]['docString_length'] > 300):
            bestI = i
            bestNumLines = docStrings[i]['score']

    print("*"*24)
    print(bestI)
    print(bestNumLines)
    print("*"*24)
    docString = docStrings[bestI]['docString']
    if len(docString)<100:
        docString="<h1>There is no content on this page.</h1>"

    title = parsed_doc.xpath(".//title")[0].text_content().strip()
    try:
        description = parsed_doc.xpath(".//meta[@name='description']")[0].get('content')
    except:
        description = ""
    url = r.url
    timeElapsed = int((time.time()-t)*1000)
    docString = docString.decode('utf-8')
    for s in docString.split('\n'):
        print(len(s))
    fileSize = 0.7 + float(sys.getsizeof(docString)/1000.0)
    fileSize = round(fileSize,1)
    return {'title':title,'description':description,'url':url,'timeElapsed':timeElapsed,'content':docString,'size':fileSize}

def getDoc2(url):
    t = time.time()
    # import urllib
    # html = urllib.urlopen(url).read()
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1'}
    r = get(url,headers=headers)
    html = r.content

    doc = Document(html,url=url)
    readable_article = doc.summary()
    readable_title = doc.short_title()
    readable_article = readable_article.replace("http","/?url=http")
    timeElapsed = int((time.time()-t)*1000)
    fileSize = 0.7 + float(sys.getsizeof(readable_article)/1000.0)
    fileSize = round(fileSize,1)
    return {'title':readable_title,'description':"",'url':url,'timeElapsed':timeElapsed,'content':readable_article,'size':fileSize}

#print(getDoc('http://www.bbc.co.uk/news/entertainment-arts-34768201'))
