# 03_xkcd_multithread_download.py
# In dieser Übung geht es darum den Download der Comics zu beschleunigen
# indem man mehrere Threads zum downloaden nutzt.

import os, threading, requests, bs4
os.chdir(os.path.dirname(__file__))

target_dir='.\\comics'
source_url='https://xkcd.com'

# Prüfe ob Seite erreichbar
url_content=requests.get(source_url)
try:
    url_content.raise_for_status()
except:
    print('URL xkcd.com kann nicht aufgerufen werden. Script wird beendet.')
    exit()

# Downloade die Comics als Thread
def download_comic(comic_url):
    file_name=comic_url.split('/')[-1]
    new_file=open(target_dir+'\\'+file_name, 'wb')
    get_comic=requests.get(comic_url)
    try:
        get_comic.raise_for_status()
        for chunk in get_comic.iter_content(10**6):
            new_file.write(chunk)
        new_file.close()
    except:
        print('Bild-URL %s ist fehlerhaft') % (comic_url)

# Sammle die Links zu den Comics und den weiterführenden Seiten
link_counter=0
threads=[]
def scrape_comic_links(url_name):
    global link_counter, threads
    while link_counter != int(comic_target_amount):
        url_content=requests.get(url_name)
        try:
            url_content.raise_for_status()
            bs4_object=bs4.BeautifulSoup(url_content.text, features='html.parser')
            bs4_next_result=bs4_object.select('a[rel="prev"]')
            next_url=bs4_next_result[0].get('href')
            bs4_comic_result=bs4_object.select('div #comic img')
            comic_url=bs4_comic_result[0].get('src')
            comic_url='https://'+comic_url.lstrip('/')
            url_name=source_url+next_url
            link_counter+=1
            # Starte Download-Thread
            thread_object=threading.Thread(name='Download_Comic', target=download_comic, args=[comic_url])
            thread_object.start()
            # Füge diesen Thread einer Liste hinzu um später zu prüfen ob alles Abgearbeitet wurde.
            threads.append(thread_object)
        except:
            print('URL nicht gefunden.')
            return
    else:
        link_counter=0
        return

while True:
    print('Wieviele Comics sollen heruntergeladen werden?')
    comic_target_amount=input()
    if comic_target_amount.isdecimal():
        scrape_comic_links(source_url)
        # Warte bis alle Prozesse abgeschlossen sind.
        for thread in threads:
            thread.join()
        print('Downloads abgeschlossen')
        break
