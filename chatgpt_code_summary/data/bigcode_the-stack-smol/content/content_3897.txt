# -*- coding: utf-8 -*-
# https://github.com/cvzi/foodemoji
import sys
import os
import timeit

try:
    import foodemoji
except ImportError:
    include = os.path.relpath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, include)
    import foodemoji
    print("Imported foodemoji from %s" % os.path.abspath(os.path.join(include, "foodemoji")))

PY2 = sys.version_info.major == 2

text = """Erbsencremesuppe
Mousse Tiramisu, Wackelpudding Kirsch (vegan)
Milch Auswahl an frischen Salaten, Gemüse-, Fisch-, Geflügel-, Schweine- und Rindfleisch- und vegetarischen und veganen Gerichten
Tagessuppe,
Fish + Chips,
Remouladensauce,
Salat der Saison
Tagessuppe,
2 Polentaschnitten mit Spinatfüllung,
Tomatensauce,
Reis,
Salat der Saison
Karotten-Ingwersuppe
Rote Grütze (vegan), Zweierlei Mousse au Chocolat
Milch
Tagessuppe,
Schweinegulasch,
Champignonsauce,
Salzkartoffeln,
Salat der Saison
Tagessuppe,
5 Cannelloni mit Ricotta-Spinat-Füllung,
Tomatensauce,
Reibekäse,
Salat der Saison
Tomatencremesuppe
Milchreis mit Kirschen, Rote Grütze (vegan)
Tagessuppe,
Feuerwurst,
Portion Senf,
Pommes frites,
Salat der Saison
Tagessuppe,
2 Kartoffelknödel,
Rahmgemüse,
Salat der Saison
Kohlrabicremesuppe Creme Brulee, Kokosmilch mit Ananas (vegan) Schlemmerbuffet je 100g Reichhaltige Auswahl an frischen Salaten, Gemüse-, Fisch-, Geflügel-, Schweine- und Rindfleisch- und vegetarischen und veganen Gerichten   Erbseneintopf, Bockwurst, Kartoffeln, Brötchen, Salat der Saison, Schokopudding Tagessuppe, Asiatische Gemüseknusperschnitte, Wasabi Currysauce, Reis, Salat der Saison, Gebrannte Grießsuppe Kokosmilch mit Ananas (vegan), Mousse au Chocolat Milch (ML) Schlemmerbuffet je 100g Reichhaltige Auswahl an frischen Salaten, Gemüse-, Fisch-, Geflügel-, Schweine- und Rindfleisch- und vegetarischen und veganen Gerichten  D: Tagessuppe, Schweinegeschnetzeltes, Pilzrahmsauce, Reis, Salat der Saison Tagessuppe, Knöpflepfanne "Allgäu", Käsesauce, Salat der Saison.
Brokkolicremesuppe
Sojajoghurt mit Früchten (vegan), Tiramisu
Milch (ML)
Tagessuppe,
paniertes Alaska-Seelachsfilet,
Dillmayonnaise,
Petersilienkartoffeln,
Salat der Saison
Tagessuppe,
veganes Geschnetzeltes „Züricher Art",
Reis,
Salat der Saison
"""

text_short = """Erbsencremesuppe
Mousse Tiramisu, Wackelpudding Kirsch (vegan)
Milch Auswahl an frischen Salaten, Gemüse-, Fisch-, Geflügel-, Schweine- und Rindfleisch- und vegetarischen und veganen Gerichten
Kohlrabicremesuppe Creme Brulee, Kokosmilch mit Ananas (vegan) Schlemmerbuffet je 100g Reichhaltige Auswahl an frischen Salaten, Gemüse-, Fisch-, Geflügel-, Schweine- und Rindfleisch- und vegetarischen und veganen Gerichten   Erbseneintopf, Bockwurst, Kartoffeln, Brötchen, Salat der Saison, Schokopudding Tagessuppe, Asiatische Gemüseknusperschnitte, Wasabi Currysauce, Reis, Salat der Saison, Gebrannte Grießsuppe Kokosmilch mit Ananas (vegan), Mousse au Chocolat Milch (ML) Schlemmerbuffet je 100g Reichhaltige Auswahl an frischen Salaten, Gemüse-, Fisch-, Geflügel-, Schweine- und Rindfleisch- und vegetarischen und veganen Gerichten  D: Tagessuppe, Schweinegeschnetzeltes, Pilzrahmsauce, Reis, Salat der Saison Tagessuppe, Knöpflepfanne "Allgäu", Käsesauce, Salat der Saison.
Salat der Saison
"""

text_one_line = "Milch Auswahl an frischen Salaten, Gemüse-, Fisch-, Geflügel-, Schweine- und Rindfleisch- und vegetarischen und veganen Gerichten"

book = ""

def _setup():
    global book

    filename = 'italienische-reise.txt'
    url = 'https://github.com/GITenberg/Italienische-Reise-Band-1_2404/raw/master/2404-8.txt'

    if not os.path.isfile(filename):
        if PY2:
            import urllib2
            furl = urllib2.urlopen(url)
            book = furl.read().decode('cp1252' ,errors='ignore')
            furl.close()
        else:
            import urllib.request
            with urllib.request.urlopen(url) as furl:
                book = furl.read().decode('utf-8' ,errors='ignore')
        with open(filename, 'wb') as fout:
            fout.write(book.encode('utf-8'))
    else:
        with open(filename, 'rb') as fin:
            book = fin.read().decode('utf-8')

def test_long_text_100():
    x = foodemoji.decorate(text)
    return x[0] == text[0]

def test_long_text_linebyline_100():
    x = foodemoji.decorate(text, line_by_line=True)
    return x[0] == text[0]

def test_short_text_300():
    x = foodemoji.decorate(text_short)
    return x[0] == text_short[0]

def test_short_text_linebyline_300():
    x = foodemoji.decorate(text_short, line_by_line=True)
    return x[0] == text_short[0]

def test_one_line_1000():
    x = foodemoji.decorate(text_one_line)
    return x[0] == text_one_line[0]

def test_one_line_linebyline_1000():
    x = foodemoji.decorate(text_one_line, line_by_line=True)
    return x[0] == text_one_line[0]

def test_book_2():
    x = foodemoji.decorate(book)
    return x[0] == book[0]

def test_book_linebyline_2():
    x = foodemoji.decorate(book, line_by_line=True)
    return x[0] == book[0]

_setup()

if __name__ == '__main__':

    for fname in sorted(list(globals().keys())):
        if fname.startswith('test_'):
            if fname.split('_')[-1].isdigit():
                N = int(fname.split('_')[-1])
            else:
                N = 100

            print("% 6dx\t\t%s():" % (N, fname))
            t = timeit.timeit('speed.%s()' % fname, setup='import speed', number=N)
            print("{:25.20f}".format(t))
