#!/usr/bin/env python
# (works in both Python 2 and Python 3)

# Offline HTML Indexer v1.32 (c) 2013-15,2020 Silas S. Brown.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a Python program for creating large indices of
# HTML text which can be queried using simple Javascript
# that works on many mobile phone browsers without needing
# an Internet connection or a Web server. This is useful if
# you want to load a dictionary or other reference onto your
# phone (or computer) for use when connectivity is not
# available.
# The input HTML should be interspersed with anchors like
# this: <a name="xyz"></a> where xyz is the index heading
# for the following text. There should be one such anchor
# before each entry and an extra anchor at the end of the
# text; everything before the first anchor is counted as the
# "header" and everything after the last as the "footer". If
# these are empty, a default "mobile friendly" HTML header
# and footer specifying UTF-8 encoding will be
# added. Anchors may be linked from other entries; these
# links are changed as necessary.
# Opening any of the resulting HTML files should display a
# textbox that lets you type the first few letters of the
# word you wish to look up; the browser will then jump to
# whatever heading is alphabetically nearest to the typed-in
# text.

# Configuration
# -------------

infile = None # None = standard input, or set a "filename"
outdir = "." # current directory by default
alphabet = "abcdefghijklmnopqrstuvwxyz" # set to None for all characters and case-sensitive
ignore_text_in_parentheses = True # or False, for parentheses in index headings
more_sensible_punctuation_sort_order = True

remove_utf8_diacritics = True # or False, for removing diacritics in index headings (not in main text);
# assumes UTF-8.  (Letters with diacritics will be treated as though they did not have any.)

max_filesize = 64*1024 # of each HTML file
# (max_filesize can be exceeded by 1 very large entry)

# Where to find history:
# on GitHub at https://github.com/ssb22/indexer
# and on GitLab at https://gitlab.com/ssb22/indexer
# and on BitBucket https://bitbucket.org/ssb22/indexer
# and at https://gitlab.developers.cam.ac.uk/ssb22/indexer
# and in China: https://gitee.com/ssb22/indexer

# ---------------------------------------------------------------

import re,sys,os,time
if type("")==type(u""): izip = zip # Python 3
else: from itertools import izip # Python 2

if infile:
    sys.stderr.write("Reading from "+infile+"... ")
    infile = open(infile)
else:
    sys.stderr.write("Reading from standard input... ")
    infile = sys.stdin
fragments = re.split(r'<a name="([^"]*)"></a>',infile.read())
# odd indices should be the tag names, even should be the HTML in between
assert len(fragments)>3, "Couldn't find 2 or more hash tags (were they formatted correctly?)"
assert len(fragments)%2, "re.split not returning groups??"
header,footer = fragments[0],fragments[-1]
if not header.strip(): header="""<html><head><meta name="mobileoptimized" content="0"><meta name="viewport" content="width=device-width"><meta http-equiv="Content-Type" content="text/html; charset=utf-8"></head><body>"""
if not footer.strip(): footer = "</body></html>"
fragments = fragments[1:-1]
sys.stderr.write("%d entries\n" % len(fragments))
def alphaOnly(x):
  if ignore_text_in_parentheses: x=re.sub(r"\([^)]*\)[;, ]*","",x)
  if alphabet: x=''.join(c for c in x.lower() if c in alphabet)
  return re.sub(r"^[@,;]*","",x) # see ohi_latex.py
if more_sensible_punctuation_sort_order:
    _ao1 = alphaOnly
    alphaOnly = lambda x: _ao1(re.sub('([;,]);+',r'\1',x.replace('-',' ').replace(',','~COM~').replace(';',',').replace('~COM~',';').replace(' ',';'))) # gives ; < , == space (useful if ; is used to separate definitions and , is used before extra words to be added at the start; better set space EQUAL to comma, not higher, or will end up in wrong place if user inputs something forgetting the comma)
    if alphabet:
      for c in '@,;':
        if not c in alphabet: alphabet += c
if remove_utf8_diacritics:
    _ao = alphaOnly ; import unicodedata
    def S(s):
        if type(u"")==type(""): return s # Python 3
        else: return s.encode('utf-8') # Python 2
    def U(s):
        if type(s)==type(u""): return s
        return s.decode('utf-8')
    alphaOnly = lambda x: _ao(S(u''.join((c for c in unicodedata.normalize('NFD',U(x)) if not unicodedata.category(c).startswith('M')))))
fragments = list(zip(map(alphaOnly,fragments[::2]), fragments[1::2]))
fragments.sort()
class ChangedLetters:
    def __init__(self): self.lastText = ""
    def __call__(self,text):
        "Find shortest prefix of text that differentiates it from previous item (empty string if no difference)"
        assert text >= self.lastText, "input must have been properly sorted"
        i = 0
        for c1,c2 in izip(self.lastText+chr(0),text):
            i += 1
            if not c1==c2:
                self.lastText = text
                return text[:i]
        assert text==self.lastText, repr(text)+"!="+repr(self.lastText)
        return "" # no difference from lastText
changedLetters = ChangedLetters() ; f2 = []
fragments.reverse()
sys.stderr.write("Minimizing prefixes... ")
while fragments:
    x,y = fragments.pop()
    x = changedLetters(x)
    if f2 and not x: f2[-1] = (f2[-1][0], f2[-1][1]+y) # combine effectively-identical ones
    else: f2.append((x,y))
sys.stderr.write("done\n")
fragments = f2
def tag(n):
    if n: return '<a name="%s"></a>' % n
    else: return ''
def old_javascript_array(array):
    "in case the browser doesn't support JSON, and to save some separator bytes"
    array = list(array) # in case it was an iterator
    sepChar = ord(' ')
    chars_used = set(''.join(array))
    assert '"' not in chars_used and '\\' not in chars_used and '<' not in chars_used and '&' not in chars_used, "Can't use special chars (unless you change this code to escape them)"
    while True:
        if chr(sepChar) not in chars_used and not chr(sepChar) in r'\"<&': break
        sepChar += 1
        assert sepChar < 127, "can't find a suitable separator char (hard-code the array instead?)"
    return '"'+chr(sepChar).join(array)+'".split("'+chr(sepChar)+'")'
js_binchop = """function(a,i) {
function inner(a,i,lo,hi) {
var mid=lo+Math.floor((hi-lo)/2);
if(mid==lo || a[mid]==i) return a[mid];
if(a[mid] > i) return inner(a,i,lo,mid);
return inner(a,i,mid,hi);
} return inner(a,i,0,a.length);
}"""
js_binchop_dx = js_binchop.replace("return a[mid]","return mid")
def js_hashjump(hashtags): return """<script><!--
var h=location.hash; if(h.length > 1) { if(h!='#_h' && h!='#_f') { var n="#"+%s(%s,h.slice(1)); if (h!=n) location.hash=n; } } else location.href="index.html"
//--></script>""" % (js_binchop,old_javascript_array(hashtags)) # (the h!=n test is needed to avoid loop on some  browsers e.g. PocketIE7)
# #_h and #_f are special hashes for header and footer, used for "Next page" and "Previous page" links
# (HTML5 defaults type to text/javascript, as do all pre-HTML5 browsers including NN2's 'script language="javascript"' thing, so we might as well save a few bytes)

__lastStartEnd = None
def htmlDoc(start,end,docNo):
    "Returns an HTML document containing fragments[start:end].  docNo is used to generate previous/next page links as appropriate.  Caches its return value in case called again with same start,end (in which case docNo is ignored on second call)."
    global __lastStartEnd,__lastDoc
    if not (start,end) == __lastStartEnd:
        __lastStartEnd = (start,end)
        __lastDoc = header+js_hashjump(x for x,y in fragments[start:end] if x)
        if start:
            assert docNo, "Document 0 should start at 0"
            __lastDoc += '<p><a name="_h" href="%d.html#_f">Previous page</a></p>' % (docNo-1,)
        __lastDoc += ''.join(tag(x)+y for x,y in fragments[start:end])
        if end<len(fragments): __lastDoc += '<p><a name="_f" href="%d.html#_h">Next page</a></p>' % (docNo+1,)
        __lastDoc += footer
    return linkSub(__lastDoc)

def linkSub(txt): return re.sub(r'(?i)<a href=("?)#',r'<a href=\1index.html#',txt) # (do link to index.html#whatever rather than directly, so link still works if docs change)

def findEnd(start,docNo):
    "Given 'start' (an index into 'fragments'), find an 'end' that produces the largest possible htmlDoc less than max_filesize.  docNo is used to generate previous/next page links as appropriate."
    eTry = len(fragments)-start
    assert eTry, "must start before the end"
    sLen = len(htmlDoc(start,start+eTry,docNo))
    if sLen > max_filesize:
        eTry = int(eTry / int(sLen / max_filesize)) # rough start point
        while eTry > 1 and len(htmlDoc(start,start+eTry,docNo)) > max_filesize:
            eTry = int(eTry/2)
        if eTry < 1: eTry = 1
    while eTry < len(fragments)-start and len(htmlDoc(start,start+eTry,docNo)) < max_filesize: eTry += 1
    return start + max(1,eTry-1)
def allRanges():
    start = docNo = 0
    while start < len(fragments):
        end = findEnd(start,docNo)
        sys.stderr.write("\rSegmenting (%d/%d)" % (end,len(fragments)))
        yield start,end
        start = end ; docNo += 1
sys.stderr.write("Segmenting")
startsList = []
for start,end in allRanges():
    open(("%s%s%d.html" % (outdir,os.sep,len(startsList))),"w").write(htmlDoc(start,end,len(startsList)))
    startsList.append(start)
if alphabet:
    assert not '"' in alphabet and not '\\' in alphabet and not '&' in alphabet and not '<' in alphabet, "Can't use special characters in alphabet (unless js_alphabet is modified to quote them)"
    js_alphabet = """var a=val.toLowerCase(),i; val="";
for(i=0; i < a.length; i++) { var c=a.charAt(i); if("%s".indexOf(c)>-1) val += c }
""" % alphabet # TODO: what if user types letters with diacritics, when remove_utf8_diacritics is set?
else: js_alphabet = ""
if more_sensible_punctuation_sort_order: js_alphabet = "val = val.replace(/-/g,' ').replace(/,/g,'~COM~').replace(/;/g,',').replace(/~COM~/g,';').replace(/ /g,';').replace(/([;,]);+/g,'$1');" + js_alphabet

def hashReload(footer):
    # If a footer refers to index.html#example, need to
    # make sure the hash script runs when clicking there
    # from the index page itself.
    strToFind = '<a href="index.html#'
    # TODO: what if it's quoted differently and/or has extra attributes?  (ohi.html does specify using " quoting though)
    while True:
        i = footer.lower().find(strToFind)
        if i==-1: return footer
        footer = footer[:i]+'<a onclick="document.forms[0].q.value=\''+footer[i+len(strToFind):footer.index('"',i+len(strToFind))]+'\';jump()" href="index.html#'+footer[i+len(strToFind):]

open(outdir+os.sep+"index.html","w").write("""%s<script><!--
function jump() {
  var val=document.forms[0].q.value; %s
  location.href=%s(%s,val)+".html#"+val
}
if(navigator.userAgent.indexOf("Opera/9.50" /* sometimes found on WM6.1 phones from 2008 */) >= 0) document.write("<p><b>WARNING:</"+"b> Your version of Opera may have trouble jumping to anchors; please try Opera 10 or above.</"+"p>")
//--></script><noscript><p><b>ERROR:</b> Javascript needs to be switched on for this form to work.</p></noscript>
<form action="#" onSubmit="jump();return false">Lookup: <input type="text" name="q"><input type="submit" value="ok"></form><script><!--
if(location.hash.length > 1) { document.forms[0].q.value = location.hash.slice(1).replace(/(\+|%%20)/g,' '); jump(); } else document.forms[0].q.focus();
//--></script>%s""" % (hashReload(linkSub(header)),js_alphabet,js_binchop_dx,old_javascript_array(fragments[s][0] for s in startsList),hashReload(linkSub(footer))))
sys.stderr.write(" %d files\n" % (len(startsList)+1))
