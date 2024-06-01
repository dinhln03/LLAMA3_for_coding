#!/usr/bin/python

import re, random, sys, difflib

random.seed(123)
for i, line in enumerate(sys.stdin.readlines()):
    if i % 1000 == 0: print >>sys.stderr, i, "..."
    if i>0 and re.search(r'^id\tsentiment', line): continue  # combined files, ignore multiple header rows
    line = re.sub(r'\n$', '', line)   # strip trailing newlines
    # "0003b8d" <tab>1<tab> }\n \n u32 cik_gfx_get_wptr(struct radeon_device *rdev,\n \t\t   struct radeon_ri
    fields = line.split('\t', 2)
    if fields[2] == '': continue  # corruption due to empty commits, i.e. no applicable code...
    fields[2] = '\\n'.join(fields[2].split('\\n')[0:25])  # keep <=25 lines
    f2 = fields[2] = re.sub(r'[^\x09,\x0A,\x20-\x7E]', '.', fields[2])  # cleanup non-ASCII
    r = random.randint(0,99)  # augment x% of the time, i.e. don't go crazy
    if fields[1] == '0':
        # no bug - harmless transforms
        res = []
        if r % 10 == 0:  # 10% of the time
            f2 = re.sub(r'/[*].*?[*]/|//.*?(\\n)', '\1', f2)
        # inject spaces and newlines
        for i in range(len(f2)-1):
            c = f2[i]
            # lines end in newlines, so no risk of running off the end
            if c == '\\':
                c2 = f2[i+1]
                if c2 == ' ' and r < 3: res.append(' ')  # add a space
                elif c2 == 'n' and r < 5: res.append('\\n\\')  # add a newline
                elif c2 == 'n' and r < 7: res.append(' \\')  # add extra trailing whitespace
                elif c2 == 't' and r < 3: res.append(' \\')  # extra space before tab
                elif c2 == 't' and r < 5: res.append('\\t ')  # extra space after tabs
                ### your ideas here ###
                else: res.append(c)
            elif c in '{}[]':
                r = random.randint(0,99)
                if r < 3: res.append(' ')  # add a space
                ### your ideas here ###
                else: res.append(c)
            else: res.append(c)
        newf2 = ''.join(res)+f2[-1]
    else:  # fields[1] == '1'
        # contains a bug - harmful transform
        r = random.randint(0,99)
        if r < 50:
            # swap if/then clauses - may introduce syntax errors
            newf2 = re.sub(r'(if[^(]*[(].+?[)][^{]*){(.+?)}(.*?then.*?){(.*?)}', r'\1{\4}\3{\2}', f2)
            # change comparison operators - since ==/!= is used for other datatypes, keep separate from </>
            # note: pick random operator to avoid real parsing
            newf2 = re.sub(r'([a-zA-Z0-9_] *)(<=?|>=?)( *[a-zA-Z0-9_])', r'\1'+['<','<=','>','>='][r%4]+r'\3', newf2)
            newf2 = re.sub(r'([a-zA-Z0-9_] *)(==|!=)( *[a-zA-Z0-9_])', r'\1'+['==','!='][r%2]+r'\3', newf2)
            newf2 = re.sub(r'([a-zA-Z0-9_] *)(&&|[|][|])( *[a-zA-Z0-9_])', r'\1'+['==','!='][r%2]+r'\3', newf2)
            # muck numbers
            # 201 - 99...99 doesn't end in 0, not binary, etc.
            newf2 = re.sub(r'([2-9][0-9]+[1-9])', str(r*random.randint(0,99)+200), newf2)
        else:
            newf2 = f2
    print '\t'.join(fields)
    if newf2 != fields[2]:
        print '\t'.join([re.sub(r'"$', 'beef"', fields[0]), fields[1], newf2])
        #print 'diff:\n' + ''.join(difflib.unified_diff(fields[2].replace('\\n','\n'), newf2.replace('\\n','\n')))
        
