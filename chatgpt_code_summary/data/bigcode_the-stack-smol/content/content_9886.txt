#!/usr/bin/python
# -*- coding: UTF-8, tab-width: 4 -*-


from sys import argv, stdin, stdout, stderr
from codecs import open as cfopen
import json




def main(invocation, *cli_args):
    json_src = stdin
    if len(cli_args) > 0:
        json_src = cfopen(cli_args[0], 'r', 'utf-8')
    data = json.load(json_src, 'utf-8')
    json_enc = dict(
        indent=2,
        sort_keys=True,     ### ### <-- some magic here ### ###
        )
    json_enc['separators'] = (',', ': ',)
        # ^-- because the default had space after comma even at end of line.

    rules = data.get('entries')
    if rules is not None:
        del data['entries']

    json_enc = json.JSONEncoder(**json_enc)
    json_iter = json_enc.iterencode(data)

    for chunk in json_iter:
        chunk = chunk.lstrip()
        if chunk == '': continue
        if chunk.startswith('"'):
            stdout.write(' ')
            if rules is not None:
                stdout.write('"entries": {\n')
                verbsep = '  '
                for verb in sorted(rules.keys()):
                    stdout.write(verbsep + json.dumps(verb) + ': [')
                    write_rule_subjs(stdout, rules[verb])
                    stdout.write(']')
                    verbsep = ',\n  '
                stdout.write('\n}, ')
            stdout.write(chunk)
            break
        stdout.write(chunk)

    for chunk in json_iter:
        if rules is not None:
            if chunk.startswith(','):
                stdout.write(',')
                chunk = chunk[1:]
            if chunk.startswith('\n'):
                chunk = ' ' + chunk.lstrip()
        stdout.write(chunk)
    stdout.write('\n')



def gen_rule_subj_hrname(subj):
    hrname = [ subj.get(role, u'\uFFFF') for role in ('o', 'd',) ]
    hrname = [ gen_rule_host_hrname(part) for part in hrname ]
    return hrname


def gen_rule_host_hrname(host):
    try:
        host = host['h']
    except: pass
    host = split_subdomains(host)
    return host


def split_subdomains(host):
    parts = host.split('.')
    major = [ parts.pop() ]
    while len(parts) > 0:
        part = parts.pop()
        major.insert(0, part)
        if len(part) > 3: break
    return '.'.join(major) + ':' + '.'.join(parts)


def write_rule_subjs(dest, subjs):
    if len(subjs) < 1: return

    for subj in subjs:
        subj['hrname'] = gen_rule_subj_hrname(subj)
    subjs.sort(key=lambda s: s['hrname'])

    props = None
    stdout.write('\n')
    for subj in subjs:
        if props is not None:
            dest.write(',\n')
        dest.write('    {')
        propsep = ' '
        del subj['hrname']
        props = [ 'o', 'd' ]
        props += [ prop for prop in sorted(subj.keys()) if prop not in props ]
        for prop in props:
            if subj.has_key(prop):
                dest.write(propsep + json.dumps(prop) + ': '
                    + json.dumps(subj[prop]))
                propsep = ', '
        dest.write(' }')
    stdout.write('\n  ')



















if __name__ == '__main__':
    main(*argv)
