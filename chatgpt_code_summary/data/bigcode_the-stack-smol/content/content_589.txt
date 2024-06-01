
from prettytable import PrettyTable
from collections import OrderedDict

def _fieldnames(rows):
    def g():
        for row in rows:
            yield from row
    d = OrderedDict((k, None) for k in g())
    return list(d.keys())

def _echo_table(rows):
    if not rows: return
    fieldnames = _fieldnames(rows)
    table = PrettyTable(fieldnames)
    table.align = 'l'
    for row in rows:
        table.add_row([row[k] or '' for k in fieldnames])
    click.echo(table.get_string())

def _echo_row(row):
    if not row: return
    table = PrettyTable(row.keys())
    table.align = 'l'
    table.add_row(row.values())
    click.echo(table.get_string())

def _echo_item(x):
    if not x: return
    click.echo(x)




import os
import logging
import click
import click_log
from . import config

_logger = logging.getLogger(__name__)
click_log.basic_config(_logger)

@click.group()
def cli():
    pass




from . import blackboard

@cli.group(name='blackboard')
def cli_blackboard():
    pass

@cli_blackboard.command(name='download', help='Download')
@click.option('--get-password', default=None, help='Command to evaluate to get password (default is to ask).')
@click.option('--netid', default=None, help='Use this NetID.')
@click.argument('link_text', type=click.STRING)
@click_log.simple_verbosity_option(_logger)
def cli_blackboard_download(netid, get_password, link_text):
    if netid is None: netid = config.NETID
    if get_password is None: get_password = config.get_password

    x = blackboard.download(netid=netid, get_password=get_password, link_text=link_text)
    _echo_item(x)

@cli_blackboard.command(name='upload', help='Upload')
@click.option('--get-password', default=None, help='Command to evaluate to get password (default is to ask).')
@click.option('--netid', default=None, help='Use this NetID.')
@click.argument('link_text', type=click.STRING)
@click.argument('path', type=click.STRING)
@click_log.simple_verbosity_option(_logger)
def cli_blackboard_upload(netid, get_password, link_text, path):
    if netid is None: netid = config.NETID
    if get_password is None: get_password = config.get_password

    blackboard.upload(netid=netid, get_password=get_password, link_text=link_text, path=path)

@cli_blackboard.command(name='webassign', help='WebAssign')
@click.option('--get-password', default=None, help='Command to evaluate to get password (default is to ask).')
@click.option('--netid', default=None, help='Use this NetID.')
@click.argument('link_text', type=click.STRING)
@click_log.simple_verbosity_option(_logger)
def cli_blackboard_webassign(netid, get_password, link_text):
    if netid is None: netid = config.NETID
    if get_password is None: get_password = config.get_password

    blackboard.webassign(netid=netid, get_password=get_password, link_text=link_text)

@cli_blackboard.command(name='combo', help='Combine the other commands')
@click.option('--get-password', default=None, help='Command to evaluate to get password (default is to ask).')
@click.option('--netid', default=None, help='Use this NetID.')
@click.option('--upload', type=click.Path(exists=True), default=None, help="CSV to upload.")
@click.option('--webassign/--no-webassign', default=False, help="Export/import WebAssign.")
@click.option('--download/--no-download', default=True, help="Download CSV.")
@click.argument('link_text', type=click.STRING)
@click_log.simple_verbosity_option(_logger)
def cli_blackboard_webassign(netid, get_password, link_text, upload, webassign, download):
    if netid is None: netid = config.NETID
    if get_password is None: get_password = config.get_password

    if not (upload is None):
        blackboard.upload(netid=netid, get_password=get_password, link_text=link_text, path=upload)

    if webassign:
        blackboard.webassign(netid=netid, get_password=get_password, link_text=link_text)

    if download:
        x = blackboard.download(netid=netid, get_password=get_password, link_text=link_text)
        _echo_item(x)










from . import ldap

@cli.group(name='ldap')
def cli_ldap():
    pass

@cli_ldap.command(name='filter', help='LDAP search with user-specified filter.')
@click.argument('filter', type=click.STRING)
@click.argument('keys', nargs=-1, type=click.STRING)
@click_log.simple_verbosity_option(_logger)
def cli_ldap_filter(filter, keys):
    rows = list(ldap.filter(filter), list(keys))
    _echo_table(rows)

@cli_ldap.command(name='search', help='Perform an LDAP search with filter: .' + ldap.SEARCH_FILTER)
@click.argument('term', type=click.STRING)
@click.argument('keys', nargs=-1, type=click.STRING)
@click_log.simple_verbosity_option(_logger)
def cli_ldap_search(term, keys):
    rows = list(ldap.search(term, list(keys)))
    _echo_table(rows)

@cli_ldap.command(name='netid', help='Filter by NetID')
@click.argument('netid', type=click.STRING)
@click.argument('keys', nargs=-1, type=click.STRING)
@click_log.simple_verbosity_option(_logger)
def cli_ldap_netid(netid, keys):
    row = ldap.netid(netid, list(keys))
    _echo_row(row)

@cli_ldap.command(name='alias', help='Filter by alias/PEA')
@click.argument('alias', type=click.STRING)
@click.argument('keys', nargs=-1, type=click.STRING)
@click_log.simple_verbosity_option(_logger)
def cli_ldap_alias(alias, keys):
    row = ldap.alias(alias, list(keys))
    _echo_row(row)

@cli_ldap.command(name='netid-to-alias', help='NetID -> alias/PEA')
@click.argument('netid', type=click.STRING)
@click_log.simple_verbosity_option(_logger)
def cli_ldap_netid_to_alias(netid):
    x = ldap.netid_to_alias(netid)
    _echo_item(x)

@cli_ldap.command(name='alias-to-netid', help='alias -> NetID')
@click.argument('alias', type=click.STRING)
@click_log.simple_verbosity_option(_logger)
def cli_ldap_alias_to_netid(alias):
    x = ldap.alias_to_netid(alias)
    _echo_item(x)






import os
import shutil
from . import coursebook

@cli.group(name='coursebook')
def cli_coursebook():
    pass

@cli_coursebook.group(name='db')
def cli_coursebook_db():
    pass

@cli_coursebook_db.command(name='update')
def cli_coursebook_db_update():
    coursebook.db_update()

@cli_coursebook_db.command(name='netid-to-address')
@click.argument('netid', type=click.STRING)
def cli_coursebook_db_netid_to_address(netid):
    X = list(coursebook.db_netid_to_address(netid))
    _echo_item(' '.join(X))

@cli_coursebook.group(name='roster')
def cli_coursebook_roster():
    pass

@cli_coursebook_roster.command(name='xlsx-to-csv', help='Convert a CourseBook roster XLSX to CSV.')
@click.option('--force/--no-force', default=False, help="Overwrite existing file.")
@click.argument('source', type=click.Path(exists=True))
@click.argument('target', type=click.Path())
def cli_coursebook_xlsx_to_csv(force, source, target):
    if os.path.exists(target) and not force:
        raise click.ClickException('File exists, maybe use --force?: ' + target)
    coursebook.roster_xlsx_to_csv(source, target)

@cli_coursebook_roster.group(name='download')
def cli_coursebook_roster_download():
    pass

@cli_coursebook_roster.command(name='download', help='Download a CourseBook roster.')
@click.option('--force/--no-force', default=False, help="Overwrite existing file.")
@click.option('--new/--no-new', default=False, help="Get a new file (don't use the cache).")
@click.option('--get-password', default=None, help='Command to evaluate to get password (default is to ask).')
@click.option('--netid', default=None, help='Use this NetID.')
@click.argument('address', nargs=-1, type=click.STRING)
def cli_coursebook_roster_download(netid, get_password, new, force, address):

    def _split(x):
        y, f = os.path.splitext(x)
        return y, f[1:]

    for x in address:
        _, f = _split(x)
        if not (f in coursebook.ROSTER_FORMAT):
            raise click.ClickException("{x}: I don't know how to download a `{f}`, only: {these}.".format(x=x, f=f, these=' '.join(coursebook.ROSTER_FORMAT)))
        # FIXME: check for proper address format
        if os.path.exists(x) and not force:
            raise click.ClickException('File exists, maybe use --force?: ' + x)

    if netid is None: netid = config.NETID
    if get_password is None: get_password = config.get_password

    if netid is None:
        raise click.ClickException('You must either specify a NetID in {config} or with --netid.'.format(config.CONFIG_FILE))

    for x in address:
        y, f = _split(x)
        z = coursebook.roster_download(netid=netid, get_password=get_password, address=y, format=f, new=new)
        shutil.copyfile(z, x)

