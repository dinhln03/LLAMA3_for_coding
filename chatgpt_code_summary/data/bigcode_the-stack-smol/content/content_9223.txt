#!/usr/bin/env python

"""
This action will display a list of volumes for an account
"""

from libsf.apputil import PythonApp
from libsf.argutil import SFArgumentParser, GetFirstLine, SFArgFormatter
from libsf.logutil import GetLogger, logargs
from libsf.sfcluster import SFCluster
from libsf.util import ValidateAndDefault, NameOrID, IPv4AddressType, BoolType, StrType, OptionalValueType, SelectionType, SolidFireIDType
from libsf import sfdefaults
from libsf import SolidFireError, UnknownObjectError
import sys
import json

@logargs
@ValidateAndDefault({
    # "arg_name" : (arg_type, arg_default)
    "account_name" : (OptionalValueType(StrType), None),
    "account_id" : (OptionalValueType(SolidFireIDType), None),
    "by_id" : (BoolType, False),
    "mvip" : (IPv4AddressType, sfdefaults.mvip),
    "username" : (StrType, sfdefaults.username),
    "password" : (StrType, sfdefaults.password),
    "output_format" : (OptionalValueType(SelectionType(sfdefaults.all_output_formats)), None),
})
def AccountListVolumes(account_name,
                       account_id,
                       by_id,
                       mvip,
                       username,
                       password,
                       output_format):
    """
    Show the list of volumes for an account

    Args:
        
        account_name:       the name of the account
        account_id:         the ID of the account
        by_id:              show volume IDs instead of names
        mvip:               the management IP of the cluster
        username:           the admin user of the cluster
        password:           the admin password of the cluster
        output_format:      the format to display the information
    """
    log = GetLogger()
    NameOrID(account_name, account_id, "account")

    log.info("Searching for accounts")
    try:
        account = SFCluster(mvip, username, password).FindAccount(accountName=account_name, accountID=account_id)
    except UnknownObjectError:
        log.error("Account does not exists")
        return False
    except SolidFireError as e:
        log.error("Could not search for accounts: {}".format(e))
        return False

    log.info("Searching for volumes")
    try:
        all_volumes = SFCluster(mvip, username, password).ListActiveVolumes()
        all_volumes += SFCluster(mvip, username, password).ListDeletedVolumes()
    except SolidFireError as e:
        log.error("Could not search for volumes: {}".format(e))
        return False
    all_volumes = {vol["volumeID"] : vol for vol in all_volumes}

    attr = "name"
    if by_id:
        attr = "volumeID"
    account_volumes = [all_volumes[vid][attr] for vid in account.volumes]

    # Display the list in the requested format
    if output_format and output_format == "bash":
        sys.stdout.write(" ".join([str(item) for item in account_volumes]) + "\n")
        sys.stdout.flush()
    elif output_format and output_format == "json":
        sys.stdout.write(json.dumps({"volumes" : account_volumes}) + "\n")
        sys.stdout.flush()
    else:
        log.info("{} volumes in account {}".format(len(account.volumes), account.username))
        if account.volumes:
            log.info("  {}".format(", ".join([str(item) for item in account_volumes])))

    return True


if __name__ == '__main__':
    parser = SFArgumentParser(description=GetFirstLine(__doc__), formatter_class=SFArgFormatter)
    parser.add_cluster_mvip_args()
    parser.add_account_selection_args()
    parser.add_argument("--byid", action="store_true", default=False, dest="by_id", help="display volume IDs instead of volume names")
    parser.add_console_format_args()
    args = parser.parse_args_to_dict()

    app = PythonApp(AccountListVolumes, args)
    app.Run(**args)
