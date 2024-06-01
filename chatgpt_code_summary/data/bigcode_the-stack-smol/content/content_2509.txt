"""Returns full pathname of backup directory."""

import os
import pytest
from pathlib import Path
from mklists.constants import CONFIGFILE_NAME
from mklists.returns import get_backupdir_path


def test_get_backupdir_path(tmp_path):
    """Returns backups Path named for default working directory."""
    os.chdir(tmp_path)
    Path(CONFIGFILE_NAME).write_text("config stuff")
    backdir = "_backups"
    datestr = "2020-01-03_1646"
    workingdir = Path("agenda")
    workingdir.mkdir()
    os.chdir(workingdir)
    actual = get_backupdir_path(backdir=backdir, now=datestr)
    expected = Path(tmp_path) / backdir / str(workingdir) / datestr
    expected_explicit = Path(tmp_path) / "_backups" / "agenda" / "2020-01-03_1646"
    assert actual == expected
    assert actual == expected_explicit


def test_get_backupdir_path_given_datadir(tmp_path):
    """Returns backups Path named for specified working directory."""
    os.chdir(tmp_path)
    Path(CONFIGFILE_NAME).write_text("config stuff")
    workingdir = Path(tmp_path).joinpath("todolists/a")
    workingdir.mkdir(parents=True, exist_ok=True)
    workingdir_shortname_expected = "todolists_a"
    backdir = "_backups"
    datestr = "2020-01-03_1646_06488910"
    actual = get_backupdir_path(datadir=workingdir, backdir=backdir, now=datestr)
    expected = Path(tmp_path) / backdir / workingdir_shortname_expected / datestr
    assert actual == expected


def test_get_backupdir_path_given_datadir_with_slash(tmp_path):
    """Returns backups Path named for specified working directory ending with slash."""
    os.chdir(tmp_path)
    Path(CONFIGFILE_NAME).write_text("config stuff")
    workingdir = Path(tmp_path).joinpath("todolists/a/")
    workingdir.mkdir(parents=True, exist_ok=True)
    workingdir_shortname_expected = "todolists_a"
    backdir = "_backups"
    datestr = "2020-01-03_1646_06488910"
    actual = get_backupdir_path(datadir=workingdir, backdir=backdir, now=datestr)
    expected = Path(tmp_path) / backdir / workingdir_shortname_expected / datestr
    assert actual == expected


def test_get_backupdir_path_raise_exception_if_rootdir_not_found(tmp_path):
    """Raises exception if no rootdir is found (rootdir is None)."""
    os.chdir(tmp_path)
    with pytest.raises(SystemExit):
        get_backupdir_path()
