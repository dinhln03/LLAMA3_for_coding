""" This module is intended to extend functionality of the code provided by original authors.
    The process is as follows:
    1. User has to provide source root path containing (possibly nested) folders with dicom files
    2. The program will recreate the structure in the destination root path and anonymize all
    dicom files.
"""

import argparse
import json
import logging
import logging.config
import random
from pathlib import Path
from typing import Optional

import pydicom

from dicomanonymizer.anonym_state import AnonState
from dicomanonymizer.dicom_utils import fix_exposure
from dicomanonymizer.simpledicomanonymizer import (
    anonymize_dicom_file,
    initialize_actions,
)
from dicomanonymizer.utils import (
    LOGS_PATH,
    PROJ_ROOT,
    ActionsDict,
    Path_Str,
    get_dirs,
    to_Path,
    try_valid_dir,
)

# setup logging (create dirs, if it is first time)
LOGS_PATH.mkdir(parents=True, exist_ok=True)
logging.config.fileConfig(
    PROJ_ROOT / "dicomanonymizer/config/logging.ini",
    defaults={"logfilename": (LOGS_PATH / "file.log").as_posix()},
    disable_existing_loggers=False,
)
logger = logging.getLogger(__name__)

_STATE_PATH = Path.home() / ".dicomanonymizer/cache"
_STATE_PATH.mkdir(parents=True, exist_ok=True)


def get_extra_rules(
    use_extra: bool,
    extra_json_path: Path_Str,
) -> Optional[ActionsDict]:
    """Helper to provide custom (project level/user level) anonymization
    rules as a mapping of tags -> action function.

    Args:
        use_extra (bool): If use extra rules.
        extra_json_path (Path_Str): Path to extra rules json file.
        It should be flat json with action as a key and list of tags as value.

    Returns:
        Optional[ActionsDict]: extra rules mapping (tags -> action function)
    """
    # Define the actions dict for additional tags (customization)
    extra_rules = None
    if use_extra:
        # default or user provided path to extra rules json file
        with open(extra_json_path, "r") as fout:
            extra_rules = json.load(fout)
        for key in extra_rules:
            tag_list = extra_rules[key]
            tag_list = [tuple(elem) for elem in tag_list]
            extra_rules[key] = tag_list
        extra_rules = initialize_actions(extra_rules)
    return extra_rules


def anonymize_dicom_folder(
    in_path: Path_Str, out_path: Path_Str, debug: bool = False, **kwargs
):
    """Anonymize dicom files in `in_path`, if `in_path` doesn't
    contain dicom files, will do nothing. Debug == True will do
    sort of dry run to check if all good for the large data storages

    Args:
        in_path (Path_Str): path to the folder containing dicom files
        out_path (Path_Str): path to the folder there anonymized copies
        will be saved
        debuf (bool): if true, will do a "dry" run
    """
    # check and prepare
    in_path = to_Path(in_path)
    try_valid_dir(in_path)

    out_path = to_Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing: {in_path}")
    # work itself
    in_files = [p for p in in_path.iterdir() if p.is_file()]

    if not in_files:
        logger.info(f"Folder {in_path} doesn't have dicom files, skip.")
        return

    if debug:
        # anonymize just one file
        f_in = random.choice(in_files)
        f_out = out_path / f_in.name
        try:
            anonymize_dicom_file(f_in, f_out)
        except Exception as e:
            logger.info(f_in)
            logger.exception(e)
            raise e
    else:
        for f_in in in_files:
            f_out = out_path / f_in.name
            try:
                anonymize_dicom_file(f_in, f_out, **kwargs)
            except Exception as e:
                logger.info(f_in)
                logger.exception(e)
                raise e


def anonymize_root_folder(
    in_root: Path_Str,
    out_root: Path_Str,
    **kwargs,
):
    """The fuction will get all nested folders from `in_root`
    and perform anonymization of all folders containg dicom-files
    Will recreate the `in_root` folders structure in the `out_root`

    Args:
        in_root (Path_Str): source root folder (presumably has
        some dicom-files inide, maybe nested)
        out_root (Path_Str): destination root folder, will create
        if not exists
    """
    in_root = to_Path(in_root)
    try_valid_dir(in_root)
    out_root = to_Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    in_dirs = get_dirs(in_root)

    state = AnonState(_STATE_PATH)
    state.init_state()
    state.load_state()

    def get_tags_callback(dataset: pydicom.Dataset):
        state.tag_counter.update(dataset.dir())

    logger.info(
        "Processed paths will be added to the cache, if cache exist and has some paths included, they will be skipped"
    )
    logger.info(
        f"if, you need to process data again delete files {_STATE_PATH}, please"
    )
    # will try to process all folders, if exception will dump state before raising
    try:
        for in_d in in_dirs:
            rel_path = in_d.relative_to(in_root)
            if str(rel_path) in state.visited_folders:
                logger.info(f"{in_d} path is in cache, skipping")
                continue
            else:
                out_d = out_root / rel_path
                anonymize_dicom_folder(
                    in_d, out_d, ds_callback=get_tags_callback, **kwargs
                )
                # update state
                state.visited_folders[str(rel_path)] = True
    except Exception as e:
        raise e
    finally:
        # before saving updated state let's flag tags not seen previously
        prev_state = AnonState(_STATE_PATH)
        prev_state.init_state()
        prev_state.load_state()
        new_tags = set(state.tag_counter.keys()).difference(
            prev_state.tag_counter.keys()
        )
        if new_tags:
            logger.warning(
                f"During the anonymization new tags: {new_tags} were present"
            )
        else:
            logger.info("No new tags werer present")
        # now we can save the current state
        state.save_state()


# Add CLI args
parser = argparse.ArgumentParser(description="Batch dicom-anonymization CLI")
parser.add_argument(
    "--type",
    type=str,
    choices=["batch", "folder"],
    default="batch",
    help="Process only one folder - folder or all nested folders - batch, default = batch",
)
parser.add_argument(
    "--extra-rules",
    default="",
    help="Path to json file defining extra rules for additional tags. Defalult in project.",
)
parser.add_argument(
    "--no-extra",
    action="store_true",
    help="Only use a rules from DICOM-standard basic de-id profile",
)
parser.add_argument(
    "--debug", action="store_true", help="Will do a dry run (one file per folder)"
)
parser.add_argument(
    "src",
    type=str,
    help="Absolute path to the folder containing dicom-files or nested folders with dicom-files",
)
parser.add_argument(
    "dst",
    type=str,
    help="Absolute path to the folder where to save anonymized copy of src",
)


def main():
    # parse args
    args = parser.parse_args()
    in_path = Path(args.src)
    out_path = Path(args.dst)
    debug = args.debug

    path = args.extra_rules
    if not path:
        path = PROJ_ROOT / "dicomanonymizer/resources/extra_rules.json"

    extra_rules = get_extra_rules(use_extra=not args.no_extra, extra_json_path=path)
    # fix known issue with dicom
    fix_exposure()
    msg = f"""
    Start a job: {args.type}, debug set to {args.debug}
    Will anonymize data at: {in_path} and save to {out_path}
    """
    logger.info(msg)
    # anonymize
    if args.type == "batch":
        anonymize_root_folder(
            in_path, out_path, debug=debug, extra_anonymization_rules=extra_rules
        )
    elif args.type == "folder":
        anonymize_dicom_folder(
            in_path, out_path, debug=debug, extra_anonymization_rules=extra_rules
        )
    logger.info("Well done!")


if __name__ == "__main__":
    main()
