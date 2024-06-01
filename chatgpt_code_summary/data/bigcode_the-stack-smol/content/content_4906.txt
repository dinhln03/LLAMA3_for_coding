import argparse
from datetime import datetime
from json import decoder
from os import path, mkdir, remove
from os.path import isfile
from threading import Thread
from time import sleep

try:
    from progress.bar import Bar
    import requests
    import termcolor

except ImportError:
    print("You are missing modules. Run \"python3 -m pip install -r requirements.txt\" to "
          "install them.")
    exit(0)


# print a message with a time stamp
def status(message):
    print("{0} {1}".format(datetime.now(), message))


# clean any temp files created during runtime
def cleanup():
    if isfile("runfile"):
        remove("runfile")


# main loop
def main():
    status("Fetching latest paste IDs...")

    # fetch latest 100 paste IDs
    fetch_limit = 100

    current_request = requests.get("https://scrape.pastebin.com/api_scraping.php?limit={0}".format(fetch_limit))
    current_json = []

    try:
        current_json = current_request.json()

    except decoder.JSONDecodeError:
        status(termcolor.colored("Unable to fetch latest paste IDs. Make sure your IP is whitelisted at "
                                 "https://pastebin.com/doc_scraping_api", "red"))
        cleanup()

        exit(0)

    status("Paste IDs fetched. Processing...")

    # clean up fetched ids
    cleaned_json = []
    for entry in current_json:
        if entry["key"] not in paste_ids:
            cleaned_json.append(entry)

    # create a progress bar and start downloading pastes if we have new ones
    if len(cleaned_json) is not 0:
        with Bar("Processing", max=len(cleaned_json), fill=">") as bar:
            for entry in cleaned_json:
                # download the raw paste data
                entry_request = requests.get("https://scrape.pastebin.com/api_scrape_item.php?i={0}"
                                             .format(entry["key"]))

                entry_content = entry_request.text
                path_file = path.join("files", "{0}.txt".format(entry["key"]))

                paste_ids.append(entry["key"])

                # if we have a provided keyword list, check for keywords
                if keywords is not None:
                    for keyword in keywords:
                        if keyword.upper() in entry_content.upper():
                            bar.suffix = "%(index)d/%(max)d " + termcolor.colored("[KEYWORD] Paste \'{0}\' contains "
                                                                                  "keyword \'{1}\'".format(entry["key"],
                                                                                                           keyword),
                                                                                  "green")

                            if args.noSorting is False:
                                path_file = path.join("files", keyword, "{0}.txt".format(entry["key"]))

                            with open(path_file, "w+", encoding='utf-8') as entry_file:
                                entry_file.write(entry_content)

                            break
                else:
                    with open(path_file, "w+", encoding='utf-8') as entry_file:
                        entry_file.write(entry_content)

                    bar.suffix = "%(index)d/%(max)d Saving paste \'{0}\'".format(entry["key"])

                bar.next()

        bar.finish()

    # otherwise, just say that we didn't have any new content
    else:
        status("No new pastes found, skipping downloads...")

    if args.infinite is False:
        if not isfile("runfile"):
            print()
            status("Runfile no longer found, exiting...")
            exit(0)

    skipped_pastes = fetch_limit - len(cleaned_json)
    if skipped_pastes != 0:
        status("Skipped {0} previously fetched pastes".format(skipped_pastes))

    status("Cleaning up internal ID list...")
    while len(paste_ids) > max_id_list_size:
        paste_ids.pop(0)

    # start 60 second loop
    status("Hibernating for 60 seconds...")
    with Bar("Hibernating", max=60, fill=">", suffix="") as bar:
        for i in range(60):
            sleep(1)
            bar.next()

        bar.finish()

    print()
    Thread(main()).start()


if __name__ == '__main__':

    AUTHOR = "SYRAPT0R"
    COPYRIGHT = "2019-2022"
    VERSION = "0.5.3"

    # parse arguments
    keywords = None

    parser = argparse.ArgumentParser(description="A script to scrape pastebin.com with optional keyword search")

    parser.add_argument("-k", "--keywords", help="A file containing keywords for the search")
    parser.add_argument("-i", "--infinite", help="Whether to run in infinite mode (Default: false)",
                        action="store_true", default=False)
    parser.add_argument("-nS", "--noSorting", help="Whether to sort keyword pastes into subdirectories",
                        action="store_true", default=False)

    args = parser.parse_args()

    status("STARTING PASTA SCRAPER {0}, (c) {1} {2}".format(VERSION, COPYRIGHT, AUTHOR))
    print()

    # make sure file directories exists
    if not path.isdir("files"):
        status(termcolor.colored("No file directory found, creating...", "yellow"))
        mkdir("files")

    if args.keywords is not None:
        try:
            with open(args.keywords, "r") as f:
                keywords = f.readlines()

        except IOError:
            status(termcolor.colored("Unable to load specified keyword file. Aborting...", "red"))
            exit(0)

        keywords = [keyword.strip() for keyword in keywords]

        # create subdirectories if required
        if args.noSorting is False:
            for keyword in keywords:
                current_path = path.join("files", keyword)
                if not path.isdir(current_path):
                    status(termcolor.colored("Creating directory {0}".format(current_path), "yellow"))
                    mkdir(current_path)

        status("Loaded {0} keywords".format(len(keywords)))

    # create paste ID index
    paste_ids = []
    max_id_list_size = 200

    # create non infinite file if needed
    if args.infinite is False:
        status("Creating run file...")

        f = open("runfile", "w+")
        f.close()
    else:
        status("Running in infinite mode...")

    # preparation done, enter main loop
    status("Entering main loop...")
    print()

    main()
