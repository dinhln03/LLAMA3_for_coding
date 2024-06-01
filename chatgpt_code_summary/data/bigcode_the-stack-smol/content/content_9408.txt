""" python site scraping tool """

import xml.etree.ElementTree as ET
from StringIO import StringIO
import unicodedata
import re
import requests
from BuildItParser import BuildItParser


def http_get(url):
    """ simple wrapper around http get """
    try:
        request = requests.get(url)
        # not concerned with returning nice utf-8, as only the urls count
        text = unicodedata.normalize('NFKD', request.text
                                     ).encode('ascii', 'ignore')
        return (text, 200)
    except requests.HTTPError as http_error:
        if request.status_code == 404:
            print "{} not found: {}".format(url, http_error)
            return ("", 404)
        else:
            # simplify all other errors as 500's
            print "error retrieving {}: {}".format(url, http_error)
            return ("", 500)


def process_html(html_page, this_parser):
    """ extract links from an html page """
    this_parser.feed(html_page)
    return {
        "int_links": this_parser.int_links,
        "ext_links": this_parser.ext_links,
        "static_links": this_parser.static_links
        }


def process_xml(xml_sitemap, regex):
    """ extract links from xml """
    site_map_paths = set([])
    url_paths = set([])
    try:
        # need to strip namespaces
        ns_stripped = ET.iterparse(StringIO(xml_sitemap))
        for _, element in ns_stripped:
            if '}' in element.tag:
                element.tag = element.tag.split('}', 1)[1]
        xml_root = ns_stripped.root
        for found_sitemap in xml_root.findall('sitemap'):
            sitemap_loc = found_sitemap.find('loc')
            new_sitemap = sitemap_loc.text
            new_path = regex.search(new_sitemap)
            if new_path is not None:
                site_map_paths.add(new_path.group(1))
        for found_url in xml_root.findall('url'):
            url_loc = found_url.find('loc')
            new_url = url_loc.text
            new_path = regex.search(new_url)
            if new_path is not None:
                new_path = new_path.group(1)
                url_paths.add(new_path)
    except Exception as XML_Error:
        print "Exception trying to parse sitemap: {}".format(XML_Error)
        raise XML_Error
    return (site_map_paths, url_paths)


def main():
    """ main function """
    site = "http://wiprodigital.com"
    site_regex = re.compile(r"{}(.+)$".format(site))

    site_structure = []

    paths_to_visit = set(["index.html", "index.php"])
    paths_visited = set([])

    sitemaps = set(["sitemap.xml"])

    sitemaps_still_to_process = True
    while sitemaps_still_to_process:
        # print "Processing paths..."
        num_sitemaps = len(sitemaps)
        for sitemap in sitemaps:
            sitemap_url = "{}/{}".format(site, sitemap)
            # print "sitemap: {}".format(sitemap_url)
            (xml, http_code) = http_get(sitemap_url)
            (sitemap_paths, url_paths) = process_xml(xml, site_regex)
            new_sitemaps = set([])
            for sitemap_path in sitemap_paths:
                new_sitemaps.add(sitemap_path)
            for url_path in url_paths:
                paths_to_visit.add(url_path)
        sitemaps = sitemaps.union(new_sitemaps)
        if num_sitemaps == len(sitemaps):
            sitemaps_still_to_process = False

    html_parser = BuildItParser(site_regex)

    paths_still_to_process = True
    while paths_still_to_process:
        num_paths = len(paths_to_visit)
        new_paths = set([])
        # print "Processing paths..."
        for path in paths_to_visit:
            if path not in paths_visited:
                page_url = "{}/{}".format(site, path)
                # print "page: {}".format(page_url)
                (page, code) = http_get(page_url)
                if code == 200:
                    new_page = process_html(page, html_parser)
                    new_page["path"] = path
                    site_structure.append(new_page)
                    for internal_link in new_page["int_links"]:
                        if internal_link not in paths_visited:
                            new_paths.add(internal_link)
            paths_visited.add(path)
        paths_to_visit = paths_to_visit.union(new_paths)
        if num_paths == len(paths_to_visit):
            # no new paths added
            paths_still_to_process = False

    print "SITE: {}".format(site)
    for page in sorted(site_structure, key=lambda p: p["path"]):
        print "PAGE: {}".format(page["path"])
        for int_link in page["int_links"]:
            print "   internal link: {}".format(int_link)
        for ext_link in page["ext_links"]:
            print "   external link: {}".format(ext_link)
        for static_link in page["static_links"]:
            print "   static link: {}".format(static_link)


if __name__ == "__main__":
    main()
