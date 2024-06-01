import wikipedia as wiki
from ..parsing import get_wiki_page_id, get_wiki_lines, get_wiki_sections

def get_wiki_references(url, outfile=None):
    """get_wiki_references.
    Extracts references from predefined sections of wiki page
    Uses `urlscan`, `refextract`, `doi`, `wikipedia`, and `re` (for ArXiv URLs)

    :param url: URL of wiki article to scrape
    :param outfile: File to write extracted references to
    """
    def _check(l):
        return (not l['doi'] or l['doi'] == l['refs'][-1]['doi']) \
            and (not l['arxiv'] or l['arxiv'] == l['refs'][-1]['arxiv'])
    page = wiki.page(get_wiki_page_id(url))
    sections = get_wiki_sections(page.content)
    lines = sum([get_wiki_lines(s, predicate=any) for s in sections.values()], [])
    links = sum([wikiparse.parse(s).external_links for s in sections.values()], [])
    summary = sum([
        [
            {
                'raw': l,
                'links': urlscan.parse_text_urls(l),
                'refs': refextract.extract_references_from_string(l),
                'doi': doi.find_doi_in_text(l),
                'arxiv': m.group(1) if (m := arxiv_url_regex.matches(l)) is not None else None
            } for l in get_wiki_lines(s, predicate=any)
        ] for s in sections.values()
    ])
    failed = [ld for ld in summary if not _check(ld)]
    if any(failed):
        logger.warning('Consistency check failed for the following lines: {}'.format(failed))
    return _serialize(summary, outfile)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
