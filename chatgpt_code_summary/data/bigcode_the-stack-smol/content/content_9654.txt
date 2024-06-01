import copy
import json
import logging
import time
from elasticsearch import Elasticsearch
import es_search_functions
from common.mongo_client import getMongoClient
from apps.apis.search.keyword_list import keyword_list


class SuggestionCacheBuilder:
    MAX_LEVEL = 2

    def __init__(self, site_id, mongo_client):
        self.site_id = site_id
        self.tasks = []
        self.mongo_client = mongo_client

    def output_entry(self, cache_entry):
        self.mongo_client.updateSearchTermsCache(self.site_id, cache_entry)

    def process_task(self, terms, terms_count, level):
        cache_entry = {"terms": terms, "count": terms_count,
                       "categories": [],
                       "more_terms": []}

        if level > self.MAX_LEVEL:
            return

        current_time = time.time()
        if current_time - self.last_logging_time > 2:
            self.last_logging_time = current_time
            time_spent = time.time() - self.start_time
            current_tasks = len(self.tasks)
            if current_tasks != 0:
                logging.debug("Time Spent: %s | TASKS: %s/%s | %s"
                                % (time_spent, self.finished_tasks, current_tasks, 
                                   self.finished_tasks/float(current_tasks + self.finished_tasks)))

        if terms:
            size_limit = 1000
        else:
            size_limit = 1000000

        filter = {"term": {"available": True}}
        keywords_facets = {'terms': {'field': 'keywords'},
                                       "facet_filter": filter}
        if size_limit:
            keywords_facets["terms"]["size"] = size_limit

        facets = {'keywords': keywords_facets,
                  'categories': {'terms': {'field': 'categories', 'size': 5},
                                 "facet_filter": filter}
                 }

        body={"facets": facets,
              "filter": filter}
        if terms:
            body["query"] = es_search_functions.construct_query(" ".join(terms))
        es = Elasticsearch()
        res =  es.search(index=es_search_functions.getESItemIndexName(self.site_id), 
                                search_type="count",
                                body=body
                                )

        suggested_categories = []
        for facet in res["facets"]["categories"]["terms"]:
            if facet["count"] < terms_count:
                suggested_category = {"category_id": facet["term"], "count": facet["count"]}
                cache_entry["categories"].append(suggested_category)

        terms_to_check = []
        for kw in res["facets"]["keywords"]["terms"]:
            keyword_status = keyword_list.getKeywordStatus(self.site_id, kw["term"])
            if keyword_status == keyword_list.WHITE_LIST and kw["count"] < terms_count:
                terms_to_check.append(kw)

        for term_to_check in terms_to_check:
            cache_entry["more_terms"].append(copy.copy(term_to_check))
            new_terms = list(terms) + [term_to_check["term"]]
            new_terms.sort()
            self.tasks.append((tuple(new_terms), term_to_check["count"], level + 1))

        self.finished_tasks += 1
        self.output_entry(cache_entry)


    def rebuild(self):
        self.start_time = time.time()
        self.last_logging_time = time.time()
        self.finished_tasks = 0
        # FIXME
        logging.debug("Start to rebuild Suggestion Cache for site: %s" % self.site_id)
        self.tasks.append((tuple(), 1000000, 0))
        while len(self.tasks) > 0:
            task = self.tasks.pop(0)
            self.process_task(*task)
        finish_time = time.time()
        logging.debug("Finished to rebuild Suggestion Cache for site: %s, total time spent: %s seconds." % (self.site_id, finish_time - self.start_time))


def rebuild_suggestion_cache(site_id):
    mongo_client = getMongoClient()
    builder = SuggestionCacheBuilder(site_id, mongo_client)
    builder.rebuild()

