import sys

sys.path.append("/home/chenyuheng/zhouyuyang/CogKTR")

from cogktr.enhancers.searcher.kilt_searcher import KiltSearcher
from cogktr.enhancers.searcher import BaseSearcher
from qwikidata.sparql import (get_subclasses_of_item,
                              return_sparql_query_results)


class WikidataSearcher(BaseSearcher):

    def __init__(self):
        super().__init__()
        self.ks = KiltSearcher()

    def search(self, wikipedia_id, step_num=1, result_num=2):

        # get wikidata_id
        wikidata_id = self.ks.get_page_by_id(wikipedia_id)["wikidata_info"]["wikidata_id"]
        triple_list = []

        # query wikidata service
        if step_num == 1:
            sparql_query = """
            SELECT DISTINCT ?label ?property1 ?property1nameLabel ?value1 ?value1Label
            WHERE {
                wd:%s ?property1 ?value1 .
                wd:%s rdfs:label ?label .
                FILTER (langMatches( lang(?label), "EN" ) )
                FILTER(STRSTARTS(STR(?property1), "http://www.wikidata.org/prop/direct/"))
                FILTER(STRSTARTS(STR(?value1), "http://www.wikidata.org/entity/"))
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                ?property1name wikibase:directClaim ?property1.
            } limit %d
            """ % (wikidata_id, wikidata_id, result_num)

            try:
                res = return_sparql_query_results(sparql_query)
                # print("finish query.")
                for triple in res['results']['bindings']:
                    subjection_id = wikidata_id
                    subjection_name = triple['label']['value']
                    predicate_id = triple['property1']['value'].split("/prop/direct/")[1]
                    predicate_name = triple['property1nameLabel']['value']
                    objection_id = triple['value1']['value'].split("/entity/")[1]
                    objection_name = triple['value1Label']['value']
                    triple_list.append({"subjection_id": subjection_id, "subjection_name": subjection_name,
                                        "predicate_id": predicate_id, "predicate_name": predicate_name,
                                        "objection_id": objection_id, "objection_name": objection_name})
                return triple_list
            except:
                print("Error in wikidata_id: {}".format(wikidata_id))

        elif step_num == 2:
            sparql_query = """
            SELECT DISTINCT ?label ?property1 ?property1nameLabel ?value1 ?value1Label ?property2 ?property2nameLabel ?value2 ?value2Label
                WHERE {
                    wd:%s ?property1 ?value1.
                    wd:%s rdfs:label ?label.
                    ?value1 ?property2 ?value2.
                    FILTER(STRSTARTS(STR(?property1), "http://www.wikidata.org/prop/direct/"))
                    FILTER(STRSTARTS(STR(?value1), "http://www.wikidata.org/entity/"))
                    FILTER(STRSTARTS(STR(?property2), "http://www.wikidata.org/prop/direct/"))
                    FILTER(STRSTARTS(STR(?value2), "http://www.wikidata.org/entity/"))
                    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                    ?property1name wikibase:directClaim ?property1.
                    ?property2name wikibase:directClaim ?property2.
                } limit %d
            """ % (wikidata_id, wikidata_id, result_num)

            try:
                res = return_sparql_query_results(sparql_query)
                for triple in res['results']['bindings']:
                    subjection_id = wikidata_id
                    subjection_name = triple['label']['value']
                    predicate1_id = triple['property1']['value'].split("/prop/direct/")[1]
                    predicate1_name = triple['property1nameLabel']['value']
                    objection1_id = triple['value1']['value'].split("/entity/")[1]
                    objection1_name = triple['value1Label']['value']
                    predicate2_id = triple['property2']['value'].split("/prop/direct/")[1]
                    predicate2_name = triple['property2nameLabel']['value']
                    objection2_id = triple['value2']['value'].split("/entity/")[1]
                    objection2_name = triple['value2Label']['value']
                    triple_list.append({"subjection_id": subjection_id, "subjection_name": subjection_name,
                                        "predicate1_id": predicate1_id, "predicate1_name": predicate1_name,
                                        "objection1_id": objection1_id, "objection1_name": objection1_name,
                                        "predicate2_id": predicate2_id, "predicate2_name": predicate2_name,
                                        "objection2_id": objection2_id, "objection2_name": objection2_name})
                return triple_list
            except:
                print("Error in wikidata_id: {}".format(wikidata_id))

        else:
            raise Exception("Step_num should not be more than 2.")


if __name__ == "__main__":
    searcher = WikidataSearcher()
    result1 = searcher.search(wikipedia_id=18978754, step_num=1, result_num=2)
    print(result1)
    print("end")
