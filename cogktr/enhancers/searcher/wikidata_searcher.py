from cogktr.enhancers.searcher.kilt_searcher import KnowledgeSource
from cogktr.enhancers.searcher import BaseSearcher
from qwikidata.sparql import (get_subclasses_of_item,
                              return_sparql_query_results)


class WikidataSearcher(BaseSearcher):
    """ WikidataSearcher can search Wikidata in order to find adjoin wiki entity.

    It will search 1-step adjoin entity or 2-step adjoin entity in wikidata Knowledge graph
    online (Wikidata sparql service) or offline (JSON dump)
    """
    def __init__(self, wikipedia_id):
        self.wikipedia_id = wikipedia_id
        ks = KnowledgeSource()
        self.wikidata_id = ks.get_page_by_id(self.wikipedia_id)["wikidata_info"]["wikidata_id"]

    def get_1step_adjoin_entity(self, online=False):
        if(online==True):
            sparql_query = """
            SELECT DISTINCT ?property ?propertynameLabel ?value ?valueLabel ?valueDescription
            WHERE {
              wd:%s ?property ?value.
              FILTER(STRSTARTS(STR(?property), "http://www.wikidata.org/prop/direct/"))
              FILTER(STRSTARTS(STR(?value), "http://www.wikidata.org/entity/"))
              SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
              ?propertyname wikibase:directClaim ?property.
            }
            """ % self.wikidata_id
            res = return_sparql_query_results(sparql_query)
            return res

        if(online==False):
            # TODO: use qwikidata search json dump
            pass

    def get_2step_adjoin_entity(self, online=False):
        if (online == True):
            sparql_query = """
                SELECT DISTINCT ?property1 ?property1nameLabel ?value1 ?value1Label ?value1Description ?property2 ?property2nameLabel ?value2 ?value2Label ?value2Description
                WHERE {
                    wd:%s ?property1 ?value1.
                    ?value1 ?property2 ?value2.
                    FILTER(STRSTARTS(STR(?property1), "http://www.wikidata.org/prop/direct/"))
                    FILTER(STRSTARTS(STR(?value1), "http://www.wikidata.org/entity/"))
                    FILTER(STRSTARTS(STR(?property2), "http://www.wikidata.org/prop/direct/"))
                    FILTER(STRSTARTS(STR(?value2), "http://www.wikidata.org/entity/"))
                    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                    ?property1name wikibase:directClaim ?property1.
                    ?property2name wikibase:directClaim ?property2.
                }
            """ % self.wikidata_id
            res = return_sparql_query_results(sparql_query)
            return res

        if (online == False):
            # TODO: use qwikidata search json dump
            pass


if __name__ =="__main__":
    g1 = WikidataSearcher(wikipedia_id=18978754)
    result1 = g1.get_1step_adjoin_entity(online=True)
    print("end")