from qwikidata.sparql import (get_subclasses_of_item,
                              return_sparql_query_results)
from tqdm import tqdm
import pandas as pd
import time

with open('wikidata_id_list.txt', 'r') as f:
    lines = f.read().splitlines()

triples = pd.DataFrame(columns=['subjection', 'predicate', 'objection'])
count = 0

for wikidata_id in tqdm(lines):
    count += 1
    sparql_query = """
    SELECT DISTINCT ?label ?property1nameLabel ?value1Label
        WHERE {
            wd:%s ?property1 ?value1 .
            wd:%s rdfs:label ?label .
            FILTER (langMatches( lang(?label), "EN" ) )
            FILTER(STRSTARTS(STR(?property1), "http://www.wikidata.org/prop/direct/"))
            FILTER(STRSTARTS(STR(?value1), "http://www.wikidata.org/entity/"))
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            ?property1name wikibase:directClaim ?property1.
        }
    """ % (wikidata_id, wikidata_id)
    try:
        res = return_sparql_query_results(sparql_query)
        for triple in res['results']['bindings']:
            subjection = triple['label']['value']
            predicate = triple['property1nameLabel']['value']
            objection = triple['value1Label']['value']
            # triple_dict = {'subjection': subjection, 'predicate': predicate, 'objection': objection}
            triple_list = [subjection, predicate, objection]
            triples.loc[len(triples.index)] = triple_list
        time.sleep(0.5)
    except:
        print("Error in line %d" % count)

# print(triples)
triples.to_csv('wikidata.spo', sep='\t', index=False)
print("end")