# Use KILT to search wikidata
KILT uses MongoDB to retrieve the wikidata. 

If you have installed the MongoDB local server, just download the data and import to the database.

```shell
wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
mongoimport --db kilt --collection knowledgesource --file kilt_knowledgesource.json
```

If you use MongoDB for the first time, you can install the MongoDB local server, tools and import the wikidata by running `setup_kilt.sh`.

Or you can choose different versions of MongoDB from https://www.mongodb.com/try/download/community manually. 
Corresponding command line tools can be found at https://www.mongodb.com/try/download/database-tools .