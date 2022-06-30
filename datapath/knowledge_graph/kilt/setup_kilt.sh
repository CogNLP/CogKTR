# install mongoDB
cd ~
mkdir mongodb
cd mongodb/
curl -O https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2004-5.0.9.tgz
tar xvf mongodb-linux-x86_64-ubuntu2004-5.0.9.tgz
mv mongodb-linux-x86_64-ubuntu2004-5.0.9 mongodb
cd mongodb
echo $PATH
export PATH=$PATH:~/mongodb/mongodb/bin

# start mongo server
mkdir data
cd bin
./mongod --dbpath ~/mongodb/mongodb/data &

# install mongo tools
cd ~
mkdir mongo-tools
cd mongo-tools/
curl -O https://fastdl.mongodb.org/tools/db/mongodb-database-tools-ubuntu2004-x86_64-100.5.2.tgz
tar xvf mongodb-database-tools-ubuntu2004-x86_64-100.5.2.tgz
mv mongodb-database-tools-ubuntu2004-x86_64-100.5.2 mongo-tools
cd mongo-tools
echo $PATH
export PATH=$PATH:~/mongo-tools/mongo-tools/bin

# download and import KILT data
wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
mongoimport --db kilt --collection knowledgesource --file kilt_knowledgesource.json

