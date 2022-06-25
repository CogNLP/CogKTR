#/bin/bash
if [ $# -eq 0 ]
then
	echo "Please provide one folder tag!"
	exit
fi
if [ $# -ne 1 ]
then
	echo "Only one folder tag search is supported!"
	exit
fi

RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
folder_tag=$1

echo "Searching folder tag ${foldedr_tag}..."
filelist=$(ls experimental_result)
for file in ${filelist}
do
if [[ ${file} == $1* ]]
then
	echo -e "In folder ${RED}${folder_tag}${BLUE}${file#${folder_tag}}${NC}:"
	cat experimental_result/${file}/log.txt | grep Evaluate
fi
done
