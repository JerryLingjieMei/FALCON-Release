DATASET=$1
REMOTE=https://people.csail.mit.edu/jerrymei/projects/falcon

#NAME="cub_fewshot_builder"
#URL="${REMOTE}/assets/${NAME}.json"
#mkdir -p "output/${NAME}"
#echo "Downloading data from ${URL}"
#curl ${URL} | tar -zx -C ${DATASET}/.augmented/CUB-200-2011/0/${NAME}/questions.json

NAME="cub_fewshot"
URL="${REMOTE}/assets/${NAME}.json"
mkdir -p "output/${NAME}"
echo "Downloading data with parsed programs from ${URL}"
curl ${URL} | tar -zx -C ${DATASET}/.augmented/CUB-200-2011/0/${NAME}/questions.json

