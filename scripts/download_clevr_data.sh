DATASET=$1
REMOTE=https://people.csail.mit.edu/jerrymei/projects/falcon

#NAME="clevr_fewshot_builder"
#URL="${REMOTE}/assets/${NAME}.json"
#mkdir -p "output/${NAME}"
#echo "Downloading data from ${URL}"
#curl ${URL} | tar -zx -C ${DATASET}/.augmented/CLEVR_v1.0/0/${NAME}/questions.json

NAME="clevr_fewshot"
URL="${REMOTE}/assets/${NAME}.json"
mkdir -p "output/${NAME}"
echo "Downloading data with parsed programs from ${URL}"
curl ${URL} | tar -zx -C ${DATASET}/.augmented/CLEVR_v1.0/0/${NAME}/questions.json

