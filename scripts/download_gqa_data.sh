DATASET=$1
REMOTE=https://people.csail.mit.edu/jerrymei/projects/falcon

#NAME="gqa_fewshot_builder"
#URL="${REMOTE}/assets/${NAME}.json"
#mkdir -p ${DATASET}/.augmented/GQA/0/${NAME}
#echo "Downloading data from ${URL}"
#curl ${URL} | tar -zx -C ${DATASET}/.augmented/GQA/0/${NAME}/questions.json

NAME="gqa_fewshot"
URL="${REMOTE}/assets/${NAME}.json"
#mkdir -p ${DATASET}/.augmented/GQA/0/${NAME}
echo "Downloading data with parsed programs from ${URL}"
curl ${URL} > ${DATASET}/.augmented/GQA/0/${NAME}/questions.json

