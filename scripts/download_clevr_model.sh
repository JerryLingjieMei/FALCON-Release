REMOTE=https://people.csail.mit.edu/jerrymei/projects/falcon

#NAME="clevr_fewshot_build_0"
#URL="${REMOTE}/assets/${NAME}_0010000.pth"
#mkdir -p "output/${NAME}/checkpoints"
#echo "Downloading weights for parser from ${URL}"
#curl ${URL} > output/${NAME}/checkpoints/model_0010000.pth
#echo "output/${NAME}/checkpoints/model_0010000.pth" > output/${NAME}/last_checkpoint
#
#NAME="clevr_support_0"
#URL="${REMOTE}/assets/${NAME}_0050000.pth"
#mkdir -p "output/${NAME}/checkpoints"
#echo "Downloading pre-training weights from ${URL}"
#curl ${URL} > output/${NAME}/checkpoints/model_0050000.pth
#echo "output/${NAME}/checkpoints/model_0050000.pth" > output/${NAME}/last_checkpoint

NAME="clevr_fewshot_graphical_0"
URL="${REMOTE}/assets/${NAME}_0010000.pth"
mkdir -p "output/${NAME}/checkpoints"
echo "Downloading weights from ${URL}"
curl ${URL} > output/${NAME}/checkpoints/model_0010000.pth
echo "output/${NAME}/checkpoints/model_0010000.pth" > output/${NAME}/last_checkpoint
