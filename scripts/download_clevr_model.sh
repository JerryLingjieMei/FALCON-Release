REMOTE=https://people.csail.mit.edu/jerrymei/projects/falcon

#NAME="clevr_fewshot_build_0"
#URL="${REMOTE}/assets/${NAME}_0010000.pth"
#mkdir -p "output/${NAME}"
#echo "Downloading weights for parser from ${URL}"
#curl ${URL} | tar -zx -C output/${NAME}/checkpoints
#echo "output/${NAME}/checkpoints/${NAME}_0010000.pth" > output/${NAME}/last_checkpoint
#
#NAME="clevr_support_0"
#URL="${REMOTE}/assets/${NAME}_0050000.pth"
#mkdir -p "output/${NAME}"
#echo "Downloading pre-training weights from ${URL}"
#curl ${URL} | tar -zx -C output/${NAME}/checkpoints
#echo "output/${NAME}/checkpoints/${NAME}_0050000.pth" > output/${NAME}/last_checkpoint

NAME="clevr_fewshot_graphical_0"
URL="${REMOTE}/assets/${NAME}_0010000.pth"
mkdir -p "output/${NAME}"
echo "Downloading weights from ${URL}"
curl ${URL} | tar -zx -C output/${NAME}/checkpoints
echo "output/${NAME}/checkpoints/${NAME}_0010000.pth" > output/${NAME}/last_checkpoint
