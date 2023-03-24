devexp=$1
testexp=$2
finalexp=$3

mkdir -p /CVPR23-FAS-WILD/anti_spoof/finalout/exp${finalexp}

cp -a /CVPR23-FAS-WILD/anti_spoof/testout/exp${testexp}/config.json /CVPR23-FAS-WILD/anti_spoof/finalout/exp${finalexp}

cat /CVPR23-FAS-WILD/anti_spoof/devout/exp${devexp}/predict.txt /CVPR23-FAS-WILD/anti_spoof/testout/exp${testexp}/predict.txt > /CVPR23-FAS-WILD/anti_spoof/finalout/exp${finalexp}/predict.txt