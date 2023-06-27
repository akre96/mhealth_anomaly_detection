ARTS_PATH=/Users/sakre/Code/dgc/Seabreeze/src/ARTS.pl
INPUT_FILE=../output/BRIGHTEN_v2_stratify.txt
OUTPUT_FILE=../output/BRIGHTEN_v2_train_test_split.txt

$ARTS_PATH \
    -i $INPUT_FILE \
    -o $OUTPUT_FILE \
	-c "2,3,4,5,6,7,8,9,10,11;2;3;4;5;6;7;8;9;10;11" \
	-b 107,46 \
	-cc 2,4