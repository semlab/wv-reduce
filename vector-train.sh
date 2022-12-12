#!/bin/bash

BIN_DIR=../opensrc/word2vec/bin
DATA_DIR=../opensrc/word2vec/data

TEXT_DATA=$DATA_DIR/text8


for (( vector_size=50; vector_size<=500; vector_size+=50 ))
do 
	vector_data=$DATA_DIR/text8-wv-$vector_size.txt
	echo "Training size: $vector_size, save to: $vector_data"
	/usr/bin/time -f "%C - %E" -a -o log.txt $BIN_DIR/word2vec -train $TEXT_DATA -output $vector_data -cbow 0 -size $vector_size -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15 #&
done 
