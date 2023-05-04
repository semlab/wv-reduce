#!/bin/bash

model=$1

WORD2VEC=../../opensrc/word2vec
GLOVE=../../opensrc/GloVe

OUTPUT_DIR=vectors/8billion/$model/train

BIN_DIR=$WORD2VEC/bin
CBOW=0

#CORPUS=data/text8
CORPUS=data/text-data-phrase2.txt
VOCAB_FILE=out/$1/vocab.txt
COOCCURRENCE_FILE=out/$1/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=out/$1/cooccurrence.shuf.bin
BUILDDIR=$GLOVE/build
#SAVE_FILE=vectors
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
#VECTOR_SIZE=50
MAX_ITER=15
WINDOW_SIZE=10
BINARY=0
NUM_THREADS=15
X_MAX=10

if [ ! -d $OUTPUT_DIR ]; then
	mkdir -p $OUTPUT_DIR
fi



train_glove(){
	if [ ! -e $VOCAB_FILE ]; then
		echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
		$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT \
			-verbose $VERBOSE < $CORPUS > $VOCAB_FILE
	fi
	if [ ! -e $COOCCURRENCE_FILE ]; then
		echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
		$BUILDDIR/cooccur -memory $MEMORY \
			-vocab-file $VOCAB_FILE \
			-verbose $VERBOSE \
			-window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
	fi
	if [ ! -e $COOCCURRENCE_SHUF_FILE ]; then
		echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
		$BUILDDIR/shuffle -memory $MEMORY \
			-verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
	fi

	output_file=$OUTPUT_DIR/$model-$vector_size.txt
	echo "$ $BUILDDIR/glove -save-file $output_file -threads $NUM_THREADS \
		-input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX \
		-iter $MAX_ITER -vector-size $vector_size -binary $BINARY \
		-vocab-file $VOCAB_FILE -verbose $VERBOSE"
	/usr/bin/time -f "%C - %E" -a -o log-$model.txt \
			$BUILDDIR/glove -save-file $output_file \
			-threads $NUM_THREADS \
			-input-file $COOCCURRENCE_SHUF_FILE \
			-x-max $X_MAX \
			-iter $MAX_ITER \
			-vector-size $vector_size  \
			-binary $BINARY \
			-vocab-file $VOCAB_FILE \
			-verbose $VERBOSE

}


train_word2vec(){
	output_file=$OUTPUT_DIR/$model-$vector_size.txt
	echo "Training size: $vector_size, save to: $output_file"
	/usr/bin/time -f "%C - %E" -a -o log-$model.txt \
			$BIN_DIR/word2vec -train $CORPUS \
			-output $output_file \
			-cbow $CBOW \
			-size $vector_size \
			-window $WINDOW_SIZE \
			-negative 25 \
			-hs 0 \
			-sample 1e-4 \
			-threads $NUM_THREADS \
			-binary $BINARY \
			-iter $MAX_ITER 
}


#train_all(){
#	output_folder=$glove_output_folder
#	train_glove
#	CBOW=0
#	output_folder=$skipgram_output_folder
#	train_word2vec
#	CBOW=1
#	output_folder=$cbow_output_folder
#	train_word2vec
#}



training() { 
	echo "Training Undefined $model" 
}
#output_folder="out/$model/train"
if [ "$model" = 'glove' ]; then
	mkdir -p $OUTPUT_DIR
	#training=$train_glove
	training() { 
		train_glove
	}
elif [ "$model" = 'skipgram' ]; then
	mkdir -p $OUTPUT_DIR
	CBOW=0
	#training=$train_word2vec
	training() { 
		train_word2vec
	}
elif [ "$model" = 'cbow' ]; then
	mkdir -p $OUTPUT_DIR
	CBOW=1
	#training=$train_word2vec
	training() { 
		train_word2vec
	}
elif [ "$model" = 'all' ]; then
	echo "Mada desu yo~ > (^_^)'"
	exit 1
	#glove_output_folder="out/glove/train"
	#skipgram_output_folder="out/skipgram/train"
	#cbow_output_folder="out/cbow/train"
	#mkdir -p $glove_output_folder
	#mkdir -p $skipgram_output_folder
	#mkdir -p $cbow_output_folder
	#training=train_all
else 
	echo "Enter glove, skipgram, cbow or all"
	exit 1
fi

for (( vector_size=50; vector_size<=1000; vector_size+=50 ))
do 
	training
done 
