
###############################################################################################
#
# Script for training good word and phrase vector model using public corpora, version 1.0.
# The training time will be from several hours to about a day.
# (edit of the word2vec 8 billion word script)
#
# Downloads about 8 billion words,
# makes phrases using two runs of word2phrase, trains
#
###############################################################################################

OUTPUT_FILE="text-data.txt"
WORD2VEC=../../../opensrc/word2vec

# This function will convert text to lowercase and remove special characters
normalize_text() {
  awk '{print tolower($0);}' | sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
  -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
  -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
  -e 's/«/ /g' | tr 0-9 " "
}


wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz
wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz
gzip -d news.2012.en.shuffled.gz
gzip -d news.2013.en.shuffled.gz
normalize_text < news.2012.en.shuffled > $OUTPUT_FILE
normalize_text < news.2013.en.shuffled >> $OUTPUT_FILE

wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar -xvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
for i in `ls 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled`; do
  normalize_text < 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/$i >> $OUTPUT_FILE
done

wget http://ebiquity.umbc.edu/redirect/to/resource/id/351/UMBC-webbase-corpus
tar -zxvf umbc_webbase_corpus.tar.gz webbase_all/*.txt
for i in `ls webbase_all`; do
  normalize_text < webbase_all/$i >> $OUTPUT_FILE
done

wget http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
bzip2 -c -d enwiki-latest-pages-articles.xml.bz2 | awk '{print tolower($0);}' | perl -e '
# Program to filter Wikipedia XML dumps to "clean" text consisting only of lowercase
# letters (a-z, converted from A-Z), and spaces (never consecutive)...
# All other characters are converted to spaces.  Only text which normally appears.
# in the web browser is displayed.  Tables are removed.  Image captions are.
# preserved.  Links are converted to normal text.  Digits are spelled out.
# *** Modified to not spell digits or throw away non-ASCII characters ***

# Written by Matt Mahoney, June 10, 2006.  This program is released to the public domain.

$/=">";                     # input record separator
while (<>) {
  if (/<text /) {$text=1;}  # remove all but between <text> ... </text>
  if (/#redirect/i) {$text=0;}  # remove #REDIRECT
  if ($text) {

    # Remove any text not normally visible
    if (/<\/text>/) {$text=0;}
    s/<.*>//;               # remove xml tags
    s/&amp;/&/g;            # decode URL encoded chars
    s/&lt;/</g;
    s/&gt;/>/g;
    s/<ref[^<]*<\/ref>//g;  # remove references <ref...> ... </ref>
    s/<[^>]*>//g;           # remove xhtml tags
    s/\[http:[^] ]*/[/g;    # remove normal url, preserve visible text
    s/\|thumb//ig;          # remove images links, preserve caption
    s/\|left//ig;
    s/\|right//ig;
    s/\|\d+px//ig;
    s/\[\[image:[^\[\]]*\|//ig;
    s/\[\[category:([^|\]]*)[^]]*\]\]/[[$1]]/ig;  # show categories without markup
    s/\[\[[a-z\-]*:[^\]]*\]\]//g;  # remove links to other languages
    s/\[\[[^\|\]]*\|/[[/g;  # remove wiki url, preserve visible text
    s/{{[^}]*}}//g;         # remove {{icons}} and {tables}
    s/{[^}]*}//g;
    s/\[//g;                # remove [ and ]
    s/\]//g;
    s/&[^;]*;/ /g;          # remove URL encoded chars

    $_=" $_ ";
    chop;
    print $_;
  }
}
' | normalize_text | awk '{if (NF>1) print;}' >> $OUTPUT_FILE

$WORD2VEC/bin/word2phrase -train $OUTPUT_FILE -output data-phrase.txt -threshold 200 -debug 2
$WORD2VEC/bin/word2phrase -train data-phrase.txt -output data-phrase2.txt -threshold 100 -debug 2