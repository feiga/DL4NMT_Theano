#! /bin/bash

mkdir -p translated/init
mkdir -p translated/regression_init
python translate.py $1 data/dic/filtered_dic_en-fr.en.pkl data/dic/filtered_dic_en-fr.fr.pkl data/test/test_en-fr.en.tok translated/$2 -p $3 -k 4
perl multi-bleu.perl data/test/test_en-fr.fr.tok < translated/$2

