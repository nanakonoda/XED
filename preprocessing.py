#!/usr/bin/env python
import argparse
from collections import Counter

from multiprocessing import Pool
from itertools import repeat

import pandas as pd

NEGATIVE_SENTIMENTS = { 1, 3, 4, 6 }
POSITIVE_SENTIMENTS = { 2, 5, 8 }
SENTIMENT_DICT = { 1 : 0,
                   3 : 0,
                   4 : 0,
                   6 : 0,
                   2 : 1,
                   5 : 1,
                   8 : 1,
                   }

# Make dictionary of annotations based on filename
def make_annotation_dict(filename):
    annotation_dict = {}
    with open(filename, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        try:
            sent, annotation = line.strip().split("\t")
            annotation = tuple(int(x) for x in annotation.split(","))
            annotation_dict.update( { sent : annotation } )

        except ValueError:
            pass

    return annotation_dict

# Read en-annotated.tsv and store English sentence with labels
def read_en_annotations():
    filename = "AnnotatedData/en-annotated.tsv"
    return make_annotation_dict(filename)

EN_WITH_ANNOTATIONS = read_en_annotations()

# Read annotations from projections
def read_projection_annotations(lang):
    filename = "Projections/" + lang + "-projections.tsv"
    return make_annotation_dict(filename)

# Create Danish reversed dictionary
def make_danish_dict():
    filename = "subtitle-retrieval/students/pairs-da.txt"
    with open(filename, "r") as f:
        lines = f.readlines()
    
    da_en_dict = { line.split("\t")[-1].strip() :
                          line.split("\t")[-2]
                            for line in lines
                            }
    return da_en_dict

DANISH_DICT = make_danish_dict()

# Create dictionary of English to L1
def make_en_lang_dict(lang):

    # Read in pairs-[lang].txt file
    filename = "subtitle-retrieval/students/pairs-" + lang + ".txt"
    with open(filename, "r") as f:
        lines = f.readlines()
    
    en_lang_dict = {}
    
    for line in lines:
        try:
            base_lang_filename, _, _, _, base_lang_sent, lang_sent = line.strip().split("\t")

            if base_lang_filename.split("/")[0] == "da":
                try:
                    en_sent = DANISH_DICT[base_lang_sent]
                    en_lang_dict.update({ en_sent : lang_sent })

                except KeyError:
                    pass
            
            else:
                en_sent = base_lang_sent
                en_lang_dict.update({ en_sent : lang_sent })

        except ValueError:
            # print(base_lang_sent)
            continue

    return en_lang_dict

# Find common sentences in dicts
def make_parallel_corpus(lang1, lang2):

    parallel_dict = {}

    en_lang1_dict = make_en_lang_dict(lang1)
    en_lang2_dict = make_en_lang_dict(lang2)

    lang1_dict = read_projection_annotations(lang1)
    lang2_dict = read_projection_annotations(lang2)

    if len(en_lang1_dict.keys()) < len(en_lang2_dict.keys()):
        loop_dict = en_lang1_dict
        other_dict = en_lang2_dict
        
    else:
        loop_dict = en_lang2_dict
        other_dict = en_lang1_dict
    
    for en_sent in loop_dict.keys():
        if en_sent in other_dict.keys():
            lang1_sent, lang2_sent = loop_dict[en_sent], other_dict[en_sent]
            
            try:
                annotation = EN_WITH_ANNOTATIONS[en_sent]
                parallel_dict.update({ (en_sent, lang1_sent, lang2_sent) : annotation })

            except KeyError:
                try:
                    annotation1 = lang1_dict[lang1_sent]
                    annotation2 = lang2_dict[lang2_sent]
                    if annotation1 == annotation2:
                        parallel_dict.update({ (en_sent, lang1_sent, lang2_sent) : annotation1 })

                except KeyError:
                    pass

    return parallel_dict

# Save parallel corpus
def save_parallel_corpus(filename, lang1, lang2):
    parallel_dict = make_parallel_corpus(lang1, lang2)

    parallel_df = []
    for i, dict_item in enumerate(parallel_dict.items()):
        sents_tuple, annotation_tuple = dict_item
        if len(sents_tuple) == 3:
            annotation_str = [str(annotation) for annotation in annotation_tuple]
            label = ",".join(annotation_str)
            parallel_row = {
                            "en_sentence" : sents_tuple[0],
                            lang1 + "_sentence" : sents_tuple[1],
                            lang2 + "_sentence" : sents_tuple[2],
                            "multi" : label,
                            "binary" : str(get_common_label(label))
            }
            
            parallel_row = pd.DataFrame(data=parallel_row, index=[i])
            parallel_df.append(parallel_row)
    parallel_df = pd.concat(parallel_df)
    parallel_df.to_csv("en_de_fr_pd.tsv", sep="\t", index=False)

def read_parallel_corpus(filename, lang1, lang2):
    df = pd.read_csv(filename, delimiter="\t")
    df.columns =["en_sentence", lang1 + "_sentence", lang2 + "_sentence", "multi", "binary"]
    df.dropna(inplace = True)
    return df

def check_sentiments(df):
    for x in df.index:
        if len(df.loc[x, "label"]) > 1:
            sentiments = df.loc[x, "label"].split(",")
            cnt = Counter([get_binary_sentiments_from_dict(s) for s in sentiments])
            if cnt[0] > 1 and cnt[1] > 1:
                print(x)

def collapse_sentiments(df):
    df["binary"] = df.apply(lambda row : get_common_label(row.label), axis=1)
    return df

def get_common_label(labels):
    cnt = Counter([get_binary_sentiments_from_dict(s) for s in labels.split(",")])
    return cnt.most_common(1)[0][0]

def get_binary_sentiments_from_dict(sentiment):
    try:
        return SENTIMENT_DICT[int(sentiment)]
    except KeyError:
        pass