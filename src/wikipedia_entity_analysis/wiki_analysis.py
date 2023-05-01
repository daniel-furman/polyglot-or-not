import json
from json import JSONDecodeError
import urllib.parse
import urllib.request

import os

import re
from ftfy import fix_text
from string import punctuation

import spacy

CODE_TO_LANG_DICT = {
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hr": "Croatian",
    "hu": "Hungarian",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "uk": "Ukrainian",
}

CODE_TO_WIKI_CLEANUP_DICT = {
    "ca": "Referèncie",
    "da": "Litteratur",
    "de": "Literatur",
    "en": "References",
    "es": "Referencias",
    "fr": "Notes et références",
    "hr": "Izvori",
    "it": "Note",
    "nl": "Literatuur",
    "pl": "Przypisy",
    "pt": "Referências",
    "ro": "Note",
    "ru": "Примечания",
    "sv": "Källor",
    "uk": "Література",
}

CODE_TO_SPACY_MODEL_DICT = {
    "ca": "ca_core_news_lg",
    "da": "da_core_news_lg",
    "de": "de_core_news_lg",
    "en": "en_core_web_lg",
    "es": "es_core_news_lg",
    "fr": "fr_core_news_lg",
    "hr": "hr_core_news_lg",
    "it": "it_core_news_lg",
    "nl": "nl_core_news_lg",
    "pl": "pl_core_news_lg",
    "pt": "pt_core_news_lg",
    "ro": "ro_core_news_lg",
    "ru": "ru_core_news_lg",
    "sv": "sv_core_news_lg",
    "uk": "uk_core_news_lg",
}


def load_spacy_models(code_to_spacy_model_dict):
    container = {}
    for lang, model in code_to_spacy_model_dict.items():
        container[lang] = spacy.load(model)

    return container


def get_mulitlingual_lookup(entity_analysis_df, code_to_lang_dict):
    # get lookup that connects the english form of an entity to its multilingual version
    # annoying that with the way the DF is set up right now, have to do manual cleanup to extract the translated forms
    # should update the other NB so that it ouptuts a well formatted json into the column
    target_entities_multiling = {}
    for row in entity_analysis_df.iterrows():
        d = row[1].alternate_forms
        for code in code_to_lang_dict.keys():
            d = d.replace("'" + code + "'", '"' + code + '"')

        d = d.replace(": '", ': "')
        d = d.replace("',", '",')
        d = d.replace("'}", '"}')
        d = d.replace('""', '"')
        try:
            d = json.loads(d)
            target_entities_multiling[row[1].entity] = d

        except JSONDecodeError:
            continue
    return target_entities_multiling


# for a given language, randomly sample <n> articles (max of 500).
# return a dict of their id and title.
def get_wikipedia_pages(lang, limit=500, debug=False):
    if limit > 500:
        limit = 500
    # construct URL for API call
    articles_url = f"https://{lang}.wikipedia.org/w/api.php?action=query&list=random&format=json&rnnamespace=0&rnlimit={str(limit)}&format=json"

    # grab data
    url = urllib.request.urlopen(articles_url)

    # read data
    data = url.read()

    # set encoding and load into obj
    encoding = url.info().get_content_charset("utf-8")
    obj = json.loads(data.decode(encoding))

    if "query" not in obj or "random" not in obj["query"]:
        if debug:
            print(f"Unable to grab articles from {lang} using URL {url}.")
        raise Exception

    mappings = obj["query"]["random"]
    ids = {}
    for m in mappings:
        ids[m["id"]] = m["title"]

    if debug:
        print(f"Fetched {len(ids)} articles from {lang} wikipedia")
    return ids


"""
for an inputted article_id:title combination
we want to hit:
https://en.wikipedia.org/w/api.php?action=query&format=json&titles=Kerala&prop=extracts&explaintext
good response - {"batchcomplete":"","query":{"pages":{"14958":{"pageid":14958,"ns":0,"title":"Kerala"
bad response  - {"batchcomplete":"","query":{"pages":{"-1":{"ns":0,"title":"Kerala","missing":""}}}}
"""


def get_article_info(article_title, pageid, lang, cleanup_str, debug=False):
    # val
    if article_title == "" or article_title is None:
        if debug:
            print("Can't parse empty title.")
        return {}

    if lang == "" or lang is None:
        if debug:
            print("Input a language.")
        return {}

    lang = lang.lower()

    url = ""

    # format title via quote escapes to ensure non-ascii chars can get handed off properly
    quoted_title = urllib.parse.quote(article_title)

    # construct url
    # where lang is the language we are requested
    # and quoted title refers to our article
    info_url = f"https://{lang}.wikipedia.org/w/api.php?action=query&format=json&titles={quoted_title}&prop=extracts&explaintext&format=json"

    if debug:
        print(f"calling {info_url} for info about {article_title} from {lang} wiki.")

    # grab data
    try:
        url = urllib.request.urlopen(info_url)
    except UnicodeDecodeError:
        print(
            f"could not decode API call for {article_title} on {lang} wiki; url is {info_url}."
        )
        return {}

    # read content
    data = url.read()

    # set encoding and load into obj
    encoding = url.info().get_content_charset("utf-8")
    obj = json.loads(data.decode(encoding))

    if "query" not in obj or "pages" not in obj["query"]:
        if debug:
            print(f"Error parsing response for {article_title} from {lang} wiki.")
        raise Exception

    # check for a 'missing'/bad response
    if -1 in obj["query"]["pages"].keys():
        if debug:
            print(f"No wiki data found for {article_title} on {lang} wiki.")
        return {}

    # get pageid of the returned article
    data_pageid = list(obj["query"]["pages"].keys())[0]

    if debug:
        print(
            f"retrieved pageid {data_pageid} corresponding to {article_title} on {lang} wiki."
        )

    # double check pageid matches the one returned by API
    if str(data_pageid) != str(pageid):
        if debug:
            print(
                f"id mismatch -- expected {pageid} but retrieved {data_pageid} for {article_title} on {lang} wiki."
            )
        return {}

    # check if text is properly returned
    if "extract" not in obj["query"]["pages"][data_pageid]:
        if debug:
            print(
                f"could not retrieve text from {pageid} {article_title} on {lang} wiki."
            )
        return {}

    # get text
    content = obj["query"]["pages"][data_pageid]["extract"]

    # fix text
    content = fix_text(content)

    # remove references and whatever is below that as well
    references_line = cleanup_str

    if "\n== " + references_line in content:
        content = content[0 : content.find("\n== " + references_line)]
    elif "\n=== " + references_line in content:
        content = content[0 : content.find("\n=== " + references_line)]
    else:
        if debug:
            print(
                f"Couldn't remove references for {article_title} with content {content} searching for {references_line}"
            )

    # light string substitutions
    content = content.replace("\n", " ")
    content = content.replace("=", " ")
    content = re.sub(r"\s{2,}", "", content)

    return {article_title: content}


# always search english and the native language in-case of translation inconsistencies
def count_entities_in_article(
    target_entities_multiling, article_content, spacy_model, lang, debug=False
):
    nlp = spacy_model

    all_entities = {}

    if article_content is None or bool(article_content) == False:
        if debug:
            print("article content is empty.")
        return {}

    article_title = list(article_content.keys())[0]
    article_text = list(article_content.values())[0]

    if (
        article_title is None
        or article_title == ""
        or article_text is None
        or article_title == ""
    ):
        if debug:
            print(f"Could not parse article content -- {article_content}")
        return {}

    if debug:
        print(f"Parsing article {article_title}.")

    doc = nlp(article_text)

    word_count = 0
    for token in doc:
        if token.text not in punctuation:
            word_count += 1

    if debug:
        print(f"{article_title} has {word_count} words.")

    for ent in doc.ents:
        if ent.label_ == "MISC":
            continue
        formatted_entity = ent.text + "___" + ent.label_
        if formatted_entity not in all_entities:
            all_entities[formatted_entity] = 1
        else:
            all_entities[formatted_entity] += 1

    if debug:
        print(
            f"{article_title} mentions {len(all_entities)} unique entities a total of {sum(list(all_entities.values()))} times."
        )

    target_entities = {}

    # look through our target entity mapping
    # which connects an english entity to the languages its translated into and that form
    for target_entity_english, translated_info in target_entities_multiling.items():
        # for all the language / entity combos
        for code, translated_entity in translated_info.items():
            # get the data for the one we care about
            if code == lang:
                # for every entity that spacy tagged
                for e in all_entities:
                    # pull out the plain text form
                    doc_entity = e.split("___")[0]
                    # if either the translated version (of lang <lang>) is in our target set
                    # or the english version is in our target set
                    # record that this article contains that target entity
                    if (
                        translated_entity == doc_entity
                        or target_entity_english == doc_entity
                    ):
                        target_entities[target_entity_english] = all_entities[e]

    if debug:
        print(
            f"{article_title} mentions {len(target_entities)} target entities a total of {sum(list(target_entities.values()))} times."
        )

    return word_count, all_entities, target_entities
