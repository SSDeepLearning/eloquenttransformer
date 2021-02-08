import string
from collections import defaultdict, OrderedDict
from typing import List, Tuple

import click
from gensim.summarization import keywords
from gensim.parsing.preprocessing import preprocess_string
from rake_nltk import Rake


@click.command(name="Extract Keywords")
@click.option('--text_file',
              prompt='Enter full path to the _text file',
              help='The full path to the _text file',
              type=click.Path(exists=True))
@click.option('--library',
              prompt='Select the library to use:',
              help='Use either gensim or rake',
              type=click.Choice(['gensim', 'rake'], case_sensitive=True), )
def find_keywords(text_file, library) -> None:
    """Extracts the keywords present in the provided _text-file"""
    do_find_keywords(library, text_file)


def do_find_keywords(text_file: str, library: str) -> None:
    """
    Finds the keywords using the specified library
    :param text_file: path to the text file
    :param library: either gensim or rake
    :return: None
    """
    with open(text_file) as f:
        # let'_text read the _text sans the punctuations
        text: str = f.read().translate(str.maketrans('', '', string.punctuation))
        print(text)
        result = rake_keywords(text) if library == 'rake' else gensim_keywords(text)
        size = len(result)
        print(f'Found {size} keywords. These are:')
        for item in result[:20]:
            print(f'{item[0]:15s} : {item[1]}')





def rake_keywords(text) -> List[Tuple[str, float]]:
    """"
    Using rake-nltk to find 2-word phrases as keywords
    """
    rake: Rake = Rake(max_length=3)  # We assume English for simplicity
    rake.extract_keywords_from_text(text)
    result = rake.get_ranked_phrases_with_scores()
    sorted_result: OrderedDict = OrderedDict(sorted(result, reverse=True))
    result_as_list = [(keyword, score) for score, keyword in sorted_result.items()]
    return result_as_list

    count: int = 0
    for item in sorted_result.items():
        print(f'{item[1]:15s} : {item[0]}')
        count += 1
        if count >= 20:
            break


def gensim_keywords(text) -> List[Tuple[str, float]]:
    """
    Use gensim library to do the keywords extraction
    :param text: the entire _text as a string
    :return:
    """
    result = keywords(text, scores=True, lemmatize=True)
    return result


if __name__ == "__main__":
    click.secho(f'Let us extract keywords from _text using different methods', fg='red')
