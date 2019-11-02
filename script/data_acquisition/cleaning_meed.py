import csv
import sys

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordTokenizer


CATEGORIES = {
    'education': {'outreach', 'recreation', 'interact'},
    'air': {'smog', 'fume', 'allergen', 'pollen', 'emission', 'exhaust', 'transit', 'ozone', 'greenhouse', 'wind'},
    'tree': {'forest', 'canopy', 'log', 'timber', 'wood', 'oak', 'cedar', 'land'},
    'water': {'river', 'riverway', 'lake', 'stream', 'creek', 'well', 'aquifer', 'groundwater', 'oil', 'watershed', 'drain', 'ocean', 'gulf', 'swamp', 'bay', 'bayou', 'pond'},
    'beauty': {'trash', 'cleanup', 'beautify', 'litter', 'recycle', 'reuse', 'parks', 'city', 'houston', 'dallas', 'austin', 'antonio'},
    'wildlife': {'fish', 'game', 'hunt', 'fisherman', 'poach', 'poacher', 'trout', 'bass', 'deer', 'armadillo', 'cattle'},
}


def clean_text(text):
    """Turns freeform text into a list of words that are lemmatized and stripped of unnecessary punctuation."""
    tokens = TreebankWordTokenizer().tokenize(text)

    wnl = WordNetLemmatizer()
    stems = [wnl.lemmatize(token.lower()) for token in tokens]

    return stems


def categorize(words, categories):
    ret = {}
    words_set = set(words)
    for catname, catset in categories.items():
        if catname in words or (words_set & catset):
            ret[catname] = 1
        else:
            ret[catname] = 0
    return ret


def main():
    load = []
    finname = sys.argv[1]
    foutname = sys.argv[2]
    with open(finname) as fin:
        fread = csv.DictReader(fin)
        for line in fread:
            load.append({'ein': line['EIN'], 'year': line['Real_Year'], 'mission': line['CombinedText']})

    categories_list = list(CATEGORIES.keys())
    for org in load:
        clean_mission = clean_text(org['mission'])
        orgcats = categorize(clean_mission, CATEGORIES)
        org['mission_clean'] = ' '.join(clean_mission)
        org.update({'category_{}'.format(cat): orgcats[cat] for cat in categories_list})

    with open(foutname, 'w') as fout:
        fwrite = csv.DictWriter(fout, fieldnames=['ein', 'year', 'mission', 'mission_clean'] + ['category_{}'.format(cat) for cat in categories_list])
        fwrite.writeheader()
        for org in load:
            fwrite.writerow(org)


if __name__ == '__main__':
    main()
