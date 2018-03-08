Specifying path to CSV dataset file:
- `python sample.py --dataset_path /path/to/dataset/lyrics.csv` or
- `python sample.py` assumes `./dataset/lyrics.csv`
The cleaner assumes `./dataset/lyrics.csv`
- due to 3.6 package issues
- will try to resolve later if it becomes a big issue

NL feature gereartor assumes `./dataset/cleaned_lyrics.csv`
- possible sources of error:
  - key errors for words not in cmudict
  - skipped over some parts of speech we probably don't care about
  - some generalizations in POS tagging (for example, types of nouns were not separated)
