Requirements:
- python 3.8.8
- numpy 1.19
- nltk 3.6.1
- matplotlib 3.3.4

How to Run:
- By default, you can run the provided script by simply installing the requirements (preferrably in a conda environment) and running
  - `python script.py`
- The default run options are described below. Configuration is done through environment variables, described below as well.
- You can run a particular configuration like so
  - `TOKENIZATION=treebank python script.py`
- To generate the election time plot as seen in the report, run:
  - `TOKENIZATION=tweet PRESERVE_CASE=False STRIP_HANDLES=False ELECTION=True python script.py`

- Accuracy
  - This is shown in `$STDOUT` against the testing data


Environment Variables:
- TOKENIZATION: tokenization policy for tweets
  - `naive`: interpret each tweet as a list of space-delimited words
  - `punctuation`: tokenize tweets based on punctuation
  - `treebank`: tokenize based on the prolific Penn Treebank Tokenizer
  - `tweet`: tokenize based on a twitter-specific NLTK-implemented tokenizer
- PRESERVE_CASE: (only for `tweet` tokenization) preserve the case of each tweet when building/evaluating the classifier
  - `True`
  - `False` or any other value
  - default (if undefined): `True`
- STRIP_HANDLES: (only for `tweet` tokenization) remove twitter handles/usernames when building/evaluating the classifier
  - `True`
  - `False` or any other value
  - default (if undefined): `False`
- ALPHA: loglikelihood smoothing factor
  - any non-negative integral value e.g. `1`,`10`,`100`,etc.
  - default (if undefined): `1`
- ELECTION: evaluate the election results, produce plots described in the report
  - `True`
  - `False` or anything else
  - default: `False`