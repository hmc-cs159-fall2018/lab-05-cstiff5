# Answers

1. FINISH
2. The command-line interface takes in --store, which is the file where the probability is saved to after training, and --source, which is the file to read the word pairs from. You would run `python3 editDistance.py --store ed.pkl --source /data/spelling/wikipedia_misspellings.txt`.
3. Unigrams and bigrams are supported by the `LanguageModel` class.
4. `LanguageModel` adds a value of `alpha` to the numerator of the probability calculation (and a corresponding alpha factor to the denominator). This means that, instead of getting a zero probability, all words with no occurrence will get the same small probability (`alpha` divided by the sum of the counts of every word, plus `alpha` times the number of words).
5. `__contains__` returns `true` if a word `w` is in the vocabulary, and `false` otherwise.
6. `get_chunks` loops through each source file, and then reads in some number of lines `chunk_size` from the file. It then joins these lines into a string separated by new lines, and returns these lines.
7. `LanguageModel` takes in a store and a source, which perform exactly as they do in EditDistance. It also takes in a value of `alpha`, which is what we used in part (4), and `--vocab`, which is the maximum size for the vocab. We would run
   `python3 LanguageMode.py --store lm.pkl /data/gutenberg/*.txt`. We do not need to explicitly pass in `alpha` and the vocabulary size, as the values we want to use are the default values.
   Note: The formatting for `autocorrect` and the reddit_ispell.txt are totally different, so I couldn't use diff and just had to do this by hand.
8. ispell seems to be generally better than our spellchecker.
9. Our model does pretty well with spelling errors that are caused by swapping letters (for example, category from catagory). However, our model is horribly inconsistent - one time it'll get the correction, and in the next it won't. ispell attempts to make corrections to phrases that have missing spaces (such at "atleast" to "at least"), but doesn't catch all such errors (such as correcting "aswell" to "swell")
10. Our model seems to have a lot more changing of correctly spelled words than ispell does.
