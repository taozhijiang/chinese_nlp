2nd International Chinese Word Segmentation Bakeoff - Data Release
Release 1, 2005-11-18

* Introduction

This directory contains the training, test, and gold-standard data
used in the 2nd International Chinese Word Segmentation Bakeoff. Also
included is the script used to score the results submitted by the
bakeoff participants and the simple segmenter used to generate the
baseline and topline data.

* File List

gold/       Contains the gold standard segmentation of the test data
            along with the training data word lists.

scripts/    Contains the scoring script and simple segmenter.

testing/    Contains the unsegmented test data.

training/   Contains the segmented training data.

doc/        Contains the instructions used in the bakeoff.

* Encoding Issues

Files with the extension ".utf8" are encoded in UTF-8 Unicode.

Files with the extension ".txt" are encoded as follows:

as_    Big Five (CP950)
hk_    Big Five/HKSCS
msr_   EUC-CN (CP936)
pku_   EUC-CN (CP936)

EUC-CN is often called "GB" or "GB2312" encoding, though technically
GB2312 is a character set, not a character encoding.

* Scoring

The script 'score' is used to generate compare two segmentations. The
script takes three arguments:

1. The training set word list
2. The gold standard segmentation
3. The segmented test file

You must not mix character encodings when invoking the scoring
script. For example:

% perl scripts/score gold/cityu_training_words.utf8 \
    gold/cityu_test_gold.utf8 test_segmentation.utf8 > score.ut8

* Licensing

The corpora have been made available by the providers for the purposes
of this competition only. By downloading the training and testing
corpora, you agree that you will not use these corpora for any other
purpose than as material for this competition. Petitions to use the
data for any other purpose MUST be directed to the original providers
of the data. Neither SIGHAN nor the ACL will assume any liability for
a participant's misuse of the data.

* Questions?

Questions or comments about these data can be sent to Tom Emerson,
tree@sighan.org.

