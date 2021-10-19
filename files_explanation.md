# Dataset files and codes

## *annotated dataset* folder

This is the main folder to find all the annotations, as well as information necessary to link all the annotations with the original dataset in the folder *data*. If you want to use the final dataset, aggregated with all of the contributor's answers, you should use *aggregated_dataset.csv*; if you want to filter your own data, for instance by annotator trust, you can use the *full_dataset.csv* and aggregate the data yourself.

- *annotated_sentences_reference.csv*: dataset with references to IDs to easily link sentences, contexts, etc. It contains the following columns: 

| Column | Explanation |
| ---- | ---- |
| Pair ID | Unique ID of the pair of sentences and the surrounding context, in the following format: nDDXXXXbYYpZZWW. See explanation above.|
| Sentence 1 ID | The ID of the first sentence of the pair to annotate in the format nDDXXXX explained below. |
| Sentence 2 ID | The ID of the second sentence of the pair to annotate in the format nDDXXXX explained below. |
| Context buffer | Buffer of +- sentences given as context around the central sentence. Note that although we performed trials with different number of sentences in the context, we fixed this number to 7 and all the data in this dataset have a buffer of 7 sentences. |
| Context start | ID of the sentence that starts the context. This is always the central sentence ID minus the number of buffer sentences (7). For instance, for n02456b07p5559, the central sentence is 456 of debate 02, and the context start is at n020449 (456-7). |
| Context end | ID of the sentence that finishes the context. This is always the central sentence ID plus the number of buffer sentences (7). For instance, for n02456b07p5559, the central sentence is 456 of debate 02, and the context end is at n020463 (456+7). |
| Topic | The topic of the pair of sentences to annotate. |
| Context tags | The full context in text form with HTML tags, as shown to the crowd annotators. The pair of sentences to annotate were highlighted in blue. |
| Sentence 1 | First sentence of the pair in text form. |
| Sentence 2 | Second sentence of the pair in text form. |
| Context | The full context in text without HTML tags. |

- *full_dataset.csv*: dataset with **full** individual annotations by each contributor, with a total of 24710 entries. It contains the following columns:

Note: when we talk about "question", we refer to an annotation task of the argumentative relation between two pairs of sentences, for which the annotators were asked "what's the relation between these two sentences?"

| Column | Explanation |
| ---- | ---- |
| \_unit\_id | Unique ID of the pairs of sentences to annotate in the annotation platform. |
| \_golden | Boolean indicating whether the annotation was gold (test) or not. If it was a golden question, it should appear several times, since they were shown to many contributors to test their performance. |
| \_missed | Boolean indicating whether the annotator answered incorrectly (missed) the test question. It only contains a boolean if the current unit was a gold question, otherwise it's empty. |
| \_tainted | Boolean indicating whether this annotation was tainted or not. This means that if the "trust" in the contributor was lost after several annotations (for instance, they started doing a good job answering gold questions, but then they started failing all of them), all of their annotations were deemed "untrusted" or "tainted". To obtain high quality annotations, these tainted annotations should be filtered out. |
| \_trust | Trust in the annotator, calculated according to their track record of correctly answered gold questions throughout the task. It's simply the ration of correctly answered gold questions over the total number of gold questions shown. |
| \_worker\_id | Unique randomised annotator ID. Originally, the IDs were already randomised to avoid identifying any contributor, but we further anonymised those IDs to strip any reference to the annotation platform. |
| confidence | Self-reported confidence in the annotation given by the current annotator (see **Annotations** section). | 
| relation | Annotated relation between pair of sentences given by the current annotator (see **Annotations** section). |
| orig\_\_golden | Boolean indicating whether a quesiton was *originally* a test or gold question. By default, all annotations with *orig\_\_golden* == TRUE will also have *\_golden* == TRUE. However, in three instances, test questions that seemed to be too confusing for the annotators (and hence not appropriate test questions) were "hidden", meaning that they were not shown again. These pairs of sentences have annotations and a *relation\_gold* entry, but have *\_golden* == FALSE, which seems inconsistent. *orig\_\_golden* == TRUE tells us that they were originally gold questions, but not at the end of the full annotation.
| context | Context shown to the annotators, including HTML tags. |
| pair\_id | Unique identifying pair ID in the format nDDXXXXbYYpZZWW. See below for more detais. |
| relation\_gold | If the current annotation is a gold question, this is the "correct answer" that provided beforehand and against which annotators were tested. If not a gold question, it's empty. |
| relation\_gold\_reason |  If the current annotation is a gold question, this is the explanation of why this was the correct answer. The annotators were shown this explanation if they failed, so they could learn from their mistakes. If not a gold question, it's empty. |
| sentence\_1 | First sentence of the pair in text form. |
| setence\_2 | Second sentence of the pair in text form. |
| speaker\_1 | Speaker of the first sentence. |
| speaker\_2 | Speaker of the second sentence. |
| topic | Topic of discussion during that part of the debate. |


- *aggregated_dataset.csv*: dataset with final annotations, aggregated from previuos dataset. Therefore, each row refers to a unique pair of sentences and does not contain information about annotators. It contains the following columns:

| Column | Explanation |
| ---- | ---- |
| \_unit\_id | Unique ID of the pairs of sentences to annotate in the annotation platform. |
| \_golden | Boolean indicating whether the annotation was gold (test) or not. If it was a golden question, it should appear several times, since they were shown to many contributors to test their performance. |
| \_unit\_state | String that indicates whether the pair of sentences were a *"golden"* pair or it was *"finalized"* (meaning the annotation had stopped after reaching sufficient agreement or maximum number of annotations). |
| \_trusted\_judgements | Number of judgements (annotations) received to compute the final annotation. Notice that gold questions were annotated many times, while normal pairs of sentences were only annotated 3-7 times.|
| confidence | **Average** self-reported confidence in the annotation (see **Annotations** section). | 
| confidence:stddev | **Standard deviation** of the self-reported confidence. | 
| relation | **Most likely** argumentative relation between the pair of sentences (see **Annotations** section). The most likely argumentative relation is decided according to the highest confidence score (next row). |
| relation:confidence | **Confidence score** in the argumentative relation between the pair of sentences. This is calculated as a weighted average of the annotators' trust when doing the judgement. |
| orig\_\_golden | Boolean indicating whether a quesiton was *originally* a test or gold question. By default, all annotations with *orig\_\_golden* == TRUE will also have *\_golden* == TRUE. However, in three instances, test questions that seemed to be too confusing for the annotators (and hence not appropriate test questions) were "hidden", meaning that they were not shown again. These pairs of sentences have annotations and a *relation\_gold* entry, but have *\_golden* == FALSE, which seems inconsistent. *orig\_\_golden* == TRUE tells us that they were originally gold questions, but not at the end of the full annotation.
| confidence\_gold |
| context | Context shown to the annotators, including HTML tags. |
| pair\_id | Unique identifying pair ID in the format nDDXXXXbYYpZZWW. See below for more detais. |
| relation\_gold | If the current annotation is a gold question, this is the "correct answer" that provided beforehand and against which annotators were tested. If not a gold question, it's empty. |
| relation\_gold\_reason |  If the current annotation is a gold question, this is the explanation of why this was the correct answer. The annotators were shown this explanation if they failed, so they could learn from their mistakes. If not a gold question, it's empty. |
| sentence\_1 | First sentence of the pair in text form. |
| setence\_2 | Second sentence of the pair in text form. |
| speaker\_1 | Speaker of the first sentence. |
| speaker\_2 | Speaker of the second sentence. |
| topic | Topic of discussion during that part of the debate. |

### Explanation of the ID format

Sentence ID's are in the format nDDXXXX. The first 2 digits (DD) after n indicate the debate number and the following 4 digits (XXXX) indicate the sentence number within that debate. For instance, n050153 refers to the sentence number 153 in the debate 05.

Pair of sentences with a context have an ID in the format nDDXXXXbYYpZZWW. The first part, nDDXXXX, represents the unique ID of central sentence of the context provided and is defined as just above. For instance, n020456 refers to the sentence number 456 in the debate 02. The following two digits after b (YY) indicate the number of buffer sentences given with the context. For instance, in n020456b07, 7 sentences before and after sentence 456 in debate 02 were given in the context. (Note that although we performed trials with different number of sentences in the context, we fixed this number to 7 and thus all the data in this dataset have a buffer of 7 sentences.) The next 4 digits after p (ZZWW) indicate the last two digits of the first sentence (ZZ) and the second sentence (WW) of the pair. For instance, n020456b07p5559, refers to the central sentence 456 within debate 02, that was presented with +- 7 sentences as context, and the pair of sentences to annotate were number 455 and 459.

## *data* folder

This folder contains all the data from the original dataset, the processed (tokenised, etc.) dataset, and the audio clips to run multimodal models. Each debate audio clip is split in different files to either decrease its size or remove applause, music, ads... The clips corresponding to each utterance are not provided, as they are a large amount of files, but instructions on how to generate them are given in the **codes** section. Some basic information about the dataset is shown here:

![image](https://user-images.githubusercontent.com/13152269/137493368-19962b44-a1b5-4962-b65c-3bce81e7f7ea.png)

Inside this folder, there are six subforlders:

* *audio sentences*: this is an empty folder, but it's where the individual clips of each utterance will be saved if the code is run.
* *original data*: it contains 10 files, 5 in .csv format and 5 in .txt format, each pair corresponding to one of the debates. These are the files in the original dataset (https://www.kaggle.com/headsortails/us-election-2020-presidential-debates). We don't provide the full auido (only the split version) due to size constraints, but they can be downloaded in the link above. The .txt files were not useful for our task. In the .csv files, the timestamp didn't correspond exactly with the audio provide, so we had to do our own text alignment, as explained in the paper and the readme file. The names of the files correspond to the debate numbers as follows:

| Debate number | File name |
| --- | --- |
| 1 | us_election_2020_1st_presidential_debate |
| 2 | us_election_2020_2nd_presidential_debate |
| 3 | us_election_2020_biden_town_hall |
| 4 | us_election_2020_trump_town_hall |
| 5 | us_election_2020_vice_presidential_debate | 

* *preprocessed full dataset*: this folder contains two .csv files. *dataset\_context\_definition.csv* contains the tokenised full dataset (split by sentences and concatenated) whose columns are their assigned unique *id*, *text*, *speaker*, a unique numeric *speaker\_id*, *audio\_file* id, the *context* and the *debate* id, considering that they've been split in different files (thus having 17 different debate files). This files is called "context definition" because the *context* or topic was manually assigned by one of the authors, taking into account what they were talking about and the sections the moderators introduced. Details of the different topics can be found on the paper. The unique ID of the sentences ar ein the format nDDXXXX explained above. Each sentence has associated an audio clip of the utterance with an ID in the format aDDXXXX, identical to the sentences ID, only changing the first letter to indicate audio. The other file, *full\_feature\_extraction\_dataset.csv* is almost identical fo the previous one, but it conaints an extra *timestamp* column, that links it to a timestamp in the folder *timestamps* (see below).
* *split audio*: this folder contains the audio files of the debates after being split in different sub-files. 
* *timestamps*: this folder contains the subdebate files after being "force-aligned" to the audio files in the previous folder (as explained in the readme file), as returned by the *aeneas* tool. The first column is the timestamp assigned to that utterance. Second and third columns are the starting and finish time of the utterance (in s). The fourth column is the text that was aligned to the audio. 
* *tokenised text*: this folder contains 10 files, 5 in .txt and 5 in .csv format, corresponding to each debate. This is the result of tokenising the original dataset found in the *original data* folder with the code **ADDCODE**. The files with the *\_plain.txt* ending are simply the sentences in plain format, separated by a line break, as this was the format necessary for the *aeneas* tool. The files with *|_split.csv* ending are the result of tokenising the dataset, and it contains only the speaker and the tokenised text.


## *codes* folder

This folder contains the Python codes needed to replicate the results of the paper. 

### *M-arg\_preprocessing.py*

This script performs the preprocessing of the original dataset and creates the audio clips to run the models. 

- Accesses the original data (files in folder *\data\original data*), tokenises the sentences (and saves them in folder *\data\tokenised text*), and assigns IDs to sentences and speakers.
- Accesses the timestamps (in folder *\data\timestamps*) and splits the audio files (in folder *data\split audio*) by utterances, with a buffer of +-2 s (by default, can be changed) and saves the clips (in folder *data\audio sentences*). Remember that the folder *audio sentences* is empty in our GitHub, and this code should be run until one (modifying the buffer time of 2 s if wanted) to obtain these clips and be able to run the models.
- Merges all of this information and creates the file *data\preprocessed full dataset\full_feature_extraction_dataset.csv*, explained above.

This script has been tested with Python 3.8, although it should work with Python >=3.0 (but it hasn't been tested). The following Python modules of at least these versions need to be installed:

| Module | Version |
| --- | --- |
| pathlib | 1.0.1 |
| pandas | 1.2.5 |
| pydub | 0.25.1 |
| nltk | 3.6.2 |

### *dataset\_statistics.py*

This Jupyter Notebook reproduces some descriptive statistics of the dataset, including the figures shown in the paper, Krippendorff's aggreement, and others.

This script has been tested with Jupyter Notebook using Python 3.8.5, although it should work with Python >=3.0 (but it hasn't been tested). The following Python modules of at least these versions need to be installed for the overall functioning of the script:

| Module | Version |
| --- | --- |
| pathlib | 1.0.1 |
| pandas | 1.2.5 |
| numpy | 1.20.2 |
| seaborn | 0.11.1 |
| matplotlib | 3.3.4 |
| joypy | 0.2.5 |
| nltk | 3.6.2 |

### *multimodal\_model.py*

This Python script runs the text-only, audio-only and multimodal models whose performance metrics are shown in the README file. Detailed information about the architecture of the models can be found in the paper. 

This script has been tested with Python 3.8, although it should work with Python >=3.0 (but it hasn't been tested). The following Python modules of at least these versions need to be installed:

| Module | Version |
| --- | --- |
| pathlib | 1.0.1 |
| pandas | 1.2.5 |
| tensorflow | 2.4.1 |
| tensorflow_hub | 0.12.0 |
| pydub | 0.25.1 |
| numpy | 1.20.2 |
| librosa | 0.8.1 |
| matplotlib | 3.3.4 |
| sklearn | 0.24.2 |
| seaborn | 0.11.1 |
| tqdm | 4.61.2 |

To run the following models, we recomend:

- text-only:
- audio-only:
- multimodal:

The script saves as results the confusion matrix of the model in .png and .svg and the classification report in .csv file. In order to not overwrite the results, we recomend modifying the variable "run_nb" with a different number each time. 
