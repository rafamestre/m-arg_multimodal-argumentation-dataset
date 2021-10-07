## M-Arg: MultiModal Argument Mining Dataset for Political Debates with Audio and Transcripts

This dataset contains audio/transcripts from 5 US 2020 presidential debates annotated for argumentation mining tasks, in particular for support and attack relations. 

### The data

The dataset contains audio/transcripts from five different debates occuring during the US 2020 presidential elections. The source material from these debates were a Kaggle competition (https://www.kaggle.com/headsortails/us-election-2020-presidential-debates, version 7, last accessed in August 9th, 2021) with a CO: Public Domain license. 

The first of the debates took place between the incumbent President Donald Trump and his Democrat opponent, former Vice-president Joe Biden, on September 29th, 2020, in Cleveland, Ohio, moderated by Fox News anchor Chris Wallace. The second debate in the files (which was actually the last one in time) also took place between both presidential candidates on October 22nd, 2020, at Belmont University, Nashville, moderated by NBC News' journalist Kristen Weller. The third and fourth presidential debates in the files were not debates in the strict sense, due to COVID-19 restrictions. After Donald Trump tested positive for the disease and declined to take part in a virtual debate, both debates were transformed into socially distanced town hall events, in which Joe Biden and Donald Trump alone answered questions from the public. The third debate used was thus Joe Biden's town hall event, moderated by George Stephanopoulos on ABC, while the fourth debate used was Donald Trump's town hall event, moderated by Savannah Guthrie on NBC. Both of them took place on the same day, October 15th, 2020. Finally, the last debate corresponds to the vice-presidential debate between the incumbent Vice-president Mike Pence and California Senator Kamala Harris, taking place on October 7th, 2020, at Kingsbury Hall, Salt Lake City, Utah and moderated by Susan Page from USA Today.

The original data was presented as audio in .mp3 files and transcriptions in both .txt and .csv files. The .csv files contained three columns: *speaker*, *minute*, and *text*. Since the timestamps did not align perfectly to the audio clips, we performed our own tokenisation and text-audio alignment. The M-Arg dataset associates each sentence with a matched timestamp in the corresponding debate audio file. To do this, each text was split into utterances, defined as single sentences delimited by a full stop, using the sentence tokenizer *PunktSentenceTokenizer* from www.nltk.org. The utterances were then force-aligned to the audio using the web application of the *aeneas* (https://aeneasweb.org/help, and GitHub https://github.com/readbeyond/aeneas/), obtaining new timestamps. The source audio files were split into different files (17 in total) to comply with the file size limit for the force alignment and to avoid segments where the debate was starting, finishing or going to a break, applause, music, etc. The column *speaker* contained the names of the speakers, which were homogenised across the different files to avoid duplicate names. For example, instances of "President Donald J. Trump" and "Donald Trump" were simply labeled as "Donald Trump". The *minute* column provided the timestamps in the format "mm:ss". However, it was observed that those timestamps did not match exactly with the audio, probably due to the fact that transcripts and audio files were obtained individually so we performed our own audio alignment. Finally, the *text* column contained utterances by the speakers, but they neither corresponded to single sentences nor full uninterrupted speeches. It is unclear what the criteria for splitting the text were in the original dataset. 

### Pairs selection

The M-Arg dataset consists of 4104 labelled pairs of sentences selected from the debates. Sections of the debates were manually labelled by the authors for their "topic", following the explanations of the moderator introducing each section, obtaining high level classifications like "foreign policy". Excerpts of 15 sentences were randomly selected (the "context") and a pair of sentences within the context was chosen to classify their relation (with their distance weighted by a Gaussian distribution to ensure they were close enough). Approximately 1500 sentences were forced to be from different speakers, to balance the dataset by increasing the possibility of finding attack relations.

In each iteration, one of the 17 debate clips was randomly selected, weighted by the total number of sentences it contained. Notice that part 1 of debate number 4 (Trum Town Hall event) was removed from the sentence pair generation task because it was a very short-lived segment between ads that did not contain any meaningful interactions between speakers and was difficult to work with. This clip is still presented in the raw data, but no annotations were made from it. Then:

  1. A random sentence from the debate clip was selected using a uniform distribution.
  
  2. From this sentence, a context surrounding it was delimited by a buffer of +- 7 sentences (15 in total). 
  
  3. If the first and the last sentence of the context were part of different "topics", steps 1 and 2 were repeated to ensure that the whole context was about the same topic.
  
  4. From this context, a random first sentence was selected by generating a random Gaussian number centered around the central sentence and with a standard deviation of 2 sentences.
  
  5. A second sentence was randomly selected from a Gaussian distribution centered around the first sentence with a standard deviation of 7 sentences. Only sentences that appeared after the first sentence were considered, to ensure that the first sentence presented to the annotators was always the first one in time.
  
  6. It was ensured that this specific pair of sentences had not been generated before by comparing it with previously generated pairs.

Ensuring that sentences were always from the same "topic" was meant to maximise that potential for finding argumentative relations, as debaters were talking about the same topic. Delimiting the context to 15 sentences and weighting the random generation of the second sentence by a Gaussian distribution ensured that the two sentences were close enough to maximise the possibility of finding argumentative relations. These two decisions were meant to decrease the number of labels of "neither" support nor attack by increasing the possibilities that "support" or "attack" relation was found.

Iterations and tests with the crowdsourced annotation helped us discover that despite these efforts the number of annotations showing "attack" were a great minority in the dataset. We also noticed, as could be expected, that most of the "attack" annotations were from pairs of sentences uttered by different people, for instance Donald Trump against Joe Biden. To balance the dataset we employed two slightly modified sentence pair generation methods: i) random pairs of sentences either from the same or different people were generated, without including interventions of the moderators (~ 2500 pairs); and ii) pairs of sentences from different speakers were generated to increase the possibility of finding "attack" relations (~ 1500 pairs; some of these also included moderators' interventions).

  1. Only pairs of sentences from "allowed speakers", namely Donald Trump, Joe Biden, Mike Pence and Kamala Harris, were selected, without differentiating between same-speaker and different-speaker cases. This was decided after most of the utterances by the moderators were not useful for this purpose, containing sentences like "Mr. President, two minutes" or "go ahead", which bear no argumentative meaning. Roughly 2500 sentences were generated in this way.
    
  2. Pairs of sentences from different speakers were generated to potentially increase the number of "attack" relations in the dataset. In debate number 1, 2 and 5, only the "allowed speakers" mentioned above were taken into consideration. However, since debates 2 and 3 were individual  with only one of those speakers interviewed by the moderator, the audience members and with responses by the candidate, pairs of sentences including the presenter and the audience members were allowed. Roughly 1000 sentences were generated in this way.

### Crowd-sourced annotation

For the task of crowdscourcing annotation, the platform Appen was used (https://appen.com/). Crowd annotators were presented with a pair of sentences, randomly selected according to the details in the previous subsection, and the topic of the debate. After this, they were provided with a short context of 15 sentences, indicating the name of the speakers, and highlighting in colour the sentence pair within the context. They were asked to focus on those two sentences and read around them to find the context in order to classify the argumentative relation between the pair. Finally, they were asked to classify their self-confidence level in the annotation they had just provided on a Likert scale, ranging from 1: "not confident at all" to 5: "very confident". Each worker was then paid per "page" of work completed, with each page containing between 4-6 tasks. The Appen platform allowed for several quality settings to maximise the quality of the annotation. These included:

* **Contributors' qualifications:** Four different levels of contributor selection, from 0 to 3, are available to launch an annotation task. The higher the level, the more experienced and accurate the contributors have been shown to be in previous jobs. Given its difficulty, our annotation was only advertised for Level 3 contributors. Although the higher the level, the longer it might take a job to finish due to shortage of contributors, a full annotation of roughly 1000 pairs of sentences took on average less than a day.
     
* **Geography and language:** In order to avoid biases from different countries or languages, the location of the contributors was filtered to include only the United States of America. As this is a web-based application, an option to disable Google Translate in their browser was also selected to avoid contributors speaking different languages relying on translation tools.
    
* **Contributors:** The annotation task was launched only to the contributors of the highest level on the platform, that were more experienced. In order to avoid biases from different countries, the location of the contributors was filtered to include only the United States of America. 

* **Test questions:** Test or "gold" questions were previously annotated by the researchers. Before starting the task for the first time, the contributors had to answer a quiz containing only a random subset of gold questions ("quiz mode"). If they answered correctly above a specified threshold, they were allowed to contribute to the task; if they did not, they were not allowed to continue. Once they passed the quiz, they completed the annotations one page at a time. In each page, one of the questions was always a test question, without them knowing which one. The contributors had to continue scoring above a confidence score by being tested against the gold questions. If, once started on the task, they fell below the confidence score threshold, they were released from the task, all of their previous annotations were tainted as "untrustworthy" and new annotations for the gaps were requested. Annotators were always paid for their job, even if their answers were deemed untrustworthy. The number of test questions thus dictated the total number of work an annotator could provide: when they had seen all the possible test questions, they stopped contributing such that they should not see the same test question twice. Our confidence score throughout the annotation was 81%, meaning that if contributors were answering correctly 80% or less of test questions they were not considered anymore for the annotation task. 
     
* **Minimum time per page:** To ensure that people were taking enough time to answer the questions, the minimum time per page was set to 90 s (each page contained 4-6 judgements). If a contributor would take less than that time to answer the questions (speeding), this indicates low quality of response , so they would be released from the task (the previous annotations in this case were not discarded). The statistics from the website informed that trusted contributors took an average of 33 s (interquartile mean) to provide a judgement.
     
* **Answer distribution:** Answer distribution rules were enabled for this task. After 20 judgements had been collected, if an annotator had judged more than 60% of relations as "support" or more than 35% as "attack", they were released from the task and all their judgements tainted. This was estimated by several initial test launches that determined it very unlikely to have distributions higher than those ones.
     
* **Dynamic judgments:** A minimum of 3 annotations per pair of sentences were requested. However, if the annotation agreement fell below our selected threshold of 70%, dynamic collection of judgements was enabled. This meant that up to 7 judgements per case could be requested.

A total of 101 test questions were used in this annotation and 104 trusted contributors participated in this task out of 287 that attempted at it. Overall, considering the quality settings (e.g. dynamic judgements, tainted answers), 21646 trusted annotations were collected (5746 belonging to gold questions and 15900 to random pairs), and a separate 1663 annotations were deemed untrustworthy.  

## Annotations

Support/attack/neither - confidence

Self-confidence

Trust in the annotation


## Multimodal classification

In order to test the validity of the dataset for relation-based argumentation mining, we tested the performance of different types of NLP models, combining audio features or not:

1. A text-only model, based on a BERT encoding.
2. An audio-only model, based on parallel CNNs and Bi-LSTMs.
3. A multimodal model, combining both text and audio inputs in parallel, as in models 1 and 2.

Audio feature extraction was performed using the Python module "librosa".  The following features were extracted: Mel-frequency cepstral coefficients (MFCCs), which are widely used features to characterise and detect voice signals; several spectral features like spectral centroids, spectral bandwidth, spectral roll-off and spectral contrast; and a 12-bit chroma vector. For each sentence, we used the timestamp to clip the audio file with a buffer of <img src="https://render.githubusercontent.com/render/math?math=\pm 2"> s to ensure the full audio of the utterance was captured. All of the features were matrices of shape *N*x*T*, where *T* is the temporal dimension (dependent on the duration of the utterance) and *N* is another dimension that can range from 1 in the case of spectral bandwitdth to 25 in the case of MFCCs. They were concatenated to form a feature matrix of 45x*T* and then they were padded with 0's to the maximum *T* dimension of the dataset (proportional to the maximum duration of an utterance).

The latest results of model performance are reported in the following table. The last model is a proof of concept to show the possibility of filtering the data by some of the parameters mentioned before (namely the trust in the annotator, the confidence in the annotation and the self-confidence score) to cure the according to the reseacher's needs. In this case, we filtered the pairs of sentences according to the annotation agreement or confidence for values higher than 0.85, improving the overall accuracy of the multimodal model, even with a smaller amount of data.

| Model | Text dropout | Audio dropout | Accuracy | Macro-F1 | weighted-F1 | Attack micro-F1 | Neither micro-F1 | Support micro-F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Audio only | | 0.2 |  <img src="https://render.githubusercontent.com/render/math?math=0.89 \pm 0.05"> | <img src="https://render.githubusercontent.com/render/math?math=0.45 \pm 0.05"> | <img src="https://render.githubusercontent.com/render/math?math=0.87 \pm 0.06"> | <img src="https://render.githubusercontent.com/render/math?math=0.28 \pm 0.08"> | <img src="https://render.githubusercontent.com/render/math?math=0.94 \pm 0.03"> | <img src="https://render.githubusercontent.com/render/math?math=0.14 \pm 0.10"> |
| Text only | 0.1 |  |  <img src="https://render.githubusercontent.com/render/math?math=0.87 \pm 0.01"> | <img src="https://render.githubusercontent.com/render/math?math=0.35 \pm 0.02"> | <img src="https://render.githubusercontent.com/render/math?math=0.82 \pm 0.01"> | <img src="https://render.githubusercontent.com/render/math?math=0.02 \pm 0.03"> | <img src="https://render.githubusercontent.com/render/math?math=0.93 \pm 0.01"> | <img src="https://render.githubusercontent.com/render/math?math=0.12 \pm 0.06"> |
| Multimodal | 0.1 | 0.1 |  <img src="https://render.githubusercontent.com/render/math?math=0.86 \pm 0.01"> | <img src="https://render.githubusercontent.com/render/math?math=0.40 \pm 0.04"> | <img src="https://render.githubusercontent.com/render/math?math=0.83 \pm 0.01"> | <img src="https://render.githubusercontent.com/render/math?math=0.08 \pm 0.01"> | <img src="https://render.githubusercontent.com/render/math?math=0.92 \pm 0.003"> | <img src="https://render.githubusercontent.com/render/math?math=0.18 \pm 0.04"> |
| Multimodal | 0.1 | 0.2 |  <img src="https://render.githubusercontent.com/render/math?math=0.88 \pm 0.04"> | <img src="https://render.githubusercontent.com/render/math?math=0.44 \pm 0.02"> | <img src="https://render.githubusercontent.com/render/math?math=0.87 \pm 0.05"> | <img src="https://render.githubusercontent.com/render/math?math=0.22 \pm 0.06"> | <img src="https://render.githubusercontent.com/render/math?math=0.93 \pm 0.02"> | <img src="https://render.githubusercontent.com/render/math?math=0.15 \pm 0.05"> |
| Multimodal | 0.1 | 0.3 |  <img src="https://render.githubusercontent.com/render/math?math=0.86 \pm 0.02"> | <img src="https://render.githubusercontent.com/render/math?math=0.37 \pm 0.04"> | <img src="https://render.githubusercontent.com/render/math?math=0.83 \pm 0.02"> | <img src="https://render.githubusercontent.com/render/math?math=0.10 \pm 0.10"> | <img src="https://render.githubusercontent.com/render/math?math=0.93 \pm 0.01"> | <img src="https://render.githubusercontent.com/render/math?math=0.10 \pm 0.07"> |
| Multimodal with annotation confidence >= 0.85 | 0.1 | 0.2 |  <img src="https://render.githubusercontent.com/render/math?math=0.91 \pm 0.03"> | <img src="https://render.githubusercontent.com/render/math?math=0.39 \pm 0.04"> | <img src="https://render.githubusercontent.com/render/math?math=0.90 \pm 0.02"> | <img src="https://render.githubusercontent.com/render/math?math=0.12 \pm 0.09"> | <img src="https://render.githubusercontent.com/render/math?math=0.95 \pm 0.02"> | <img src="https://render.githubusercontent.com/render/math?math=0.10 \pm 0.04"> |

Note: The performance values presented in the publication below are a snapshot of the models' performance at the time of writing and might difer slightly from these ones. This table contains the most recent results after performing several iterations of the training. The results above show the average for *N*=4 repetitions (except the last one, for which *N*=3) with the standard deviation. Despite the small differences with the data reported in the publication, they still fall within the standard deviation and the conclusions remain. 

## Publications

> Rafael Mestre, Razvan Milicin, Stuart E. Middleton, Matt Ryan, Jiatong Zhu, Timothy J. Norman. 2021. M-Arg: Multimodal Argument Mining Dataset for Political Debates with Audio and Transcripts. 


## Dataset files and codes

Folders:

- Annotated dataset: contains three files in .csv format with the full dataset, the full annotated dataset (individual annotations) and the aggregated dataset (with average judgements). Use this to train your models.
- Codes: contains the codes to reproduce the data pre-processing (tokenisation, audio clipping), the plots in the paper (descriptive statistics) and the models outlined above.
- Data: contains all the original data, the audio (full and clipped by utterances) and further data at different levels of pre-processing, for transparency.

Please see file XXXX for extended details of the dataset columns and how to use the codes.

Details:

In annotated dataset:

- annotated_sentences_reference.csv: dataset with references to IDs to easily link sentences, contexts, etc. Contains the following columns: 

| Column | Explanation |
| ---- | ---- |
| Pair ID | Unique ID of the pair of sentences and the surrounding context, in the following format: nDDXXXXbYYpZZWW. The first 2 digits (DD) after n indicate the debate and the following 4 digits (XXXX) indicate the sentence number within that debate. This is the unique ID of the central sentence of the context. For instance, n020456 refers to the sentence number 456 in the debate 02. The following two digits after b (YY) indicate the number of buffer sentences given with the context. For instance, in n020456b07, 7 sentences before and after sentence 456 in debate 02 were given in the context. (Note that although we performed trials with different number of sentences in the context, we fixed this number to 7 and all the data in this dataset have a buffer of 7 sentences.) The next 4 digits after p (ZZWW) indicate the last two digits of the first sentence (ZZ) and the second sentence (WW) of the pair. For instance, n020456b07p5559, refers to the central sentence 456 within debate 02, that was presented with +- 7 sentences as context, and the pair of sentences to annotate were number 455 and 459. |
| Sentence 1 ID | The ID of the first sentence of the pair to annotate in the format nDDXXXX explained above. |
| Sentence 2 ID | The ID of the second sentence of the pair to annotate in the format nDDXXXX explained above. |
| Context buffer | Buffer of +- sentences given as context around the central sentence. Note that although we performed trials with different number of sentences in the context, we fixed this number to 7 and all the data in this dataset have a buffer of 7 sentences. |
| Context start | ID of the sentence that starts the context. This is always the central sentence ID minus the number of buffer sentences (7). For instance, for n02456b07p5559, the central sentence is 456 of debate 02, and the context start is at n020449 (456-7). |
| Context end | ID of the sentence that finishes the context. This is always the central sentence ID plus the number of buffer sentences (7). For instance, for n02456b07p5559, the central sentence is 456 of debate 02, and the context end is at n020463 (456+7). |
| Topic | The topic of the pair of sentences to annotate. |
| Context tags | The full context in text form with HTML tags, as shown to the crowd annotators. The pair of sentences to annotate were highlighted in blue. |
| Sentence 1 | First sentence of the pair in text form. |
| Sentence 2 | Second sentence of the pair in text form. |
| Context | The full context in text without HTML tags. |

- full_dataset.csv: 
- aggregated_dataset.csv: 


