### Dataset files and codes

## *annotated dataset* folder


- *annotated_sentences_reference.csv*: dataset with references to IDs to easily link sentences, contexts, etc. It contains the following columns: 

| Column | Explanation |
| ---- | ---- |
| Pair ID | Unique ID of the pair of sentences and the surrounding context, in the following format: nDDXXXXbYYpZZWW. The first 2 digits (DD) after n indicate the debate and the following 4 digits (XXXX) indicate the sentence number within that debate. This is the unique ID of the central sentence of the context. For instance, n020456 refers to the sentence number 456 in the debate 02. The following two digits after b (YY) indicate the number of buffer sentences given with the context. For instance, in n020456b07, 7 sentences before and after sentence 456 in debate 02 were given in the context. (Note that although we performed trials with different number of sentences in the context, we fixed this number to 7 and thus all the data in this dataset have a buffer of 7 sentences.) The next 4 digits after p (ZZWW) indicate the last two digits of the first sentence (ZZ) and the second sentence (WW) of the pair. For instance, n020456b07p5559, refers to the central sentence 456 within debate 02, that was presented with +- 7 sentences as context, and the pair of sentences to annotate were number 455 and 459. |
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

- *full_dataset.csv*: dataset with **full** individual annotations by each contributor, with a total of 24710 entries. It contains the following columns:

| Column | Explanation |
| ---- | ---- |
| \_unit\_id | Unique ID of the pairs of sentences to annotate in the annotation platform. |
| \_golden | Boolean indicating whether the annotation was gold (test) or not. If it was a golden question, it should appear several times, since they were shown to many contributors to test their performance. |
| \_missed | Boolean indicating whether the annotator answered incorrectly (missed) the test question. It only contains a boolean if the current unit was a gold question, otherwise it's empty. |
| \_trust | Trust in the annotator, calculated according to their track record of correctly answered gold questions throughout the task. It's simply the ration of correctly answered gold questions over the total number of gold questions shown. |
| \_worker\_id | Unique randomised annotator ID. Originally, the IDs were already randomised to avoid identifying any contributor, but we further anonymised those IDs to strip any reference to the annotation platform. |
| confidence | Self-reported confidence in the annotation given by the current annotator (see **Annotations** section). | 
| relation | Annotated relation between pair of sentences given by the current annotator (see **Annotations** section). |
| orig\_\_golden |
| context | Context shown to the annotators, including HTML tags. |
| pair\_id | Unique identifying pair ID in the format nDDXXXXbYYpZZWW. See above for more detais. |
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
| \_unit\_state |  |
| \_trusted\_judgements | Number of judgements (annotations) received to compute the final annotation. Notice that gold questions were annotated many times, while normal pairs of sentences were only annotated 3-7 times.|
| confidence | **Average** self-reported confidence in the annotation (see **Annotations** section). | 
| confidence:stddev | **Standard deviation** of the self-reported confidence. | 
| relation | **Most likely** argumentative relation between the pair of sentences (see **Annotations** section). The most likely argumentative relation is decided according to the highest confidence score (next row). |
| relation:confidence | **Confidence score** in the argumentative relation between the pair of sentences. This is calculated as a weighted average of the annotators' trust when doing the judgement. |
| orig\_\_golden |
| confidence\_gold |
| context | Context shown to the annotators, including HTML tags. |
| pair\_id | Unique identifying pair ID in the format nDDXXXXbYYpZZWW. See above for more detais. |
| relation\_gold | If the current annotation is a gold question, this is the "correct answer" that provided beforehand and against which annotators were tested. If not a gold question, it's empty. |
| relation\_gold\_reason |  If the current annotation is a gold question, this is the explanation of why this was the correct answer. The annotators were shown this explanation if they failed, so they could learn from their mistakes. If not a gold question, it's empty. |
| sentence\_1 | First sentence of the pair in text form. |
| setence\_2 | Second sentence of the pair in text form. |
| speaker\_1 | Speaker of the first sentence. |
| speaker\_2 | Speaker of the second sentence. |
| topic | Topic of discussion during that part of the debate. |


