# -*- coding: utf-8 -*-
"""
This script performs the following actions:
    - Accesses the original data, tokenises the sentences, and assigns IDs to 
      sentences and speakers.
    - Accesses the timestamps and splits the audio files by utterances,
      with a buffer of +-2 s (by default, can be changed) and saves the clips.
    - Merges all of this information and creates the file
      full_feature_extraction_dataset.csv. 
      
This script should be run at least one to obtain the individual clips of
each utterance, which were not provided in the GitHub repository. 

From the paper:
    
    Rafael Mestre, Razvan Milicin, Stuart E. Middleton, Matt Ryan, 
    Jiatong Zhu, Timothy J. Norman. 2021. M-Arg: Multimodal Argument 
    Mining Dataset for Political Debates with Audio and Transcripts.
    8th Workshop on Argument Mining, 2021 at 2021 Conference on 
    Empirical Methods in Natural Language Processing (EMNLP).


License:
    BSD 4-clause with attribution License

    Copyright (c) 2021 University of Southampton


@authors: Rafael Mestre (R.Mestre@soton.ac.uk)*
          Razvan Milicin (razvan.milicin@gmail.com)

*corresponding author
"""

from pathlib import Path
import pandas as pd
import nltk
from pydub import AudioSegment

#Download the punkt tokeniser from nltk
nltk.download('punkt')

filepath = Path(r'..\data\original data') #Or change to your own directory

#Look for the files .csv files
files = filepath.glob('*.csv')
#files = [f for f in files if '_split' not in f.stem] #We take only the original ones
#that is, the ones that don't have "_split" in them because they haven't been processed

#The following empty dictionary will assign a unique label
#to each speaker
speakers_to_id = {}

#Speakers map to homogenise their names into a single one,
#since each transcription used different names, sometimes with typos
#Audience members are merged into the same category
speakers_map = {
    'Chris Wallace': 'Chris Wallace',
    'Vice President Joe Biden': 'Joe Biden',
    'President Donald J. Trump': 'Donald Trump',
    'Chris Wallace:': 'Chris Wallace',
    'Kristen Welker': 'Kristen Welker',
    'Donald Trump': 'Donald Trump',
    'Joe Biden': 'Joe Biden',
    'George Stephanopoulos': 'George Stephanopoulos',
    'Nicholas Fed': 'Audience Member 1',
    'Kelly Lee': 'Audience Member 2',
    'Anthony Archer': 'Audience Member 3',
    'Voice Over': 'Voice Over',
    'Cedric Humphrey': 'Audience Member 4',
    'George Stephanopoulus': 'George Stephanopoulos',
    'Angelia Politarhos': 'Audience Member 5',
    'Speaker 1': 'Voice Over',
    'Nathan Osburn': 'Audience Member 6',
    'Andrew Lewis': 'Audience Member 7',
    'Speaker 2': 'Voice Over',
    'Michele Ellison': 'Audience Member 8',
    'Mark Hoffman': 'Audience Member 9',
    'Mieke Haeck': 'Audience Member 10',
    'Speaker 3': 'Voice Over',
    'Keenan Wilson': 'Audience Member 11',
    'Savannah Guthrie': 'Savannah Guthrie',
    'President Trump': 'Donald Trump',
    'Jacqueline Lugo': 'Audience Member 12',
    'Barbara Peña': 'Audience Member 13',
    'Isabella Peña': 'Audience Member 14',
    'Savannah': 'Savannah Guthrie',
    'Cristy Montesinos Alonso': 'Audience Member 15',
    'Adam Schucher': 'Audience Member 16',
    'Moriah Geene': 'Audience Member 17',
    'Cindy Velez': 'Audience Member 18',
    'Paulette Dale': 'Audience Member 19',
    'Susan Page': 'Susan Page',
    'Kamala Harris': 'Kamala Harris',
    'Mike Pence': 'Mike Pence',
    'Kamala Harris ': 'Kamala Harris'    
    }


#Loop to tokenise and split the utterances into sentences
#Each speaker is assigned a unique ID of 2 numbers
idx = 0

filepath_split = Path(r'..\data\tokenised text')

for i in files:

    speaker = []
    minute = []
    text = []

    #We read the original debate .csv file
    df = pd.read_csv(i, 
        header=0, 
        names=['speaker', 'minute', 'text'])
    #The original file contained "speaker", "minute" and "text" columns
    #We ignore the "minute" column because it doesn't correspond exactly
    #to the audio files. We will use our own timestamps to align it to audio.
    
    #Each row contains an utterance. We loop through them and tokenise them
    #in individual sentences
    for r in df.index:
        #Tokenised text
        split_text = nltk.sent_tokenize(df['text'].iloc[r])
        text += split_text
        #Match speakers to their names using the dictionary above
        try:
            speaker_name = speakers_map[df['speaker'].iloc[r]]
        except: #In theory this exception is never raised
            speaker_name = 'Speaker not found'
        speaker += [speaker_name]*len(split_text)
        #Assign IDs to speakers
        #A dictionary of speakers_to_id is filled for later use
        if not (speaker_name in speakers_to_id):
            if idx < 10:
                speakers_to_id[speaker_name] = '0' + str(idx)
            else:
                speakers_to_id[speaker_name] = str(idx)
            idx = idx + 1

    #Dataframe with the speaker and the tokenised text is saved into a file
    #with the suffix "_split"
    df_split = pd.DataFrame({'speaker': speaker, 'text': text})
    #We save the dataframe
    df_split.to_csv(Path(filepath_split,i.stem+'_split.csv'),index=False)
    
    #We also create a version in plan text (used by the force aligner tool)
    with open(Path(filepath_split, i.stem+'_plain.txt'),'w',encoding='utf8') as f:
        for t in df_split['text']:
            f.write(t+'\n')




#The debates were divided into segments to reduce their size
#or to avoid applauses, music, ads, etc. so that the force aligner
#could work properly.
#To each sub-debate, we give it a unique ID
debate_id_map = {
    'us_election_2020_1st_presidential_debate_part1_timestamp.csv' : '00',
    'us_election_2020_1st_presidential_debate_part2_timestamp.csv': '01',
    'us_election_2020_2nd_presidential_debate_part1_timestamp.csv':'02',
    'us_election_2020_2nd_presidential_debate_part2_timestamp.csv':'03',
    'us_election_2020_biden_town_hall_part1_timestamp.csv':'04',
    'us_election_2020_biden_town_hall_part2_timestamp.csv':'05',
    'us_election_2020_biden_town_hall_part3_timestamp.csv':'06',
    'us_election_2020_biden_town_hall_part4_timestamp.csv':'07',
    'us_election_2020_biden_town_hall_part5_timestamp.csv':'08',
    'us_election_2020_biden_town_hall_part6_timestamp.csv':'09',
    'us_election_2020_biden_town_hall_part7_timestamp.csv':'10',
    'us_election_2020_trump_town_hall_1_timestamp.csv':'11',
    'us_election_2020_trump_town_hall_2_timestamp.csv':'12',
    'us_election_2020_trump_town_hall_3_timestamp.csv':'13',
    'us_election_2020_trump_town_hall_4_timestamp.csv':'14',
    'us_election_2020_vice_presidential_debate_1_timestamp.csv':'15',
    'us_election_2020_vice_presidential_debate_2_timestamp.csv':'16'
    }

#Each timestamped sub-debate is linked to the original full debate
file_map_timestamp = {
    'us_election_2020_1st_presidential_debate_part1_timestamp.csv': 'us_election_2020_1st_presidential_debate_split.csv',
    'us_election_2020_1st_presidential_debate_part2_timestamp.csv':'us_election_2020_1st_presidential_debate_split.csv',
    'us_election_2020_2nd_presidential_debate_part1_timestamp.csv':'us_election_2020_2nd_presidential_debate_split.csv',
    'us_election_2020_2nd_presidential_debate_part2_timestamp.csv':'us_election_2020_2nd_presidential_debate_split.csv',
    'us_election_2020_biden_town_hall_part1_timestamp.csv':'us_election_2020_biden_town_hall_split.csv',
    'us_election_2020_biden_town_hall_part2_timestamp.csv':'us_election_2020_biden_town_hall_split.csv',
    'us_election_2020_biden_town_hall_part3_timestamp.csv':'us_election_2020_biden_town_hall_split.csv',
    'us_election_2020_biden_town_hall_part4_timestamp.csv':'us_election_2020_biden_town_hall_split.csv',
    'us_election_2020_biden_town_hall_part5_timestamp.csv':'us_election_2020_biden_town_hall_split.csv',
    'us_election_2020_biden_town_hall_part6_timestamp.csv':'us_election_2020_biden_town_hall_split.csv',
    'us_election_2020_biden_town_hall_part7_timestamp.csv':'us_election_2020_biden_town_hall_split.csv',
    'us_election_2020_trump_town_hall_1_timestamp.csv':'us_election_2020_trump_town_hall_split.csv',
    'us_election_2020_trump_town_hall_2_timestamp.csv':'us_election_2020_trump_town_hall_split.csv',
    'us_election_2020_trump_town_hall_3_timestamp.csv':'us_election_2020_trump_town_hall_split.csv',
    'us_election_2020_trump_town_hall_4_timestamp.csv':'us_election_2020_trump_town_hall_split.csv',
    'us_election_2020_vice_presidential_debate_1_timestamp.csv':'us_election_2020_vice_presidential_debate_split.csv',
    'us_election_2020_vice_presidential_debate_2_timestamp.csv':'us_election_2020_vice_presidential_debate_split.csv'}

speakers_to_id['error'] = 'error'


'''Read topic map'''

#The topic is taken from the annotated context file with all the merged datasets
filepath_topic = Path(r'..\data\preprocessed full dataset')
df_topic = pd.read_csv(Path(filepath_topic,'dataset_context_definition.csv'))

#Filepath to load the timestamps
filepath_timestamps = Path(r'..\data\timestamps')

#Load all the timestamps
files = filepath_timestamps.glob('*.csv')

#Creation of the dataframe that will contain all the data
big_df = pd.DataFrame(columns = ['id','text','speaker','speaker_id','audio_file','context','debate'])


#Filepaths where the audio is stored and where it will be saved as utterance clips
filepath_audio = Path(r'..\data\split audio')
filepath_audio_save = Path(r'..\data\audio sentences')

#Each audio file is clipped with a buffer of +- 2 s (custom)
buffer = 2 #In seconds

#For each sub-debate timestamp file
for i in debate_id_map:
    #We load the timestamps
    df = pd.read_csv(Path(filepath_timestamps,i),header=None,names=['id','start','end','text'])

    #We load the corresponding mp3 audio file
    debate_mp3 = AudioSegment.from_mp3(Path(filepath_audio, i.split('_timestamp')[0]+'.mp3'))
    
    #Since the timestamped files don't contain the speaker, we need to load
    #the corresponding debate file with the speaker names (generated above)
    speak_df = pd.read_csv(Path(filepath_split,file_map_timestamp[i]),
                           header=0,names=['speaker','text'])

    #Debate index
    f_idx = debate_id_map[i]
    idx = 0
    # We iterate through each sub-debate and we clip each sentence audio with
    # a buffer defined above
    for index, row in df.iterrows():
        #Extract audio clip with buffer
        if row['start'] < buffer:
            extract = debate_mp3[row['start']*1000:(row['end']+buffer)*1000]
        else:
            #Exceptions to deal with the first and last clip
            try:
                extract = debate_mp3[(row['start']-buffer)*1000:(row['end']+buffer)*1000]
            except:
                extract = debate_mp3[(row['start']-buffer)*1000:row['end']*1000]
                
        #We recreate the sentence ID, based on a two-digit number for the debate
        # and a four-digit number for the sentence itself
        str_idx = str(idx)
        while len(str_idx) < 4:
            str_idx = '0' + str_idx
        str_idx = str(f_idx) + str_idx
        
        idx = idx+1
        #Using the previuosly loaded dataframe, we extract the speaker of such sentence
        try:
            speaker = speak_df[speak_df['text']==row['text']]['speaker'].iloc[0]
        except:
            print(row['text'])
            speaker = 'error'
        
        #We add the context by looking in the context dataset for the sentence ID
        context = df_topic[df_topic['id'] == 'n' + str_idx]['context'].values[0]
        if context == 'Ignore': context = None
        
        #Adding timestamp
        timestamp = row['id']
        #We create a full dataframe with all the information
        big_df = big_df.append({'id' : 'n' + str_idx, 'text' : row['text'],
                                'speaker' : speaker,'speaker_id' : speakers_to_id[speaker], 
                                'audio_file' : ('a' + str_idx), 'context': context,
                                'debate': str(f_idx), 'timestamp': timestamp}, 
                                ignore_index = True)
        #Sentences IDs contain the letter "n" before the number. Audio clips IDs
        #are the same ID starting with an "a"

        #Finally, we save the audio clips
        #extract.export(Path(filepath_audio_save, 'a'+str_idx+'.wav'), format="wav")

#We save the full dataset
big_df.to_csv(Path(filepath_topic,'full_feature_extraction_dataset.csv'),index=False)

