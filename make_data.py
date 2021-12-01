import pandas as pd 
import string 

import shutil
import os

from eKantipur_scrap import scrap_news
scrap_news()


df = pd.read_csv('df.csv')
df_updated =  pd.DataFrame(columns=['Content'])

def sentence_tokenize(text):
    sentences = text.strip().split(u"।")
    sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in sentences]
    return sentences

for i in range(len(df['Content'])):
    sentence_list = sentence_tokenize(str(df['Content'][i]))

    for k in range(len(sentence_list)):
        temp_df = {'Content': str(sentence_list[k])}    
        df_updated = df_updated.append(temp_df, ignore_index = True, sort=False)
    
df_updated.reset_index(drop=True, inplace=True)
df_updated.columns = [''] * len(df_updated.columns)
df_updated.to_csv('english_corpus.csv',  header=False, index=False)


def move_delete_files():
    target_dir = str(os.getcwd()) + '/mimicAudio/backend/prompts'
    source_dir = str(os.getcwd()) + '/english_corpus.csv'
    target_file = target_dir + "/english_corpus.csv"
    if os.path.exists(target_file):
        os.remove(target_file)
    else:
        print("Can not delete the file as it doesn't exists")

    shutil.move(source_dir, target_dir)
    source_dir_old = str(os.getcwd()) + '/df.csv'
    os.remove(source_dir_old)

move_delete_files()
