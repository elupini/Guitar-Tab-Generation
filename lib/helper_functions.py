import jams
import numpy as np
from matplotlib import lines as mlines, pyplot as plt
import tempfile
import librosa
import sox
import os
import pandas as pd

def add_fret_positions(midi_note,string):
    
    str_midi_dict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
    fret = int(round(midi_note - str_midi_dict[string]))
    
    return fret


def generate_data(jam):
    annos = jam.search(namespace='note_midi')
    i = 0
    times = []
    durations = []
    midi_notes = []
    string = []
    for string_tran in annos:
        #print(i)
        for note in string_tran:
            times.append(note[0])
            durations.append(note[1])
            midi_notes.append(note[2])
            string.append(i)
        i+=1
    data = {'start_time':times,'duration':durations,'midi_note':midi_notes,'string':string}
    df = pd.DataFrame(data = data)
    df['fret_position'] = df.apply(lambda x: add_fret_positions(x.midi_note,x.string),axis=1)
    return df

def baseline_helper(midi_note):
    
    str_midi_dict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
    string = 0
    
    for i in range(5,-1,-1):
        #print("String Number: ",i)
        #print("Midi Sub Value: ",int(round(midi_note - str_midi_dict[i])))
        if(int(round(midi_note - str_midi_dict[i]))>=0):
            #print("I Entry value: i")
            string = i
            fret_pos = int(round(midi_note - str_midi_dict[i]))
            break
    
    return string, fret_pos


# First "dumb" prediction which picks the lowest freq option
def baseline_prediction(df):
    
    df['prediction'] = df.apply(lambda x: baseline_helper(x.midi_note), axis=1)
    
    strings = []
    fret_pos = []
    
    for i in range(len(df)):
        strings.append(df.iloc[i].prediction[0])
        fret_pos.append(df.iloc[i].prediction[1])
    
    df['string'] = strings
    df['fret_position'] = fret_pos
    
    return df


def evaluate_model(pred, target):
    num_correct = 0
    for i in range(len(pred)):
        if(pred[i]==target[i]):
            num_correct +=1
    return num_correct/len(pred)


def tabularize_result(jam,df_pred):
    
    fig = plt.figure(figsize=(8,3), dpi=300)
    tablaturize_jams_v2(jam,df_pred)
    plt.xlim(4.9, 10) # this is the time window in seconds that I'm plotting
    
def tablaturize_jams_v2(jam, df, save_path=None):
    str_midi_dict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
    string_dict = {0: 'E', 1: 'A', 2: 'D', 3: 'G', 4: 'B', 5: 'e'}
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}
    s = 0

    handle_list = []
    
    for s in range(0,6):
        handle_list.append(mlines.Line2D([], [], color=style_dict[s],
                                         label=string_dict[s]))
    
    for i in range(len(df)):
        plt.scatter(df.iloc[i].start_time, (df.iloc[i].string)+1, marker="${}$".format(int(df.iloc[i].fret_position)), color =
            style_dict[df.iloc[i].string])

    # plot Beat
    anno_b = jam.search(namespace='beat_position')[0]
    for b in anno_b.data:
        t = b.time
        plt.axvline(t, linestyle='dotted', color='k', alpha=0.5)
        if int(b.value['position']) == 1:
            plt.axvline(t, linestyle='-', color='k', alpha=0.8)

    handle_list.append(mlines.Line2D([], [], color='k',
                                     label='downbeat'))
    handle_list.append(mlines.Line2D([], [], color='k', linestyle='dotted',
                                     label='beat'))
    plt.xlabel('Time (sec)')
    plt.ylabel('String Number')
    # plt.title(jam.file_metadata.title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
               handles=handle_list, ncol=8)
    plt.xlim(-0.5, jam.file_metadata.duration)
    # fig.set_size_inches(6, 3)
    if save_path:
        plt.savefig(save_path)
        
