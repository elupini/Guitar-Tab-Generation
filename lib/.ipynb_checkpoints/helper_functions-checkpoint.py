import jams
import numpy as np
from matplotlib import lines as mlines, pyplot as plt
import tempfile
import librosa
import sox
import os
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

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


def tabularize_result(filepath,df_pred,window=None):
    
    jam = jams.load(filepath)
    fig = plt.figure(figsize=(8,3), dpi=300)
    tablaturize_jams_v2(jam,df_pred)
    if window:
        plt.xlim(window[0], window[1]) # this is the time window in seconds that I'm plotting
    else:
        plt.show()
    
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

def get_shortest_distance(df):
    
    predictions = []
    
    # Get initial prediction
    predictions.append((baseline_helper(df['midi_note'][0])))
    
    for i in range(1,len(df)):
        fret = int(df.iloc[i]['midi_note'])
        
        possible_voicings = calculate_possible_frets(fret)
        
        distances = []
        
        for v in possible_voicings:
            distances.append(cdist([[v[0],v[1]]], [[predictions[i-1][0],predictions[i-1][1]]], metric='cityblock'))
        #if(len(distances)==0):
            #print('possible voicings: ',possible_voicings)
            #print('predictions: ',predictions)
            #print('midi: ', fret)
            #print('df row: ',df.iloc[i])
        correct_voicing = possible_voicings[np.asarray(distances).argmin()]
        
        predictions.append(correct_voicing)
    
    strings = []
    frets = []
    
    for p in predictions:
        strings.append(p[0])
        frets.append(p[1])
    strings = {'string_pred':strings}
    frets = {'fret_pred':frets}

    df['string'] = strings['string_pred']
    df['fret_position'] = frets['fret_pred']
    
    return df

def calculate_possible_frets(midi_note):
    
    str_midi_dict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
    possible_voicings = []
    
    if(midi_note<40):
        possible_voicings.append((0,0))

    for key,_ in str_midi_dict.items():

        if(((midi_note - str_midi_dict[key]) >=0) and (midi_note - str_midi_dict[key]) <=22):

            possible_voicings.append((key,int(round(midi_note - str_midi_dict[key]))))
            
    return possible_voicings

def smart_shortest_distance(df):
    path_dict = {}
    path_dict[0] = []
    path_dict[1] = []
    path_dict[2] = []

    # Create 3 initial voicing options
    voicings = calculate_possible_frets(df.iloc[0]['midi_note'])
    voicings.reverse()
    if(len(voicings)==1):
        path_dict[0].append((voicings[0],0))
        path_dict[1].append((voicings[0],0))
        path_dict[2].append((voicings[0],0))
    elif(len(voicings)==2):
         path_dict[0].append((voicings[0],0))
         path_dict[1].append((voicings[0],0))
         path_dict[2].append((voicings[-1],0))
    else:
         middle_idx = int(len(voicings)/2)
         path_dict[0].append((voicings[0],0))
         path_dict[1].append((voicings[middle_idx],0))
         path_dict[2].append((voicings[-1],0))

    # For each note
    for i in range(1,len(df)):

        voicings = calculate_possible_frets(df.iloc[i]['midi_note'])

        distances = []
        possible_paths = []

        for key in path_dict.keys():

            for v in voicings:
                #print(i)
                #print(key)
                #print(path_dict[key][-1][0])
                # Replace this with generic cost function that can be passed in
                d = cdist([[v[0],v[1]]], [[path_dict[key][-1][0][0],path_dict[key][-1][0][1]]], metric='cityblock') + path_dict[key][-1][1]
                d = int(d)
                distances.append(d)
                tmp_path = path_dict[key].copy()
                tmp_path.append((v,d))
                #print('key: ',key)
                #print(path_dict[key])
                #print(tmp_path)
                possible_paths.append(tmp_path)
        #print(distances)
        distances = np.asarray(distances)
        best_path_idxs = distances.argsort()[:15]
        #print(possible_paths)
        for count, value in enumerate(best_path_idxs):
            #print(count)
            #print(value)
            path_dict[count] = possible_paths[value]
            
    strings = []
    frets = []
    
    final_costs = []
    for key in path_dict.keys():
        final_costs.append(path_dict[key][-1][-1])
    final_costs = np.asarray(final_costs)
    best_idx = final_costs.argmin()
        
    tmp_predictions = path_dict[best_idx]
    
    predictions = []
    for p in tmp_predictions:
        predictions.append(p[0])
    
    for p in predictions:
        strings.append(p[0])
        frets.append(p[1])
    strings = {'string_pred':strings}
    frets = {'fret_pred':frets}

    df['string'] = strings['string_pred']
    df['fret_position'] = frets['fret_pred']
    
    return df

def smart_shortest_distance_2(df):
    path_dict = {}

    # Create 3 initial voicing options
    voicings = calculate_possible_frets(df.iloc[0]['midi_note'])
    voicings.reverse()

    for i in range(len(voicings)):
        path_dict[i] = []
    
    for i in range(len(path_dict)):
        path_dict[i].append((voicings[i],0))
    
    #print(path_dict)

    # For each note
    for i in range(1,len(df)):

        voicings = calculate_possible_frets(df.iloc[i]['midi_note'])

        distances = []
        possible_paths = []

        for key in path_dict.keys():

            for v in voicings:
                #print(i)
                #print(key)
                #print(path_dict[key][-1][0])
                # Replace this with generic cost function that can be passed in
                d = cdist([[v[0],v[1]]], [[path_dict[key][-1][0][0],path_dict[key][-1][0][1]]], metric='cityblock') + path_dict[key][-1][1]
                d = int(d)
                distances.append(d)
                tmp_path = path_dict[key].copy()
                tmp_path.append((v,d))
                #print('key: ',key)
                #print(path_dict[key])
                #print(tmp_path)
                possible_paths.append(tmp_path)
        #print(distances)
        distances = np.asarray(distances)
        best_path_idxs = distances.argsort()[:16]
        #print(possible_paths)
        for count, value in enumerate(best_path_idxs):
            #print(count)
            #print(value)
            path_dict[count] = possible_paths[value]
            
    strings = []
    frets = []
    
    final_costs = []
    for key in path_dict.keys():
        final_costs.append(path_dict[key][-1][-1])
    final_costs = np.asarray(final_costs)
    best_idx = final_costs.argmin()
        
    tmp_predictions = path_dict[best_idx]
    
    predictions = []
    for p in tmp_predictions:
        predictions.append(p[0])
    
    for p in predictions:
        strings.append(p[0])
        frets.append(p[1])
    strings = {'string_pred':strings}
    frets = {'fret_pred':frets}
    
    #print(len(df))
    #if(len(strings['string_pred'])==1):
        #print(tmp_predictions)
        #print(path_dict)
    #print(len(frets))

    df['string'] = strings['string_pred']
    df['fret_position'] = frets['fret_pred']
    
    return df

# Takes a dataframe
# Returns the total distance cost of traversing the tab
def get_path_cost(df):
    total_cost = 0
    for i in range(1,len(df)-1):
        total_cost += int(cdist([[df.iloc[i-1]['string'],df.iloc[i-1]['fret_position']]], [[df.iloc[i]['string'],df.iloc[i]['fret_position']]], metric='cityblock'))
    return total_cost

# Helper function to determine if track is actually monophonic
def is_monophonic(df):
    num_violations = 0
    for i in range(1,len(df)):
        if(np.abs(df.iloc[i]['start_time']-df.iloc[i-1]['end_time'])<0.005):
            num_violations+=1
    if(num_violations>int(len(df)/5)):
        return False
    else:
        return True
    
# Load Data
def load_data(directory):
    data_unfiltered = os.listdir(directory)
    data = []
    for wav in data_unfiltered:
        if('solo' in wav):
            data.append(wav)
    return data

# Runs the shortest distance experiment on ALL the data
def shortest_distance_experiment(directory):
    data = load_data(directory)
    exp_3_scores = []
    costs = []
    for jam in data:
        wav = jams.load(directory+'/'+ jam)
        df = generate_data(wav)
        df = df.sort_values(by=['start_time']).reset_index(drop=True)
        df['end_time'] = df.start_time + df.duration
        if(is_monophonic(df)==False):
            continue
        df_train = df[['start_time','duration','midi_note']]
        df_target = df[['string','fret_position']]
        p = smart_shortest_distance_2(df_train)
        costs.append(get_path_cost(p))
        exp_3_scores.append(evaluate_model(p['string'], df_target['string']))
    return np.asarray(exp_3_scores).mean(), np.asarray(costs).mean()

# Predicts shortest distance on one file given the filepath
def predict_shortest_distance(filepath):
    wav = jams.load(filepath)
    df = generate_data(wav)
    df_train = df[['start_time','duration','midi_note']]
    df_target = df[['string','fret_position']]
    p = smart_shortest_distance_2(df_train)
    return p

def baseline_experiment(directory):
    data = load_data(directory)
    scores = []
    costs = []
    for jam in data:
        wav = jams.load(directory +'/'+ jam)
        df = generate_data(wav)
        df_train = df[['start_time','duration','midi_note']]
        df_target = df['string']
        df_pred = baseline_prediction(df_train)
        costs.append(get_path_cost(df_pred))
        scores.append(evaluate_model(df_pred['string'], df_target))
    return np.asarray(scores).mean(),np.asarray(costs).mean()

def naive_shortest_distance(directory):
    data = load_data(directory)
    exp_2_scores = []
    costs = []
    for jam in data:
        wav = jams.load(directory + '/' + jam)
        df = generate_data(wav)
        df = df.sort_values(by=['start_time']).reset_index(drop=True)
        df['end_time'] = df.start_time + df.duration
        if(is_monophonic(df)==False):
            continue
        df_train = df[['start_time','duration','midi_note']]
        df_target = df[['string','fret_position']]
        p = get_shortest_distance(df_train)
        costs.append(get_path_cost(p))
        exp_2_scores.append(evaluate_model(p['string'], df_target['string']))
    return np.asarray(exp_2_scores).mean(), np.asarray(costs).mean()

def ml_load_data():
    
    data_df = pd.DataFrame(columns=['start_time','duration midi_note','string','fret_position'])

    data_unfiltered = os.listdir('data')
    data = []
    track_indexes = []
    for wav in data_unfiltered:
        if('solo' in wav):
            data.append(wav)

    for jam in data:
        wav = jams.load('data/'+ jam)
        df = generate_data(wav)
        df = df.sort_values(by=['start_time']).reset_index(drop=True)
        track_indexes.append(len(df))
        data_df=data_df.append(df)
        data_df = data_df.reset_index(drop=True)
        
        track_idxs = [track_indexes[0]]
    for i in range(1,len(track_indexes)-1):
        track_idxs.append(track_idxs[i-1] + track_indexes[i])
        
    return data_df, track_idxs

def ml_experiment(data_df,track_idxs):
    train_df = data_df[['start_time','duration','midi_note']]
    target_df = data_df['string']
    target = target_df.astype('int')
    scaler = MinMaxScaler()
    train_df = scaler.fit_transform(train_df)
    
    halfway = int(len(track_idxs)/2)
    train_scores = []
    test_scores = []
    for i in range(halfway,len(track_idxs)):
        X_train = train_df[0:track_idxs[i]]
        X_test = train_df[track_idxs[i]:]
        y_train = target[0:track_idxs[i]]
        y_test = target[track_idxs[i]:]
        clf = MLPClassifier(random_state=1, max_iter=300)
        clf.fit(X_train, y_train)
        train_accuracy = clf.score(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        train_scores.append(train_accuracy)
        test_scores.append(accuracy)
    return train_scores,test_scores