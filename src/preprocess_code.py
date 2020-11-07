''' SET-UP '''
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

''' LOAD DATA - TRAIN and TEST '''
file_train = 'train.json'
file_test = 'test.json'
file_sub = 'sample_submission.csv'
df_train, df_test, df_sub = load_data(file_train, file_test, file_sub)

''' RUN PREPROCESSING FUNCTION TO AUGMENT X DATAFRAMES '''
X_train, Y_train, X_test, Y_test = preprocess_RNA(df_train, df_test)
X_train.head()


''' FUNCTIONS '''
# OVERARCHING PRE-PROCESSING FUNCTION
def preprocess_RNA(data_train, data_test):
    # Split the dataframes into X, Y components
    X_train, Y_train, X_test, Y_test = X_Y_split(df_train, df_test)
    
    # Add feature and paired counts to the X_train and X_test
    X_train = paired_counts(feature_counts(X_train))
    X_test = paired_counts(feature_counts(X_test))
    
    # Add one-hot and integer encoded features
    X_train = one_hot_matrix_sequence(X_train)
    X_test = one_hot_matrix_sequence(X_test)

    return X_train, Y_train, X_test, Y_test

# LOAD DATA
def load_data(file_train, file_test, file_sub):
    df_train = pd.read_json(file_train, lines=True)
    df_test = pd.read_json(file_test, lines=True)
    #test_pub = df_test[df_test["seq_length"] == 107]
    #test_pri = df_test[df_test["seq_length"] == 130]
    df_sub = pd.read_csv(file_sub)
    return df_train, df_test, df_sub

# SPLIT DATASETS INTO X, Y
def X_Y_split(df_train, df_test):
    X_train = df_train.iloc[:,2:9]
    Y_train = df_train.iloc[:,9:]
    X_test = df_test.iloc[:,2:9]
    Y_test = df_test.iloc[:,9:]
    
    return X_train, Y_train, X_test, Y_test

# FEATURE ENGINEERING - COUNTS FOR STRINGS
def feature_counts(X_train):
    G = list()
    A = list()
    C = list()
    U = list()

    # Count numbers of each base
    for sequence in X_train['sequence']:
        base_counts = {e:sequence.count(e) for e in set(sequence)}
        G.append(base_counts['G'])
        A.append(base_counts['A'])
        C.append(base_counts['C'])
        U.append(base_counts['U'])
    X_train['G'] = G
    X_train['A'] = A
    X_train['C'] = C
    X_train['U'] = U


    unpaired = list()
    paired = list()
    # Count the number of paired bases
    for structure in X_train['structure']:
        pair_counts = {e:structure.count(e) for e in set(structure)}
        unpaired.append(pair_counts['.'])
        paired.append(pair_counts['('] + pair_counts[')'])
    X_train['paired'] = paired
    X_train['unpaired'] = unpaired

    # Count of predicted loop types
    #S: paired "Stem" M: Multiloop I: Internal loop 
    #B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    Stem = list()
    Multiloop = list()
    Internal_loop = list()
    Bulge = list()
    Hairpin = list()
    Dangling_end = list()
    External_loop = list()
    for loop_type in X_train['predicted_loop_type']:
        type_counts = {e:loop_type.count(e) for e in set(loop_type)}
        if 'S' in type_counts:
            Stem.append(type_counts['S'])
        else:
            Stem.append(0)

        if 'M' in type_counts:
            Multiloop.append(type_counts['M'])
        else:
            Multiloop.append(0)

        if 'I' in type_counts:
            Internal_loop.append(type_counts['I'])
        else:
            Internal_loop.append(0)

        if 'B' in type_counts:
            Bulge.append(type_counts['B'])
        else:
            Bulge.append(0)

        if 'H' in type_counts:
            Hairpin.append(type_counts['H'])
        else:
            Hairpin.append(0)

        if 'E' in type_counts:
            Dangling_end.append(type_counts['E'])
        else:
            Dangling_end.append(0)

        if 'X' in type_counts:
            External_loop.append(type_counts['X'])
        else:
            External_loop.append(0)
    X_train['S'] = Stem
    X_train['I'] = Internal_loop
    X_train['M'] = Multiloop
    X_train['B'] = Bulge
    X_train['H'] = Hairpin
    X_train['D'] = Dangling_end
    X_train['X'] = External_loop
    return X_train   

def paired_counts(X_train):
    AC = list()
    AG = list()
    AU = list()
    CA = list()
    CG = list()
    CU = list()
    GA = list()
    GC = list()
    GU = list()
    UA = list()
    UC = list()
    UG = list()

    for i in range(len(X_train)):
        # Get a list of all the paired bases in a given sequence
        row = X_train.iloc[i]
        sequence = row['sequence']
        structure = row['structure']
        pair_queue = list()
        base_queue = list()
        base_pairs_list = list()

        for i in range(len(sequence)):
            if structure[i] == '(':
                pair_queue.append(i)
                base_queue.append(sequence[i]) #append the letter of base pair thats bonded
            elif structure[i] == ')':
                pair_queue.pop() # if you run into a mating pair, pop off the last first half of pair
                base_pairs_list.append(base_queue[-1] + sequence[i])
                base_queue.pop()

        # Get counts of given base pairs in the sequence
        pair_counts = {e:base_pairs_list.count(e) for e in set(base_pairs_list)}

        if 'AC' in pair_counts:
            AC.append(pair_counts['AC'])
        else:
            AC.append(0)
        if 'AG' in pair_counts:
            AG.append(pair_counts['AG'])
        else:
            AG.append(0)
        if 'AU' in pair_counts:
            AU.append(pair_counts['AU'])
        else:
            AU.append(0)

        if 'CA' in pair_counts:
            CA.append(pair_counts['CA'])
        else:
            CA.append(0)
        if 'CG' in pair_counts:
            CG.append(pair_counts['CG'])
        else:
            CG.append(0)
        if 'CU' in pair_counts:
            CU.append(pair_counts['CU'])
        else:
            CU.append(0)

        if 'GA' in pair_counts:
            GA.append(pair_counts['GA'])
        else:
            GA.append(0)
        if 'GC' in pair_counts:
            GC.append(pair_counts['GC'])
        else:
            GC.append(0)
        if 'GU' in pair_counts:
            GU.append(pair_counts['GU'])
        else:
            GU.append(0)

        if 'UA' in pair_counts:
            UA.append(pair_counts['UA'])
        else:
            UA.append(0)
        if 'UC' in pair_counts:
            UC.append(pair_counts['UC'])
        else:
            UC.append(0)
        if 'UG' in pair_counts:
            UG.append(pair_counts['UG'])
        else:
            UG.append(0)
    X_train['AC'] = np.array(AC) + np.array(CA)
    X_train['AG'] = np.array(AG) + np.array(GA)
    X_train['AU'] = np.array(AU) + np.array(UA)
    X_train['CG'] = np.array(CG) + np.array(GC)
    X_train['CU'] = np.array(CU) + np.array(UC)
    X_train['GU'] = np.array(GU) + np.array(UG) 
    return X_train 

# CONVERT STRING SEQUENCE TO LIST OF LETTERS
def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^acgu]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array

# ONEHOT AND INTEGER ENCODING
def one_hot_encoder(my_array):
    label_encoder = LabelEncoder()
    label_encoder.fit(np.array(['a','c','g','u','z']))
    integer_encoded1 = label_encoder.transform(my_array)
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
    integer_encoded = integer_encoded1.reshape(len(integer_encoded1), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)    
    return onehot_encoded, integer_encoded1

def one_hot_matrix_sequence(X):
    one_hot_column = list()
    integer_column = list()
    for i in range(len(X)):
        sequence = X['sequence'][i]
        sequence = string_to_array(sequence)
        one_hot_encoding, integer_encoding = one_hot_encoder(sequence)
        one_hot_column.append(one_hot_encoding)
        integer_column.append(integer_encoding)
    X['onehot_vectors'] = one_hot_column
    X['integer_encoded'] = integer_column
    return X