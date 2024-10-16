import pandas

def separate_target(data):

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    return X, y

