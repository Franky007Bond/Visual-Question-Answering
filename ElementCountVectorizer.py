import numpy as np
from scipy import sparse

class ElementCountVectorizer():
    '''
    Counts unique elements in lists and returns results in a sparse matrix
    In contrast to sklearn.CountVectorizer it does not count single words but
    single list elements as tokens
    Inspired by MyCountVectorizer by SongDark: https://github.com/SongDark/MyCountVectorizer
    '''
    def __init__(self, pass_stop=True):
        self.pass_stop = pass_stop
    
    def fit(self, data):
        '''
        INPUT
        data:      input vector of lists containing several elements
        '''
        self.elements_ = set()
        for line in data:
            for x in line:
                if self.pass_stop:
                    if len(x)==1:
                        continue
                self.elements_.add(x)
        self.elements_ = np.sort(list(self.elements_))
        self.labels_ = np.arange(len(self.elements_)).astype(int)
        self.dict_ = {}
        for i in range(len(self.elements_)):
            self.dict_[str(self.elements_[i])] = self.labels_[i]
    
    def transform(self, data):
        '''
        INPUT
        data:      input vector of lists containing several elements
        OUTPUT
        matrix     sparse matrix with # of rows equal to input vector length 
                   and # of columns equal to # of unique elements
        '''
        rows = []
        cols = []
        for i in range(len(data)):
            for x in data[i]:
                if self.pass_stop:
                    if len(x)==1:
                        continue
                rows.append(i)
                cols.append(self.dict_[x])
        vals = np.ones((len(rows),)).astype(int)

        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(data), len(self.labels_)))