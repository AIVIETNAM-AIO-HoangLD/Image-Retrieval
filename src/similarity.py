import numpy as np

class Similarity:

    def __init__(self, data, query):
        self.query = query
        self.data = data
        
    
    def abosolute_difference(self, pic):
        return np.sum(np.abs(pic.astype(np.int32) - self.query.astype(np.int32)))
    
    def get_f1_score(self):
        scores = []
        for pic in self.data:
            score = self.abosolute_difference(pic)
            scores.append(score)
        normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return normalized_scores

