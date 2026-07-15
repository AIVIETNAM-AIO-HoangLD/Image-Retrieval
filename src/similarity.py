import numpy as np
import cv2
class Similarity:
    def __init__(self, data, query):
        self.query = query
        self.data = data

    def abosolute_difference(self, pic):
        if pic.ndim != 1: 
            return np.sum(cv2.absdiff(pic, self.query))
        else: return np.sum(np.abs(pic - self.query))
    
    def get_L1_score(self):
            scores = []
            for pic in self.data:
                score = self.abosolute_difference(pic)
                scores.append(score)
            #normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            return scores

    def mean_square_differences(self, pic):
        if pic.ndim != 1:
            return np.sqrt(np.sum((pic - self.query)**2))
        else: return np.sqrt(np.sum((pic - self.query)**2))
    
    def get_L2_score(self):
        scores = []
        for pic in self.data:
            score = self.mean_square_differences(pic)
            scores.append(score)
        return scores

    def cosine_similarity(self, pic):
        query_norm = np.sqrt(np.sum(self.query**2))
        data_norm = np.sqrt(np.sum(pic**2))
        if self.query.ndim != 1:
            return np.dot(pic.flatten(), self.query.flatten())/ (query_norm * data_norm)
        else:
            return np.dot(pic, self.query) / (query_norm * data_norm)

    
    def get_cosine_similarity(self):
        if self.query.ndim !=1:
            scores = []
            for pic in self.data:
                score = self.cosine_similarity(pic)
                scores.append(score)
            return scores
        else:
            return self.cosine_similarity(self.data)
    
    def correlation_coefficient(self, pic):
        query_mean = self.query - np.mean(self.query)
        data_mean = pic - np.mean(pic)
        query_norm = np.sqrt(np.sum(self.query**2))
        data_norm = np.sqrt(np.sum(pic**2))
        return np.sum(data_mean * query_mean) / (query_norm * data_norm)
    
    def get_correlation_coefficient(self):
        scores = []
        for pic in self.data:
            score = self.correlation_coefficient(pic)
            scores.append(score)
        return scores



    

    


