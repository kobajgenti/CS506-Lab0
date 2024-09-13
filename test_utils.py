## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = np.dot(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    ### YOUR CODE HERE
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])

    ### (1(4)+2(5)+3(6))/((1^2+2^2+3^2)^0.5*(4^2+5^2+6^2)^0.5) = 0.9746
    result = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)) 
    
    expected_result = 0.9746318 ### YOUR CODE HERE
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    ### YOUR CODE HERE
    m1 = np.matrix([[1, 2],[3,4]])
    v2 = np.array([4, 6])

    distances = np.linalg.norm(m1 - v2, axis=1)
    result = np.argmin(distances) ### YOUR CODE HERE
    
    expected_index = 1 ### YOUR CODE HERE
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
