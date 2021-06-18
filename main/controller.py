# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18

@author: Aatish Suman
"""

from flask import Flask
from similarity_model import UserNotFoundException, SimilarityModel
from flask_cors import cross_origin

app = Flask(__name__)

@app.route('/getSimilarUsers/<user_handle>', methods=['GET'])
@cross_origin()
def get_prediction(user_handle):
    try:
        response = model.get_similar_users(int(user_handle))
        print(response)
        return response
    except (UserNotFoundException) as error:
        print(error)
        return 'User does not exist', 404
    except (Exception) as error:
        print(error)
        return error, 500

model = SimilarityModel()
model.build_similarity_matrix_from_model()
app.run(host='0.0.0.0')