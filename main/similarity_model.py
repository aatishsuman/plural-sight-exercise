# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18

@author: Aatish Suman
"""

from configparser import ConfigParser
import sqlalchemy
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.accuracy import rmse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import reduce

class UserNotFoundException(Exception):
    pass

class SimilarityModel:
    def __init__(self):
        self.DATABASE_CONFIG = 'database.ini'
        
        # gets the normalized user-assessment data
        self.USER_ASSESSMENT_QUERY = 'with norm as (select assessment_tag, avg(user_assessment_score) score_avg, stddev(user_assessment_score) score_stddev from user_assessment_scores group by assessment_tag) select user_handle, user_assessment_scores.assessment_tag, coalesce((user_assessment_score - norm.score_avg) / (norm.score_stddev + 1), 0) score_scaled from user_assessment_scores left join norm on user_assessment_scores.assessment_tag = norm.assessment_tag'
        
        # gets the normalized user-course views data
        self.USER_COURSE_QUERY = 'with norm as (select course_id, avg(view_time_seconds) view_time_avg, stddev(view_time_seconds) view_time_stddev from user_course_views group by course_id), data as (select user_handle, course_id, sum(view_time_seconds) view_time from user_course_views group by user_handle, course_id) select data.user_handle, data.course_id, coalesce((data.view_time - norm.view_time_avg) / (norm.view_time_stddev + 1), 0) view_time_scaled from data left join norm on data.course_id = norm.course_id'
        
        # gets the grouped user-interest data
        self.USER_INTEREST_QUERY = "select user_handle, string_agg(interest_tag, ', ') interest_tags from user_interests group by user_handle"
        
        # gets the user factors
        self.USER_FACTORS_QUERY = 'select * from user_factors'
        
        self.N_EPOCHS, self.N_FACTORS, self.LR_ALL, self.REG_ALL = 40, 175, 0.007, 0.03 # model parameters
        self.ASSESSMENT_WEIGHT, self.COURSE_WEIGHT, self.INTEREST_WEIGHT = 0.5, 0.2, 0.3 # similarity weights
    
    def get_config(self, filename, section='postgresql'):
        '''Reads the DB params from the disk'''
        
        parser = ConfigParser()
        parser.read(filename)
        db_params = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db_params[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, filename))
        return db_params
    
    def get_query_results(self, queries):
        '''Runs the queries and returns the results as dataframes'''
        
        engine = None
        try:
            params = self.get_config(self.DATABASE_CONFIG)
            engine = sqlalchemy.create_engine('postgresql://' + params['user'] + ':' + params['password'] + '@' + params['host'] + ':' + params['port'] + '/' + params['database'])
            return tuple([pd.read_sql(query, con=engine) for query in queries])
        except (Exception) as error:
            print(error)
        finally:
            if engine is not None:
                engine.dispose()
    
    def fetch_data(self):
        '''Gets the scores, course-views and the interests data from the DB'''
        
        self.user_assessment_scores_df, self.user_course_views_df, self.user_interests_df = self.get_query_results([self.USER_ASSESSMENT_QUERY, self.USER_COURSE_QUERY, self.USER_INTEREST_QUERY])
        print('User-assessment score, course views, interests shape', self.user_assessment_scores_df.shape, self.user_course_views_df.shape, self.user_interests_df.shape)
    
    def build_assessment_similarity_matrix(self):
        '''Builds the assessment similarity matrix using normalized scores'''
        
        user_assessment_score_vectors = pd.pivot(self.user_assessment_scores_df, index='user_handle', columns='assessment_tag', values='score_scaled').fillna(0)
        print('User-assessment score matrix shape -', user_assessment_score_vectors.shape)
        return pd.DataFrame(cosine_similarity(user_assessment_score_vectors), columns=user_assessment_score_vectors.index, index=user_assessment_score_vectors.index)
    
    def build_model(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42):
        '''Builds the collaborative filtering model to get the user and course factors'''
        
        reader = Reader(rating_scale=(min(self.user_course_views_df['view_time_scaled']), max(self.user_course_views_df['view_time_scaled'])))
        data = Dataset.load_from_df(self.user_course_views_df, reader)
        trainset = data.build_full_trainset()
        model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, random_state=random_state)
        model.fit(trainset)
        return model, rmse(model.test(trainset.build_testset())), trainset
    
    def save_factors(self, user_factors, course_factors):
        '''Saves the user and course factors to the DB'''
        
        engine = None
        try:
            params = self.get_config(self.DATABASE_CONFIG)
            engine = sqlalchemy.create_engine('postgresql://' + params['user'] + ':' + params['password'] + '@' + params['host'] + ':' + params['port'] + '/' + params['database'])
            
            user_factors.to_sql('user_factors', engine, if_exists='replace')
            print('Table user_factors successfully created.')
            
            course_factors.to_sql('course_factors', engine, if_exists='replace')
            print('Table course_factors successfully created.')
        except (Exception) as error:
            print(error)
        finally:
            if engine is not None:
                engine.dispose()
                print('Database connection closed.')
    
    def get_factors(self, model, trainset):
        '''Gets the user and course factors from the model and saves them to the DB'''
        
        user_factors = model.pu
        user_factors = np.hstack((user_factors, model.bu[..., None]))
        user_factors = pd.DataFrame.from_dict(dict(zip([trainset.to_raw_uid(user_id) for user_id in trainset.all_users()], user_factors)), orient='index')
        
        course_factors = model.qi
        course_factors = np.hstack((course_factors, model.bi[..., None]))
        course_factors = pd.DataFrame.from_dict(dict(zip([trainset.to_raw_iid(course_id) for course_id in trainset.all_items()], course_factors)), orient='index')
        self.save_factors(user_factors, course_factors)
        return user_factors, course_factors
    
    def build_course_similarity_matrix(self):
        '''Builds collaborative filtering model and the course similarity matrix using normalized view times'''
        
        model, rmse_value, trainset = self.build_model(n_factors=self.N_FACTORS, n_epochs=self.N_EPOCHS, lr_all=self.LR_ALL, reg_all=self.REG_ALL)
        user_factors, course_factors = self.get_factors(model, trainset)
        print('User and course factors matrix shape -', user_factors.shape, course_factors.shape)
        return pd.DataFrame(cosine_similarity(user_factors), columns=user_factors.index, index=user_factors.index)
    
    def build_course_similarity_matrix_from_model(self):
        '''Builds the course similarity matrix using normalized view times from trained model'''
        
        user_factors = self.get_query_results([self.USER_FACTORS_QUERY])[0]
        user_handles = user_factors[user_factors.columns[0]]
        user_factors.drop(user_factors.columns[0], axis=1, inplace=True)
        print('User-factors matrix shape -', user_factors.shape)
        return pd.DataFrame(cosine_similarity(user_factors), columns=user_handles, index=user_handles)
    
    def build_interest_similarity_matrix(self):
        '''Builds the interest similarity matrix using TF-IDF values'''
        
        vectorizer = TfidfVectorizer(tokenizer=lambda text: text.split(', '))
        vectors = vectorizer.fit_transform(self.user_interests_df['interest_tags'])
        print('User-interest matrix shape -', vectors.shape)
        return pd.DataFrame(cosine_similarity(vectors), columns=self.user_interests_df['user_handle'], index=self.user_interests_df['user_handle'])
    
    def build_similarity_matrix(self):
        '''Fetches the user data from the DB and builds the overall similarity matrix using weighted similarities'''
        
        self.fetch_data()
        self.assessment_similarity_df = self.build_assessment_similarity_matrix()
        self.course_similarity_df = self.build_course_similarity_matrix()
        self.interest_similarity_df = self.build_interest_similarity_matrix()
        self.similarity_df = reduce(lambda a, b: a.add(b, fill_value=0), [self.assessment_similarity_df * self.ASSESSMENT_WEIGHT, self.course_similarity_df * self.COURSE_WEIGHT, self.interest_similarity_df * self.INTEREST_WEIGHT])
    
    def build_similarity_matrix_from_model(self):
        '''Fetches the user data and the trained model and builds the overall similarity matrix using weighted similarities'''
        
        self.fetch_data()
        self.assessment_similarity_df = self.build_assessment_similarity_matrix()
        self.course_similarity_df = self.build_course_similarity_matrix_from_model()
        self.interest_similarity_df = self.build_interest_similarity_matrix()
        self.similarity_df = reduce(lambda a, b: a.add(b, fill_value=0), [self.assessment_similarity_df * self.ASSESSMENT_WEIGHT, self.course_similarity_df * self.COURSE_WEIGHT, self.interest_similarity_df * self.INTEREST_WEIGHT])
    
    def get_similar_users(self, user_handle):
        '''Gets similar users for the user_handle'''
        
        if user_handle not in self.similarity_df:
            raise UserNotFoundException()
        
        user_top_assessment_tags = list(self.user_assessment_scores_df[self.user_assessment_scores_df['user_handle'] == user_handle].sort_values(by='score_scaled', ascending=False)['assessment_tag'])
        user_top_assessment_tags = user_top_assessment_tags if len(user_top_assessment_tags) < 3 else user_top_assessment_tags[:3]
        
        user_highest_course_views = list(self.user_course_views_df[self.user_course_views_df['user_handle'] == user_handle].sort_values(by='view_time_scaled', ascending=False)['course_id'])
        user_highest_course_views = user_highest_course_views if len(user_highest_course_views) < 3 else user_highest_course_views[:3]
        
        user_interest_tags = list(self.user_interests_df[self.user_interests_df['user_handle'] == user_handle]['interest_tags'])
        
        similar_users = self.assessment_similarity_df[user_handle].drop(labels=user_handle).sort_values(ascending=False)[:10] if user_handle in self.assessment_similarity_df else []
        similar_score_users = {'users': list(similar_users.index), 'similarities': list(similar_users.values)} if len(similar_users) > 0 else {}
        similar_users = self.course_similarity_df[user_handle].drop(labels=user_handle).sort_values(ascending=False)[:10] if user_handle in self.course_similarity_df else []
        similar_course_views_users = {'users': list(similar_users.index), 'similarities': list(similar_users.values)} if len(similar_users) > 0 else {}
        similar_users = self.interest_similarity_df[user_handle].drop(labels=user_handle).sort_values(ascending=False)[:10] if user_handle in self.interest_similarity_df else []
        similar_interest_users = {'users': list(similar_users.index), 'similarities': list(similar_users.values)} if len(similar_users) > 0 else {}
        similar_users = self.similarity_df[user_handle].drop(labels=user_handle).sort_values(ascending=False)[:10]
        similar_overall_users = {'users': list(similar_users.index), 'similarities': list(similar_users.values)}
        
        return {'user_top_assessment_tags': user_top_assessment_tags, 
                'user_highest_course_views': user_highest_course_views, 
                'user_interest_tags': user_interest_tags,
               'similar_score_users': similar_score_users,
               'similar_course_views_users': similar_course_views_users,
               'similar_interest_users': similar_interest_users,
               'similar_overall_users': similar_overall_users}

def main():
    similarity_model = SimilarityModel()
    similarity_model.build_similarity_matrix()

if __name__ == '__main__':
    main()