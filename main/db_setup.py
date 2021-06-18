# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17

@author: Aatish Suman
"""

from configparser import ConfigParser
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sqlalchemy
import pandas as pd

class DBSetup:
    def __init__(self):
        self.SERVER_CONFIG = 'server.ini' # file containing the server credentials
        self.DATABASE_CONFIG = 'database.ini' # file containing the database credentials
        self.DB = 'pluralsight' # name of the database
        self.DATA_URL = 'https://raw.githubusercontent.com/aatishsuman/plural-sight-exercise/master/data/' # github URL user data is stored
        self.FILE_NAMES = ['course_tags', 'user_assessment_scores', 'user_course_views', 'user_interests']
    
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
    
    def create_database(self):
        '''Creates the DB'''
        
        conn = None
        try:
            params = self.get_config(self.SERVER_CONFIG)
    
            print('Connecting to the PostgreSQL server...')
            conn = psycopg2.connect(**params)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
    
            cur.execute('DROP DATABASE IF EXISTS {}'.format(self.DB))
            cur.execute('CREATE DATABASE {}'.format(self.DB))
            print('Database {} successfully created.'.format(self.DB))
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
                print('Database connection closed.')
    
    def create_tables(self):
        '''Creates the user data tables'''
        
        engine = None
        try:
            params = self.get_config(self.DATABASE_CONFIG)
            engine = sqlalchemy.create_engine('postgresql://' + params['user'] + ':' + params['password'] + '@' + params['host'] + ':' + params['port'] + '/' + params['database'])
            for file_name in self.FILE_NAMES:            
                data = pd.read_csv(self.DATA_URL + file_name + '.csv')
                data.to_sql(file_name, engine, if_exists='replace')
                print('Table {} successfully created.'.format(file_name))
        except (Exception) as error:
            print(error)
        finally:
            if engine is not None:
                engine.dispose()
                print('Database connection closed.')

def main():
    db_setup = DBSetup()
    db_setup.create_database()
    db_setup.create_tables()

if __name__ == '__main__':
    main()