# pluralsight-exercise

Analyzing Pluralsight user data.

## Directions for use

Softwares needed - Python, PostgreSQL
Python packages needed - pandas, numpy, sklearn, scikit-surprise, psycopg2, sqlalchemy, flask, flask-cors

### Instructions for running the API
Only the 3 files (controller.py, db_setup.py, similarity_model.py) in the main folder are needed.
1.	Creating the PostgreSQL database and tables –<br/>
  a.	Download PostgreSQL from here - https://www.postgresql.org/download/.<br/>
  b.	Create 2 INI files – server.ini and database.ini containing server and database parameters as shown here - https://www.postgresqltutorial.com/postgresql-python/connect/. Store these files in the same location as the other files.<br/>
  c.	Run the db_setup.py file.<br/>
2.	Training the model –<br/>
  a.	Run the similarity_model.py file.<br/>
3.	Running the API –<br/>
  a.	Run the controller.py file.
