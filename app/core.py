from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
import os

app = Flask(__name__)
app.secret_key = 'whatever'
app.config['SQLALCHEMY_DATABASE_URI'] =  os.environ.get('SQLALCHEMY_DATABASE_URI', 'mysql+pymysql://root@localhost:3306/fd_reviews')
db = SQLAlchemy(app)

def set_sql_mode(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("SET SESSION sql_mode=(SELECT REPLACE(@@sql_mode, 'ONLY_FULL_GROUP_BY', ''))")
    cursor.close()

with app.app_context():
    engine = db.get_engine()
    event.listen(engine, 'connect', set_sql_mode)

from app.views.index import index
from app.views.admin import admin

app.register_blueprint(index, url_prefix='/')
app.register_blueprint(admin, url_prefix='/admin')
