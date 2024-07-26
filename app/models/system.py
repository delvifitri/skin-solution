from datetime import datetime

from sqlalchemy import func
from app.core import db
from dataclasses import dataclass

@dataclass
class Setting(db.Model):
    id: int
    name: str
    value: str

    __tablename__ = 'settings'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    value = db.Column(db.String(255), nullable=False)

@dataclass
class User(db.Model):
    id: int
    username: str
    password: str
    email: str

    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False)

@dataclass
class TrainingResult(db.Model):
    id: int
    tag: str
    name: str
    raw_data: str
    cleaned_data: str
    processed_data: str
    dtr1_train: str
    dtr2_train: str
    svm_train: str
    dtr1_test: str
    dtr2_test: str
    svm_test: str
    dtr1_compare: str
    dtr2_compare: str
    svm_compare: str
    r2_1: float
    r2_2: float
    mae1: float
    mae2: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: str
    train_date: str
    active: bool

    __tablename__ = 'training_results'
    id = db.Column(db.Integer, primary_key=True)
    tag = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    raw_data = db.Column(db.Text)
    cleaned_data = db.Column(db.Text)
    processed_data = db.Column(db.Text)
    dtr1_train = db.Column(db.Text)
    dtr2_train = db.Column(db.Text)
    svm_train = db.Column(db.Text)
    dtr1_test = db.Column(db.Text)
    dtr2_test = db.Column(db.Text)
    svm_test = db.Column(db.Text)
    dtr1_compare = db.Column(db.Text)
    dtr2_compare = db.Column(db.Text)
    svm_compare = db.Column(db.Text)
    r2_1 = db.Column(db.Float, nullable=False)
    r2_2 = db.Column(db.Float, nullable=False)
    mae1 = db.Column(db.Float, nullable=False)
    mae2 = db.Column(db.Float, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    confusion_matrix = db.Column(db.Text)
    train_date = db.Column(db.DateTime, default=func.now())
    active = db.Column(db.Boolean, nullable=False, default=False)