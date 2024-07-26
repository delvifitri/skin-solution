import hashlib
from flask import Blueprint, redirect, render_template, request, session
from functools import wraps
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
from werkzeug.security import check_password_hash
from app.core import db
from app.models.system import Setting, TrainingResult
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import joblib
from io import BytesIO
from tqdm import tqdm

from app.lbsa.sentistrength_id import SentiStrengthID

admin = Blueprint('admin', __name__)

def logged_in(f):
    @wraps(f)
    def decorated_func(*args, **kwargs):
        if session.get('logged_in'):
            return f(*args, **kwargs)
        else:
            return redirect(f'/admin/login')
    return decorated_func

@admin.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect('/admin')
    if request.method == 'POST':
        password = request.form.get('password')
        pwhashed = Setting.query.filter_by(name='ADMIN_PASSWORD').first().value
        print(check_password_hash(pwhashed, password))
        if check_password_hash(pwhashed, password):
            session['logged_in'] = True
            return redirect('/admin')
        else:
            return render_template('admin/login.html', error="Invalid password")
    return render_template('admin/login.html')

@admin.route('/logout')
@logged_in
def logout():
    session.pop('logged_in', None)
    return redirect('/admin/login')

@admin.route('/')
@logged_in
def home():
    return render_template('admin/home.html')

@admin.route('/models')
@logged_in
def models():
    tag = request.args.get('tag')
    models = TrainingResult.query.filter_by(tag=tag).order_by(TrainingResult.id.desc()).all()
    category = {
        'facial-wash': 'Facial Wash',
        'toner': 'Toner',
        'serum-essence': 'Serum & Essence',
        'cream-1': 'Moisturizer (Cream)',
        'lotion-emulsion': 'Moisturizer (Lotion)',
        'sun-protection-1': 'Sunscreen',
    }[tag]
    return render_template('admin/models.html', models=models, tag=tag, category=category)

@admin.route('/model/active')
@logged_in
def activate():
    name = request.args.get('name')
    tag = request.args.get('tag')
    models = TrainingResult.query.filter_by(tag=tag).order_by(TrainingResult.id.desc()).all()
    for model in models:
        if model.name == name:
            model.active = True
        else:
            model.active = False
    db.session.commit()
    return "Success"

@admin.route('/model/<int:id>')
@logged_in
def training_result(id):
    result = TrainingResult.query.get(id)
    status = request.args.get('status')
    return render_template('admin/training_result.html', result=result, message="Pelatihan selesai" if status == 1 else f"Anda sudah pernah melatih model dengan dataset yang sama pada {result.train_date}" if status == -1 else None)

@admin.route('/training', methods=['POST'])
@logged_in
def training():
    file = request.files['dataset']
    file.save('dataset.csv')

    with open('dataset.csv', 'r', encoding='utf-8') as f:
        checksum = hashlib.sha256(f.read().encode()).hexdigest()

    previous_result = TrainingResult.query.filter_by(name=checksum).first()
    if previous_result:
        return redirect(f'/admin/model/{previous_result.id}?status=-1')
    
    tag = request.form.get('tag')
    ss = SentiStrengthID()
    def process(row):
        row['rating_text'], row['processed_text'] = ss.score(row['text'])
        return row

    df = pd.read_csv('dataset.csv')
    table_df = df.head().to_html(header="true").replace('border="1"', '')
    table_df += f'<p>Total data: {len(df)}</p>'
    print("Dataset loaded, total rows:", len(df))

    df_cleaned = df.dropna()
    del df
    df_cleaned = df_cleaned[df_cleaned.is_recommended != 0]
    df_cleaned.replace(r"(.*) Skin", r"\1", inplace=True, regex=True)
    table_df_cleaned = df_cleaned.head().to_html(header="true").replace('border="1"', '')
    table_df_cleaned += f'<p>Total data: {len(df_cleaned)}</p>'
    print("Dataset cleaned, total rows:", len(df_cleaned))


    # create a copy with additional column
    df_processed = df_cleaned.copy()
    del df_cleaned
    df_processed.loc[:, ['rating_text', 'processed_text']] = np.nan, ''
    for index, row in tqdm(df_processed.iterrows()):
        df_processed.loc[index] = process(row)
    table_df_processed = df_processed.head().to_html(header="true").replace('border="1"', '')
    table_df_processed += f'<p>Total data: {len(df_processed)}</p>'
    print("Dataset processed")
    
    dtr_X = df_processed[['product_id', 'age_range', 'skin_type']]
    dtr_Xd = pd.get_dummies(dtr_X)
    del dtr_X

    dtr1_y = df_processed['rating']
    X_train1, X_test1, y_train1, y_test1 = train_test_split(dtr_Xd, dtr1_y, test_size=0.2, random_state=42)
    dtr1_train = X_train1.merge(y_train1, left_index=True, right_index=True)
    dtr1_test = X_test1.merge(y_test1, left_index=True, right_index=True)
    table_dtr1_train = dtr1_train.head().to_html(header="true").replace('border="1"', '')
    table_dtr1_train += f'<p>Total data: {len(dtr1_train)}</p>'
    table_dtr1_test = dtr1_test.head().to_html(header="true").replace('border="1"', '')
    table_dtr1_test += f'<p>Total data: {len(dtr1_test)}</p>'
    del X_train1, y_train1, dtr1_train, dtr1_test

    dtr1 = DecisionTreeRegressor()
    dtr1.fit(dtr_Xd, dtr1_y)
    del dtr1_y
    print("Model 1 trained")
    dtr1_pred = dtr1.predict(X_test1)
    del X_test1
    r2_1 = r2_score(y_test1, dtr1_pred)
    mae1 = mean_absolute_error(y_test1, dtr1_pred)
    dtr1_compare = pd.DataFrame({'actual': y_test1, 'predicted': dtr1_pred})
    table_dtr1_compare = dtr1_compare.head().to_html(header="true").replace('border="1"', '')
    del y_test1, dtr1_pred, dtr1_compare

    dtr2_y = df_processed['rating_text']
    X_train2, X_test2, y_train2, y_test2 = train_test_split(dtr_Xd, dtr2_y, test_size=0.2, random_state=42)
    dtr2_train = X_train2.merge(y_train2, left_index=True, right_index=True)
    dtr2_test = X_test2.merge(y_test2, left_index=True, right_index=True)
    table_dtr2_train = dtr2_train.head().to_html(header="true").replace('border="1"', '')
    table_dtr2_train += f'<p>Total data: {len(dtr2_train)}</p>'
    table_dtr2_test = dtr2_test.head().to_html(header="true").replace('border="1"', '')
    table_dtr2_test += f'<p>Total data: {len(dtr2_test)}</p>'
    del X_train2, y_train2, dtr2_train, dtr2_test

    dtr2 = DecisionTreeRegressor()
    dtr2.fit(dtr_Xd, dtr2_y)
    del dtr2_y
    print("Model 2 trained")
    dtr2_pred = dtr2.predict(X_test2)
    del X_test2
    r2_2 = r2_score(y_test2, dtr2_pred)
    mae2 = mean_absolute_error(y_test2, dtr2_pred)
    dtr2_compare = pd.DataFrame({'actual': y_test2, 'predicted': dtr2_pred})
    table_dtr2_compare = dtr2_compare.head().to_html(header="true").replace('border="1"', '')
    del y_test2, dtr2_pred, dtr2_compare

    del dtr_Xd

    svm_X = df_processed[['rating', 'rating_text']]
    svm_y = df_processed['is_recommended']

    del df_processed

    svm = SVC()
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(svm_X, svm_y, test_size=0.2, random_state=42)
    svm_train = X_train_svm.merge(y_train_svm, left_index=True, right_index=True)
    svm_test = X_test_svm.merge(y_test_svm, left_index=True, right_index=True)
    svm.fit(X_train_svm, y_train_svm)
    table_svm_train = svm_train.head().to_html(header="true").replace('border="1"', '')
    table_svm_train += f'<p>Total data: {len(svm_train)}</p>'
    table_svm_test = svm_test.head().to_html(header="true").replace('border="1"', '')
    table_svm_test += f'<p>Total data: {len(svm_test)}</p>'
    del X_train_svm, y_train_svm, svm_train, svm_test

    svm_pred = svm.predict(X_test_svm)
    del X_test_svm

    acc = accuracy_score(y_test_svm, svm_pred)
    precision = precision_score(y_test_svm, svm_pred)
    recall = recall_score(y_test_svm, svm_pred)
    fscore = f1_score(y_test_svm, svm_pred)
    cm = confusion_matrix(y_test_svm, svm_pred)
    svm_compare = pd.DataFrame({'actual': y_test_svm, 'predicted': svm_pred})
    table_svm_compare = svm_compare.head().to_html(header="true").replace('border="1"', '')
    del y_test_svm, svm_pred, svm_compare

    # create confusion matrix
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
    disp = disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title('Confusion Matrix')
    cm_plot = BytesIO()
    fig.savefig(cm_plot, format='png')

    result = TrainingResult(
        tag=tag,
        name=checksum,
        raw_data=table_df,
        cleaned_data=table_df_cleaned,
        processed_data=table_df_processed,
        dtr1_train=table_dtr1_train,
        dtr2_train=table_dtr2_train,
        svm_train=table_svm_train,
        dtr1_test=table_dtr1_test,
        dtr2_test=table_dtr2_test,
        svm_test=table_svm_test,
        dtr1_compare=table_dtr1_compare,
        dtr2_compare=table_dtr2_compare,
        svm_compare=table_svm_compare,
        r2_1=r2_1,
        r2_2=r2_2,
        mae1=mae1,
        mae2=mae2,
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1_score=fscore,
        confusion_matrix=base64.encodestring(cm_plot.getvalue()).decode('utf-8'),
        active=False
    )
    db.session.add(result)
    db.session.commit()

    joblib.dump(dtr1, f'models/dtr1/{checksum}.pkl')
    joblib.dump(dtr2, f'models/dtr2/{checksum}.pkl')
    joblib.dump(svm, f'models/svm/{checksum}.pkl')

    return redirect(f'/admin/model/{result.id}?status=1')

