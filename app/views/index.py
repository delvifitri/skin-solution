import math
from flask import Blueprint, render_template, request, jsonify
from sqlalchemy import func
from app.models.skincare import Product, Review
from app.core import db
import joblib
import pandas as pd
from functools import lru_cache
from flask_paginate import Pagination

from app.models.system import TrainingResult
index = Blueprint('index', __name__)

@index.route('/')
def home():
    return render_template('index/home.html')

@index.route('/process')
def process():
    age_range = request.args.get('age_range')
    skin_type = request.args.get('skin_type')
    tag = request.args.get('tag')

    return _show_process(tag, age_range, skin_type)

@index.route('/recommendations')
def recommendations():
    age_range = request.args.get('age_range')
    skin_type = request.args.get('skin_type')
    tag = request.args.get('tag')
    products = _get_recommendations(tag, age_range, skin_type)
    page = request.args.get('page', type=int, default=1)
    pagination = Pagination(page=page, total=len(products), record_name='products', per_page=10)
    products = products[(page - 1) * 10:(page * 10 if page * 10 < len(products) else len(products))]
    return render_template('index/result.html', products=products, age_range=age_range, skin_type=skin_type, tag=tag, pagination=pagination)

@lru_cache(maxsize=128)
def _show_process(tag, age_range, skin_type):
    products = Product.query.filter_by(tag=tag).all()

    products_encoded = _encode_data([(product.id,) for product in products], age_range, skin_type, df=False)
    
    df_encoded = _encode_data([(product.id,) for product in products], age_range, skin_type)

    ratings = _predict_ratings(df_encoded, tag)
    text_ratings = _predict_text_ratings(df_encoded, tag)

    is_recommended = _predict_recommendation(ratings, text_ratings, tag)


    filtered_products = {product.id: {
        'product': product,
        'rating': ratings[i],
        'rating_text': text_ratings[i],
        'avg_rating': (ratings[i] + text_ratings[i]) / 2
    } for i, product in enumerate(products) if is_recommended[i] == 1}

    filtered_ids = filtered_products.keys()
    # get review count for each product
    reviews_count = (
        db.session.query(
            Product,
            func.count(Review.id).label('review_count')
        )
        .outerjoin(Review, Product.id == Review.product_id)
        .filter(
            Product.id.in_(filtered_ids),
            Review.age_range.like(f'%{age_range}%'),
            Review.skin_type.like(f'%{skin_type}%')
        )
        .group_by(Product.id, Product.name)
        .all()
    )

    for product, review_count in reviews_count:
        filtered_products[product.id]['review_count'] = review_count
        filtered_products[product.id]['wilson_score'] = sum(_wilson_score(filtered_products[product.id]['avg_rating'], review_count, z=1.96)) / 2 * 5
    
    filtered_products2 = {k: v for k, v in filtered_products.items() if 'wilson_score' in v}
    filtered_products2 = {k: v for k, v in sorted(filtered_products2.items(), key=lambda item: item[1]['wilson_score'], reverse=True)}

    return render_template('index/process.html', products=products, age_range=age_range, skin_type=skin_type, products_encoded=products_encoded, ratings=ratings, text_ratings=text_ratings, is_recommended=is_recommended, enumerate=enumerate, filtered_products=filtered_products, filtered_products2=filtered_products2)

@index.route('/reviews')
def reviews():
    age_range = request.args.get('age_range')
    skin_type = request.args.get('skin_type')
    product_id = request.args.get('product_id')
    page = request.args.get('page', 1, type=int)
    reviews = Review.query.filter_by(product_id=product_id, age_range=age_range, skin_type=skin_type).paginate()
    return jsonify([review for review in reviews])


def _encode_data(ids, age_range, skin_type, df=True):
    data = [{
        'product_id': x[0],
        'age_range_18 and Under': 1 if age_range == '18 and Under' else 0,
        'age_range_19 - 24': 1 if age_range == '19 - 24' else 0,
        'age_range_25 - 29': 1 if age_range == '25 - 29' else 0,
        'age_range_30 - 34': 1 if age_range == '30 - 34' else 0,
        'age_range_35 - 39': 1 if age_range == '35 - 39' else 0,
        'age_range_40 - 44': 1 if age_range == '40 - 44' else 0,
        'age_range_45 and Above': 1 if age_range == '45 and Above' else 0,
        'skin_type_Combination': 1 if skin_type == 'Combination' else 0,
        'skin_type_Dry': 1 if skin_type == 'Dry' else 0,
        'skin_type_Normal': 1 if skin_type == 'Normal' else 0,
        'skin_type_Oily': 1 if skin_type == 'Oily' else 0,
    } for x in ids]
    if not df:
        return data
    return pd.DataFrame(data)

@lru_cache(maxsize=128)
def _get_recommendations(tag, age_range, skin_type):
    product_ids = Product.query.filter_by(tag=tag).with_entities(Product.id).all()
    data = _encode_data(product_ids, age_range, skin_type)
    recommendations = _recommend_products(data, tag)
    products_with_reviews_count = (
        db.session.query(
            Product,
            func.count(Review.id).label('review_count')
        )
        .outerjoin(Review, Product.id == Review.product_id)
        .filter(
            Product.id.in_(recommendations.keys()),
            Review.age_range.like(f'%{age_range}%'),
            Review.skin_type.like(f'%{skin_type}%')
        )
        .group_by(Product.id)
        .all()
    )
    print(len(products_with_reviews_count))
    products = []
    for product, review_count in products_with_reviews_count:
        product.score = sum(_wilson_score(recommendations[product.id], review_count, z=1.96)) / 2 * 5
        products.append(product)
    products.sort(key=lambda x: x.score, reverse=True)
    return products

def _recommend_products(data, tag):
    ratings = _predict_ratings(data, tag)
    text_ratings = _predict_text_ratings(data, tag)
    recommendations = _predict_recommendation(ratings, text_ratings, tag)
    result = {product_id: (ratings[i] + text_ratings[i]) / 2 for i, product_id in enumerate(data['product_id']) if recommendations[i] == 1}
    return result

def _predict_ratings(data, tag):
    model = _load_model(tag, 'dtr1')
    return model.predict(data)

def _predict_text_ratings(ratings, tag):
    model = _load_model(tag, 'dtr2')
    return model.predict(ratings)

def _predict_recommendation(ratings, text_ratings, tag):
    model = _load_model(tag, 'svm')
    df = pd.DataFrame({
        'rating': ratings,
        'rating_text': text_ratings
    })
    return model.predict(df)

def _load_model(tag, model_name):
    model = TrainingResult.query.filter_by(tag=tag, active=True).first()
    return joblib.load(f'models/{model_name}/{tag}.pkl') if model.name == 'default' else joblib.load(f'models/{model_name}/{model.name}.pkl')

def _wilson_score(p, n, z=1.96):
    p = p / 5
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return lower_bound, upper_bound
