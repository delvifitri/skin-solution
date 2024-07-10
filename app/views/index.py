from flask import Blueprint, render_template, request, jsonify
from sqlalchemy import func
from app.models.skincare import Product, Review
from app.core import db
import joblib
import pandas as pd
index = Blueprint('index', __name__)

@index.route('/')
def home():
    return render_template('index/home.html')

@index.route('/recommendations', methods=['POST'])
def recommendations():
    age_range = request.form.get('age_range')
    skin_type = request.form.get('skin_type')
    tag = request.form.get('tag')
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
            Review.age_range == age_range,
            Review.skin_type == skin_type
        )
        .group_by(Product.id)
        .all()
    )
    max_review_count = max([review_count for _, review_count in products_with_reviews_count])
    products = []
    for product, review_count in products_with_reviews_count:
        product.score = 0.7 * recommendations[product.id] + 0.3 * review_count / max_review_count * 5
        products.append(product)
    products.sort(key=lambda x: x.score, reverse=True)
    return render_template('index/result.html', products=products if len(products) < 10 else products[:10], age_range=age_range, skin_type=skin_type)

@index.route('/reviews')
def reviews():
    age_range = request.args.get('age_range')
    skin_type = request.args.get('skin_type')
    product_id = request.args.get('product_id')
    reviews = Review.query.filter_by(product_id=product_id, age_range=age_range, skin_type=skin_type).all()
    return jsonify(reviews)


def _encode_data(ids, age_range, skin_type):
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
    return pd.DataFrame(data)

def _recommend_products(data, tag):
    ratings = _predict_ratings(data, tag)
    text_ratings = _predict_text_ratings(data, tag)
    recommendations = _predict_recommendation(ratings, text_ratings, tag)
    result = {product_id: (ratings[i] * 0.6 + text_ratings[i] * 0.4) for i, product_id in enumerate(data['product_id']) if recommendations[i] == 1}
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
    return joblib.load(f'models/{model_name}/{tag}.pkl')