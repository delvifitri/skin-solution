from app.core import db
from dataclasses import dataclass

@dataclass
class Product(db.Model):
    id: int
    name: str
    price: float
    brand: str
    tag: str
    image: str
    rating: float
    description: str
    category: str

    __tablename__ = 'products'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    price = db.Column(db.Float)
    brand = db.Column(db.String(255))
    tag = db.Column(db.String(255), nullable=False)
    image = db.Column(db.String(255))
    rating = db.Column(db.Float)
    description = db.Column(db.Text)
    category = db.Column(db.String(255))

@dataclass
class Review(db.Model):
    id: int
    product_id: int
    username: str
    age_range: str
    skin_type: str
    rating: float
    text: str
    is_recommended: bool

    __tablename__ = 'reviews'
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    username = db.Column(db.String(255), nullable=False)
    age_range = db.Column(db.String(255))
    skin_type = db.Column(db.String(255))
    rating = db.Column(db.Float)
    text = db.Column(db.Text)
    is_recommended = db.Column(db.Boolean)
    product = db.relationship('Product', backref='reviews')

