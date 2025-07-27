from app import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    infant_name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False)
