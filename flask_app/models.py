from flask_app import db, bcrypt

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    # nickname = db.Column(db.String(50), nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

from flask_app import db
from flask_app.models import User  # if needed

class SavedPaper(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(500), nullable=False)
    authors = db.Column(db.String(500))
    year = db.Column(db.String(10))
    url = db.Column(db.String(1000))
    citations = db.Column(db.Integer)
    score = db.Column(db.Float)

    user = db.relationship('User', backref=db.backref('saved_papers', lazy=True))