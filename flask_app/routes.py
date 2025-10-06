from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_app import queries, db, bcrypt
from flask_app.models import User

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html', queries=queries)

# 🔹 Signup Route
@bp.route('/signup', methods=['POST'])
def signup():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    # Check if user already exists
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        flash('Email already registered. Please log in.', 'warning')
        return redirect(url_for('main.index'))

    # Hash password and save new user
    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(name=name, email=email, password=hashed_pw)
    db.session.add(new_user)
    db.session.commit()

    # ✅ Automatically log the user in
    session['user_id'] = new_user.id
    session['user_name'] = new_user.name

    flash(f'Welcome, {new_user.name}! Your account has been created.', 'success')
    return redirect(url_for('main.index'))

# 🔹 Login Route
@bp.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        session['user_id'] = user.id
        session['user_name'] = user.name
        flash(f'Welcome back, {user.name}!', 'success')
        return redirect(url_for('main.index'))
    else:
        flash('Invalid email or password.', 'danger')
        return redirect(url_for('main.index'))

# 🔹 Logout Route
@bp.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('main.index'))
