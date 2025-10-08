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

# 🔹 Profile Route
@bp.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('main.login'))

    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        name = request.form.get('name')
        nickname = request.form.get('nickname')
        if name:
            user.name = name
        user.nickname = nickname
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        session['user_name'] = nickname if nickname else name  # update displayed name
        return redirect(url_for('main.profile'))

    # If GET, show profile
    bookmarks = []  # replace with your actual bookmark query later
    return render_template('profile.html', user=user, bookmarks=bookmarks)

@bp.route('/bookmarks')
def bookmarks():
    if 'user_id' not in session:
        return redirect(url_for('main.login'))

    user_id = session['user_id']
    # Example query – adjust depending on your database structure
    bookmarks = bookmarks.query.filter_by(user_id=user_id).all()
    
    return render_template('bookmarks.html', bookmarks=bookmarks)
