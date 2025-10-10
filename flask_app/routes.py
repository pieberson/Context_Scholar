from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from flask_app import queries, db, bcrypt
from flask_app.models import User, SavedPaper

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
        # Update displayed name in session
        session['user_name'] = nickname if nickname else name
        return redirect(url_for('main.profile'))

    # GET request – just show profile page
    return render_template('profile.html', user=user)

@bp.route('/bookmarks')
def bookmarks():
    if 'user_id' not in session:
        return redirect(url_for('main.login'))

    user_id = session['user_id']
    bookmarks_list = SavedPaper.query.filter_by(user_id=user_id).all()
    
    return render_template('bookmarks.html', bookmarks=bookmarks_list)
    
    return render_template('bookmarks.html', bookmarks=bookmarks)

@bp.route('/save-paper', methods=['POST'])
def save_paper():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first.'}), 401

    data = request.get_json()

    # Avoid duplicates
    existing = SavedPaper.query.filter_by(user_id=session['user_id'], url=data.get('url')).first()
    if existing:
        return jsonify({'status': 'exists', 'message': 'This paper is already saved.'})

    new_paper = SavedPaper(
        user_id=session['user_id'],
        title=data.get('title'),
        authors=data.get('authors'),
        year=data.get('year'),
        url=data.get('url'),
        citations=data.get('citations'),
        score=data.get('score')
    )

    db.session.add(new_paper)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Paper saved successfully!'})

from flask import request, jsonify

@bp.route('/remove-bookmark', methods=['POST'])
def remove_bookmark():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first.'}), 401

    data = request.get_json()
    paper_id = data.get('id')
    paper = SavedPaper.query.filter_by(id=paper_id, user_id=session['user_id']).first()

    if not paper:
        return jsonify({'status': 'error', 'message': 'Paper not found.'})

    db.session.delete(paper)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Paper removed from bookmarks.'})
