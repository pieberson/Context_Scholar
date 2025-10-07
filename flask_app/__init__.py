from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import pandas as pd
from collections import defaultdict

db = SQLAlchemy()
bcrypt = Bcrypt()

# --- Load CSV Files ---
print("Loading corpus, queries, and qrels...")

corpus_df = pd.read_csv("corpus_complete.csv")
queries_df = pd.read_csv("queries.csv")
qrels_df = pd.read_csv("qrels.csv")

corpus = {
    row["doc_id"]: {
        "title": row["title"],
        "text": row["text"],
        "citations": int(row.get("citations", 0))
    }
    for _, row in corpus_df.iterrows()
}

queries = {
    str(row["query_id"]): row["text"]
    for _, row in queries_df.iterrows()
}

qrels = defaultdict(dict)
for _, row in qrels_df.iterrows():
    qrels[str(row["query_id"])][row["doc_id"]] = int(row["score"])

def create_app():
    app = Flask(__name__, 
    template_folder='../templates',
    static_folder='../static')

    # Database config
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/flask_users'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'your_secret_key_here'

    db.init_app(app)
    bcrypt.init_app(app)

    # Import routes (after app + db are initialized)
    from flask_app import routes
    app.register_blueprint(routes.bp)

    return app
