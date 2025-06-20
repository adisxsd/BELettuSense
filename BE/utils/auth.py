import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE = 'database.db'

def register_user(username, password):
    hashed_password = generate_password_hash(password)
    try:
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        # Kemungkinan username sudah ada
        return False
    except Exception as e:
        print("Error during registration:", e)
        return False

def login_user (username, password):
    try:
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
            user = c.fetchone()
            if user and check_password_hash(user[1], password):
                return user[0]  # return user ID
            else:
                return None
    except Exception as e:
        print("Login error:", e)
        return None

def get_user_by_id(user_id):
    try:
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
            return c.fetchone()
    except Exception as e:
        print("Error getting user by id:", e)
        return None
