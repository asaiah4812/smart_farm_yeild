from functools import wraps
from flask import flash, redirect, url_for
from flask_login import current_user

def requires_roles(*roles):
    """Decorator to check if user has required roles"""
    def wrapper(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if current_user.role not in roles:
                flash('You do not have permission to access this page.')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return wrapper