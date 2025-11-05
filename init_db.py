# Create a simple script to initialize the database
# Save this as init_db.py and run it:

from app import app, db
with app.app_context():
    db.create_all()
    print("Database tables created successfully!")