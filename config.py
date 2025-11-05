import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # File upload configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.path.join(basedir, 'static/uploads')
    
    # ML Model paths
    ML_MODELS_DIR = os.path.join(basedir, 'ml_model/models')
    DATA_DIR = os.path.join(basedir, 'ml_model/data')
    
    # Default crops supported
    SUPPORTED_CROPS = ['wheat', 'corn', 'rice', 'soybean', 'potato']
    
    @staticmethod
    def init_app(app):
        # Create necessary directories
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)
        if not os.path.exists(Config.ML_MODELS_DIR):
            os.makedirs(Config.ML_MODELS_DIR)
        if not os.path.exists(Config.DATA_DIR):
            os.makedirs(os.path.join(Config.DATA_DIR, 'raw'))
            os.makedirs(os.path.join(Config.DATA_DIR, 'processed'))

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}