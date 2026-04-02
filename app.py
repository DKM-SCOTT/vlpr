"""
VLPR (Vehicle License Plate Recognition) System
A Flask-based application for detecting and recognizing Kenyan license plates
"""

import os
import sys
import cv2
import numpy as np
import uuid
import csv
import re
import torch
import gc
import tempfile
from io import StringIO
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, make_response, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import func
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Configure matplotlib for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# OpenCV configuration for headless environments
os.environ['OPENCV_OPENCL_RUNTIME'] = ''
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

# Import YOLO with fallback installation
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

from database import db, User, Plate

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')

# Database configuration
database_url = os.environ.get('DATABASE_URL', 'sqlite:///vlpr.db')
if database_url and database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# File upload configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PLATES_FOLDER'] = 'static/plates_detected'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Model paths - ordered by priority
MODEL_PATHS = [
    Path('models/best.pt'),
    Path('model/best.pt'),
    Path('/opt/render/project/src/models/best.pt'),
    Path('/app/models/best.pt'),
    Path(__file__).parent / 'models' / 'best.pt',
]

# Find the first valid model path
MODEL_PATH = None
for path in MODEL_PATHS:
    if path.exists():
        MODEL_PATH = str(path)
        print(f"✓ Found model at: {MODEL_PATH}")
        break

if MODEL_PATH is None:
    print("⚠ Warning: No model file found in any expected location")
    for path in MODEL_PATHS:
        print(f"  - {path} (exists: {path.exists()})")
    MODEL_PATH = str(MODEL_PATHS[0])  # Use default path for error messages

app.config['MODEL_PATH'] = MODEL_PATH

# ============================================================================
# INITIALIZE DIRECTORIES
# ============================================================================

def init_directories():
    """Create all necessary directories"""
    directories = [
        app.config['UPLOAD_FOLDER'],
        app.config['PLATES_FOLDER'],
        'static/css',
        'static/js',
        'templates',
        'models',
        '/tmp/static/uploads',
        '/tmp/static/plates_detected',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create symlinks for Render compatibility
    try:
        if not Path('static/uploads').exists() and Path('/tmp/static/uploads').exists():
            Path('static/uploads').symlink_to('/tmp/static/uploads')
        if not Path('static/plates_detected').exists() and Path('/tmp/static/plates_detected').exists():
            Path('static/plates_detected').symlink_to('/tmp/static/plates_detected')
    except Exception as e:
        print(f"⚠ Could not create symlinks: {e}")

init_directories()

# ============================================================================
# INITIALIZE EXTENSIONS
# ============================================================================

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Global model variables
yolo_model = None
easyocr_reader = None

# Kenyan license plate patterns
KENYAN_PLATE_PATTERNS = [
    r'^[A-Z]{3}\s?\d{3}[A-Z]$',      # KBA 123A
    r'^[A-Z]{2}\s?\d{5}$',            # KB 12345
    r'^[A-Z]{2}\s?\d{3}[A-Z]$',       # KB 123A
    r'^[A-Z]{3}\s?\d{4}$',            # KBA 1234
    r'^[A-Z]{2}\s?\d{4}[A-Z]$',       # KB 1234A
    r'^CD\s?\d{1,4}$',                # CD 1234 (Diplomatic)
]

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_yolo_model():
    """Load YOLO model for plate detection"""
    global yolo_model
    
    try:
        if yolo_model is not None:
            return True
        
        model_path = app.config['MODEL_PATH']
        print(f"\n=== Loading YOLO Model ===")
        print(f"Model path: {model_path}")
        
        if not Path(model_path).exists():
            print(f"✗ Model not found at: {model_path}")
            return False
        
        # Load model
        yolo_model = YOLO(model_path)
        
        # Force CPU usage
        if hasattr(yolo_model, 'model'):
            yolo_model.model.to('cpu')
        
        print(f"✓ YOLO model loaded successfully")
        print(f"  Device: CPU")
        print(f"  Model size: {Path(model_path).stat().st_size / 1024 / 1024:.2f} MB")
        return True
        
    except Exception as e:
        print(f"✗ Error loading YOLO model: {e}")
        import traceback
        traceback.print_exc()
        yolo_model = None
        return False

def load_easyocr():
    """Load EasyOCR for text extraction"""
    global easyocr_reader
    
    try:
        if easyocr_reader is not None:
            return True
        
        print("\n=== Loading EasyOCR ===")
        
        # Install EasyOCR if not available
        try:
            import easyocr
        except ImportError:
            print("Installing easyocr...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr"])
            import easyocr
        
        # Initialize reader
        easyocr_reader = easyocr.Reader(
            ['en'],
            gpu=False,
            model_storage_directory='/tmp/easyocr',
            download_enabled=True
        )
        
        print("✓ EasyOCR loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error loading EasyOCR: {e}")
        easyocr_reader = None
        return False

def ensure_models_loaded():
    """Ensure both models are loaded, return True if both are available"""
    yolo_ok = load_yolo_model()
    ocr_ok = load_easyocr()
    return yolo_ok and ocr_ok

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_kenyan_plate_text(text):
    """Clean and format Kenyan license plate text"""
    if not text:
        return "UNKNOWN"
    
    # Remove special characters and convert to uppercase
    text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
    
    # Format based on length
    if len(text) == 7:
        return f"{text[:3]} {text[3:6]} {text[6]}"
    elif len(text) == 6:
        if text[:3].isalpha() and text[3:].isdigit():
            return f"{text[:3]} {text[3:]}"
        elif text[:2].isalpha() and text[2:].isdigit():
            return f"{text[:2]} {text[2:]}"
        return text
    elif len(text) == 5:
        if text[0].isalpha() and text[1:4].isdigit() and text[4].isalpha():
            return f"{text[0]} {text[1:4]} {text[4]}"
        elif text[:3].isalpha() and text[3:].isdigit():
            return f"{text[:3]} {text[3:]}"
        return text
    
    return text

def is_kenyan_plate(plate_number):
    """Check if a plate number matches Kenyan patterns"""
    clean_plate = plate_number.replace(' ', '')
    for pattern in KENYAN_PLATE_PATTERNS:
        if re.match(pattern, clean_plate):
            return True
    return False

def save_uploaded_file(file):
    """Save uploaded file and return the saved path"""
    filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    filepath = Path(app.config['UPLOAD_FOLDER']) / filename
    file.save(str(filepath))
    return str(filepath), filename

def cleanup_file(filepath):
    """Safely delete a file"""
    try:
        if filepath and Path(filepath).exists():
            Path(filepath).unlink()
    except Exception as e:
        print(f"Error cleaning up file {filepath}: {e}")

def detect_plate_yolo(image_path, filename):
    """Detect license plate using YOLO and extract text with EasyOCR"""
    
    # Ensure models are loaded
    if not ensure_models_loaded():
        return {'success': False, 'error': 'Models not loaded properly'}
    
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return {'success': False, 'error': 'Could not read image'}
        
        height, width = img.shape[:2]
        
        # Run YOLO inference
        print(f"Running YOLO inference...")
        results = yolo_model(image_path, verbose=False)
        
        # Check for detections
        if len(results) == 0 or len(results[0].boxes) == 0:
            return {'success': False, 'error': 'No license plate detected'}
        
        # Get best detection
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        confidence = float(confidences[best_idx])
        
        x1, y1, x2, y2 = box
        print(f"Plate detected with confidence: {confidence:.2f}")
        
        # Add padding
        padding_x = int((x2 - x1) * 0.1)
        padding_y = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(width, x2 + padding_x)
        y2 = min(height, y2 + padding_y)
        
        # Extract plate region
        plate_img = img[y1:y2, x1:x2].copy()
        
        # Save plate image
        plate_filename = f"plate_{filename}"
        plate_path = Path(app.config['PLATES_FOLDER']) / plate_filename
        cv2.imwrite(str(plate_path), plate_img)
        
        # Save annotated image
        display_filename = f"display_{filename}"
        display_path = Path(app.config['UPLOAD_FOLDER']) / display_filename
        cv2.imwrite(str(display_path), img)
        
        # Draw rectangle on image
        img_with_rect = img.copy()
        cv2.rectangle(img_with_rect, (x1, y1), (x2, y2), (0, 255, 0), 3)
        rect_filename = f"rect_{filename}"
        rect_path = Path(app.config['UPLOAD_FOLDER']) / rect_filename
        cv2.imwrite(str(rect_path), img_with_rect)
        
        # OCR Processing
        plate_text = "UNKNOWN"
        ocr_confidence = 0.0
        
        try:
            # Convert to grayscale
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img
            
            # Resize if too small
            h, w = gray.shape
            if w < 200:
                scale = 400 / max(w, 1)
                new_w = int(w * scale)
                new_h = int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Run OCR
            ocr_results = easyocr_reader.readtext(
                gray,
                paragraph=False,
                text_threshold=0.7,
                low_text=0.4,
                width_ths=0.7,
                height_ths=0.5,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            
            # Process OCR results
            if ocr_results:
                all_text = []
                all_conf = []
                for result in ocr_results:
                    if len(result) >= 3:
                        all_text.append(result[1])
                        all_conf.append(result[2])
                
                if all_text:
                    plate_text = " ".join(all_text)
                    ocr_confidence = sum(all_conf) / len(all_conf)
                    plate_text = clean_kenyan_plate_text(plate_text)
            
        except Exception as e:
            print(f"OCR error: {e}")
            plate_text = "OCR_ERROR"
        
        # Calculate final confidence
        if ocr_confidence > 0:
            final_confidence = confidence * 0.4 + ocr_confidence * 0.6
        else:
            final_confidence = confidence * 0.5
        
        # Add text to annotated image
        img_with_text = cv2.imread(str(rect_path))
        if img_with_text is not None:
            cv2.putText(img_with_text, f"Plate: {plate_text}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img_with_text, f"Conf: {final_confidence*100:.1f}%", (x1, y1-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(str(rect_path), img_with_text)
        
        # Cleanup
        del img, plate_img, img_with_rect
        gc.collect()
        
        return {
            'success': True,
            'original_image': f'/uploads/display_{filename}',
            'detected_image': f'/uploads/{rect_filename}',
            'plate_image': f'/plates_detected/{plate_filename}',
            'plate_text': plate_text,
            'confidence': float(final_confidence),
            'yolo_confidence': float(confidence),
            'ocr_confidence': float(ocr_confidence),
            'is_kenyan_plate': is_kenyan_plate(plate_text)
        }
        
    except Exception as e:
        print(f"Error in detect_plate_yolo: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()
        return {'success': False, 'error': str(e)}

# ============================================================================
# FLASK ROUTES
# ============================================================================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.context_processor
def utility_processor():
    return {'now': datetime.now}

# Static file routes
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/plates_detected/<filename>')
def plates_detected_file(filename):
    return send_from_directory(app.config['PLATES_FOLDER'], filename)

# Health check endpoints
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': str(datetime.now()),
        'model_loaded': yolo_model is not None,
        'ocr_loaded': easyocr_reader is not None,
        'model_path_exists': Path(app.config['MODEL_PATH']).exists()
    })

@app.route('/debug')
def debug():
    """Debug endpoint for troubleshooting"""
    import sys
    import shutil
    
    models_content = []
    if Path('models').exists():
        models_content = [str(f) for f in Path('models').iterdir()]
    
    return jsonify({
        'status': 'running',
        'working_directory': str(Path.cwd()),
        'python_version': sys.version,
        'model': {
            'path': app.config['MODEL_PATH'],
            'exists': Path(app.config['MODEL_PATH']).exists(),
            'loaded': yolo_model is not None,
            'models_folder_content': models_content
        },
        'ocr': {
            'loaded': easyocr_reader is not None
        },
        'folders': {
            'uploads_exists': Path(app.config['UPLOAD_FOLDER']).exists(),
            'plates_exists': Path(app.config['PLATES_FOLDER']).exists()
        }
    })

@app.route('/model_status')
@login_required
def model_status():
    return jsonify({
        'yolo_loaded': yolo_model is not None,
        'ocr_loaded': easyocr_reader is not None,
        'model_path': app.config['MODEL_PATH'],
        'model_exists': Path(app.config['MODEL_PATH']).exists(),
        'models_folder_exists': Path('models').exists(),
        'models_content': [str(f) for f in Path('models').iterdir()] if Path('models').exists() else [],
        'working_directory': str(Path.cwd())
    })

# Authentication routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password)
        )
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred. Please try again.', 'danger')
            print(f"Registration error: {e}")
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# Main application routes
@app.route('/dashboard')
@login_required
def dashboard():
    plates = Plate.query.filter_by(user_id=current_user.id).order_by(Plate.detected_at.desc()).all()
    
    today = datetime.now().date()
    today_count = sum(1 for p in plates if p.detected_at.date() == today)
    avg_confidence = sum(p.confidence for p in plates) / len(plates) if plates else 0
    kenyan_count = sum(1 for p in plates if is_kenyan_plate(p.plate_number))
    
    return render_template('dashboard.html',
                         plates=plates,
                         today_count=today_count,
                         avg_confidence=avg_confidence,
                         kenyan_plates=kenyan_count)

@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                flash('No image uploaded', 'danger')
                return redirect(request.url)
            
            file = request.files['image']
            if file.filename == '':
                flash('No image selected', 'danger')
                return redirect(request.url)
            
            if file:
                # Save uploaded file
                filepath, filename = save_uploaded_file(file)
                
                # Run detection
                result = detect_plate_yolo(filepath, filename)
                
                # Cleanup original file
                cleanup_file(filepath)
                
                if result['success']:
                    # Save to database
                    plate = Plate(
                        plate_number=result['plate_text'],
                        image_path=result['original_image'],
                        plate_image_path=result['plate_image'],
                        confidence=result['confidence'],
                        user_id=current_user.id
                    )
                    db.session.add(plate)
                    db.session.commit()
                    
                    return render_template('detect.html', result=result, success=True)
                else:
                    flash(f'Detection failed: {result.get("error", "No license plate detected")}', 'warning')
                    return render_template('detect.html', error=True)
                    
        except Exception as e:
            print(f"Error in detect route: {e}")
            import traceback
            traceback.print_exc()
            flash(f'An error occurred: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template('detect.html')

@app.route('/plate/<int:plate_id>')
@login_required
def plate_detail(plate_id):
    plate = Plate.query.get_or_404(plate_id)
    if plate.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    return render_template('plate_detail.html',
                         plate=plate,
                         is_kenyan=is_kenyan_plate(plate.plate_number))

@app.route('/delete_plate/<int:plate_id>', methods=['POST'])
@login_required
def delete_plate(plate_id):
    plate = Plate.query.get_or_404(plate_id)
    if plate.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Access denied'})
    
    # Delete associated files
    for path in [plate.image_path, plate.plate_image_path]:
        if path:
            filename = Path(path).name
            folder = app.config['UPLOAD_FOLDER'] if 'uploads' in path else app.config['PLATES_FOLDER']
            filepath = Path(folder) / filename
            cleanup_file(str(filepath))
    
    db.session.delete(plate)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Plate deleted successfully'})

@app.route('/profile')
@login_required
def profile():
    plates = Plate.query.filter_by(user_id=current_user.id).all()
    
    now = datetime.now()
    month_start = datetime(now.year, now.month, 1)
    week_start = now - timedelta(days=now.weekday())
    
    kenyan_count = sum(1 for p in plates if is_kenyan_plate(p.plate_number))
    
    stats = {
        'total_plates': len(plates),
        'month_plates': sum(1 for p in plates if p.detected_at >= month_start),
        'week_plates': sum(1 for p in plates if p.detected_at >= week_start),
        'avg_confidence': sum(p.confidence for p in plates) / len(plates) if plates else 0,
        'kenyan_plates': kenyan_count,
        'foreign_plates': len(plates) - kenyan_count
    }
    
    recent_plates = Plate.query.filter_by(user_id=current_user.id)\
                               .order_by(Plate.detected_at.desc())\
                               .limit(5).all()
    
    return render_template('profile.html', stats=stats, recent_plates=recent_plates)

@app.route('/search')
@login_required
def search():
    query = request.args.get('query', '')
    date_filter = request.args.get('date_filter', 'all')
    confidence_filter = request.args.get('confidence', 'all')
    plate_type = request.args.get('plate_type', 'all')
    
    plates_query = Plate.query.filter_by(user_id=current_user.id)
    
    if query:
        plates_query = plates_query.filter(Plate.plate_number.contains(query.upper()))
    
    now = datetime.now()
    if date_filter == 'today':
        plates_query = plates_query.filter(func.date(Plate.detected_at) == now.date())
    elif date_filter == 'week':
        plates_query = plates_query.filter(Plate.detected_at >= now - timedelta(days=now.weekday()))
    elif date_filter == 'month':
        plates_query = plates_query.filter(Plate.detected_at >= datetime(now.year, now.month, 1))
    elif date_filter == 'year':
        plates_query = plates_query.filter(Plate.detected_at >= datetime(now.year, 1, 1))
    
    if confidence_filter == '90':
        plates_query = plates_query.filter(Plate.confidence >= 0.9)
    elif confidence_filter == '80':
        plates_query = plates_query.filter(Plate.confidence >= 0.8)
    elif confidence_filter == '70':
        plates_query = plates_query.filter(Plate.confidence >= 0.7)
    
    plates = plates_query.order_by(Plate.detected_at.desc()).all()
    
    if plate_type != 'all':
        plates = [p for p in plates if is_kenyan_plate(p.plate_number) == (plate_type == 'kenyan')]
    
    return render_template('search.html', plates=plates, query=query, plate_type=plate_type)

@app.route('/export_data')
@login_required
def export_data():
    plates = Plate.query.filter_by(user_id=current_user.id).order_by(Plate.detected_at.desc()).all()
    
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Plate Number', 'Detection Date', 'Confidence', 'Is Kenyan Plate', 'Image Path'])
    
    for plate in plates:
        cw.writerow([
            plate.plate_number,
            plate.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
            f"{plate.confidence * 100:.1f}%",
            'Yes' if is_kenyan_plate(plate.plate_number) else 'No',
            plate.image_path
        ])
    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=vlpr_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output.headers["Content-type"] = "text/csv"
    
    return output

@app.route('/analytics')
@login_required
def analytics():
    plates = Plate.query.filter_by(user_id=current_user.id).order_by(Plate.detected_at).all()
    
    dates = []
    counts = []
    kenyan_counts = []
    foreign_counts = []
    
    if plates:
        date_data = {}
        for plate in plates:
            date_str = plate.detected_at.strftime('%Y-%m-%d')
            if date_str not in date_data:
                date_data[date_str] = {'total': 0, 'kenyan': 0, 'foreign': 0}
            
            date_data[date_str]['total'] += 1
            if is_kenyan_plate(plate.plate_number):
                date_data[date_str]['kenyan'] += 1
            else:
                date_data[date_str]['foreign'] += 1
        
        dates = list(date_data.keys())
        counts = [date_data[d]['total'] for d in dates]
        kenyan_counts = [date_data[d]['kenyan'] for d in dates]
        foreign_counts = [date_data[d]['foreign'] for d in dates]
    
    total_plates = len(plates)
    kenyan_plates = sum(1 for p in plates if is_kenyan_plate(p.plate_number))
    foreign_plates = total_plates - kenyan_plates
    
    return render_template('analytics.html',
                         dates=dates,
                         counts=counts,
                         kenyan_counts=kenyan_counts,
                         foreign_counts=foreign_counts,
                         total_plates=total_plates,
                         kenyan_plates=kenyan_plates,
                         foreign_plates=foreign_plates)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    
    if username != current_user.username and User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already exists'})
    
    if email != current_user.email and User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already exists'})
    
    current_user.username = username
    current_user.email = email
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Profile updated successfully'})

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    data = request.get_json()
    
    if not check_password_hash(current_user.password, data.get('current_password')):
        return jsonify({'success': False, 'message': 'Current password is incorrect'})
    
    new_password = data.get('new_password')
    if len(new_password) < 6:
        return jsonify({'success': False, 'message': 'New password must be at least 6 characters'})
    
    current_user.password = generate_password_hash(new_password)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Password changed successfully'})

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("\n" + "="*50)
        print("VLPR SYSTEM INITIALIZED")
        print("="*50)
        print(f"Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
        print(f"Model path: {app.config['MODEL_PATH']}")
        print(f"Model exists: {Path(app.config['MODEL_PATH']).exists()}")
        print("="*50 + "\n")
        
        # Attempt to pre-load models (optional, will retry on demand if fails)
        print("Pre-loading models...")
        load_yolo_model()
        load_easyocr()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)