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
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, make_response, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import func
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# CRITICAL FIX 1: Set matplotlib to use non-interactive backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'  # Use temp directory for cache

# CRITICAL FIX 2: Set environment variables for OpenCV and other libraries
os.environ['OPENCV_OPENCL_RUNTIME'] = ''
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

from database import db, User, Plate

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')

# Use PostgreSQL if available (for Render), otherwise SQLite
database_url = os.environ.get('DATABASE_URL', 'sqlite:///vlpr.db')
if database_url and database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'  # Use /tmp for Render
app.config['PLATES_FOLDER'] = '/tmp/plates_detected'  # Use /tmp for Render
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# Check multiple possible model paths
possible_paths = [
    os.path.join('models', 'best.pt'),
    os.path.join('model', 'best.pt'),
    '/opt/render/project/src/models/best.pt',
]

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

app.config['MODEL_PATH'] = model_path or os.path.join('models', 'best.pt')

# Create directories (using /tmp for Render)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLATES_FOLDER'], exist_ok=True)
os.makedirs('/tmp/static/uploads', exist_ok=True)
os.makedirs('/tmp/static/plates_detected', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Create symbolic links for static files if needed
if not os.path.exists('static/uploads'):
    os.symlink('/tmp/static/uploads', 'static/uploads')
if not os.path.exists('static/plates_detected'):
    os.symlink('/tmp/static/plates_detected', 'static/plates_detected')

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Initialize model variables as None - will load on demand
yolo_model = None
easyocr_reader = None

KENYAN_PLATE_PATTERNS = [
    r'^[A-Z]{3}\s?\d{3}[A-Z]$',      
    r'^[A-Z]{2}\s?\d{5}$',            
    r'^[A-Z]{2}\s?\d{3}[A-Z]$',       
    r'^[A-Z]{3}\s?\d{4}$',            
    r'^[A-Z]{2}\s?\d{4}[A-Z]$',       
    r'^CD\s?\d{1,4}$',                
]

def load_yolo_model():
    """Load YOLOv8 model for plate detection using ultralytics"""
    global yolo_model
    try:
        # Clean up existing model if any
        if yolo_model is not None:
            del yolo_model
            gc.collect()
        
        model_path = app.config['MODEL_PATH']
        print(f"Looking for YOLO model at: {model_path}")
        
        if os.path.exists(model_path):
            print(f"✅ Model file found! Loading YOLO model...")
            
            # Load model with minimal memory footprint
            yolo_model = YOLO(model_path)
            
            device = 'cpu'  # Force CPU to save memory
            print(f"Using device: {device}")
            
            print(f"✅ YOLO model loaded successfully from {model_path}")
            return True
        else:
            print(f"❌ Model file NOT FOUND at {model_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in models folder: {os.listdir('models') if os.path.exists('models') else 'models folder not found'}")
            return False
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_easyocr():
    """Load EasyOCR for text extraction"""
    global easyocr_reader
    try:
        # Clean up existing reader if any
        if easyocr_reader is not None:
            del easyocr_reader
            gc.collect()
        
        import easyocr
        print("Loading EasyOCR reader for Kenyan plates...")
        # Use /tmp for model storage to avoid filling up disk
        easyocr_reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='/tmp/easyocr')
        print("✅ EasyOCR loaded successfully!")
        return True
    except ImportError:
        print("⚠️ EasyOCR not installed. Installing now...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr"])
            import easyocr
            easyocr_reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='/tmp/easyocr')
            print("✅ EasyOCR installed and loaded successfully!")
            return True
        except Exception as e:
            print(f"⚠️ EasyOCR could not be installed: {e}")
            return False
    except Exception as e:
        print(f"⚠️ EasyOCR could not be loaded: {e}")
        return False

# Don't load models at startup - they'll load on first request
print("\n" + "="*50)
print("STARTING VLPR SYSTEM (Optimized Mode)")
print("="*50)
print("Models will load on first detection request to save memory")
print("="*50 + "\n")

# FIX 3: Add a simple health check endpoint that doesn't load any models
@app.route('/health')
def health():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'time': str(datetime.now()),
        'model_loaded': yolo_model is not None,
        'ocr_loaded': easyocr_reader is not None
    })

@app.before_request
def before_request():
    """Check if models need to be loaded before certain requests"""
    global yolo_model, easyocr_reader
    
    # Only load models for detection endpoints
    if request.endpoint == 'detect' and request.method == 'POST':
        if yolo_model is None:
            print("Loading YOLO model on-demand...")
            load_yolo_model()
        
        if easyocr_reader is None:
            print("Loading EasyOCR on-demand...")
            load_easyocr()

@app.context_processor
def utility_processor():
    return {'now': datetime.now}

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/plates_detected/<filename>')
def plates_detected_file(filename):
    return send_from_directory(app.config['PLATES_FOLDER'], filename)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
        
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        email_exists = User.query.filter_by(email=email).first()
        if email_exists:
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        
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

@app.route('/dashboard')
@login_required
def dashboard():
    plates = Plate.query.filter_by(user_id=current_user.id).order_by(Plate.detected_at.desc()).all()
    
    today = datetime.now().date()
    today_count = sum(1 for plate in plates if plate.detected_at.date() == today)
    avg_confidence = sum(p.confidence for p in plates) / len(plates) if plates else 0
    
    kenyan_plates = 0
    for plate in plates:
        clean_plate = plate.plate_number.replace(' ', '')
        for pattern in KENYAN_PLATE_PATTERNS:
            if re.match(pattern, clean_plate):
                kenyan_plates += 1
                break
    
    return render_template('dashboard.html', 
                         plates=plates, 
                         today_count=today_count, 
                         avg_confidence=avg_confidence,
                         kenyan_plates=kenyan_plates)

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
                filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                print(f"✅ File saved to: {filepath}")
                
                # Check if models are loaded
                if yolo_model is None:
                    print("Loading YOLO model on-demand...")
                    if not load_yolo_model():
                        print("❌ Failed to load YOLO model")
                        flash('Error loading detection model', 'danger')
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        return redirect(request.url)
                
                if easyocr_reader is None:
                    print("Loading EasyOCR on-demand...")
                    if not load_easyocr():
                        print("❌ Failed to load EasyOCR")
                        flash('Error loading OCR model', 'danger')
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        return redirect(request.url)
                
                print("Starting plate detection...")
                result = detect_plate_yolo(filepath, filename)
                print(f"Detection result: {result.get('success', False)}")
                
                if result['success']:
                    print(f"Plate detected: {result['plate_text']}")
                    
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
                    print("✅ Plate saved to database")
                    
                    # Clean up uploaded file after successful processing
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        print(f"🗑️ Cleaned up: {filepath}")
                    
                    return render_template('detect.html', result=result, success=True)
                else:
                    print(f"❌ Detection failed: {result.get('error', 'Unknown error')}")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    flash(f'Detection failed: {result.get("error", "No license plate detected")}', 'warning')
                    return render_template('detect.html', error=True)
        except Exception as e:
            print(f"❌ Unhandled exception in detect route: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(f'An error occurred: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template('detect.html')

def clean_kenyan_plate_text(text):
    """Clean and format Kenyan license plate text"""
    if not text:
        return "UNKNOWN"
    
    text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
    print(f"Cleaning plate text: '{text}'")
    
    if len(text) == 7:
        formatted = f"{text[:3]} {text[3:6]} {text[6]}"
        print(f"Formatted as 7-char: '{formatted}'")
        return formatted
    elif len(text) == 6:
        if text[:3].isalpha() and text[3:].isdigit():
            formatted = f"{text[:3]} {text[3:]}"
        elif text[:2].isalpha() and text[2:].isdigit():
            formatted = f"{text[:2]} {text[2:]}"
        else:
            formatted = text
        print(f"Formatted as 6-char: '{formatted}'")
        return formatted
    elif len(text) == 5:
        if text[0].isalpha() and text[1:4].isdigit() and text[4].isalpha():
            formatted = f"{text[0]} {text[1:4]} {text[4]}"
        elif text[:3].isalpha() and text[3:].isdigit():
            formatted = f"{text[:3]} {text[3:]}"
        else:
            formatted = text
        print(f"Formatted as 5-char: '{formatted}'")
        return formatted
    else:
        print(f"No formatting applied: '{text}'")
        return text

def detect_plate_yolo(image_path, filename):
    """Detect license plate using YOLO model and extract text with EasyOCR"""
    global yolo_model, easyocr_reader
    
    # Ensure models are loaded
    if yolo_model is None:
        if not load_yolo_model():
            return {'success': False, 'error': 'YOLO model not loaded'}
    
    if easyocr_reader is None:
        if not load_easyocr():
            return {'success': False, 'error': 'OCR not loaded'}
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {'success': False, 'error': 'Could not read image'}
        
        height, width = img.shape[:2]
        
        print(f"Running YOLO inference on {image_path}...")
        results = yolo_model(image_path, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            print("No detections from YOLO")
            del img
            gc.collect()
            return {'success': False, 'error': 'No license plate detected'}
        
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        
        best_idx = np.argmax(confidences)
        box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        confidence = float(confidences[best_idx])
        
        x1, y1, x2, y2 = box
        
        print(f"Plate detected with confidence: {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
        
        padding_x = int((x2 - x1) * 0.1)
        padding_y = int((y2 - y1) * 0.1)
        
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(width, x2 + padding_x)
        y2 = min(height, y2 + padding_y)
        
        plate_img = img[y1:y2, x1:x2].copy()
        
        plate_filename = f"plate_{filename}"
        plate_path = os.path.join(app.config['PLATES_FOLDER'], plate_filename)
        cv2.imwrite(plate_path, plate_img)
        
        # Create a copy of original image for display
        original_display_path = os.path.join(app.config['UPLOAD_FOLDER'], f"display_{filename}")
        cv2.imwrite(original_display_path, img)
        
        # Create detected image with rectangle
        img_with_rect = img.copy()
        cv2.rectangle(img_with_rect, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        rect_filename = f"rect_{filename}"
        rect_path = os.path.join(app.config['UPLOAD_FOLDER'], rect_filename)
        cv2.imwrite(rect_path, img_with_rect)
        
        # OCR processing
        plate_text = "UNKNOWN"
        ocr_confidence = 0.0
        
        if easyocr_reader is not None:
            try:
                print("Starting OCR processing...")
                
                if len(plate_img.shape) == 3:
                    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = plate_img
                
                # Only resize if needed
                h, w = gray.shape
                if w < 200:
                    scale = 400 / max(w, 1)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                print("Running EasyOCR...")
                
                ocr_results = easyocr_reader.readtext(
                    gray,
                    paragraph=False,
                    text_threshold=0.7,
                    low_text=0.4,
                    width_ths=0.7,
                    height_ths=0.5,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
                
                print(f"OCR returned {len(ocr_results)} results")
                
                if ocr_results and len(ocr_results) > 0:
                    all_text = []
                    all_conf = []
                    
                    for result in ocr_results:
                        if len(result) >= 3:
                            text = result[1]
                            conf = result[2]
                            all_text.append(text)
                            all_conf.append(conf)
                            print(f"  Detected: '{text}' (conf: {conf:.2f})")
                    
                    if all_text:
                        plate_text = " ".join(all_text)
                        ocr_confidence = sum(all_conf) / len(all_conf) if all_conf else 0.7
                        print(f"📝 Combined text: '{plate_text}'")
                        
                        plate_text_before = plate_text
                        plate_text = clean_kenyan_plate_text(plate_text)
                        
                        print(f"✨ Before: '{plate_text_before}'")
                        print(f"✅ After: '{plate_text}'")
                    else:
                        print("❌ Could not parse OCR results")
                        plate_text = "PARSE_ERROR"
                        ocr_confidence = 0.0
                else:
                    print("❌ No text detected")
                    plate_text = "NO_TEXT"
                    ocr_confidence = 0.0
                    
            except Exception as e:
                print(f"❌ OCR error: {type(e).__name__}: {e}")
                plate_text = "OCR_ERROR"
                ocr_confidence = 0.0
        else:
            print("❌ EasyOCR not loaded")
            plate_text = "OCR_NOT_AVAILABLE"
            ocr_confidence = 0.0
        
        # Calculate final confidence
        if ocr_confidence > 0:
            final_confidence = confidence * 0.4 + ocr_confidence * 0.6
        else:
            final_confidence = confidence * 0.5
        
        # Update the detected image with the actual plate text
        img_with_rect2 = cv2.imread(rect_path)
        if img_with_rect2 is not None:
            cv2.putText(img_with_rect2, f"Plate: {plate_text}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img_with_rect2, f"Conf: {final_confidence*100:.1f}%", (x1, y1-35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(rect_path, img_with_rect2)
            del img_with_rect2
        
        # Check if Kenyan plate
        is_kenyan = False
        clean_plate = plate_text.replace(' ', '')
        for pattern in KENYAN_PLATE_PATTERNS:
            if re.match(pattern, clean_plate):
                is_kenyan = True
                break
        
        # Clean up memory
        del img
        del plate_img
        if 'gray' in locals():
            del gray
        if 'img_with_rect' in locals():
            del img_with_rect
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
            'is_kenyan_plate': is_kenyan
        }
        
    except Exception as e:
        print(f"Error in detect_plate_yolo: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()
        return {'success': False, 'error': str(e)}

@app.route('/plate/<int:plate_id>')
@login_required
def plate_detail(plate_id):
    plate = Plate.query.get_or_404(plate_id)
    if plate.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    is_kenyan = False
    clean_plate = plate.plate_number.replace(' ', '')
    for pattern in KENYAN_PLATE_PATTERNS:
        if re.match(pattern, clean_plate):
            is_kenyan = True
            break
    
    return render_template('plate_detail.html', plate=plate, is_kenyan=is_kenyan)

@app.route('/delete_plate/<int:plate_id>', methods=['POST'])
@login_required
def delete_plate(plate_id):
    plate = Plate.query.get_or_404(plate_id)
    if plate.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Access denied'})
    
    try:
        if plate.image_path:
            img_filename = os.path.basename(plate.image_path)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            if os.path.exists(img_path):
                os.remove(img_path)
        
        if plate.plate_image_path:
            plate_filename = os.path.basename(plate.plate_image_path)
            plate_path = os.path.join(app.config['PLATES_FOLDER'], plate_filename)
            if os.path.exists(plate_path):
                os.remove(plate_path)
    except Exception as e:
        print(f"Error deleting files: {e}")
    
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
    
    kenyan_plates = 0
    for plate in plates:
        clean_plate = plate.plate_number.replace(' ', '')
        for pattern in KENYAN_PLATE_PATTERNS:
            if re.match(pattern, clean_plate):
                kenyan_plates += 1
                break
    
    stats = {
        'total_plates': len(plates),
        'month_plates': sum(1 for p in plates if p.detected_at >= month_start),
        'week_plates': sum(1 for p in plates if p.detected_at >= week_start),
        'avg_confidence': sum(p.confidence for p in plates) / len(plates) if plates else 0,
        'kenyan_plates': kenyan_plates,
        'foreign_plates': len(plates) - kenyan_plates
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
        week_start = now - timedelta(days=now.weekday())
        plates_query = plates_query.filter(Plate.detected_at >= week_start)
    elif date_filter == 'month':
        month_start = datetime(now.year, now.month, 1)
        plates_query = plates_query.filter(Plate.detected_at >= month_start)
    elif date_filter == 'year':
        year_start = datetime(now.year, 1, 1)
        plates_query = plates_query.filter(Plate.detected_at >= year_start)
    
    if confidence_filter == '90':
        plates_query = plates_query.filter(Plate.confidence >= 0.9)
    elif confidence_filter == '80':
        plates_query = plates_query.filter(Plate.confidence >= 0.8)
    elif confidence_filter == '70':
        plates_query = plates_query.filter(Plate.confidence >= 0.7)
    
    plates = plates_query.order_by(Plate.detected_at.desc()).all()
    
    if plate_type != 'all':
        filtered_plates = []
        for plate in plates:
            clean_plate = plate.plate_number.replace(' ', '')
            is_kenyan = False
            for pattern in KENYAN_PLATE_PATTERNS:
                if re.match(pattern, clean_plate):
                    is_kenyan = True
                    break
            
            if (plate_type == 'kenyan' and is_kenyan) or (plate_type == 'foreign' and not is_kenyan):
                filtered_plates.append(plate)
        plates = filtered_plates
    
    return render_template('search.html', plates=plates, query=query, plate_type=plate_type)

@app.route('/export_data')
@login_required
def export_data():
    plates = Plate.query.filter_by(user_id=current_user.id).order_by(Plate.detected_at.desc()).all()
    
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Plate Number', 'Detection Date', 'Confidence', 'Is Kenyan Plate', 'Image Path'])
    
    for plate in plates:
        is_kenyan = False
        clean_plate = plate.plate_number.replace(' ', '')
        for pattern in KENYAN_PLATE_PATTERNS:
            if re.match(pattern, clean_plate):
                is_kenyan = True
                break
        
        cw.writerow([
            plate.plate_number,
            plate.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
            f"{plate.confidence * 100:.1f}%",
            'Yes' if is_kenyan else 'No',
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
    confidences = []
    kenyan_counts = []
    foreign_counts = []
    
    if plates:
        date_counts = {}
        date_kenyan = {}
        date_foreign = {}
        
        for plate in plates:
            date_str = plate.detected_at.strftime('%Y-%m-%d')
            date_counts[date_str] = date_counts.get(date_str, 0) + 1
            
            is_kenyan = False
            clean_plate = plate.plate_number.replace(' ', '')
            for pattern in KENYAN_PLATE_PATTERNS:
                if re.match(pattern, clean_plate):
                    is_kenyan = True
                    break
            
            if is_kenyan:
                date_kenyan[date_str] = date_kenyan.get(date_str, 0) + 1
            else:
                date_foreign[date_str] = date_foreign.get(date_str, 0) + 1
        
        dates = list(date_counts.keys())
        counts = list(date_counts.values())
        kenyan_counts = [date_kenyan.get(d, 0) for d in dates]
        foreign_counts = [date_foreign.get(d, 0) for d in dates]
        confidences = [p.confidence for p in plates[-10:]]
    
    total_plates = len(plates)
    kenyan_plates = 0
    for plate in plates:
        clean_plate = plate.plate_number.replace(' ', '')
        for pattern in KENYAN_PLATE_PATTERNS:
            if re.match(pattern, clean_plate):
                kenyan_plates += 1
                break
    foreign_plates = total_plates - kenyan_plates
    
    return render_template('analytics.html', 
                         dates=dates, 
                         counts=counts, 
                         confidences=confidences,
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
    
    if username != current_user.username:
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return jsonify({'success': False, 'message': 'Username already exists'})
    
    if email != current_user.email:
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            return jsonify({'success': False, 'message': 'Email already exists'})
    
    current_user.username = username
    current_user.email = email
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Profile updated successfully'})

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    data = request.get_json()
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    
    if not check_password_hash(current_user.password, current_password):
        return jsonify({'success': False, 'message': 'Current password is incorrect'})
    
    if len(new_password) < 6:
        return jsonify({'success': False, 'message': 'New password must be at least 6 characters'})
    
    current_user.password = generate_password_hash(new_password)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Password changed successfully'})

@app.route('/model_status')
@login_required
def model_status():
    model_exists = os.path.exists(app.config['MODEL_PATH'])
    models_folder_exists = os.path.exists('models')
    models_content = os.listdir('models') if models_folder_exists else []
    
    return jsonify({
        'yolo_loaded': yolo_model is not None,
        'ocr_loaded': easyocr_reader is not None,
        'model_path': app.config['MODEL_PATH'],
        'model_exists': model_exists,
        'models_folder_exists': models_folder_exists,
        'models_content': models_content,
        'working_directory': os.getcwd()
    })

@app.route('/debug')
def debug():
    """Debug endpoint to check system status"""
    import sys
    import shutil
    
    # Check model file
    model_exists = os.path.exists(app.config['MODEL_PATH'])
    models_content = []
    if os.path.exists('models'):
        models_content = os.listdir('models')
    
    # Check upload folders
    uploads_writable = os.access(app.config['UPLOAD_FOLDER'], os.W_OK) if os.path.exists(app.config['UPLOAD_FOLDER']) else False
    plates_writable = os.access(app.config['PLATES_FOLDER'], os.W_OK) if os.path.exists(app.config['PLATES_FOLDER']) else False
    
    # Get disk space info
    try:
        disk_usage = shutil.disk_usage('/')
        free_space_mb = disk_usage.free / (1024 * 1024)
        total_space_mb = disk_usage.total / (1024 * 1024)
    except:
        free_space_mb = 'Unknown'
        total_space_mb = 'Unknown'
    
    status = {
        'status': 'running',
        'model': {
            'path': app.config['MODEL_PATH'],
            'exists': model_exists,
            'models_folder_content': models_content,
            'yolo_loaded': yolo_model is not None,
            'ocr_loaded': easyocr_reader is not None
        },
        'folders': {
            'uploads_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
            'uploads_writable': uploads_writable,
            'plates_exists': os.path.exists(app.config['PLATES_FOLDER']),
            'plates_writable': plates_writable,
            'uploads_path': app.config['UPLOAD_FOLDER'],
            'plates_path': app.config['PLATES_FOLDER']
        },
        'system': {
            'cwd': os.getcwd(),
            'python_version': sys.version,
            'free_disk_space_mb': free_space_mb,
            'total_disk_space_mb': total_space_mb,
            'environment': os.environ.get('FLASK_ENV', 'production')
        }
    }
    return jsonify(status)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)