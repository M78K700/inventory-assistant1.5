from google.cloud import vision
import os
from PIL import Image
import io
import tempfile
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clean_json_string(s):
    """Clean JSON string by removing control characters and properly escaping newlines"""
    # Remove control characters
    s = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s)
    # Properly escape newlines in private key
    s = s.replace('\\n', '\\\\n')
    return s

def get_vision_client():
    """Initialize and return a Vision API client with proper credentials"""
    try:
        # Try to get credentials from environment variable
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if credentials_path and os.path.exists(credentials_path):
            # Local development with file
            return vision.ImageAnnotatorClient()
        else:
            # Streamlit Cloud deployment
            # Get credentials from Streamlit secrets
            credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
            if credentials_json:
                try:
                    # Clean the JSON string
                    credentials_json = clean_json_string(credentials_json.strip())
                    
                    # Parse the JSON string to ensure it's valid
                    credentials_dict = json.loads(credentials_json)
                    
                    # Create temporary credentials file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                        json.dump(credentials_dict, temp_file, indent=2)
                        temp_file.flush()
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name
                    
                    # Initialize the client
                    client = vision.ImageAnnotatorClient()
                    
                    # Clean up the temporary file
                    os.unlink(temp_file.name)
                    
                    return client
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in GOOGLE_CREDENTIALS_JSON: {str(e)}")
            else:
                raise ValueError("Google Cloud credentials not found in environment variables or Streamlit secrets")
    except Exception as e:
        raise Exception(f"Error initializing Vision API client: {str(e)}")

def detect_labels(image_path):
    """Detect labels in an image using Google Cloud Vision API"""
    client = vision.ImageAnnotatorClient()
    
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    
    # Sort labels by confidence score
    sorted_labels = sorted(labels, key=lambda x: x.score, reverse=True)
    return [label.description for label in sorted_labels]

def detect_text(image_path):
    """Detect text in an image using Google Cloud Vision API"""
    client = vision.ImageAnnotatorClient()
    
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    if texts:
        return texts[0].description
    return ""

def detect_objects(image_path):
    """Detect objects in an image using Google Cloud Vision API"""
    client = vision.ImageAnnotatorClient()
    
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations
    
    # Sort objects by confidence score
    sorted_objects = sorted(objects, key=lambda x: x.score, reverse=True)
    return [obj.name for obj in sorted_objects]

def process_product_image(image):
    """Process product image using Google Cloud Vision API"""
    try:
        # Get credentials from environment variable
        credentials_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
        
        if not credentials_json:
            raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable is not set")
        
        try:
            # Parse the JSON string to ensure it's valid
            credentials_dict = json.loads(credentials_json)
            
            # Create a temporary file for credentials
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(credentials_dict, temp_file, indent=2)
                temp_file.flush()
                temp_file_path = temp_file.name
            
            try:
                # Set the credentials file path
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path
                
                # Initialize the Vision API client
                client = vision.ImageAnnotatorClient()
                
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format or 'JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Create Vision API image object
                vision_image = vision.Image(content=img_byte_arr)
                
                # Perform label detection
                response = client.label_detection(image=vision_image)
                labels = [label.description for label in response.label_annotations]
                
                # Perform text detection
                text_response = client.text_detection(image=vision_image)
                texts = [text.description for text in text_response.text_annotations]
                
                # Perform object detection
                object_response = client.object_localization(image=vision_image)
                objects = [obj.name for obj in object_response.localized_object_annotations]
                
                return {
                    'labels': labels,
                    'texts': texts,
                    'objects': objects
                }
                
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in GOOGLE_CREDENTIALS_JSON: {str(e)}")
                
    except Exception as e:
        raise Exception(f"Error initializing Vision API client: {str(e)}") 