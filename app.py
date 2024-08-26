import os
import shutil
import unicodedata
from flask import Flask, request, send_file, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import io

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
CLASSIFIED_FOLDER = 'classified_images'
NO_OBJECTS_FOLDER = 'no_objects_detected'
ZIP_FILENAME = 'classified_images.zip'

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    labels = get_model_labels()

    if request.method == 'POST':
        clear_directories([UPLOAD_FOLDER, CLASSIFIED_FOLDER, NO_OBJECTS_FOLDER])

        # 업로드 폴더가 없으면 생성
        create_directories([UPLOAD_FOLDER, CLASSIFIED_FOLDER, NO_OBJECTS_FOLDER])

        files = request.files.getlist('file')
        include_labels = request.form.getlist('include_labels')

        for file in files:
            filename = secure_and_normalize_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # 파일 저장
            file.save(filepath)

            # 이미지 파일 열기 및 객체 탐지
            image = Image.open(filepath)
            category = detect_objects(image, include_labels=include_labels)
            
            if category in ["No objects detected", "No included objects detected"]:
                save_image_to_no_objects_folder(image, filename, NO_OBJECTS_FOLDER)
            else:
                save_image_by_category(image, category, filename, CLASSIFIED_FOLDER)
        
        # 두 개의 폴더를 모두 압축
        zip_filepath = zip_directories([CLASSIFIED_FOLDER, NO_OBJECTS_FOLDER], ZIP_FILENAME)
        return send_file(zip_filepath, as_attachment=True)

    return render_template('index.html', labels=labels)

def detect_objects(image, include_labels=None):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(image)
    detections = results.pandas().xyxy[0]

    if detections.empty:
        return "No objects detected"

    if include_labels:
        detections = detections[detections['name'].isin(include_labels)]

    if detections.empty:
        return "No included objects detected"

    detections['area'] = (detections['xmax'] - detections['xmin']) * (detections['ymax'] - detections['ymin'])
    largest_object = detections.loc[detections['area'].idxmax()]['name']
    return largest_object

def save_image_to_no_objects_folder(image, image_name, base_dir=NO_OBJECTS_FOLDER):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_name += '.jpg'

    no_objects_dir = os.path.join(base_dir)
    os.makedirs(no_objects_dir, exist_ok=True)
    image_path = os.path.join(no_objects_dir, image_name)
    image.save(image_path)
    return image_path

def save_image_by_category(image, category, image_name, base_dir=CLASSIFIED_FOLDER):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_name += '.jpg'

    category_dir = os.path.join(base_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    image_path = os.path.join(category_dir, image_name)
    image.save(image_path)
    return image_path

def zip_directories(directories, output_filename):
    # 압축 파일 이름에서 확장자(.zip)를 제거
    base_name = output_filename.split('.zip')[0]

    # 임시 디렉토리를 생성하여 파일을 모두 모읍니다.
    temp_dir = 'temp_for_zip'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # 각 디렉토리의 내용을 임시 디렉토리로 복사합니다.
    for directory in directories:
        dest_dir = os.path.join(temp_dir, os.path.basename(directory))
        if os.path.exists(directory):
            shutil.copytree(directory, dest_dir)

    # 임시 디렉토리를 압축 파일로 만듭니다.
    shutil.make_archive(base_name, 'zip', root_dir=temp_dir)

    # 임시 디렉토리를 삭제합니다.
    shutil.rmtree(temp_dir)

    # 압축 파일 경로 반환
    return f"{base_name}.zip"

def clear_directories(directories):
    """기존의 디렉토리 내의 파일들만 삭제하는 함수"""
    for directory in directories:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

def create_directories(directories):
    """필요한 디렉토리가 없으면 생성하는 함수"""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def secure_and_normalize_filename(filename):
    """파일명을 안전하게 변환하는 함수"""
    filename = unicodedata.normalize('NFC', filename)
    filename = secure_filename(filename)
    return filename

def get_model_labels():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    labels = list(model.names.values())
    return labels

if __name__ == '__main__':
    app.run(debug=True)
