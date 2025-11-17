#!/usr/bin/env python3
"""
СИСТЕМА ДЕТЕКЦИИ ЛИЦ - ТОЛЬКО ВСТРОЕННЫЕ МОДЕЛИ OPENCV
"""

import sys
import time
sys.path.append('/usr/lib/python3/dist-packages')

import cv2
import numpy as np
from picamera2 import Picamera2

def load_face_detector():
    """Загружаем детектор лиц из встроенных моделей OpenCV"""
    
    # 1. Пробуем DNN детектор (встроен в OpenCV)
    try:
        # Эти файлы обычно есть в OpenCV
        prototxt = "deploy.prototxt"
        model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        
        # Если файлов нет, используем встроенные пути
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        print("✅ DNN детектор загружен")
        return {"type": "dnn", "model": net}
    except:
        print("⚠️ DNN детектор недоступен")
    
    # 2. Всегда доступный вариант - Haar Cascades
    print("🔄 Используем Haar Cascades")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    return {"type": "haar", "model": face_cascade, "eye_model": eye_cascade}

def detect_faces_dnn(model, frame, confidence_threshold=0.7):
    """Детекция лиц с помощью DNN"""
    h, w = frame.shape[:2]
    
    # Подготавливаем входное изображение
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            
            # Проверяем валидность координат
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2, confidence))
    
    return faces

def detect_faces_haar(model, frame):
    """Детекция лиц с помощью Haar Cascades"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Детекция лиц
    faces = model.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    result = []
    for (x, y, w, h) in faces:
        result.append((x, y, x + w, y + h, 1.0))
    
    return result

def detect_eyes_haar(eye_model, frame, face_roi):
    """Детекция глаз в области лица"""
    x, y, w, h = face_roi
    face_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)
    
    eyes = eye_model.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    
    # Конвертируем координаты глаз относительно всего изображения
    eyes_global = []
    for (ex, ey, ew, eh) in eyes:
        eyes_global.append((x + ex, y + ey, ew, eh))
    
    return eyes_global

def ensure_rgb(frame):
    """Гарантирует, что frame будет в RGB формате"""
    if len(frame.shape) == 2:  # Grayscale
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:  # RGBA
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    elif frame.shape[2] == 3:  # RGB
        return frame
    else:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def main():
    print("👤 СИСТЕМА ДЕТЕКЦИИ ЛИЦ И ГЛАЗ - OPENCV")
    print("=" * 50)
    
    # Используем ваш рабочий подход с камерой
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    
    print("🔄 Загружаем детекторы...")
    
    # Загружаем модели
    detector = load_face_detector()
    model_type = detector["type"]
    model = detector["model"]
    
    # Для Haar добавляем детектор глаз
    eye_model = None
    if model_type == "haar":
        eye_model = detector["eye_model"]
    
    print(f"✅ Используется: {model_type.upper()} детектор")
    
    frame_count = 0
    start_time = time.time()
    detection = True
    fps = 0
    total_faces = 0
    total_eyes = 0
    
    try:
        while True:
            frame = picam2.capture_array()
            frame_count += 1
            
            # Гарантируем RGB формат
            frame_rgb = ensure_rgb(frame)
            display_frame = frame_rgb.copy()
            
            # Детекция лиц
            if detection:
                start_detect = time.time()
                
                if model_type == "dnn":
                    faces = detect_faces_dnn(model, frame_rgb, 0.7)
                    eyes = []  # DNN не детектирует глаза напрямую
                else:  # haar
                    faces = detect_faces_haar(model, frame_rgb)
                    eyes = []
                    # Для каждого лица детектируем глаза
                    for (x, y, x2, y2, conf) in faces:
                        face_roi = (x, y, x2-x, y2-y)
                        face_eyes = detect_eyes_haar(eye_model, frame_rgb, face_roi)
                        eyes.extend(face_eyes)
                
                detect_time = time.time() - start_detect
                
                # Отображаем лица
                face_count = 0
                for (x1, y1, x2, y2, conf) in faces:
                    color = (0, 255, 0) if model_type == "dnn" else (255, 0, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Подпись для лица
                    if model_type == "dnn":
                        label = f"Face {conf:.2f}"
                    else:
                        label = "Face"
                    
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    face_count += 1
                    total_faces += 1
                
                # Отображаем глаза (только для Haar)
                eye_count = 0
                if model_type == "haar":
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(display_frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 1)
                        cv2.putText(display_frame, "Eye", (ex, ey-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                        eye_count += 1
                        total_eyes += 1
                
                if face_count > 0:
                    status = f"Лиц: {face_count} | Глаз: {eye_count} | Время: {detect_time*1000:.1f}мс"
                    print(status, end='\r')
            
            # FPS расчет
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                status = "DETECTING" if detection else "VIEW ONLY"
                print(f"📊 FPS: {fps:.1f} | {status} | ", end='\r')
            
            # Интерфейс
            model_name = "DNN" if model_type == "dnn" else "Haar Cascades"
            cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display_frame, f'Model: {model_name}', (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display_frame, 'Detection: ON' if detection else 'Detection: OFF', 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                       (0, 255, 0) if detection else (0, 0, 255), 1)
            
            if model_type == "haar":
                cv2.putText(display_frame, "Eyes: Yellow", (10, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            
            cv2.putText(display_frame, "Press 'Q' to quit, 'D' to toggle", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            cv2.imshow(f'Face & Eye Detection - {model_name}', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                detection = not detection
                print(f"\n🔍 Детекция: {'ВКЛ' if detection else 'ВЫКЛ'}")
                
    except KeyboardInterrupt:
        print("\n⏹️ Остановлено")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print(f"\n📊 ИТОГО: {total_faces} лиц, {total_eyes} глаз")
        print("👋 Завершено")

if __name__ == "__main__":
    main()