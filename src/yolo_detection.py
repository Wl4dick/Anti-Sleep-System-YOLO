#!/usr/bin/env python3
"""
ФИНАЛЬНАЯ СИСТЕМА ДЕТЕКЦИИ ЛИЦ - исправленные ошибки
"""

import sys
import time
sys.path.append('/usr/lib/python3/dist-packages')

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

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
    print("👤 ФИНАЛЬНАЯ СИСТЕМА ДЕТЕКЦИИ ЛИЦ")
    print("=" * 50)
    
    # Минимальное разрешение для скорости
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    
    print("🔄 Загружаем модель для детекции лиц...")
    
    try:
        # Используем специализированную модель для лиц
        model = YOLO('yolov8n-face.pt')
        print("✅ Модель для лиц загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return
    
    frame_count = 0
    start_time = time.time()
    detection = True
    fps = 0  # Инициализируем переменную fps
    
    try:
        while True:
            frame = picam2.capture_array()
            frame_count += 1
            
            # Гарантируем RGB формат
            frame_rgb = ensure_rgb(frame)
            display_frame = frame_rgb.copy()
            
            # Детекция лиц
            if detection and frame_count % 2 == 0:  # Каждый 2-й кадр
                # Конвертируем в BGR для YOLO
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Детекция лиц с оптимизацией
                results = model(frame_bgr, 
                              verbose=False, 
                              conf=0.5,      # Более высокая уверенность = быстрее
                              imgsz=192,     # Маленький размер для скорости
                              max_det=3,     # Максимум 3 обнаружения
                              half=True)     # Половинная точность если возможно
                
                # Отображаем результаты
                face_count = 0
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            
                            # Рисуем прямоугольник лица
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Подпись
                            label = f"Face {conf:.2f}"
                            cv2.putText(display_frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            
                            face_count += 1
                
                if face_count > 0:
                    print(f"👤 Найдено лиц: {face_count}", end='\r')
            
            # FPS расчет
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                status = "DETECTING" if detection else "VIEW ONLY"
                print(f"📊 FPS: {fps:.1f} | {status} | ", end='\r')
            
            # Интерфейс
            cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display_frame, 'Face Detection ON' if detection else 'Face Detection OFF', 
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                       (0, 255, 0) if detection else (0, 0, 255), 1)
            cv2.putText(display_frame, "Press 'Q' to quit, 'D' to toggle detection", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            cv2.imshow('Final Face Detection System', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                detection = not detection
                print(f"\n🔍 Детекция лиц: {'ВКЛ' if detection else 'ВЫКЛ'}")
                
    except KeyboardInterrupt:
        print("\n⏹️ Остановлено")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("\n👋 Завершено")

if __name__ == "__main__":
    main()