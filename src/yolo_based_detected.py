#!/usr/bin/env python3
"""
ЧИСТЫЙ YOLO ДЕТЕКТОР ЛИЦ - ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ
"""

import sys
import time
sys.path.append('/usr/lib/python3/dist-packages')

import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

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
    print("🎯 ЧИСТЫЙ YOLO ДЕТЕКТОР ЛИЦ - ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    
    # Инициализация камеры
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    
    print("🔄 Загружаем YOLO модель...")
    
    try:
        # Используем специальную модель для детекции лиц
        # Если нет специальной модели, используем стандартную YOLO
        try:
            model = YOLO('yolov8n-face.pt')  # Специальная модель для лиц
            print("✅ YOLO face detection модель загружена")
            face_class_id = 0  # В face моделях обычно один класс - лицо
        except:
            model = YOLO('yolov8n.pt')  # Стандартная YOLO
            print("✅ Стандартная YOLO модель загружена")
            face_class_id = 0  # В COCO класс 0 = человек
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return
    
    frame_count = 0
    start_time = time.time()
    detection = True
    fps = 0
    total_faces = 0
    
    # Статистика производительности
    detection_times = []
    
    print("🚀 Запуск детекции...")
    print("💡 Используется только YOLO, без каскадов Хаара")
    
    try:
        while True:
            frame = picam2.capture_array()
            frame_count += 1
            
            # Конвертируем в RGB
            frame_rgb = ensure_rgb(frame)
            display_frame = frame_rgb.copy()
            
            # Детекция только лиц
            if detection:
                start_detect = time.time()
                
                # Конвертируем в BGR для YOLO
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Детекция с оптимизацией для Raspberry Pi
                results = model(frame_bgr, 
                              verbose=False, 
                              conf=0.5,      # Порог уверенности
                              imgsz=160,     # Маленький размер для скорости
                              max_det=3,     # Максимум 3 обнаружения
                              half=False,    # Полная точность на CPU
                              device='cpu')  # Явно указываем CPU
                
                face_count = 0
                
                # Обрабатываем результаты детекции
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            cls = int(box.cls[0])
                            
                            # Фильтруем только лица (класс 0)
                            if cls == face_class_id:
                                # Рисуем прямоугольник лица
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Подпись с уверенностью
                                label = f"Face {conf:.2f}"
                                cv2.putText(display_frame, label, (x1, y1-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                                
                                face_count += 1
                                total_faces += 1
                
                detect_time = time.time() - start_detect
                detection_times.append(detect_time)
                
                # Выводим статистику в реальном времени
                if face_count > 0:
                    avg_detect_time = np.mean(detection_times[-10:]) * 1000  # мс
                    status = f"Лиц: {face_count} | Время: {detect_time*1000:.1f}мс"
                    print(status, end='\r')
            
            # Расчет FPS
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                
                # Статистика каждую секунду
                if detection_times:
                    avg_time = np.mean(detection_times) * 1000
                    min_time = np.min(detection_times) * 1000
                    max_time = np.max(detection_times) * 1000
                    
                    status = "DETECTING" if detection else "VIEW ONLY"
                    print(f"📊 FPS: {fps:.1f} | {status} | Детекция: {avg_time:.1f}мс", end='\r')
            
            # Интерфейс
            cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display_frame, 'Pure YOLO Face Detection', (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display_frame, 'Detection: ON' if detection else 'Detection: OFF', 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                       (0, 255, 0) if detection else (0, 0, 255), 1)
            
            # Текущая статистика на кадре
            if detection_times:
                current_detect = detection_times[-1] * 1000 if detection_times else 0
                cv2.putText(display_frame, f'Current: {current_detect:.1f}ms', (10, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            cv2.putText(display_frame, "Press 'Q' to quit, 'D' to toggle", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            cv2.imshow('Pure YOLO Face Detection - Performance Test', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                detection = not detection
                print(f"\n🔍 Детекция: {'ВКЛ' if detection else 'ВЫКЛ'}")
                
    except KeyboardInterrupt:
        print("\n⏹️ Остановлено пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        
        # Финальная статистика
        if detection_times:
            avg_detect = np.mean(detection_times) * 1000
            max_detect = np.max(detection_times) * 1000
            min_detect = np.min(detection_times) * 1000
            
            print(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
            print(f"   FPS: {fps:.1f}")
            print(f"   Время детекции: {avg_detect:.1f}мс")
            print(f"   Min: {min_detect:.1f}мс, Max: {max_detect:.1f}мс")
            print(f"   Всего обнаружено лиц: {total_faces}")
        
        print("👋 Тестирование завершено")

if __name__ == "__main__":
    main()