#!/usr/bin/env python3
"""
СУПЕР-БЫСТРАЯ ДЕТЕКЦИЯ ЛИЦ - Haar Cascades
"""

import sys
import time
sys.path.append('/usr/lib/python3/dist-packages')

import cv2
from picamera2 import Picamera2

def main():
    print("⚡ СУПЕР-БЫСТРАЯ ДЕТЕКЦИЯ ЛИЦ - Haar Cascades")
    print("=" * 50)
    
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    
    # Загружаем каскады для лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("✅ Система запущена - Haar Cascades")
    
    frame_count = 0
    start_time = time.time()
    detection = True
    fps = 0
    
    try:
        while True:
            frame = picam2.capture_array()
            frame_count += 1
            
            # Конвертируем в grayscale для Haar cascades
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            display_frame = frame.copy()
            
            # Быстрая детекция лиц
            if detection:
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(50, 50),  # Больше минимальный размер для скорости
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Рисуем прямоугольники вокруг лиц
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, 'Face', (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if len(faces) > 0:
                    print(f"👤 Найдено лиц: {len(faces)}", end='\r')
            
            # FPS
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                print(f"📊 FPS: {fps:.1f} | Лица: {'ВКЛ' if detection else 'ВЫКЛ'}", end='\r')
            
            # Интерфейс
            cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display_frame, 'Haar Cascades - FAST', (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(display_frame, "Press 'Q' to quit", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            cv2.imshow('Super Fast Face Detection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                detection = not detection
                
    except KeyboardInterrupt:
        print("\n⏹️ Остановлено")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("\n👋 Завершено")

if __name__ == "__main__":
    main()