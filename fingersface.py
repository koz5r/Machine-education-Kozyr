import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import pickle

class FaceFingerCounter:
    def __init__(self, owner_name="Owner"):
        self.owner_name = owner_name
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_encoder = LabelEncoder()
        self.face_data = []
        self.labels = []
        self.is_trained = False
        
    def collect_training_data(self, num_samples=100):
        cap = cv2.VideoCapture(0)
        sample_count = 0
        
        print("Сбор данных. Нажмите 'q' для выхода")
        
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_roi = gray[y:y+h, x:x+w]
                
                if len(face_roi) > 0:
                    face_resized = cv2.resize(face_roi, (100, 100))
                    
                    if sample_count < num_samples:
                        self.face_data.append(face_resized.flatten())
                        self.labels.append(self.owner_name)
                        sample_count += 1
                        cv2.putText(frame, f"Collected: {sample_count}/{num_samples}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Collect Faces', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(self.face_data) > 0:
            self.train_model()
    
    def train_model(self):
        labels_encoded = self.label_encoder.fit_transform(self.labels)
        X_train = np.array(self.face_data).reshape(len(self.face_data), 100, 100)
        
        self.recognizer.train(X_train, np.array(labels_encoded))
        self.is_trained = True
        
        with open('face_model.pkl', 'wb') as f:
            pickle.dump(self.recognizer, f)
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print("Модель обучена и сохранена!")
    
    def load_model(self):
        try:
            with open('face_model.pkl', 'rb') as f:
                self.recognizer = pickle.load(f)
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.is_trained = True
            print("Модель загружена!")
        except:
            print("Модель не найдена. Сначала обучите модель!")
    
    def count_fingers(self, hand_roi):
        hsv = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
        
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            
            hull = cv2.convexHull(max_contour, returnPoints=False)
            defects = cv2.convexityDefects(max_contour, hull)
            
            finger_count = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])
                    
                    a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    
                    angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c)) * 57
                    
                    if angle <= 90 and d > 20000:
                        finger_count += 1
                        cv2.circle(hand_roi, far, 5, [0, 0, 255], -1)
            
            return finger_count + 1 if finger_count > 0 else 0
        
        return 0
    
    def run(self):
        if not self.is_trained:
            print("Модель не обучена!")
            choice = input("Хотите обучить модель? (y/n): ")
            if choice.lower() == 'y':
                self.collect_training_data()
            else:
                self.load_model()
        
        cap = cv2.VideoCapture(0)
        owner_detected = False
        finger_count = 0
        
        print("Запуск. Нажмите 'q' для выхода")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            owner_face = None
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (100, 100))
                
                if self.is_trained:
                    try:
                        label, confidence = self.recognizer.predict(face_resized)
                        name = self.label_encoder.inverse_transform([label])[0]
                        
                        if confidence < 100:
                            owner_detected = (name == self.owner_name)
                            color = (0, 255, 0) if owner_detected else (0, 0, 255)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, name if owner_detected else "Unknown", 
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                            
                            if owner_detected:
                                owner_face = (x, y, w, h)
                    except:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            if owner_detected and owner_face:
                x, y, w, h = owner_face
                hand_roi = frame[y+h:y+h+200, max(0, x-100):min(frame.shape[1], x+w+100)]
                
                if hand_roi.size > 0:
                    finger_count = self.count_fingers(hand_roi)
                    
                    cv2.rectangle(frame, (max(0, x-100), y+h), 
                                (min(frame.shape[1], x+w+100), min(frame.shape[0], y+h+200)), 
                                (255, 255, 0), 2)
                    
                    cv2.putText(frame, f"Fingers: {finger_count}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.putText(frame, f"Owner detected: {owner_detected}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if not owner_detected:
                cv2.putText(frame, "Show owner's face first!", 
                          (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Face and Finger Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = FaceFingerCounter(owner_name="Owner")
    recognizer.run()