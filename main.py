from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import json
import base64
from typing import Dict

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class ImageRequest(BaseModel):
    image: str

class ASLRecognizer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Load reference landmarks
        try:
            with open('reference_landmarks.json', 'r') as f:
                self.reference_landmarks = json.load(f)
            print("Loaded reference landmarks successfully")
        except Exception as e:
            print(f"Error loading reference landmarks: {e}")
            self.reference_landmarks = {}

    def extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            coords = []
            for landmark in landmarks.landmark:
                coords.append([landmark.x, landmark.y])
            return coords
        return None

    def compare_landmarks(self, landmarks1, landmarks2):
        """Calculate similarity between two sets of landmarks"""
        landmarks1 = np.array(landmarks1)
        landmarks2 = np.array(landmarks2)
        
        # Calculate Euclidean distance between corresponding points
        distances = np.sqrt(np.sum((landmarks1 - landmarks2) ** 2, axis=1))
        return np.mean(distances)

    def recognize_sign(self, image) -> tuple[str, float]:
        try:
            landmarks = self.extract_landmarks(image)
            if not landmarks:
                return "No hand detected", 0.0

            best_match = None
            best_confidence = 0.0

            for sign, reference in self.reference_landmarks.items():
                distance = self.compare_landmarks(landmarks, reference)
                confidence = 1 / (1 + distance)  # Convert distance to confidence score

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = sign

            if best_match:
                return best_match, best_confidence
            return "No match found", 0.0

        except Exception as e:
            print(f"Error in recognize_sign: {e}")
            return "Error processing image", 0.0

# Initialize recognizer
recognizer = ASLRecognizer()

@app.get("/")
async def root():
    return {"message": "ASL Translator API"}

@app.post("/detect-sign")
async def detect_sign(request: ImageRequest):
    try:
        # Extract base64 image data
        image_data = request.image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        # Decode image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Process image
        sign, confidence = recognizer.recognize_sign(image)

        # Return results
        return {
            "sign": sign,
            "confidence": float(confidence)
        }

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting ASL Translator API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)