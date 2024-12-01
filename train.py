import cv2
import mediapipe as mp
import json
import numpy as np

class ASLReferenceCreator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
    def extract_landmarks(self, image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]  # Get first hand
            # Convert landmarks to list of (x, y) coordinates
            coords = []
            for landmark in landmarks.landmark:
                coords.append([landmark.x, landmark.y])
            return coords
        return None

    def create_reference_file(self, sign_images):
        """
        sign_images: dict of sign_name: [list_of_image_paths]
        """
        reference_data = {}
        
        for sign, images in sign_images.items():
            sign_landmarks = []
            for image_path in images:
                landmarks = self.extract_landmarks(image_path)
                if landmarks:
                    sign_landmarks.append(landmarks)
            
            if sign_landmarks:
                # Store average landmarks for this sign
                reference_data[sign] = np.mean(sign_landmarks, axis=0).tolist()
        
        # Save to JSON file
        with open('reference_landmarks.json', 'w') as f:
            json.dump(reference_data, f)

# Usage
creator = ASLReferenceCreator()
sign_images = {
    'hello': ['reference_images/hello8.jpg', 'reference_images/hello9.jpg'],
    'good': ['reference_images/good4.jpg', 'reference_images/good5.jpg'],
    'yes': ['reference_images/yes5.jpg', 'reference_images/yes5.jpg'],
    'no': ['reference_images/no1.jpg', 'reference_images/no7.jpg'],
    'thank_you': ['reference_images/thankyou2.jpg', 'reference_images/thankyou1.jpg']
}
creator.create_reference_file(sign_images)