import cv2
import typing
import numpy as np
import asyncio
import time
from selfieSegmentation import MPSegmentation

frame_count = 0

class Engine:
    def __init__(
        self, 
        webcam_id: int = 0,
        custom_objects: typing.Iterable = [],
        frame_interval: int = 5  # Process one frame every 5 seconds
    ) -> None:
        self.webcam_id = webcam_id
        self.custom_objects = custom_objects
        self.frame_interval = frame_interval
        self.processing_delay = 1.5  # Extra time for processing

    def custom_processing(self, frame: np.ndarray) -> np.ndarray:
        """Apply custom object processing (FaceNet + MediaPipe)"""
        for custom_object in self.custom_objects:
            frame = custom_object(frame)
        return frame

    def display_frame(self, frame: np.ndarray) -> None:
        """Display the processed frame with recognition results."""
        cv2.imshow("Recognition Result", frame)
        cv2.waitKey(500)  # Display frame for a shorter duration
        cv2.destroyAllWindows()

    async def process_webcam(self) -> None:
        frame_count = 0
        """Capture frames asynchronously, allowing time for processing."""
        while True:
            start_time = time.time()  # Measure execution time
            
            cap = cv2.VideoCapture(self.webcam_id)
            if not cap.isOpened():
                print(f"Webcam with ID ({self.webcam_id}) can't be opened")
                await asyncio.sleep(self.frame_interval)
                continue
            
            success, frame = cap.read()
            cap.release()
            
            if not success or frame is None:
                print("Ignoring empty camera frame.")
                await asyncio.sleep(self.frame_interval)
                continue

            # Resize frame to reduce processing load
            frame = cv2.resize(frame, (320, 240))  # Reduce size for faster processing
            
            # Run FaceNet & MediaPipe in separate thread
            if frame_count % 3 == 0:  # Process every 3rd frame
                frame = await asyncio.to_thread(self.custom_processing, frame)
                frame = await asyncio.to_thread(self.custom_processing, frame)
                print("Frame processed.")  # Debug message to confirm processing

            frame_count += 1

            self.display_frame(frame)

            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.frame_interval - elapsed_time)
            
            print(f"Processing took {elapsed_time:.2f} sec. Waiting {sleep_time:.2f} sec before next capture.")
            await asyncio.sleep(sleep_time + self.processing_delay)

    def run(self):
        asyncio.run(self.process_webcam())
