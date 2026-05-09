import cv2
import math
import numpy as np
import csv
from ultralytics import YOLO
import mediapipe as mp


class PadelConfig:
    """Configuration and thresholds for the Padel Analysis System."""
    SHOT_SPEED_THRESHOLD = 12
    DIRECTION_CHANGE_THRESHOLD = 35
    MIN_DISTANCE_TO_PLAYER = 120
    COOLDOWN_FRAMES = 25
    TRAJECTORY_LEN = 5
    WIDTH = 640
    HEIGHT = 480


class PadelAnalyzer:
    def __init__(self, ball_model_path, player_model_path, racket_model_path):
        """Initialize the YOLO and MediaPipe models and set up state."""
        self.ball_model = YOLO(ball_model_path)
        self.player_model = YOLO(player_model_path)
        self.racket_model = YOLO(racket_model_path)
        
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
        
        self.ball_history = []
        self.player_data = {}
        self.shot_log = []

    @staticmethod
    def get_center(bbox):
        """Calculate the center of a bounding box."""
        return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

    @staticmethod
    def classify_shot(landmarks):
        """Classify the type of shot based on pose landmarks."""
        lw, rw = landmarks[15], landmarks[16]
        ls, rs = landmarks[11], landmarks[12]
        avg_wrist_y = (lw.y + rw.y) / 2
        avg_shoulder_y = (ls.y + rs.y) / 2
        
        if avg_wrist_y < (avg_shoulder_y - 0.12): 
            return "SMASH"
        if abs(avg_wrist_y - avg_shoulder_y) < 0.08: 
            return "VOLLEY"
        
        return "FOREHAND" if rw.x > rs.x else "BACKHAND"

    def process_players(self, frame, display_frame):
        """Track players and draw bounding boxes."""
        player_results = self.player_model.track(
            frame, persist=True, tracker="botsort.yaml", conf=0.3, classes=[0], verbose=False
        )[0]
        
        current_players = {}
        if player_results.boxes.id is not None:
            boxes = player_results.boxes.xyxy.cpu().numpy().tolist()
            ids = player_results.boxes.id.cpu().numpy().astype(int).tolist()
            
            for box, t_id in zip(boxes, ids):
                bbox = list(map(int, box))
                current_players[t_id] = bbox
                
                if t_id not in self.player_data:
                    self.player_data[t_id] = {"shots": 0, "cooldown": 0}
                
                # Draw Player Box & ID Label
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                label = f"Player {t_id}"
                cv2.putText(display_frame, label, (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
        return current_players

    def process_rackets(self, frame, display_frame):
        """Detect and draw rackets."""
        racket_results = self.racket_model.predict(frame, conf=0.25, classes=[38], verbose=False)[0]
        for r_box in racket_results.boxes.xyxy.cpu().numpy():
            rx1, ry1, rx2, ry2 = map(int, r_box)
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

    def process_ball(self, frame, display_frame):
        """Detect and draw the ball, returning its current center coordinate."""
        ball_res = self.ball_model.predict(frame, conf=0.15, verbose=False)[0]
        curr_ball = None
        
        if len(ball_res.boxes) > 0:
            bx1, by1, bx2, by2 = ball_res.boxes.xyxy[0].cpu().numpy()
            curr_ball = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
            cv2.circle(display_frame, curr_ball, 5, (0, 0, 255), -1)
            
        return curr_ball

    def detect_shots(self, frame, curr_ball, current_players, frame_idx, timestamp):
        """Identify if a shot occurred based on ball trajectory and proximity to players."""
        if curr_ball:
            self.ball_history.append(curr_ball)
            if len(self.ball_history) > PadelConfig.TRAJECTORY_LEN: 
                self.ball_history.pop(0)

        if len(self.ball_history) >= 3:
            p_old, p_mid, p_new = self.ball_history[0], self.ball_history[-2], self.ball_history[-1]
            
            v_in = (p_mid[0] - p_old[0], p_mid[1] - p_old[1])
            v_out = (p_new[0] - p_mid[0], p_new[1] - p_mid[1])
            
            mag_in = math.sqrt(v_in[0]**2 + v_in[1]**2)
            mag_out = math.sqrt(v_out[0]**2 + v_out[1]**2)

            if mag_in > 0 and mag_out > 0:
                dot = v_in[0]*v_out[0] + v_in[1]*v_out[1]
                angle = math.degrees(math.acos(max(-1, min(1, dot / (mag_in * mag_out)))))
                
                if mag_out > PadelConfig.SHOT_SPEED_THRESHOLD and angle > PadelConfig.DIRECTION_CHANGE_THRESHOLD:
                    best_candidate, min_dist = None, float('inf')

                    for p_id, bbox in current_players.items():
                        px, py = self.get_center(bbox)
                        dist = math.sqrt((p_mid[0] - px)**2 + (p_mid[1] - py)**2)
                        
                        if dist < PadelConfig.MIN_DISTANCE_TO_PLAYER and self.player_data[p_id]["cooldown"] == 0:
                            if dist < min_dist:
                                min_dist, best_candidate = dist, p_id

                    if best_candidate is not None:
                        self.register_shot(frame, best_candidate, current_players[best_candidate], min_dist, frame_idx, timestamp)

    def register_shot(self, frame, p_id, bbox, distance, frame_idx, timestamp):
        """Register a detected shot, classify it via pose, and log the event."""
        crop = frame[max(0, bbox[1]-30):bbox[3]+30, max(0, bbox[0]-30):bbox[2]+30]
        shot_type = "HIT"
        
        if crop.size > 0:
            res = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                shot_type = self.classify_shot(res.pose_landmarks.landmark)

        self.player_data[p_id]["shots"] += 1
        self.player_data[p_id]["cooldown"] = PadelConfig.COOLDOWN_FRAMES
        
        self.shot_log.append({
            "frame": frame_idx, 
            "timestamp_sec": timestamp,
            "player_id": p_id, 
            "shot_type": shot_type, 
            "distance": round(distance, 2)
        })

    def update_cooldowns(self):
        """Decrement cooldowns for all players."""
        for p_id in self.player_data:
            if self.player_data[p_id]["cooldown"] > 0: 
                self.player_data[p_id]["cooldown"] -= 1

    def save_report(self, output_csv_path):
        """Save the logged shots to a CSV file."""
        if self.shot_log:
            keys = self.shot_log[0].keys()
            with open(output_csv_path, "w", newline="") as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.shot_log)
            print("Report and Video successfully saved.")
        else:
            print("No shots logged. Report not saved.")

    def process_video(self, input_path, output_path, report_path):
        """Main loop to process the video frame by frame."""
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (PadelConfig.WIDTH, PadelConfig.HEIGHT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(frame, (PadelConfig.WIDTH, PadelConfig.HEIGHT))
            display_frame = frame.copy()
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = round(frame_idx / fps, 2)

            current_players = self.process_players(frame, display_frame)
            self.process_rackets(frame, display_frame)
            curr_ball = self.process_ball(frame, display_frame)
            
            self.detect_shots(frame, curr_ball, current_players, frame_idx, timestamp)
            self.update_cooldowns()

            out.write(display_frame)
            cv2.imshow("Padel Analysis System", display_frame)
            
            if cv2.waitKey(1) & 0xFF == 27: 
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        self.save_report(report_path)


if __name__ == "__main__":
    analyzer = PadelAnalyzer(
        ball_model_path="runs/detect/train/weights/best.pt",
        player_model_path="yolov8s.pt",
        racket_model_path="yolov8s.pt"
    )
    
    analyzer.process_video(
        input_path="./input/input.mp4",
        output_path="./output/padel_analysis_output.mp4",
        report_path="./output/padel_report.csv"
    )