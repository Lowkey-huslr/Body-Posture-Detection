import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        landmarks = results.pose_landmarks.landmark
        nose_x, nose_y = int(nose_landmark.x * frame.shape[1]), int(nose_landmark.y * frame.shape[0])
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        cv2.circle(frame, (nose_x, nose_y), 10, (0, 255, 0), -1)
        cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)
        distance = ((nose_x - frame_center_x)**2 + (nose_y - frame_center_y)**2)**0.5
        cv2.putText(frame, f"Distance: {distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.line(frame,(220,240),(420,240),(0,0,255),4)
        cv2.line(frame,(220,0),(220,480),(0,0,255),4)
        cv2.line(frame,(420,0),(420,480),(0,0,255),4)
    cv2.imshow("Posture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
