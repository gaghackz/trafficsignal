import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from ultralytics import YOLO
from sklearn.neural_network import MLPRegressor
import time
import math

# ==========================================
# 1. NEURAL NETWORK SETUP (Predictive AI)
# ==========================================
print("Training Neural Network on historical traffic patterns...")

# Synthetic Historical Data: [Hour of the Day (0-24)]
# We simulate low traffic at night, and spikes at 8 AM and 5 PM (17:00)
X_train = np.array([[0], [4], [8], [12], [15], [17], [20], [23]])
# Expected baseline vehicles for those hours
y_train = np.array([2, 5, 45, 20, 25, 50, 15, 5]) 

# Create and train a Multi-Layer Perceptron (Neural Network)
traffic_nn = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=2000, random_state=42)
traffic_nn.fit(X_train, y_train)

print("Neural Network Ready!")

# ==========================================
# 2. FUZZY LOGIC SETUP (Decision AI)
# ==========================================
def build_fuzzy_system():
    traffic = ctrl.Antecedent(np.arange(0, 51, 1), 'traffic_density')
    waiting = ctrl.Antecedent(np.arange(0, 61, 1), 'waiting_time')
    green = ctrl.Consequent(np.arange(0, 61, 1), 'green_time')

    traffic['low'] = fuzz.trimf(traffic.universe, [0, 0, 25])
    traffic['medium'] = fuzz.trimf(traffic.universe, [10, 25, 40])
    traffic['high'] = fuzz.trimf(traffic.universe, [30, 50, 50])

    waiting['short'] = fuzz.trimf(waiting.universe, [0, 0, 25])
    waiting['medium'] = fuzz.trimf(waiting.universe, [15, 30, 45])
    waiting['long'] = fuzz.trimf(waiting.universe, [35, 60, 60])

    green['short'] = fuzz.trimf(green.universe, [0, 0, 25])
    green['medium'] = fuzz.trimf(green.universe, [15, 30, 45])
    green['long'] = fuzz.trimf(green.universe, [35, 60, 60])

    rule1 = ctrl.Rule(traffic['high'] | waiting['long'], green['long'])
    rule2 = ctrl.Rule(traffic['medium'] & waiting['medium'], green['medium'])
    rule3 = ctrl.Rule(traffic['low'] & waiting['short'], green['short'])
    rule4 = ctrl.Rule(traffic['low'] & waiting['medium'], green['medium'])
    rule5 = ctrl.Rule(traffic['medium'] & waiting['short'], green['short'])

    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    return ctrl.ControlSystemSimulation(system)

fuzzy_sim = build_fuzzy_system()

# ==========================================
# 3. COMPUTER VISION & ROI SETUP
# ==========================================
model = YOLO('yolov8n.pt') 
video_path = "traffic_video.mp4" 
cap = cv2.VideoCapture(video_path)

start_time = time.time()

# Detection Zone (Adjust these if needed to fit your lane)
zone_points = np.array([
    [150, 600], [850, 600], [650, 150], [450, 150]
], np.int32).reshape((-1, 1, 2))

# Simulation clock (start at 7:30 AM to watch the NN predict the 8 AM rush hour)
simulated_hour = 7.5 

print("Starting Video Simulation... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 600))
    annotated_frame = frame.copy()
    cv2.polylines(annotated_frame, [zone_points], isClosed=True, color=(0, 255, 255), thickness=2)

    # 1. Vision: Get live count from YOLO
    results = model(frame, classes=[2, 3, 5, 7], verbose=False)
    live_vehicles_in_zone = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        if cv2.pointPolygonTest(zone_points, (cx, cy), False) >= 0:
            live_vehicles_in_zone += 1
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

    # 2. Prediction: Ask Neural Network for anticipated traffic based on time of day
    # We speed up time in the simulation so you can see it change
    simulated_hour = (simulated_hour + 0.005) % 24 
    nn_prediction = traffic_nn.predict([[simulated_hour]])[0]
    anticipated_traffic = max(0, int(nn_prediction)) # Prevent negative predictions

    # 3. Hybrid Calculation: Blend Live Count + NN Prediction
    # 70% weight to real-time live sight, 30% weight to historical NN trend
    hybrid_traffic_density = int((live_vehicles_in_zone * 0.7) + (anticipated_traffic * 0.3))
    
    current_time = time.time()
    simulated_waiting_time = int(current_time - start_time)
    
    capped_vehicle_count = min(hybrid_traffic_density, 50) 
    capped_waiting_time = min(simulated_waiting_time, 60)

    # 4. Fuzzy Logic Execution
    try:
        fuzzy_sim.input['traffic_density'] = capped_vehicle_count
        fuzzy_sim.input['waiting_time'] = capped_waiting_time
        fuzzy_sim.compute()
        green_time = fuzzy_sim.output['green_time']
    except Exception:
        green_time = 0.0

    # ==========================================
    # 5. DRAW HYBRID INFO OVERLAY
    # ==========================================
    # Make the UI box a bit taller to fit the new NN data
    cv2.rectangle(annotated_frame, (10, 10), (550, 190), (0, 0, 0), -1)

    # Convert decimal hour to HH:MM format for the display
    display_hour = math.floor(simulated_hour)
    display_min = int((simulated_hour - display_hour) * 60)
    time_str = f"{display_hour:02d}:{display_min:02d}"

    # Display Text
    cv2.putText(annotated_frame, f"Simulated Time: {time_str} | Wait: {capped_waiting_time}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(annotated_frame, f"1. Live Vision Count: {live_vehicles_in_zone}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"2. NN Trend Predict: {anticipated_traffic} (Historical)", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(annotated_frame, f"3. Hybrid Fuzzy Input: {hybrid_traffic_density}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 255), 2)
    cv2.putText(annotated_frame, f"Final Green Time: {green_time:.1f}s", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) 

    cv2.imshow("Hybrid Neuro-Fuzzy Traffic Simulation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()