import pygame
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.neural_network import MLPRegressor
import time
import math
import random
import sys

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

    green['short'] = fuzz.trimf(green.universe, [5, 10, 25])   
    green['medium'] = fuzz.trimf(green.universe, [15, 30, 45])
    green['long'] = fuzz.trimf(green.universe, [35, 60, 60])

    rule1 = ctrl.Rule(traffic['high'] | waiting['long'], green['long'])
    rule2 = ctrl.Rule(traffic['medium'] & waiting['medium'], green['medium'])
    rule3 = ctrl.Rule(traffic['low'] & waiting['short'], green['short'])
    rule4 = ctrl.Rule(traffic['low'] & waiting['medium'], green['medium'])
    rule5 = ctrl.Rule(traffic['medium'] & waiting['short'], green['short'])

    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    return ctrl.ControlSystemSimulation(system)

# Scene Coordinates
ROAD_Y = 300
ROAD_HEIGHT = 100
STOP_LINE_X = 700
ZONE_START_X = 200

class Car:
    def __init__(self):
        self.width = 40
        self.height = 20
        self.x = -self.width
        self.y = ROAD_Y + (ROAD_HEIGHT // 2) - (self.height // 2) + random.randint(-10, 10)
        self.max_speed = random.uniform(3.0, 5.0)
        self.speed = self.max_speed
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

    def move(self, light_state, car_ahead):
        target_speed = self.max_speed

        if light_state == "RED" and self.x + self.width <= STOP_LINE_X:
            dist_to_stop = STOP_LINE_X - (self.x + self.width)
            if dist_to_stop < 20:
                target_speed = 0

        if car_ahead:
            dist_to_car = car_ahead.x - (self.x + self.width)
            if dist_to_car < 25:
                target_speed = min(target_speed, car_ahead.speed)
                if dist_to_car < 10:
                    target_speed = 0

        if self.speed < target_speed:
            self.speed += 0.1  
        elif self.speed > target_speed:
            self.speed -= 0.3  
            
        self.speed = max(0.0, min(self.speed, self.max_speed))
        self.x += self.speed

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (int(self.x), int(self.y), self.width, self.height))
        pygame.draw.rect(surface, (0, 0, 0), (int(self.x), int(self.y), self.width, self.height), 2)


def main():
    # ==========================================
    # 1. NEURAL NETWORK SETUP
    # ==========================================
    print("Training Neural Network on historical traffic patterns...")
    X_train = np.array([[0], [4], [8], [12], [15], [17], [20], [23]])
    y_train = np.array([2, 5, 45, 20, 25, 50, 15, 5]) 

    # FIX: Added solver='lbfgs' to instantly fix the ConvergenceWarning and freeze!
    traffic_nn = MLPRegressor(hidden_layer_sizes=(10, 10), solver='lbfgs', max_iter=2000, random_state=42)
    traffic_nn.fit(X_train, y_train)
    print("Neural Network Ready!")

    # ==========================================
    # 2. FUZZY LOGIC & PYGAME SETUP
    # ==========================================
    fuzzy_sim = build_fuzzy_system()
    
    pygame.init()
    WIDTH, HEIGHT = 1024, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Hybrid AI Traffic Simulation")
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 24)
    large_font = pygame.font.Font(None, 36)

    # ==========================================
    # 3. MASTER LOOP
    # ==========================================
    cars = []
    simulated_hour = 7.5 

    light_state = "RED"
    state_start_time = time.time()
    red_phase_duration = 15.0
    allocated_green_time = 0.0

    last_ai_update_time = 0.0
    anticipated_traffic = 0
    target_green_time = 10.0
    hybrid_traffic_density = 0
    capped_waiting_time = 0
    
    display_green_time = 10.0
    stopwatch_display = 0.0
    countdown_display = 0.0

    running = True
    while running:
        # Handle Window Closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        if not running:
            break

        current_time = time.time()
        elapsed_time = current_time - state_start_time

        simulated_hour = (simulated_hour + 0.005) % 24 
        live_vehicles_in_zone = sum(1 for c in cars if ZONE_START_X <= c.x + c.width and c.x <= STOP_LINE_X)

        # AI Calculations (Only updates twice a second to prevent freezing)
        if current_time - last_ai_update_time >= 0.5:
            nn_prediction = traffic_nn.predict([[simulated_hour]])[0]
            anticipated_traffic = max(0, int(nn_prediction))
            
            hybrid_traffic_density = int((live_vehicles_in_zone * 0.7) + (anticipated_traffic * 0.3))
            capped_vehicle_count = min(hybrid_traffic_density, 50) 
            
            if light_state == "RED":
                simulated_waiting_time = int(elapsed_time)
                capped_waiting_time = min(simulated_waiting_time, 60)

                try:
                    fuzzy_sim.input['traffic_density'] = capped_vehicle_count
                    fuzzy_sim.input['waiting_time'] = capped_waiting_time
                    fuzzy_sim.compute()
                    target_green_time = fuzzy_sim.output['green_time']
                except Exception:
                    target_green_time = 10.0
                    
            last_ai_update_time = current_time

        # Dynamic Spawning
        spawn_chance = max(0.01, anticipated_traffic / 1000.0) 
        if random.random() < spawn_chance:
            if not cars or cars[-1].x > 20:
                cars.append(Car())

        # STATE MACHINE LOGIC
        if light_state == "RED":
            display_green_time = target_green_time
            stopwatch_display = 0.0
            countdown_display = 0.0

            if elapsed_time >= red_phase_duration:
                light_state = "GREEN"
                allocated_green_time = target_green_time
                state_start_time = time.time()

        elif light_state == "GREEN":
            capped_waiting_time = 0 
            green_stopwatch = elapsed_time
            time_remaining = max(0.0, allocated_green_time - green_stopwatch)

            display_green_time = allocated_green_time
            stopwatch_display = green_stopwatch
            countdown_display = time_remaining

            if green_stopwatch >= allocated_green_time:
                light_state = "RED"
                state_start_time = time.time()

        # Update Car Physics
        for i, car in enumerate(cars):
            car_ahead = cars[i-1] if i > 0 else None
            car.move(light_state, car_ahead)

        cars = [c for c in cars if c.x < WIDTH + 50]

        # RENDERING
        screen.fill((30, 30, 30)) 

        pygame.draw.rect(screen, (80, 80, 80), (0, ROAD_Y, WIDTH, ROAD_HEIGHT))
        pygame.draw.rect(screen, (100, 100, 50), (ZONE_START_X, ROAD_Y, STOP_LINE_X - ZONE_START_X, ROAD_HEIGHT))
        pygame.draw.rect(screen, (255, 255, 255), (STOP_LINE_X, ROAD_Y, 10, ROAD_HEIGHT))

        for car in cars:
            car.draw(screen)

        light_color = (255, 50, 50) if light_state == "RED" else (50, 255, 50)
        pygame.draw.circle(screen, light_color, (STOP_LINE_X + 40, ROAD_Y - 50), 30)
        pygame.draw.circle(screen, (200, 200, 200), (STOP_LINE_X + 40, ROAD_Y - 50), 30, 3)

        overlay_rect = pygame.Rect(10, 10, 600, 230)
        pygame.draw.rect(screen, (10, 10, 10), overlay_rect, border_radius=10)
        pygame.draw.rect(screen, (100, 100, 100), overlay_rect, 2, border_radius=10)

        display_hour = math.floor(simulated_hour)
        display_min = int((simulated_hour - display_hour) * 60)
        time_str = f"{display_hour:02d}:{display_min:02d}"

        def draw_text(text, x, y, color=(255, 255, 255), use_large=False):
            f = large_font if use_large else font
            surface = f.render(text, True, color)
            screen.blit(surface, (x, y))

        draw_text(f"Traffic Simulation Time: {time_str} | Active Wait: {capped_waiting_time}s", 25, 20, (200, 200, 200))
        draw_text(f"1. Virtual Sensor Count (Zone): {live_vehicles_in_zone}", 25, 50)
        draw_text(f"2. NN Historical Trend Predict: {anticipated_traffic}", 25, 75, (100, 200, 255))
        draw_text(f"3. Hybrid AI Result: {hybrid_traffic_density} vehicles", 25, 100, (255, 150, 255))
        
        draw_text(f"Calculated Green Time: {display_green_time:.1f}s", 25, 140, (50, 255, 50))
        
        if light_state == "GREEN":
            draw_text(f"[System] Internal Stopwatch: {stopwatch_display:.1f}s", 25, 170, (150, 150, 150))
            ui_color = (255, 255, 50) if countdown_display <= 3.0 else (255, 255, 255)
            draw_text(f"[UI] Time Remaining: {countdown_display:.1f}s", 25, 195, ui_color, use_large=True)

        pygame.display.flip()
        clock.tick(60) 

    pygame.quit()
    sys.exit()

# FIX: This block prevents Windows from accidentally running the code twice!
if __name__ == "__main__":
    main()