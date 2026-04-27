import pygame
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.neural_network import MLPRegressor
import time
import random
import sys

# ==========================================
# 1. CONFIGURATION & SIMULATION CONSTANTS
# ==========================================
WIDTH, HEIGHT = 1024, 768
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
ROAD_WIDTH = 120
LANE_WIDTH = ROAD_WIDTH // 2
STOP_LINE_OFFSET = 150

MIN_GREEN = 10.0
MAX_GREEN = 60.0
YELLOW_TIME = 3.0
ALL_RED_TIME = 2.0
CAR_SIZE = (20, 40)
MAX_SPEED = 4.0

TOTAL_CARS = 400  # Increased to allow gridlock to build up

BLACK, GRAY, WHITE = (10, 10, 10), (50, 50, 50), (255, 255, 255)
RED, YELLOW, GREEN, CYAN = (255, 50, 50), (255, 200, 0), (50, 255, 50), (0, 255, 255)
UI_BG = (15, 20, 25, 220) 

NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3

# ==========================================
# 2. STATS LOGGER
# ==========================================
class StatsLogger:
    def __init__(self):
        self.total_spawned = 0
        self.total_cleared = 0
        self.total_accumulated_delay = 0.0
        self.max_wait_time = 0.0
        self.start_time = time.time()
        print("🚥 Stats Logger Initialized. Unoptimized Benchmark Running...")

    def log_spawn(self):
        self.total_spawned += 1

    def log_cleared_vehicle(self, vehicle):
        self.total_cleared += 1
        final_wait = vehicle.total_historical_wait + vehicle.wait_duration
        self.total_accumulated_delay += final_wait
        if final_wait > self.max_wait_time:
            self.max_wait_time = final_wait

    def print_final_report(self):
        real_duration = time.time() - self.start_time
        avg_wait = self.total_accumulated_delay / self.total_cleared if self.total_cleared > 0 else 0

        print("\n" + "="*50)
        print("🚦 UNOPTIMIZED SIMULATION - FINAL STATISTICS 🚦")
        print("="*50)
        print(f"Total Real Time Elapsed: {real_duration:.2f} seconds")
        print(f"Total Vehicles Cleared:  {self.total_cleared} / {TOTAL_CARS}")
        print("-" * 50)
        print(f"Total Accumulated Delay: {self.total_accumulated_delay:.1f} seconds")
        print(f"Average Wait Time/Car:   {avg_wait:.1f} seconds")
        print(f"Max Wait Time (Worst):   {self.max_wait_time:.1f} seconds")
        print(f"Intersection Throughput: {(self.total_cleared / real_duration):.2f} cars/sec")
        print("="*50 + "\n")

# ==========================================
# 3. STATIC FUZZY SYSTEM (UNOPTIMIZED)
# ==========================================
def build_fuzzy_system():
    active_traffic = ctrl.Antecedent(np.arange(0, 51, 1), 'active_traffic')
    competing_wait = ctrl.Antecedent(np.arange(0, 121, 1), 'competing_wait')
    green_time = ctrl.Consequent(np.arange(MIN_GREEN, MAX_GREEN + 1, 1), 'green_time')

    t_m_s, t_h_s = 15.0, 30.0  
    w_m_s, w_l_s = 40.0, 80.0  
    g_m, g_l = 30.0, 50.0    

    active_traffic['low'] = fuzz.trimf(active_traffic.universe, [0, 0, t_m_s])
    active_traffic['medium'] = fuzz.trimf(active_traffic.universe, [t_m_s - 5, (t_m_s+t_h_s)//2, t_h_s])
    active_traffic['high'] = fuzz.trimf(active_traffic.universe, [t_h_s - 5, 50, 50])

    competing_wait['short'] = fuzz.trimf(competing_wait.universe, [0, 0, w_m_s])
    competing_wait['medium'] = fuzz.trimf(competing_wait.universe, [w_m_s - 10, 60, w_l_s])
    competing_wait['long'] = fuzz.trimf(competing_wait.universe, [w_l_s - 10, 120, 120])

    green_time['short'] = fuzz.trimf(green_time.universe, [MIN_GREEN, MIN_GREEN, g_m])
    green_time['medium'] = fuzz.trimf(green_time.universe, [MIN_GREEN, g_m, g_l])
    green_time['long'] = fuzz.trimf(green_time.universe, [g_m, MAX_GREEN, MAX_GREEN])

    rule1 = ctrl.Rule(active_traffic['low'], green_time['short'])
    rule2 = ctrl.Rule(active_traffic['high'] & competing_wait['short'], green_time['long'])
    rule3 = ctrl.Rule(active_traffic['high'] & competing_wait['long'], green_time['medium'])
    rule4 = ctrl.Rule(active_traffic['medium'] & competing_wait['short'], green_time['medium'])
    rule5 = ctrl.Rule(active_traffic['medium'] & competing_wait['long'], green_time['short'])
    
    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    return ctrl.ControlSystemSimulation(system), [t_m_s, t_h_s, w_m_s, w_l_s, g_m, g_l]

# ==========================================
# 4. NEURAL NETWORK & VEHICLES
# ==========================================
def train_traffic_nn():
    X = np.array([[0], [2], [5], [7], [8], [9], [12], [14], [16], [17], [18], [21], [23]])
    y = np.array([0.1, 0.05, 0.2, 0.8, 1.0, 0.9, 0.6, 0.7, 0.9, 1.0, 0.8, 0.3, 0.15])
    nn = MLPRegressor(hidden_layer_sizes=(10, 10), solver='lbfgs', max_iter=1000)
    nn.fit(X, y)
    return nn

class Vehicle:
    def __init__(self, direction):
        self.direction = direction
        if direction == NORTH: self.x, self.y = CENTER_X + LANE_WIDTH // 2, -CAR_SIZE[1]
        elif direction == SOUTH: self.x, self.y = CENTER_X - LANE_WIDTH // 2, HEIGHT + CAR_SIZE[1]
        elif direction == EAST: self.x, self.y = -CAR_SIZE[1], CENTER_Y - LANE_WIDTH // 2
        elif direction == WEST: self.x, self.y = WIDTH + CAR_SIZE[1], CENTER_Y + LANE_WIDTH // 2

        self.w, self.h = CAR_SIZE if direction in [NORTH, SOUTH] else (CAR_SIZE[1], CAR_SIZE[0])
        self.speed = MAX_SPEED
        self.state = "MOVING"
        self.color = (random.randint(100,255), random.randint(100,255), random.randint(100,255))
        self.stop_time = time.time()
        self.wait_duration = 0
        self.total_historical_wait = 0.0 

    def update_physics(self, light_state, vehicle_ahead):
        target_speed = MAX_SPEED
        dist_to_stop = 9999
        if light_state != "GREEN":
            if self.direction == NORTH: dist_to_stop = (CENTER_Y - STOP_LINE_OFFSET) - (self.y + self.h)
            elif self.direction == SOUTH: dist_to_stop = self.y - (CENTER_Y + STOP_LINE_OFFSET)
            elif self.direction == EAST: dist_to_stop = (CENTER_X - STOP_LINE_OFFSET) - (self.x + self.w)
            elif self.direction == WEST: dist_to_stop = self.x - (CENTER_X + STOP_LINE_OFFSET)

        if 0 < dist_to_stop < 15: target_speed = 0
        elif dist_to_stop < 60: target_speed = MAX_SPEED * (dist_to_stop / 60)

        if vehicle_ahead:
            if self.direction == NORTH: dist_to_car = vehicle_ahead.y - (self.y + self.h)
            elif self.direction == SOUTH: dist_to_car = self.y - (vehicle_ahead.y + vehicle_ahead.h)
            elif self.direction == EAST: dist_to_car = vehicle_ahead.x - (self.x + self.w)
            elif self.direction == WEST: dist_to_car = self.x - (vehicle_ahead.x + vehicle_ahead.w)

            if dist_to_car < 20: target_speed = 0
            elif dist_to_car < 60: target_speed = min(target_speed, vehicle_ahead.speed * 0.8)

        if self.speed < target_speed: self.speed += 0.2
        elif self.speed > target_speed: self.speed -= 0.6
        self.speed = max(0, min(self.speed, MAX_SPEED))

        if self.direction == NORTH: self.y += self.speed
        elif self.direction == SOUTH: self.y -= self.speed
        elif self.direction == EAST: self.x += self.speed
        elif self.direction == WEST: self.x -= self.speed

        if self.speed < 0.1:
            if self.state != "STOPPED":
                self.state = "STOPPED"
                self.stop_time = time.time()
            self.wait_duration = time.time() - self.stop_time
        else:
            if self.state == "STOPPED":
                self.total_historical_wait += self.wait_duration
            self.state = "MOVING"
            self.wait_duration = 0

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (int(self.x), int(self.y), self.w, self.h))

# ==========================================
# 5. CONTROLLER & MAIN LOOP
# ==========================================
class TrafficLightController:
    def __init__(self, fuzzy_sim):
        self.fuzzy_sim = fuzzy_sim
        self.current_phase = 0
        self.state = "GREEN"
        self.state_start_time = time.time()
        self.green_duration = MIN_GREEN
        self.update_needed = True
        self.diag_active_queue = 0
        self.diag_competing_wait = 0

    def get_light_states(self):
        states = ["RED"] * 4
        if self.state == "GREEN":
            states[0 if self.current_phase == 0 else 2] = "GREEN"
            states[1 if self.current_phase == 0 else 3] = "GREEN"
        elif self.state == "YELLOW":
            states[0 if self.current_phase == 0 else 2] = "YELLOW"
            states[1 if self.current_phase == 0 else 3] = "YELLOW"
        return states

    def update(self, vehicles_by_dir):
        self.time_elapsed = time.time() - self.state_start_time
        if self.state == "GREEN":
            if self.update_needed:
                d1, d2 = (0, 1) if self.current_phase == 0 else (2, 3)
                o1, o2 = (2, 3) if self.current_phase == 0 else (0, 1) 
                self.diag_active_queue = len(vehicles_by_dir[d1]) + len(vehicles_by_dir[d2])
                self.diag_competing_wait = 0
                for d in [o1, o2]:
                    for v in vehicles_by_dir[d]:
                        if v.state == "STOPPED": 
                            self.diag_competing_wait = max(self.diag_competing_wait, v.wait_duration)
                try:
                    self.fuzzy_sim.input['active_traffic'] = min(self.diag_active_queue, 50)
                    self.fuzzy_sim.input['competing_wait'] = min(self.diag_competing_wait, 120)
                    self.fuzzy_sim.compute()
                    self.green_duration = max(MIN_GREEN, min(self.fuzzy_sim.output['green_time'], MAX_GREEN))
                except:
                    self.green_duration = MIN_GREEN
                self.update_needed = False

            if self.time_elapsed >= self.green_duration:
                self.state = "YELLOW"
                self.state_start_time = time.time()
        elif self.state == "YELLOW" and self.time_elapsed >= YELLOW_TIME:
            self.state = "ALL_RED"
            self.state_start_time = time.time()
        elif self.state == "ALL_RED" and self.time_elapsed >= ALL_RED_TIME:
            self.current_phase = (self.current_phase + 1) % 2
            self.state = "GREEN"
            self.state_start_time = time.time()
            self.update_needed = True

def draw_ui_panel(screen, font, title_font, sim_hour, intensity_factor, controller, params, logger):
    panel = pygame.Surface((380, 410), pygame.SRCALPHA)
    pygame.draw.rect(panel, UI_BG, panel.get_rect(), border_radius=10)
    screen.blit(panel, (15, 15))
    y_offset, x_offset = 25, 30

    def add_text(text, color, is_title=False):
        nonlocal y_offset
        f = title_font if is_title else font
        screen.blit(f.render(text, True, color), (x_offset, y_offset))
        y_offset += 30 if is_title else 22

    add_text("UNOPTIMIZED BENCHMARK", RED, True)
    add_text(f"Progress: {logger.total_cleared} / {TOTAL_CARS} Cars", CYAN)
    y_offset += 5

    add_text("--- 1. NEURAL NETWORK (sklearn) ---", YELLOW)
    add_text(f"Time of Day: {int(sim_hour):02d}:00", WHITE)
    add_text(f"Raw Output (Traffic Vol): {intensity_factor:.2f}x", WHITE)
    y_offset += 10

    phase_name = "NORTH/SOUTH" if controller.current_phase == 0 else "EAST/WEST"
    add_text("--- 2. FUZZY LOGIC (skfuzzy) ---", YELLOW)
    add_text(f"Active Phase: {phase_name}", WHITE)
    add_text(f"Input A (Active Cars): {controller.diag_active_queue}", WHITE)
    add_text(f"Input B (Opposing Wait): {controller.diag_competing_wait:.1f} sec", WHITE)
    
    if controller.state == "GREEN":
        rem = controller.green_duration - controller.time_elapsed
        add_text(f"Output (Total Green): {controller.green_duration:.1f} sec", GREEN)
        add_text(f"Remaining Green: {rem:.1f} sec", GREEN)
    else:
        add_text(f"Remaining Green: SWITCHING", YELLOW)
    y_offset += 10

    add_text("--- 3. STATIC PARAMETERS ---", RED)
    add_text("Hardcoded human guesses.", WHITE)

def main():
    traffic_predictor = train_traffic_nn()
    fuzzy_sim, static_params = build_fuzzy_system()
    tl_controller = TrafficLightController(fuzzy_sim)
    logger = StatsLogger()
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Traffic Intersection - UNOPTIMIZED Benchmark")
    clock, ui_font, ui_title = pygame.time.Clock(), pygame.font.Font(None, 24), pygame.font.Font(None, 32)
    vehicles, sim_hour, running = [], 7.0, True
    
    while running:
        dt = clock.tick(60) / 1000.0 
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
        sim_hour = (sim_hour + (dt / 5)) % 24 
        nn_raw = traffic_predictor.predict([[sim_hour]])[0]
        intensity_factor = max(0.1, min(1.5, nn_raw))
        
        # BENCHMARK SPAWN LOGIC - ASYMMETRIC GRIDLOCK (90% North/South, 10% East/West)
        if logger.total_spawned < TOTAL_CARS:
            spawn_chance_per_sec = 5.0  
            if random.random() < spawn_chance_per_sec * dt:
                # 45 N, 45 S, 5 E, 5 W (Huge main highway, tiny side road)
                direction = random.choices([NORTH, SOUTH, EAST, WEST], weights=[45, 45, 5, 5])[0]
                vehicles.append(Vehicle(direction))
                logger.log_spawn()

        vehicles_by_dir = [[] for _ in range(4)]
        for v in vehicles: vehicles_by_dir[v.direction].append(v)
            
        tl_controller.update(vehicles_by_dir)
        light_states = tl_controller.get_light_states()

        for v in vehicles:
            ahead = None
            lane_mates = vehicles_by_dir[v.direction]
            idx = lane_mates.index(v)
            if idx > 0: ahead = lane_mates[idx-1]
            v.update_physics(light_states[v.direction], ahead)

        active_vehicles = []
        for v in vehicles:
            if -100 < v.x < WIDTH+100 and -100 < v.y < HEIGHT+100:
                active_vehicles.append(v)
            else:
                logger.log_cleared_vehicle(v)
        vehicles = active_vehicles

        # AUTO-SHUTDOWN WHEN BENCHMARK IS COMPLETE
        if logger.total_cleared >= TOTAL_CARS:
            running = False

        # RENDERING
        screen.fill((20, 100, 20))
        pygame.draw.rect(screen, GRAY, (0, CENTER_Y - ROAD_WIDTH//2, WIDTH, ROAD_WIDTH)) 
        pygame.draw.rect(screen, GRAY, (CENTER_X - ROAD_WIDTH//2, 0, ROAD_WIDTH, HEIGHT)) 
        for v in vehicles: v.draw(screen)
        for i, pos in enumerate([(CENTER_X + 80, CENTER_Y - 150), (CENTER_X - 80, CENTER_Y + 150), 
                                 (CENTER_X - 150, CENTER_Y - 80), (CENTER_X + 150, CENTER_Y + 80)]):
            col = GREEN if light_states[i] == "GREEN" else YELLOW if light_states[i] == "YELLOW" else RED
            pygame.draw.circle(screen, col, pos, 15)

        draw_ui_panel(screen, ui_font, ui_title, sim_hour, intensity_factor, tl_controller, static_params, logger)
        pygame.display.flip()
        
    logger.print_final_report()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()