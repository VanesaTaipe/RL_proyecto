import pickle, random, numpy as np, pygame, time
from collections import Counter
from Snake import SnakeGame
from foosd import FoodSequenceController, ControlledSnakeGameQL

# --- Parametros ---
BOARD = 16
CELL = 20
N_GAMES = 10
SEED_BASE = 42
MAX_FOODS = 20  # Maximo de comidas por episodio

# CONFIGURACION POR SECUENCIA DE COMIDA (IDENTICA A DQN)
HEAD_ORIGIN_GRID = (BOARD//2 - 4, BOARD//2)    # (4, 8) - cabeza a la izquierda
FIRST_FOOD_ORIGIN_GRID = (BOARD//2, BOARD//2)  # (8, 8) - primera comida

# Parametros anti-loop
MAX_STEPS_PER_EPISODE = 100
EPSILON = 0.1
STUCK_THRESHOLD = 50
POSITION_REPEAT_LIMIT = 5

print(f"Configuracion Q-Learning POR SECUENCIA DE COMIDA:")
print(f"   Tablero: {BOARD}x{BOARD}")
print(f"   Cabeza inicial: {HEAD_ORIGIN_GRID}")
print(f"   Primera comida: {FIRST_FOOD_ORIGIN_GRID}")
print(f"   Semilla base: {SEED_BASE}")
print(f"   Numero de juegos: {N_GAMES}")
print(f"   Limite de pasos: {MAX_STEPS_PER_EPISODE}")
print(f"   Maximo de comidas: {MAX_FOODS}")
print(f"   SECUENCIA: Comida #N sera identica para ambos algoritmos")


# Cargar Q-table
Q = pickle.load(open("qtable.pkl", "rb"))

# Pygame
pygame.init()
screen = pygame.display.set_mode((BOARD*CELL, BOARD*CELL))
clock = pygame.time.Clock()

def draw(game):
    screen.fill((0, 0, 0))
    head_pos = game.snake.getHead().getPosition()
    node = game.snake.getTail()
    while node:
        x, y = node.getPosition()
        is_head = (x, y) == head_pos
        color = (0, 255, 0) if is_head else (0, 0, 255)
        pygame.draw.rect(screen, color, (x*CELL, y*CELL, CELL, CELL))
        node = node.parent
    
    fy, fx = game.foodIndex
    pygame.draw.rect(screen, (255, 0, 0), (fx*CELL, fy*CELL, CELL, CELL))
    pygame.display.flip()

def calculate_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class AntiLoopAgent:
    """Agente Q-Learning con mecanismos anti-loop"""
    
    def __init__(self, q_table, epsilon=0.1):
        self.Q = q_table
        self.epsilon = epsilon
        self.position_history = Counter()
        self.last_positions = []
        self.steps_without_progress = 0
        self.last_score = 0
        
    def reset_episode(self):
        """Reset para nuevo episodio"""
        self.position_history.clear()
        self.last_positions.clear()
        self.steps_without_progress = 0
        self.last_score = 0
    
    def select_action(self, game, state):
        """Seleccion de accion con anti-loop"""
        current_pos = game.snake.getHead().getPosition()
        current_score = game.length
        
        # Actualizar historial de posiciones
        self.position_history[current_pos] += 1
        self.last_positions.append(current_pos)
        
        # Mantener solo las ultimas 20 posiciones
        if len(self.last_positions) > 20:
            old_pos = self.last_positions.pop(0)
            self.position_history[old_pos] -= 1
            if self.position_history[old_pos] <= 0:
                del self.position_history[old_pos]
        
        # Verificar progreso
        if current_score > self.last_score:
            self.steps_without_progress = 0
            self.last_score = current_score
        else:
            self.steps_without_progress += 1
        
        # ESTRATEGIA ANTI-LOOP
        
        # 1. Si esta muy atascado, accion completamente aleatoria
        if self.steps_without_progress > STUCK_THRESHOLD:
            valid_actions = [a for a in range(4) if game.checkValid(a)]
            if valid_actions:
                action = random.choice(valid_actions)
                return action
        
        # 2. Si esta en una posicion muy repetida, evitarla
        if self.position_history[current_pos] > POSITION_REPEAT_LIMIT:
            valid_actions = []
            for a in range(4):
                if game.checkValid(a):
                    next_pos = self._get_next_position(current_pos, a)
                    # Preferir posiciones menos visitadas
                    if self.position_history[next_pos] < POSITION_REPEAT_LIMIT:
                        valid_actions.append(a)
            
            if valid_actions:
                action = random.choice(valid_actions)
                return action
        
        # 3. Exploracion epsilon-greedy mejorada
        if random.random() < self.epsilon:
            # Exploracion inteligente: evitar posiciones muy visitadas
            valid_actions = []
            for a in range(4):
                if game.checkValid(a):
                    next_pos = self._get_next_position(current_pos, a)
                    if self.position_history[next_pos] <= 2:  # No muy visitada
                        valid_actions.append(a)
            
            if valid_actions:
                action = random.choice(valid_actions)
                return action
            else:
                # Si todas estan muy visitadas, exploracion normal
                valid_actions = [a for a in range(4) if game.checkValid(a)]
                if valid_actions:
                    return random.choice(valid_actions)
        
        # 4. Explotacion normal con Q-table
        q_values = self.Q[state]
        valid_actions = [a for a in range(4) if game.checkValid(a)]
        
        if valid_actions:
            # Elegir la mejor accion valida
            best_action = valid_actions[0]
            best_q = q_values[best_action]
            for action in valid_actions:
                if q_values[action] > best_q:
                    best_action = action
                    best_q = q_values[action]
            return best_action
        
        # Ultimo recurso: cualquier accion valida
        for a in range(4):
            if game.checkValid(a):
                return a
        
        return 0  # Fallback
    
    def _get_next_position(self, current_pos, action):
        """Calcula la siguiente posicion dada una accion"""
        x, y = current_pos
        if action == 0:  # UP
            return (x, y - 1)
        elif action == 1:  # RIGHT
            return (x + 1, y)
        elif action == 2:  # DOWN
            return (x, y + 1)
        elif action == 3:  # LEFT
            return (x - 1, y)
        return current_pos

# --- EJECUTAR EPISODIOS ---
results = {}
total_start_time = time.time()
verification_log = {}

# Crear agente anti-loop
agent = AntiLoopAgent(Q, epsilon=EPSILON)

for i in range(N_GAMES):
    print(f"\nIniciando episodio Q-Learning {i+1}/{N_GAMES}")
    
    # CALCULAR SEMILLA UNICA PARA ESTE EPISODIO (IDENTICA A DQN)
    episode_seed = SEED_BASE + i
    print(f"   Usando semilla: {episode_seed}")
    
    # Reset del agente para nuevo episodio
    agent.reset_episode()
    
    # CREAR CONTROLADOR DE SECUENCIA CENTRALIZADO (IDENTICO A DQN)
    food_controller = FoodSequenceController(episode_seed, MAX_FOODS, BOARD)
    
    # CREAR JUEGO CON CONTROLADOR CENTRALIZADO
    controlled_game = ControlledSnakeGameQL(
        width=BOARD, 
        height=BOARD,
        head_origin=HEAD_ORIGIN_GRID,
        food_controller=food_controller
    )
    
    state = controlled_game.calcStateNum()
    
    # Verificar posiciones iniciales
    initial_food_grid = controlled_game.foodIndex  # (fila, columna)
    initial_head_grid = controlled_game.snake.getHead().getPosition()  # (columna, fila)
    initial_food_xy = (initial_food_grid[1], initial_food_grid[0])  # Convertir a (x, y)
    
    print(f"   Comida inicial (x,y): {initial_food_xy}")
    print(f"   Cabeza inicial (x,y): {initial_head_grid}")
    
    # Variables del episodio
    done = False
    steps = 0
    episode_start = time.time()
    last_food_time = time.time()
    last_len = 1
    frames_nofood = 0
    max_frames_nofood = 0
    foods_eaten = 0
    state_counter = Counter()
    
    # Log para verificacion POR NUMERO DE COMIDA
    episode_food_log = []
    
    # Metricas completas (identicas a DQN)
    metrics = {
        # Metricas basicas
        "final_score": 0,
        "total_steps": 0,
        "episode_duration": 0,
        "foods_eaten": 0,
        
        # Trayectorias
        "head_trajectory": [],
        "food_trajectory": [],
        "length_evolution": [],
        
        # Metricas de eficiencia
        "steps_per_food": [],
        "time_per_food": [],
        "frames_between_foods": [],
        "max_frames_without_food": 0,
        "average_frames_between_foods": 0,
        
        # Metricas de comportamiento
        "state_repetitions": 0,
        "max_state_repetition": 0,
        "unique_states_visited": 0,
        "exploration_ratio": 0,
        
        # Metricas de posicion
        "death_position": None,
        "death_position_grid": None,
        "snake_length_at_death": 0,
        "distance_to_food_at_death": 0,
        
        # Metricas avanzadas
        "food_positions": [],
        "wall_collisions": 0,
        "self_collisions": 0,
        "direction_changes": 0,
        "last_direction": None,
        
        # Estados especificos
        "initial_state": None,
        "final_state": None,
        "states_sequence": [],
        
        # METRICAS PARA CONTROL POR SECUENCIA
        "episode_seed": episode_seed,
        "food_sequence": food_controller.get_sequence(),
        "control_method": "food_sequence"
    }
    
    # Estado inicial
    metrics["initial_state"] = state
    last_action = None
    
    print(f"   Iniciando con limite de {MAX_STEPS_PER_EPISODE} pasos y {MAX_FOODS} comidas")
    
    # --- LOOP PRINCIPAL CON CONTROL POR SECUENCIA ---
    while not done and steps < MAX_STEPS_PER_EPISODE:
        # Eventos de pygame
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                exit(0)
        
        # OBTENER POSICIONES ACTUALES DEL CONTROLADOR CENTRALIZADO
        head_pos = controlled_game.snake.getHead().getPosition()  # (x, y)
        food_pos = controlled_game.foodIndex  # (fila, columna) -> convertir a (x, y)
        food_pos_xy = (food_pos[1], food_pos[0])  # Convertir (fila, col) a (x, y)
        current_food_number = controlled_game.get_current_food_number()
        
        # REGISTRAR PARA VERIFICACION POR NUMERO DE COMIDA
        episode_food_log.append({
            "step": steps + 1,
            "food_number": current_food_number,
            "food_position": food_pos_xy
        })
        
        # Guardar estado
        state_tuple = int(state)
        state_counter[state_tuple] += 1
        
        # SELECCION DE ACCION ANTI-LOOP
        action = agent.select_action(controlled_game, state)
        
        # Detectar cambios de direccion
        if last_action is not None and last_action != action:
            metrics["direction_changes"] += 1
        last_action = action
        
        # Ejecutar accion
        new_state, reward, done, score = controlled_game.makeMove(action)
        steps += 1
        
        # Limite de comidas usando controlador centralizado
        if controlled_game.get_current_food_number() > MAX_FOODS:
            print(f"   Limite de comidas alcanzado ({MAX_FOODS}), terminando episodio")
            done = True
        
        # DETECCION DE PROGRESO
        if score > last_len:
            foods_eaten += 1
            food_time = time.time() - last_food_time
            steps_for_food = frames_nofood + 1
            
            metrics["time_per_food"].append(food_time)
            metrics["steps_per_food"].append(steps_for_food)
            metrics["frames_between_foods"].append(frames_nofood)
            
            if frames_nofood > max_frames_nofood:
                max_frames_nofood = frames_nofood
            
            # Reset contadores
            last_food_time = time.time()
            frames_nofood = 0
            last_len = score
            
            print(f"   Comida #{current_food_number} en {food_pos_xy} - Longitud: {score}, Pasos: {steps_for_food}")
        else:
            frames_nofood += 1
        
        # Log cada 100 pasos para monitoreo
        if steps % 100 == 0:
            current_pos = controlled_game.snake.getHead().getPosition()
            repeat_count = agent.position_history[current_pos]
            print(f"   Paso {steps}: Score={score}, Pos={current_pos}, Comida #{current_food_number}, Repeticiones={repeat_count}")
        
        # Actualizar trayectorias
        metrics["head_trajectory"].append(head_pos)
        metrics["food_trajectory"].append(food_pos_xy)
        metrics["length_evolution"].append(score)
        metrics["states_sequence"].append(state_tuple)
        
        # Guardar posicion de comida cuando aparece nueva
        if food_pos_xy not in metrics["food_positions"]:
            metrics["food_positions"].append(food_pos_xy)
        
        # Actualizar estado
        state = new_state
        
        draw(controlled_game)
        clock.tick(5)  
    
    # --- FINALIZACION FORZADA SI ES NECESARIO ---
    if steps >= MAX_STEPS_PER_EPISODE:
        print(f"   LIMITE DE PASOS ALCANZADO: Terminando episodio en paso {steps}")
        done = True
    
    episode_duration = time.time() - episode_start
    
    # GUARDAR LOG DE VERIFICACION POR SECUENCIA CON CLAVES CORRECTAS
    verification_log[f"QL_episode_{i+1}"] = episode_food_log
    
    # Calcular metricas finales (identico al DQN)
    metrics["final_score"] = score
    metrics["total_steps"] = steps
    metrics["episode_duration"] = episode_duration
    metrics["foods_eaten"] = foods_eaten
    metrics["max_frames_without_food"] = max_frames_nofood
    metrics["state_repetitions"] = sum(count - 1 for count in state_counter.values() if count > 1)
    metrics["max_state_repetition"] = max(state_counter.values()) if state_counter else 0
    metrics["unique_states_visited"] = len(state_counter)
    metrics["exploration_ratio"] = len(state_counter) / steps if steps > 0 else 0
    
    # Posicion de muerte
    final_head_pos = controlled_game.snake.getHead().getPosition()
    metrics["death_position"] = final_head_pos  # (x, y)
    metrics["death_position_grid"] = final_head_pos  # Ya esta en grilla
    metrics["snake_length_at_death"] = score
    
    # Distancia a la comida al morir
    final_food_pos = controlled_game.foodIndex
    final_food_xy = (final_food_pos[1], final_food_pos[0])
    metrics["distance_to_food_at_death"] = calculate_distance(final_head_pos, final_food_xy)
    
    # Estados
    metrics["final_state"] = state
    
    # Promedios
    if metrics["frames_between_foods"]:
        metrics["average_frames_between_foods"] = np.mean(metrics["frames_between_foods"])
    
    # GUARDAR RESULTADOS CON CLAVES CORRECTAS
    results[f"QL_episode_{i+1}"] = metrics
    
    # Resumen del episodio
    efficiency = (foods_eaten / steps * 100) if steps > 0 else 0
    max_food_reached = controlled_game.get_current_food_number()
    print(f"Episodio {i+1} completado:")
    print(f"   Score: {score} | Pasos: {steps}/{MAX_STEPS_PER_EPISODE} | Duracion: {episode_duration:.2f}s")
    print(f"   Comidas: {foods_eaten}/{MAX_FOODS} | Maxima comida alcanzada: #{max_food_reached}")
    print(f"   Eficiencia: {efficiency:.2f}% | Estados unicos: {len(state_counter)}")
    print(f"   Posiciones unicas visitadas: {len(agent.position_history)}")

# --- FINALIZACION ---
total_duration = time.time() - total_start_time
pygame.quit()

# GUARDAR ARCHIVOS CON NOMBRES CORRECTOS PARA VALIDACION
print(f"\nGUARDANDO RESULTADOS...")
pickle.dump(results, open("metrics_qlearning_food_sequence.pkl", "wb"))
pickle.dump(verification_log, open("verification_qlearning_food_sequence.pkl", "wb"))
print("metrics_qlearning_food_sequence.pkl guardado")
print("verification_qlearning_food_sequence.pkl guardado")

# Resumen final
print(f"\nRESUMEN FINAL Q-LEARNING (CONTROL POR SECUENCIA DE COMIDA):")
print(f"   Total de episodios: {N_GAMES}")
print(f"   Tiempo total: {total_duration:.2f}s")
print(f"   Tiempo promedio por episodio: {total_duration/N_GAMES:.2f}s")

# Estadisticas globales
all_scores = [results[f"QL_episode_{i+1}"]["final_score"] for i in range(N_GAMES)]
all_steps = [results[f"QL_episode_{i+1}"]["total_steps"] for i in range(N_GAMES)]
all_foods = [results[f"QL_episode_{i+1}"]["foods_eaten"] for i in range(N_GAMES)]

print(f"   Score promedio: {np.mean(all_scores):.2f} ± {np.std(all_scores):.2f}")
print(f"   Score maximo: {max(all_scores)}")
print(f"   Pasos promedio: {np.mean(all_steps):.0f} ± {np.std(all_steps):.0f}")
print(f"   Comidas promedio: {np.mean(all_foods):.1f} ± {np.std(all_foods):.1f}")

print(f"\nARCHIVOS GUARDADOS:")
print(f"   metrics_qlearning_food_sequence.pkl")
print(f"   verification_qlearning_food_sequence.pkl")
print(f"\nCONTROL POR SECUENCIA: Comida #N sera identica para ambos algoritmos")
print(f"Maximo de {MAX_FOODS} comidas por episodio")
