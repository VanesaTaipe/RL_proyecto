import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pickle, numpy as np, pygame, torch, time
from model import Linear_QNet
from snake_gameai import SnakeGameAI
from collections import namedtuple, Counter
from foosd import FoodSequenceController, ControlledSnakeGameDQN

# --- Parametros ---
BOARD = 16
BLOCK = 20
N_GAMES = 10
SEED_BASE = 42
MAX_FOODS = 20  # Maximo de comidas por episodio

# CONFIGURACION POR SECUENCIA DE COMIDA
HEAD_ORIGIN_GRID = (BOARD//2 - 4, BOARD//2)    # (4, 8) - cabeza a la izquierda
FIRST_FOOD_ORIGIN_GRID = (BOARD//2, BOARD//2)  # (8, 8) - primera comida

print(f"Configuracion DQN POR SECUENCIA DE COMIDA:")
print(f"   Tablero: {BOARD}x{BOARD}")
print(f"   Cabeza inicial: {HEAD_ORIGIN_GRID}")
print(f"   Primera comida: {FIRST_FOOD_ORIGIN_GRID}")
print(f"   Semilla base: {SEED_BASE}")
print(f"   Numero de juegos: {N_GAMES}")
print(f"   Maximo de comidas: {MAX_FOODS}")
print(f"   SECUENCIA: Comida #N sera identica para ambos algoritmos")

device = torch.device("cpu")
model = Linear_QNet(11, 256, 3).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
print(f"   Dispositivo: {device}")

# Pygame
pygame.init()
screen = pygame.display.set_mode((BOARD*BLOCK, BOARD*BLOCK))
clock = pygame.time.Clock()

Point = namedtuple('Point', 'x y')

# Convertir a pixeles
head_origin = Point(HEAD_ORIGIN_GRID[0]*BLOCK, HEAD_ORIGIN_GRID[1]*BLOCK)

def get_state(game):
    head = game.snake[0]
    def p(dx, dy): return Point(head.x + dx*BLOCK, head.y + dy*BLOCK)
    
    dir_l = game.direction.name == "LEFT"
    dir_r = game.direction.name == "RIGHT"
    dir_u = game.direction.name == "UP"
    dir_d = game.direction.name == "DOWN"
    
    return list(map(int, [
        # Peligro adelante
        (dir_r and game.is_collision(p(1, 0))) or
        (dir_l and game.is_collision(p(-1, 0))) or
        (dir_u and game.is_collision(p(0, -1))) or
        (dir_d and game.is_collision(p(0, 1))),
        
        # Peligro derecha
        (dir_u and game.is_collision(p(1, 0))) or
        (dir_d and game.is_collision(p(-1, 0))) or
        (dir_l and game.is_collision(p(0, -1))) or
        (dir_r and game.is_collision(p(0, 1))),
        
        # Peligro izquierda
        (dir_d and game.is_collision(p(1, 0))) or
        (dir_u and game.is_collision(p(-1, 0))) or
        (dir_r and game.is_collision(p(0, -1))) or
        (dir_l and game.is_collision(p(0, 1))),
        
        # Direccion actual
        dir_l, dir_r, dir_u, dir_d,
        
        # Posicion relativa de la comida
        game.food.x < head.x,  # Comida a la izquierda
        game.food.x > head.x,  # Comida a la derecha
        game.food.y < head.y,  # Comida arriba
        game.food.y > head.y   # Comida abajo
    ]))

def get_action(state):
    s = torch.tensor(state, dtype=torch.float).to(device)
    out = model(s)
    idx = torch.argmax(out).item()
    return [1 if i == idx else 0 for i in range(3)]

def draw(game):
    screen.fill((0, 0, 0))
    for idx, pt in enumerate(game.snake):
        color = (0, 255, 0) if idx == 0 else (0, 0, 255)
        pygame.draw.rect(screen, color, (pt.x, pt.y, BLOCK, BLOCK))
    pygame.draw.rect(screen, (255, 0, 0), (game.food.x, game.food.y, BLOCK, BLOCK))
    pygame.display.flip()

def calculate_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# --- EJECUTAR EPISODIOS ---
results = {}
total_start_time = time.time()
verification_log = {}

for i in range(N_GAMES):
    print(f"\nIniciando episodio DQN {i+1}/{N_GAMES}")
    
    # CALCULAR SEMILLA UNICA PARA ESTE EPISODIO
    episode_seed = SEED_BASE + i
    print(f"   Usando semilla: {episode_seed}")
    
    # CREAR CONTROLADOR DE SECUENCIA CENTRALIZADO
    food_controller = FoodSequenceController(episode_seed, MAX_FOODS, BOARD)
    
    # CREAR JUEGO CON CONTROLADOR CENTRALIZADO
    controlled_game = ControlledSnakeGameDQN(
        w=BOARD*BLOCK, 
        h=BOARD*BLOCK,
        head_origin=head_origin,
        food_controller=food_controller  
    )
    
    # Verificar posiciones iniciales
    initial_food_grid = (int(controlled_game.food.x/BLOCK), int(controlled_game.food.y/BLOCK))
    initial_head_grid = (int(controlled_game.snake[0].x/BLOCK), int(controlled_game.snake[0].y/BLOCK))
    print(f"   Comida inicial (grilla): {initial_food_grid}")
    print(f"   Cabeza inicial (grilla): {initial_head_grid}")
    
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
    
    # LOG PARA VERIFICACION POR NUMERO DE COMIDA
    episode_food_log = []
    
    # Metricas completas
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
    initial_state = get_state(controlled_game)
    metrics["initial_state"] = tuple(initial_state)
    last_action = None
    
    # --- LOOP PRINCIPAL DEL EPISODIO ---
    while not done:
        # Eventos de pygame
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                exit(0)
        
        # OBTENER POSICIONES ACTUALES DEL CONTROLADOR CENTRALIZADO
        head_pos_grid = (int(controlled_game.snake[0].x/BLOCK), int(controlled_game.snake[0].y/BLOCK))
        food_pos_grid = (int(controlled_game.food.x/BLOCK), int(controlled_game.food.y/BLOCK))
        current_food_number = controlled_game.get_current_food_number()
        
        # REGISTRAR PARA VERIFICACION POR NUMERO DE COMIDA
        episode_food_log.append({
            "step": steps + 1,
            "food_number": current_food_number,
            "food_position": food_pos_grid
        })
        
        # Obtener estado y accion
        state = get_state(controlled_game)
        state_tuple = tuple(state)
        state_counter[state_tuple] += 1
        action = get_action(state)
        
        # Detectar cambios de direccion
        current_action = action.index(1) if 1 in action else None
        if last_action is not None and last_action != current_action:
            metrics["direction_changes"] += 1
        last_action = current_action
        
        # Ejecutar accion
        _, done, length = controlled_game.play_step(action)
        steps += 1
        
        # Limite de pasos para evitar loops infinitos
        if steps > 100:
            print(f"   Limite de pasos alcanzado ({steps}), terminando episodio")
            done = True
        
        # Limite de comidas usando controlador centralizado
        if controlled_game.get_current_food_number() > MAX_FOODS:
            print(f"   Limite de comidas alcanzado ({MAX_FOODS}), terminando episodio")
            done = True
        
        # Actualizar trayectorias
        metrics["head_trajectory"].append(head_pos_grid)
        metrics["food_trajectory"].append(food_pos_grid)
        metrics["length_evolution"].append(length)
        metrics["states_sequence"].append(state_tuple)
        
        # Detectar si comio
        if length > last_len:
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
            last_len = length
            
            print(f"   Comida #{current_food_number} en {food_pos_grid} - Longitud: {length}, Pasos: {steps_for_food}")
        else:
            frames_nofood += 1
        
        # Guardar posicion de comida cuando aparece nueva
        if food_pos_grid not in metrics["food_positions"]:
            metrics["food_positions"].append(food_pos_grid)
        
        # Dibujar
        draw(controlled_game)
        clock.tick(5)  
    
    # --- FINALIZACION DEL EPISODIO ---
    episode_duration = time.time() - episode_start
    
    # Calcular metricas finales
    metrics["final_score"] = length
    metrics["total_steps"] = steps
    metrics["episode_duration"] = episode_duration
    metrics["foods_eaten"] = foods_eaten
    metrics["max_frames_without_food"] = max_frames_nofood
    metrics["state_repetitions"] = sum(count - 1 for count in state_counter.values() if count > 1)
    metrics["max_state_repetition"] = max(state_counter.values()) if state_counter else 0
    metrics["unique_states_visited"] = len(state_counter)
    metrics["exploration_ratio"] = len(state_counter) / steps if steps > 0 else 0
    
    # Posicion de muerte
    final_head_pos_grid = (int(controlled_game.snake[0].x/BLOCK), int(controlled_game.snake[0].y/BLOCK))
    metrics["death_position"] = final_head_pos_grid
    metrics["death_position_grid"] = final_head_pos_grid
    metrics["snake_length_at_death"] = length
    
    # Distancia a la comida al morir
    final_food_pos_grid = (int(controlled_game.food.x/BLOCK), int(controlled_game.food.y/BLOCK))
    metrics["distance_to_food_at_death"] = calculate_distance(final_head_pos_grid, final_food_pos_grid)
    
    # Estados
    final_state = get_state(controlled_game) if not done else state
    metrics["final_state"] = tuple(final_state)
    
    # Promedios
    if metrics["frames_between_foods"]:
        metrics["average_frames_between_foods"] = np.mean(metrics["frames_between_foods"])
    
    # GUARDAR RESULTADOS CON CLAVES CORRECTAS
    verification_log[f"DQN_episode_{i+1}"] = episode_food_log
    results[f"DQN_episode_{i+1}"] = metrics
    
    # Resumen del episodio
    efficiency = (foods_eaten / steps * 100) if steps > 0 else 0
    max_food_reached = controlled_game.get_current_food_number()
    print(f"Episodio {i+1} completado:")
    print(f"   Score: {length} | Pasos: {steps} | Duracion: {episode_duration:.2f}s")
    print(f"   Comidas: {foods_eaten}/{MAX_FOODS} | Maxima comida alcanzada: #{max_food_reached}")
    print(f"   Eficiencia: {efficiency:.2f}% | Estados unicos: {len(state_counter)}")

# --- FINALIZACION ---
total_duration = time.time() - total_start_time
pygame.quit()

# GUARDAR ARCHIVOS CON NOMBRES CORRECTOS PARA VALIDACION
print(f"\nGUARDANDO RESULTADOS...")
pickle.dump(results, open("metrics_dqn_food_sequence.pkl", "wb"))
pickle.dump(verification_log, open("verification_dqn_food_sequence.pkl", "wb"))
print("metrics_dqn_food_sequence.pkl guardado")
print("verification_dqn_food_sequence.pkl guardado")

# Resumen final
print(f"\nRESUMEN FINAL DQN (CONTROL POR SECUENCIA DE COMIDA):")
print(f"   Total de episodios: {N_GAMES}")
print(f"   Tiempo total: {total_duration:.2f}s")
print(f"   Tiempo promedio por episodio: {total_duration/N_GAMES:.2f}s")

# Estadisticas globales
all_scores = [results[f"DQN_episode_{i+1}"]["final_score"] for i in range(N_GAMES)]
all_steps = [results[f"DQN_episode_{i+1}"]["total_steps"] for i in range(N_GAMES)]
all_foods = [results[f"DQN_episode_{i+1}"]["foods_eaten"] for i in range(N_GAMES)]

print(f"   Score promedio: {np.mean(all_scores):.2f} ± {np.std(all_scores):.2f}")
print(f"   Score maximo: {max(all_scores)}")
print(f"   Pasos promedio: {np.mean(all_steps):.0f} ± {np.std(all_steps):.0f}")
print(f"   Comidas promedio: {np.mean(all_foods):.1f} ± {np.std(all_foods):.1f}")

print(f"\nARCHIVOS GUARDADOS:")
print(f"   metrics_dqn_food_sequence.pkl")
print(f"   verification_dqn_food_sequence.pkl")
print(f"\nCONTROL POR SECUENCIA: Comida #N sera identica para ambos algoritmos")
print(f"Maximo de {MAX_FOODS} comidas por episodio")
