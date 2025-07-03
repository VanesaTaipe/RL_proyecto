
import os
import subprocess
import threading
import time
import sys
from datetime import datetime

# Configurar entorno desde el inicio
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

class DualSnakeRunner:
    def __init__(self):
        self.dqn_process = None
        self.qlearning_process = None
        self.start_time = None
        
    def run_dqn_script(self):
        """Ejecuta el script DQN con control por secuencia de comida"""
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando DQN...")
            
            # Código DQN actualizado
            dqn_code = '''# SOLUCION PARA ERROR OPENMP
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pickle, random, numpy as np, pygame, torch, time
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

# Configuracion inicial
random.seed(SEED_BASE)
np.random.seed(SEED_BASE)
torch.manual_seed(SEED_BASE)

# Modelo DQN - Forzar CPU
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

# --- EJECUTAR EPISODIOS ---
print(f"\\nDQN: Iniciando {N_GAMES} episodios...")

for i in range(N_GAMES):
    print(f"\\nDQN Episodio {i+1}/{N_GAMES}")
    
    # CALCULAR SEMILLA UNICA PARA ESTE EPISODIO
    episode_seed = SEED_BASE + i
    print(f"   Semilla: {episode_seed}")
    
    # CREAR CONTROLADOR DE SECUENCIA CENTRALIZADO
    food_controller = FoodSequenceController(episode_seed, MAX_FOODS, BOARD)
    
    # CREAR JUEGO CON CONTROLADOR CENTRALIZADO
    controlled_game = ControlledSnakeGameDQN(
        w=BOARD*BLOCK, 
        h=BOARD*BLOCK,
        head_origin=head_origin,
        food_controller=food_controller
    )
    
    # Variables del episodio
    done = False
    steps = 0
    foods_eaten = 0
    episode_start = time.time()
    
    # Loop principal del episodio
    while not done:
        # Eventos de pygame
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                exit(0)
        
        # Obtener estado y accion
        state = get_state(controlled_game)
        action = get_action(state)
        
        # Ejecutar accion
        _, done, length = controlled_game.play_step(action)
        steps += 1
        
        # Limite de pasos para evitar loops infinitos
        if steps > 1000:
            print(f"   Limite de pasos alcanzado, terminando episodio")
            done = True
        
        # Limite de comidas
        if controlled_game.get_current_food_number() > MAX_FOODS:
            print(f"   Limite de comidas alcanzado ({MAX_FOODS})")
            done = True
        
        # Detectar si comio
        if length > foods_eaten + 1:
            foods_eaten = length - 1
            current_food_num = controlled_game.get_current_food_number() - 1
            food_pos = (int(controlled_game.food.x/BLOCK), int(controlled_game.food.y/BLOCK))
            print(f"   Comida #{current_food_num} comida en {food_pos} - Longitud: {length}")
        
        # Dibujar
        draw(controlled_game)
        clock.tick(10)
    
    # Resumen del episodio
    episode_duration = time.time() - episode_start
    print(f"DQN Episodio {i+1} completado:")
    print(f"   Score final: {length}")
    print(f"   Pasos totales: {steps}")
    print(f"   Comidas comidas: {foods_eaten}")
    print(f"   Duracion: {episode_duration:.2f}s")

pygame.quit()
print(f"\\nDQN completado exitosamente!")
'''
            
            # Escribir y ejecutar
            with open('temp_dqn.py', 'w', encoding='utf-8') as f:
                f.write(dqn_code)
            
            result = subprocess.run([sys.executable, 'temp_dqn.py'], 
                                  capture_output=True, text=True)
            
            if result.stdout:
                for line in result.stdout.split('\\n'):
                    if line.strip():
                        print(f"[DQN] {line}")
            
            if result.stderr:
                print(f"[DQN ERROR] {result.stderr}")
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] DQN terminado")
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error en DQN: {e}")
        finally:
            if os.path.exists('temp_dqn.py'):
                os.remove('temp_dqn.py')
    
    def run_qlearning_script(self):
        """Ejecuta el script Q-Learning con control por secuencia de comida"""
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando Q-Learning...")
            
            # Código Q-Learning actualizado
            qlearning_code = '''import pickle, random, numpy as np, pygame, time
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
MAX_STEPS_PER_EPISODE = 1000
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

# Configuracion inicial
random.seed(SEED_BASE)
np.random.seed(SEED_BASE)

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
print(f"\\nQ-Learning: Iniciando {N_GAMES} episodios...")

# Crear agente anti-loop
agent = AntiLoopAgent(Q, epsilon=EPSILON)

for i in range(N_GAMES):
    print(f"\\nQ-Learning Episodio {i+1}/{N_GAMES}")
    
    # CALCULAR SEMILLA UNICA PARA ESTE EPISODIO (IDENTICA A DQN)
    episode_seed = SEED_BASE + i
    print(f"   Semilla: {episode_seed}")
    
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
    
    # Variables del episodio
    done = False
    steps = 0
    foods_eaten = 0
    episode_start = time.time()
    last_len = 1
    
    # Loop principal del episodio
    while not done and steps < MAX_STEPS_PER_EPISODE:
        # Eventos de pygame
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                exit(0)
        
        # SELECCION DE ACCION ANTI-LOOP
        action = agent.select_action(controlled_game, state)
        
        # Ejecutar accion
        new_state, reward, done, score = controlled_game.makeMove(action)
        steps += 1
        
        # Limite de comidas usando controlador centralizado
        if controlled_game.get_current_food_number() > MAX_FOODS:
            print(f"   Limite de comidas alcanzado ({MAX_FOODS})")
            done = True
        
        # DETECCION DE PROGRESO
        if score > last_len:
            foods_eaten += 1
            current_food_num = controlled_game.get_current_food_number() - 1
            food_pos = controlled_game.foodIndex
            food_pos_xy = (food_pos[1], food_pos[0])
            print(f"   Comida #{current_food_num} comida en {food_pos_xy} - Longitud: {score}")
            last_len = score
        
        # Log cada 100 pasos para monitoreo
        if steps % 100 == 0:
            current_pos = controlled_game.snake.getHead().getPosition()
            repeat_count = agent.position_history[current_pos]
            current_food_num = controlled_game.get_current_food_number()
            print(f"   Paso {steps}: Score={score}, Pos={current_pos}, Comida #{current_food_num}, Repeticiones={repeat_count}")
        
        # Actualizar estado
        state = new_state
        
        draw(controlled_game)
        clock.tick(5)  
    
    # Finalizacion del episodio
    if steps >= MAX_STEPS_PER_EPISODE:
        print(f"   LIMITE DE PASOS ALCANZADO: Terminando episodio en paso {steps}")
        done = True
    
    episode_duration = time.time() - episode_start
    
    # Resumen del episodio
    print(f"Q-Learning Episodio {i+1} completado:")
    print(f"   Score final: {score}")
    print(f"   Pasos totales: {steps}/{MAX_STEPS_PER_EPISODE}")
    print(f"   Comidas comidas: {foods_eaten}")
    print(f"   Duracion: {episode_duration:.2f}s")
    print(f"   Posiciones unicas visitadas: {len(agent.position_history)}")

pygame.quit()
print(f"\\nQ-Learning completado exitosamente!")
'''
            
            # Escribir y ejecutar
            with open('temp_qlearning.py', 'w', encoding='utf-8') as f:
                f.write(qlearning_code)
            
            result = subprocess.run([sys.executable, 'temp_qlearning.py'], 
                                  capture_output=True, text=True)
            
            if result.stdout:
                for line in result.stdout.split('\\n'):
                    if line.strip():
                        print(f"[Q-Learning] {line}")
            
            if result.stderr:
                print(f"[Q-Learning ERROR] {result.stderr}")
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Q-Learning terminado")
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error en Q-Learning: {e}")
        finally:
            if os.path.exists('temp_qlearning.py'):
                os.remove('temp_qlearning.py')
    
    def run_parallel(self):
        """Ejecuta ambos algoritmos en paralelo"""
        print("=" * 80)
        print("EJECUTOR DUAL - SNAKE DQN vs Q-LEARNING")
        print("=" * 80)
        print(f"Iniciando ejecucion paralela: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 80)
        
        self.start_time = time.time()
        
        # Crear hilos
        dqn_thread = threading.Thread(target=self.run_dqn_script, name="DQN")
        qlearning_thread = threading.Thread(target=self.run_qlearning_script, name="Q-Learning")
        
        # Iniciar
        dqn_thread.start()
        time.sleep(2)  # Pequeño delay
        qlearning_thread.start()
        
        # Esperar
        dqn_thread.join()
        qlearning_thread.join()
        
        total_time = time.time() - self.start_time
        
        print("\\n" + "=" * 80)
        print("EJECUCION DUAL COMPLETADA")
        print("=" * 80)
        print(f"Tiempo total: {total_time:.2f} segundos")
        print("Ambos algoritmos ejecutados exitosamente")
        print("=" * 80)
    
    def run_sequential(self):
        """Ejecuta de forma secuencial"""
        print("=" * 80)
        print("EJECUTOR SECUENCIAL")
        print("=" * 80)
        
        self.start_time = time.time()
        
        print("\\n1. Ejecutando DQN...")
        self.run_dqn_script()
        
        print("\\n2. Ejecutando Q-Learning...")
        self.run_qlearning_script()
        
        total_time = time.time() - self.start_time
        print(f"\\nTiempo total: {total_time:.2f} segundos")

def main():
    """Funcion principal"""
    runner = DualSnakeRunner()
    
    print("\\nEJECUTOR DUAL SNAKE")
    print("=" * 40)
    print("1. Paralelo")
    print("2. Secuencial") 
    print("3. Solo DQN")
    print("4. Solo Q-Learning")
    print("5. Salir")
    print("-" * 40)
    
    try:
        choice = input("Opcion (1-5): ").strip()
        
        if choice == '1':
            runner.run_parallel()
        elif choice == '2':
            runner.run_sequential()
        elif choice == '3':
            runner.run_dqn_script()
        elif choice == '4':
            runner.run_qlearning_script()
        elif choice == '5':
            print("Hasta luego!")
            return
        else:
            print("Opcion invalida")
    except KeyboardInterrupt:
        print("\\nInterrumpido por el usuario")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()