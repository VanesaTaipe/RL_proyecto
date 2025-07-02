import random
import numpy as np

class FoodSequenceController:
    """
    Controlador centralizado que asegura sincronizacion perfecta
    """
    def __init__(self, episode_seed, max_foods=20, board_size=16):
        self.episode_seed = episode_seed
        self.max_foods = max_foods
        self.board_size = board_size
        
        # CONTADOR CENTRALIZADO - Una sola fuente de verdad
        self.current_food_number = 1
        
        # Generar secuencia fija de posiciones de comida
        self.food_sequence = self._generate_food_sequence()
        
        print(f"   Controlador de secuencia inicializado:")
        print(f"   Semilla: {episode_seed}, Max comidas: {max_foods}")
        print(f"   Secuencia: {self.food_sequence[:5]}...")  # Solo mostrar primeras 5
    
    def _generate_food_sequence(self):
        """Genera secuencia fija de 20 posiciones de comida"""
        
        # Guardar estado actual del random
        current_state = random.getstate()
        # Usar semilla especifica para este episodio
        random.seed(self.episode_seed)
        np.random.seed(self.episode_seed)
        
        # Generar secuencia deterministica
        food_sequence = []
        
        # Comida #1 siempre en (8, 8)
        food_sequence.append((8, 8))
        
        # Generar resto de comidas (#2 a #20)
        for i in range(1, self.max_foods):
            x = random.randint(0, self.board_size - 1)
            y = random.randint(0, self.board_size - 1)
            food_sequence.append((x, y))
        
        # Restaurar estado del random
        random.setstate(current_state)
        
        return food_sequence
    
    def get_food_position(self, food_number):
        """
        Obtiene la posicion para un numero especifico de comida
        food_number: 1, 2, 3, ..., 20
        """
        if 1 <= food_number <= len(self.food_sequence):
            return self.food_sequence[food_number - 1]
        else:
            # Si se pasa del limite, repetir la ultima
            return self.food_sequence[-1]
    
    def get_current_food_position(self):
        """
        METODO NUEVO: Obtiene la posicion de la comida actual
        """
        return self.get_food_position(self.current_food_number)
    
    def advance_to_next_food(self):
        """
        METODO NUEVO: Avanza al siguiente numero de comida
        Este metodo debe ser llamado cuando CUALQUIER algoritmo come
        """
        self.current_food_number += 1
        print(f"   Avanzando a comida #{self.current_food_number}")
        return self.get_current_food_position()
    
    def get_current_food_number(self):
        """Retorna el numero actual de comida"""
        return self.current_food_number
    
    def reset_for_new_episode(self):
        """
        METODO NUEVO: Reset para nuevo episodio
        """
        self.current_food_number = 1
        print(f"   Reset: Volviendo a comida #1")
    
    def get_sequence(self):
        """Retorna la secuencia completa para verificacion"""
        return self.food_sequence.copy()

class ControlledSnakeGameDQN:
    """DQN con control centralizado por numero de comida"""
    
    def __init__(self, w, h, head_origin, food_controller):
        from snake_gameai import SnakeGameAI
        from collections import namedtuple
        
        Point = namedtuple('Point', 'x y')
        
        # REFERENCIA AL CONTROLADOR CENTRALIZADO
        self.food_controller = food_controller
        self.block_size = 20
        
        # Crear juego estandar
        self.game = SnakeGameAI(w=w, h=h)
        
        # CONFIGURAR POSICION INICIAL DE LA CABEZA
        self.game.head = Point(head_origin.x, head_origin.y)
        self.game.snake = [self.game.head,
                          Point(self.game.head.x-self.block_size, self.game.head.y),
                          Point(self.game.head.x-(2*self.block_size), self.game.head.y)]
        
        # NO AVANZAR AUTOMATICAMENTE - Solo obtener posicion actual
        food_pos = food_controller.get_current_food_position()
        initial_food_point = Point(food_pos[0] * self.block_size, food_pos[1] * self.block_size)
        self.game.food = initial_food_point
        
        # Sobrescribir el metodo _place__food
        self.game._place__food = self._controlled_place_food
        
        print(f"   DQN iniciado con comida #{food_controller.get_current_food_number()} en {food_pos}")
    
    def _controlled_place_food(self):
        """Colocar siguiente comida usando controlador centralizado"""
        from collections import namedtuple
        Point = namedtuple('Point', 'x y')
        
        # USAR CONTROLADOR CENTRALIZADO
        food_pos = self.food_controller.advance_to_next_food()
        food_point = Point(food_pos[0] * self.block_size, food_pos[1] * self.block_size)
        
        # Verificar colisiones y ajustar si es necesario
        if food_point in self.game.snake:
            safe_pos = self._find_safe_position_near(food_pos)
            food_point = Point(safe_pos[0] * self.block_size, safe_pos[1] * self.block_size)
            print(f"   DQN: Comida #{self.food_controller.get_current_food_number()} reubicada de {food_pos} a {safe_pos}")
        
        self.game.food = food_point
        print(f"   DQN: Comida #{self.food_controller.get_current_food_number()} colocada en {food_pos}")
    
    def _find_safe_position_near(self, target_pos):
        """Encuentra posicion segura cerca de la posicion objetivo"""
        snake_positions_grid = [(int(p.x/self.block_size), int(p.y/self.block_size)) 
                               for p in self.game.snake]
        
        # Buscar en circulos concentricos
        for radius in range(1, 5):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        new_x = target_pos[0] + dx
                        new_y = target_pos[1] + dy
                        
                        if (0 <= new_x < 16 and 0 <= new_y < 16 and
                            (new_x, new_y) not in snake_positions_grid):
                            return (new_x, new_y)
        
        # Ultimo recurso: esquinas
        for corner in [(0, 0), (0, 15), (15, 0), (15, 15)]:
            if corner not in snake_positions_grid:
                return corner
        
        return target_pos
    
    def get_current_food_number(self):
        """Retorna el numero actual de comida del controlador centralizado"""
        return self.food_controller.get_current_food_number()
    
    def play_step(self, action):
        """Ejecutar paso del juego"""
        return self.game.play_step(action)
    
    def reset(self):
        """Reset del juego - NO RESETEAR EL CONTROLADOR AQUI"""
        # No resetear el controlador de comida, solo el juego
        result = self.game.reset()
        
        # Volver a configurar la posicion inicial de la cabeza
        from collections import namedtuple
        Point = namedtuple('Point', 'x y')
        
        # La posicion de cabeza debe mantenerse segun configuracion original
        head_origin = Point(80, 160)  # (4, 8) * 20 = (80, 160)
        self.game.head = head_origin
        self.game.snake = [self.game.head,
                          Point(self.game.head.x-self.block_size, self.game.head.y),
                          Point(self.game.head.x-(2*self.block_size), self.game.head.y)]
        
        # Configurar comida segun el estado actual del controlador
        food_pos = self.food_controller.get_current_food_position()
        initial_food_point = Point(food_pos[0] * self.block_size, food_pos[1] * self.block_size)
        self.game.food = initial_food_point
        
        return result
    
    def __getattr__(self, name):
        """Delegar atributos al juego subyacente"""
        return getattr(self.game, name)

class ControlledSnakeGameQL:
    """Q-Learning con control centralizado por numero de comida"""
    
    def __init__(self, width, height, head_origin, food_controller):
        # REFERENCIA AL CONTROLADOR CENTRALIZADO
        self.food_controller = food_controller
        self.board_size = width
        
        # Crear juego con parametros estandar
        self.game = self._create_modified_snake_game(width, height, head_origin, food_controller)
        
        # Sobrescribir el metodo spawnFood
        self.game.spawnFood = self._controlled_spawn_food
        
        print(f"   Q-Learning iniciado con comida #{food_controller.get_current_food_number()} en {food_controller.get_current_food_position()}")
    
    def _create_modified_snake_game(self, width, height, head_origin, food_controller):
        """Crear SnakeGame con configuracion inicial personalizada"""
        from Snake import SnakeGame, Snake, BodyNode
        import numpy as np
        
        # Crear juego estandar
        game = SnakeGame(width, height)
        
        # CONFIGURAR POSICION INICIAL DE LA CABEZA
        # Limpiar el tablero de la posicion por defecto
        game.board[game.board == game.headVal] = 0
        
        # Configurar nueva posicion de cabeza
        start_x, start_y = head_origin
        game.board[start_y, start_x] = game.headVal  # SnakeGame usa [y, x]
        game.snake = Snake(start_x, start_y)
        
        # NO AVANZAR AUTOMATICAMENTE - Solo obtener posicion actual
        food_pos = food_controller.get_current_food_position()
        food_yx = (food_pos[1], food_pos[0])  # Convertir (x, y) a (y, x)
        
        # Limpiar comida anterior
        game.board[game.board == game.foodVal] = 0
        
        # Colocar comida inicial
        game.foodIndex = food_yx
        game.board[food_yx] = game.foodVal
        
        # Recalcular estado
        game.calcState()
        
        return game
    
    def _controlled_spawn_food(self):
        """Spawn de siguiente comida usando controlador centralizado"""
        
        # USAR CONTROLADOR CENTRALIZADO
        food_pos = self.food_controller.advance_to_next_food()
        
        # Verificar colisiones con la serpiente
        snake_positions = self._get_all_snake_positions()
        
        if food_pos in snake_positions:
            safe_pos = self._find_safe_position_near(food_pos, snake_positions)
            food_pos = safe_pos
            print(f"   Q-Learning: Comida #{self.food_controller.get_current_food_number()} reubicada por colision a {safe_pos}")
        
        # SnakeGame usa (fila, columna) = (y, x)
        food_yx = (food_pos[1], food_pos[0])
        
        # Limpiar comida anterior
        self.game.board[self.game.board == self.game.foodVal] = 0
        
        # Colocar nueva comida
        self.game.foodIndex = food_yx
        self.game.board[food_yx] = self.game.foodVal
        
        print(f"   Q-Learning: Comida #{self.food_controller.get_current_food_number()} colocada en {food_pos}")
    
    def _get_all_snake_positions(self):
        """Obtiene todas las posiciones de la serpiente"""
        positions = []
        current_node = self.game.snake.getTail()
        while current_node is not None:
            positions.append(current_node.getPosition())
            current_node = current_node.parent
        return positions
    
    def _find_safe_position_near(self, target_pos, snake_positions):
        """Encuentra posicion segura cerca de la posicion objetivo"""
        
        # Buscar en circulos concentricos
        for radius in range(1, 5):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        new_x = target_pos[0] + dx
                        new_y = target_pos[1] + dy
                        
                        if (0 <= new_x < self.board_size and 0 <= new_y < self.board_size and
                            (new_x, new_y) not in snake_positions):
                            return (new_x, new_y)
        
        # Ultimo recurso
        for corner in [(0, 0), (0, self.board_size-1), (self.board_size-1, 0), (self.board_size-1, self.board_size-1)]:
            if corner not in snake_positions:
                return corner
        
        return target_pos
    
    def get_current_food_number(self):
        """Retorna el numero actual de comida del controlador centralizado"""
        return self.food_controller.get_current_food_number()
    
    def makeMove(self, action):
        """Ejecutar movimiento"""
        return self.game.makeMove(action)
    
    def calcStateNum(self):
        """Calcular numero de estado"""
        return self.game.calcStateNum()
    
    def __getattr__(self, name):
        """Delegar atributos al juego subyacente"""
        return getattr(self.game, name)

# EJEMPLO DE USO CORRECTO:
def example_synchronized_setup():
    """Ejemplo de configuracion sincronizada correcta"""
    
    SEED_BASE = 42
    episode_num = 1
    episode_seed = SEED_BASE + episode_num - 1
    
    # UN SOLO CONTROLADOR COMPARTIDO
    food_controller = FoodSequenceController(episode_seed, max_foods=20)
    
    # AMBOS JUEGOS USAN EL MISMO CONTROLADOR
    from collections import namedtuple
    Point = namedtuple('Point', 'x y')
    
    dqn_game = ControlledSnakeGameDQN(w=320, h=320, head_origin=Point(160, 160), 
                                      food_controller=food_controller)
    
    ql_game = ControlledSnakeGameQL(width=16, height=16, head_origin=(8, 8), 
                                   food_controller=food_controller)
    
    print("\nRESULTADO ESPERADO:")
    print("Ambos algoritmos veran EXACTAMENTE las mismas posiciones de comida")
    print("Comida #1: Misma posicion para ambos")
    print("Comida #7: Misma posicion para ambos") 
    print("Independientemente del rendimiento de cada algoritmo")
    
    return food_controller, dqn_game, ql_game

if __name__ == "__main__":
    example_synchronized_setup()