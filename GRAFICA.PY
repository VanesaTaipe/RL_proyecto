import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
plt.rcParams['figure.figsize'] = (20, 15)
plt.rcParams['font.size'] = 10

print("ANÁLISIS COMPLETO UNIFICADO DE TODAS LAS COMIDAS - DQN vs Q-Learning")
print("=" * 80)

class UnifiedFoodAnalyzer:
    """Analizador unificado que combina todas las funcionalidades"""
    
    def __init__(self):
        self.data = None
        self.all_food_info = {}
        self.df = None
        self.food_analysis = {}
        self.validation_results = {}
        self.episode_stats = []
        
    def load_all_data(self):
        """Cargar todos los datos disponibles"""
        try:
            # Cargar datos de verificación
            with open("verification_qlearning_food_sequence.pkl", "rb") as f:
                ql_verification = pickle.load(f)
            print("✓ Datos de verificación Q-Learning cargados")
            
            with open("verification_dqn_food_sequence.pkl", "rb") as f:
                dqn_verification = pickle.load(f)
            print("✓ Datos de verificación DQN cargados")
            
            # Cargar métricas
            with open("metrics_qlearning_food_sequence.pkl", "rb") as f:
                ql_metrics = pickle.load(f)
            print("✓ Métricas Q-Learning cargadas")
            
            with open("metrics_dqn_food_sequence.pkl", "rb") as f:
                dqn_metrics = pickle.load(f)
            print("✓ Métricas DQN cargadas")
            
            self.data = {
                'ql_verification': ql_verification,
                'dqn_verification': dqn_verification,
                'ql_metrics': ql_metrics,
                'dqn_metrics': dqn_metrics
            }
            
            return True
            
        except FileNotFoundError as e:
            print(f"Error: No se pudo encontrar el archivo {e.filename}")
            return False

    def verify_food_1_presence(self):
        """Verificar que la comida #1 esté presente en ambos algoritmos"""
        print("\nVERIFICANDO PRESENCIA DE COMIDA #1...")
        
        ql_verification = self.data['ql_verification']
        dqn_verification = self.data['dqn_verification']
        
        food_1_status = {'found': 0, 'synced': 0, 'total': 0}
        
        for episode_key in ql_verification.keys():
            if episode_key in dqn_verification:
                food_1_status['total'] += 1
                episode_num = episode_key.split('_')[-1]
                
                # Verificar Q-Learning
                ql_food_1 = [entry for entry in ql_verification[episode_key] 
                            if entry.get('food_number') == 1]
                
                # Verificar DQN
                dqn_food_1 = [entry for entry in dqn_verification[episode_key] 
                             if entry.get('food_number') == 1]
                
                if ql_food_1 and dqn_food_1:
                    food_1_status['found'] += 1
                    ql_pos = ql_food_1[0].get('food_position')
                    dqn_pos = dqn_food_1[0].get('food_position')
                    sync_status = "SYNC" if ql_pos == dqn_pos else " DIFF"
                    
                    if ql_pos == dqn_pos:
                        food_1_status['synced'] += 1
                    
                    print(f"   Episodio {episode_num}: QL={ql_pos}, DQN={dqn_pos} - {sync_status}")
                else:
                    print(f"   Episodio {episode_num}: Comida #1 faltante")
        
        sync_rate = (food_1_status['synced'] / food_1_status['total'] * 100) if food_1_status['total'] > 0 else 0
        print(f"\n RESUMEN COMIDA #1:")
        print(f"   • Encontrada: {food_1_status['found']}/{food_1_status['total']} episodios")
        print(f"   • Sincronizada: {food_1_status['synced']}/{food_1_status['total']} episodios")
        print(f"   • Tasa de sincronización: {sync_rate:.1f}%")
        
        return food_1_status

    def extract_complete_food_info(self):
        """Extraer información completa de todas las comidas"""
        print("\nEXTRAYENDO INFORMACIÓN COMPLETA DE TODAS LAS COMIDAS...")
        
        # Obtener episodios comunes
        ql_episodes = set(self.data['ql_verification'].keys())
        dqn_episodes = set(self.data['dqn_verification'].keys())
        
        ql_episode_nums = set([int(ep.split('_')[-1]) for ep in ql_episodes])
        dqn_episode_nums = set([int(ep.split('_')[-1]) for ep in dqn_episodes])
        
        common_episodes = sorted(ql_episode_nums.intersection(dqn_episode_nums))
        
        for ep_num in common_episodes:
            print(f"Procesando episodio {ep_num}...")
            
            ql_key = f"QL_episode_{ep_num}"
            dqn_key = f"DQN_episode_{ep_num}"
            
            # Datos de verificación
            ql_verification = self.data['ql_verification'].get(ql_key, [])
            dqn_verification = self.data['dqn_verification'].get(dqn_key, [])
            
            # Métricas del episodio
            ql_metrics = self.data['ql_metrics'].get(ql_key, {})
            dqn_metrics = self.data['dqn_metrics'].get(dqn_key, {})
            
            #  VERIFICACIÓN ESPECÍFICA DE COMIDA #1 PARA ESTE EPISODIO
            ql_food_1_count = len([e for e in ql_verification if e.get('food_number') == 1])
            dqn_food_1_count = len([e for e in dqn_verification if e.get('food_number') == 1])
            print(f"   Comida #1: QL={ql_food_1_count}, DQN={dqn_food_1_count}")
            
            # Procesar información de comidas
            episode_info = {
                'episode': ep_num,
                'ql_foods': self.extract_detailed_food_info(ql_verification, ql_metrics, 'Q-Learning'),
                'dqn_foods': self.extract_detailed_food_info(dqn_verification, dqn_metrics, 'DQN'),
                'ql_sequence': ql_metrics.get('food_sequence', []),
                'dqn_sequence': dqn_metrics.get('food_sequence', []),
                'ql_score': ql_metrics.get('final_score', 0),
                'dqn_score': dqn_metrics.get('final_score', 0),
                'ql_total_steps': ql_metrics.get('total_steps', 0),
                'dqn_total_steps': dqn_metrics.get('total_steps', 0)
            }
            
            self.all_food_info[ep_num] = episode_info

    def extract_detailed_food_info(self, verification_data, metrics_data, model_name):
        """Extraer información detallada de comidas para un modelo"""
        food_info = {}
        
        if not verification_data:
            return food_info
        
        # Agrupar por número de comida
        foods_by_number = defaultdict(list)
        
        for entry in verification_data:
            food_num = entry.get('food_number', 0)
            foods_by_number[food_num].append(entry)
        
        # Procesar cada comida
        for food_num, entries in foods_by_number.items():
            first_appearance = min(entries, key=lambda x: x.get('step', 0))
            last_appearance = max(entries, key=lambda x: x.get('step', 0))
            
            food_info[food_num] = {
                'number': food_num,
                'position': first_appearance.get('food_position', (0, 0)),
                'first_step': first_appearance.get('step', 0),
                'last_step': last_appearance.get('step', 0),
                'steps_visible': last_appearance.get('step', 0) - first_appearance.get('step', 0) + 1,
                'total_appearances': len(entries),
                'was_eaten': False,
                'steps_to_eat': None
            }
        
        # Determinar qué comidas fueron comidas usando length_evolution
        length_evolution = metrics_data.get('length_evolution', [])
        if length_evolution:
            food_eaten_steps = []
            for i in range(1, len(length_evolution)):
                if length_evolution[i] > length_evolution[i-1]:
                    food_eaten_steps.append(i)
            
            # Asociar pasos de comer con números de comida de manera más precisa
            for eat_step in food_eaten_steps:
                # Encontrar qué comida estaba activa en ese paso
                active_food = None
                for food_num, info in food_info.items():
                    if info['first_step'] <= eat_step <= info['last_step']:
                        active_food = food_num
                        break
                
                if active_food:
                    food_info[active_food]['was_eaten'] = True
                    food_info[active_food]['steps_to_eat'] = eat_step - food_info[active_food]['first_step']
        
        return food_info

    def create_complete_food_table(self):
        """Crear tabla completa con información de todas las comidas"""
        print("\n CREANDO TABLA COMPLETA DE TODAS LAS COMIDAS...")
        
        food_records = []
        
        for ep_num, episode_data in self.all_food_info.items():
            # Obtener todas las comidas únicas en este episodio
            all_food_numbers = set()
            all_food_numbers.update(episode_data['ql_foods'].keys())
            all_food_numbers.update(episode_data['dqn_foods'].keys())
            
            for food_num in sorted(all_food_numbers):
                ql_info = episode_data['ql_foods'].get(food_num, {})
                dqn_info = episode_data['dqn_foods'].get(food_num, {})
                
                # Verificar secuencia predefinida
                sequence_pos = None
                if episode_data['ql_sequence'] and food_num <= len(episode_data['ql_sequence']):
                    sequence_pos = episode_data['ql_sequence'][food_num - 1] if food_num > 0 else None
                
                record = {
                    'Episodio': ep_num,
                    'Comida_Numero': food_num,
                    'Secuencia_Predefinida': sequence_pos,
                    
                    # Q-Learning
                    'QL_Posicion': ql_info.get('position', 'N/A'),
                    'QL_Primer_Paso': ql_info.get('first_step', 'N/A'),
                    'QL_Ultimo_Paso': ql_info.get('last_step', 'N/A'),
                    'QL_Pasos_Visible': ql_info.get('steps_visible', 'N/A'),
                    'QL_Fue_Comida': ql_info.get('was_eaten', False),
                    'QL_Pasos_Para_Comer': ql_info.get('steps_to_eat', 'N/A'),
                    
                    # DQN
                    'DQN_Posicion': dqn_info.get('position', 'N/A'),
                    'DQN_Primer_Paso': dqn_info.get('first_step', 'N/A'),
                    'DQN_Ultimo_Paso': dqn_info.get('last_step', 'N/A'),
                    'DQN_Pasos_Visible': dqn_info.get('steps_visible', 'N/A'),
                    'DQN_Fue_Comida': dqn_info.get('was_eaten', False),
                    'DQN_Pasos_Para_Comer': dqn_info.get('steps_to_eat', 'N/A'),
                    
                    # Comparaciones
                    'Posiciones_Coinciden': ql_info.get('position') == dqn_info.get('position') if ql_info.get('position') != 'N/A' and dqn_info.get('position') != 'N/A' else False,
                    'Ambos_La_Comieron': ql_info.get('was_eaten', False) and dqn_info.get('was_eaten', False),
                    
                    # Información del episodio
                    'QL_Score_Final': episode_data['ql_score'],
                    'DQN_Score_Final': episode_data['dqn_score'],
                    'QL_Pasos_Totales': episode_data['ql_total_steps'],
                    'DQN_Pasos_Totales': episode_data['dqn_total_steps']
                }
                
                food_records.append(record)
        
        # Crear DataFrame
        self.df = pd.DataFrame(food_records)
        
        # 🔍 VERIFICACIÓN ESPECIAL DE COMIDA #1
        food_1_data = self.df[self.df['Comida_Numero'] == 1]
        print(f"\n VERIFICACIÓN FINAL DE COMIDA #1:")
        print(f"   Registros de comida #1 encontrados: {len(food_1_data)}")
        
        if len(food_1_data) > 0:
            sync_count = food_1_data['Posiciones_Coinciden'].sum()
            print(f"   Episodios con comida #1 sincronizada: {sync_count}/{len(food_1_data)}")
            print(f"   Tasa de sincronización comida #1: {sync_count/len(food_1_data)*100:.1f}%")
            
            # Mostrar algunas posiciones de comida #1
            print(f"   Muestras de posiciones comida #1:")
            for _, row in food_1_data.head(5).iterrows():
                ep = row['Episodio']
                ql_pos = row['QL_Posicion']
                dqn_pos = row['DQN_Posicion']
                sync = "correcto" if row['Posiciones_Coinciden'] else "error"
                print(f"     Ep {ep}: QL={ql_pos}, DQN={dqn_pos} {sync}")
        else:
            print(f"    ERROR: No se encontraron registros de comida #1")
        
        # Guardar en CSV
        self.df.to_csv('analisis_completo_unificado_todas_las_comidas.csv', index=False)
        print(f"\n✓ Tabla completa guardada en 'analisis_completo_unificado_todas_las_comidas.csv'")
        print(f"   Total de registros: {len(self.df)}")

    def analyze_food_by_number(self):
        """Analizar cada número de comida específico"""
        print("\nANÁLISIS POR NÚMERO DE COMIDA...")
        
        for food_num in sorted(self.df['Comida_Numero'].unique()):
            food_data = self.df[self.df['Comida_Numero'] == food_num]
            
            analysis = {
                'numero': food_num,
                'episodios_aparece': len(food_data),
                'posiciones_unicas': len(set([str(pos) for pos in food_data['QL_Posicion'] if pos != 'N/A'])),
                'sincronizacion_perfecta': food_data['Posiciones_Coinciden'].sum(),
                'ql_success_rate': (food_data['QL_Fue_Comida'].sum() / len(food_data)) * 100,
                'dqn_success_rate': (food_data['DQN_Fue_Comida'].sum() / len(food_data)) * 100,
                'ambos_exito_rate': (food_data['Ambos_La_Comieron'].sum() / len(food_data)) * 100,
                'posiciones_mas_comunes': Counter([str(pos) for pos in food_data['QL_Posicion'] if pos != 'N/A']).most_common(3)
            }
            
            # Calcular tiempo promedio para comer
            ql_times = [t for t in food_data['QL_Pasos_Para_Comer'] if t != 'N/A' and isinstance(t, (int, float))]
            dqn_times = [t for t in food_data['DQN_Pasos_Para_Comer'] if t != 'N/A' and isinstance(t, (int, float))]
            
            analysis['ql_avg_time_to_eat'] = np.mean(ql_times) if ql_times else 'N/A'
            analysis['dqn_avg_time_to_eat'] = np.mean(dqn_times) if dqn_times else 'N/A'
            
            self.food_analysis[food_num] = analysis

    def extract_individual_episode_foods(self):
        """Extraer datos de comidas para mapas individuales por episodio"""
        ql_foods = {}
        dqn_foods = {}
        
        for episode_num, episode_data in self.all_food_info.items():
            ql_episode_foods = {}
            dqn_episode_foods = {}
            
            for food_num, food_info in episode_data['ql_foods'].items():
                ql_episode_foods[food_num] = food_info['position']
            
            for food_num, food_info in episode_data['dqn_foods'].items():
                dqn_episode_foods[food_num] = food_info['position']
            
            ql_foods[episode_num] = ql_episode_foods
            dqn_foods[episode_num] = dqn_episode_foods
        
        return ql_foods, dqn_foods

    def get_food_colors(self):
        """Obtener colores únicos y brillantes para cada número de comida"""
        return [
            '#FF0000',  # 1: Rojo brillante
            '#00FF00',  # 2: Verde brillante  
            '#0000FF',  # 3: Azul brillante
            '#FFFF00',  # 4: Amarillo brillante
            '#FF00FF',  # 5: Magenta brillante
            '#00FFFF',  # 6: Cian brillante
            '#FF8000',  # 7: Naranja brillante
            '#8000FF',  # 8: Violeta brillante
            '#80FF00',  # 9: Verde lima brillante
            '#FF0080',  # 10: Rosa brillante
            '#0080FF',  # 11: Azul cielo brillante
            '#FF8080',  # 12: Rosa claro brillante
            '#80FF80',  # 13: Verde claro brillante
            '#8080FF',  # 14: Azul claro brillante
            '#FFFF80',  # 15: Amarillo claro brillante
            '#FF80FF',  # 16: Magenta claro brillante
            '#80FFFF',  # 17: Cian claro brillante
            '#FFC080',  # 18: Durazno brillante
            '#C080FF',  # 19: Lavanda brillante
            '#80FFC0'   # 20: Menta brillante
        ]

    def create_individual_episode_food_map(self, episode_num, dqn_foods, ql_foods):
        """Crear mapa completo de comidas para un episodio específico - FUNCIÓN PRINCIPAL"""
        
        print(f"Creando mapa detallado para Episodio {episode_num}...")
        
        # Configurar figura optimizada para mostrar todas las comidas
        fig, (ax_dqn, ax_ql) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'EPISODIO {episode_num} - MAPA COMPLETO DE TODAS LAS COMIDAS', 
                     fontsize=24, fontweight='bold', y=0.95)
        
        board_size = 16
        colors = self.get_food_colors()
        
        # Obtener todas las comidas en este episodio
        all_food_numbers = sorted(set(dqn_foods.keys()).union(set(ql_foods.keys())))
        max_food = max(all_food_numbers) if all_food_numbers else 0
        
        # Análisis de sincronización
        differences = []
        matches = 0
        total_foods = len(all_food_numbers)
        
        # Verificar sincronización comida por comida
        for food_num in all_food_numbers:
            dqn_pos = dqn_foods.get(food_num)
            ql_pos = ql_foods.get(food_num)
            
            if dqn_pos and ql_pos:
                if dqn_pos == ql_pos:
                    matches += 1
                else:
                    differences.append(f"#{food_num}")
            elif dqn_pos and not ql_pos:
                differences.append(f"#{food_num}(DQN)")
            elif ql_pos and not dqn_pos:
                differences.append(f"#{food_num}(QL)")
        
        sync_rate = (matches / total_foods * 100) if total_foods > 0 else 100
        
        # Configurar ambos subplots (DQN y Q-Learning)
        for ax, foods, title, algorithm in [(ax_dqn, dqn_foods, "DQN", "dqn"), 
                                           (ax_ql, ql_foods, "Q-Learning", "ql")]:
            
            # Configuración del tablero de juego
            ax.set_xlim(-0.5, board_size - 0.5)
            ax.set_ylim(-0.5, board_size - 0.5)
            ax.set_aspect('equal')
            
            # Fondo blanco limpio para mejor contraste
            ax.set_facecolor('white')
            
            # Grid de referencia muy sutil para facilitar lectura
            for i in range(board_size + 1):
                ax.axhline(y=i-0.5, color='lightgray', linewidth=0.3, alpha=0.4)
                ax.axvline(x=i-0.5, color='lightgray', linewidth=0.3, alpha=0.4)
            
            # Bordes del tablero más prominentes
            ax.add_patch(patches.Rectangle((-0.5, -0.5), board_size, board_size, 
                                         fill=False, edgecolor='black', linewidth=4))
            
            # Título con información completa del algoritmo
            food_count = len(foods)
            max_food_algo = max(foods.keys()) if foods else 0
            ax.set_title(f'{title}\n{food_count} comidas detectadas | Máxima: #{max_food_algo}', 
                        fontweight='bold', fontsize=16, pad=20)
            
            # Etiquetas de ejes más claras
            ax.set_xlabel('Coordenada X (Posición Horizontal)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Coordenada Y (Posición Vertical)', fontsize=14, fontweight='bold')
            
            # Ticks más espaciados para claridad
            ax.set_xticks(range(0, board_size, 2))
            ax.set_yticks(range(0, board_size, 2))
            ax.tick_params(labelsize=12, width=2)
            
            # DIBUJAR TODAS LAS COMIDAS CON DETALLES COMPLETOS
            for food_num in sorted(foods.keys()):
                pos = foods[food_num]
                if isinstance(pos, tuple) and len(pos) == 2:
                    x, y = pos
                    if 0 <= x < board_size and 0 <= y < board_size:
                        # Color único basado en número de comida
                        color = colors[food_num - 1] if food_num <= len(colors) else '#333333'
                        
                        # TRATAMIENTO ESPECIAL PARA COMIDA #1
                        if food_num == 1:
                            radius = 0.5  # Más grande
                            edge_color = 'gold'  # Borde dorado
                            edge_width = 4  # Borde más grueso
                            text_size = 18  # Texto más grande
                            coord_bg = 'yellow'  # Fondo amarillo para coordenadas
                            coord_color = 'red'  # Texto rojo para coordenadas
                        else:
                            radius = 0.45
                            edge_color = 'white'
                            edge_width = 3
                            text_size = 14
                            coord_bg = 'white'
                            coord_color = 'black'
                        
                        # Círculo principal para la comida
                        circle = patches.Circle((x, y), radius, facecolor=color, 
                                              edgecolor=edge_color, linewidth=edge_width, alpha=0.9)
                        ax.add_patch(circle)
                        
                        # Número de comida en el centro del círculo
                        text_color = 'white' if food_num <= 10 else 'black'
                        ax.text(x, y, str(food_num), ha='center', va='center', 
                               fontsize=text_size, fontweight='bold', color=text_color)
                        
                        # Coordenadas debajo de cada comida
                        ax.text(x, y-0.8, f'({x},{y})', ha='center', va='center', 
                               fontsize=9, fontweight='bold', color=coord_color,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=coord_bg, alpha=0.8))
        
        # INFORMACIÓN DE SINCRONIZACIÓN PROMINENTE
        sync_text = f"SINCRONIZACIÓN ENTRE ALGORITMOS: {sync_rate:.1f}%\n"
        sync_text += f"Comidas con posiciones idénticas: {matches}/{total_foods}\n"
        
        if not differences:
            sync_text += "TODAS LAS POSICIONES SON PERFECTAMENTE IDÉNTICAS"
            text_color = 'darkgreen'
            bg_color = 'lightgreen'
        else:
            sync_text += f" DIFERENCIAS DETECTADAS EN: {', '.join(differences[:8])}"
            if len(differences) > 8:
                sync_text += f"\n... y {len(differences) - 8} comidas más"
            text_color = 'darkorange'
            bg_color = 'lightyellow'
        
        # Colocar información de sincronización en posición prominente
        fig.text(0.5, 0.12, sync_text, ha='center', va='center', 
                fontsize=14, color=text_color, weight='bold',
                bbox=dict(boxstyle='round,pad=1.0', facecolor=bg_color, alpha=0.9))
        
        # ESTADÍSTICAS DETALLADAS DEL EPISODIO
        stats_text = f"""ESTADÍSTICAS DETALLADAS - EPISODIO {episode_num}:
• DQN detectó: {len(dqn_foods)} comidas (máxima: #{max(dqn_foods.keys()) if dqn_foods else 0})
• Q-Learning detectó: {len(ql_foods)} comidas (máxima: #{max(ql_foods.keys()) if ql_foods else 0})
• Total de comidas únicas: {total_foods}
• Diferencia en detección: {abs(len(dqn_foods) - len(ql_foods))} comidas
• Comidas sincronizadas: {matches}
• Comidas con diferencias: {len(differences)}"""
        
        fig.text(0.02, 0.02, stats_text, ha='left', va='bottom', 
                fontsize=12, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # LEYENDA DE COLORES PARA LAS COMIDAS
        legend_text = "LEYENDA DE COMIDAS DETECTADAS:\n"
        
        # Mostrar solo las comidas que aparecen en este episodio
        foods_in_episode = sorted(all_food_numbers)
        for i, food_num in enumerate(foods_in_episode[:10]):  # Primeras 10 comidas
            color_name = f"#{food_num}"
            if food_num == 1:
                color_name += "  (INICIAL - DESTACADA)"
            legend_text += f"{color_name}  "
            if (i + 1) % 5 == 0:
                legend_text += "\n"
        
        if len(foods_in_episode) > 10:
            legend_text += f"\n... y {len(foods_in_episode) - 10} comidas adicionales"
        
        # Agregar información sobre el código de colores
        legend_text += f"\n\nCÓDIGO VISUAL:"
        legend_text += f"\n• Círculos numerados = Posición de comida"
        legend_text += f"\n• Borde dorado = Comida #1 (inicial)"
        legend_text += f"\n• (x,y) = Coordenadas exactas"
        
        fig.text(0.98, 0.02, legend_text, ha='right', va='bottom', 
                fontsize=11, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Ajustar layout para que no se sobrepongan elementos
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, top=0.88)
        
        # Guardar imagen individual con nombre descriptivo
        filename = f'episodio_{episode_num:02d}_mapa_completo_unificado.png'
        plt.savefig(filename, dpi=250, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Mapa detallado guardado: {filename}")
        print(f"   └─ {total_foods} comidas mapeadas, Sincronización: {sync_rate:.1f}%")
        
        return {
            'episode': episode_num,
            'total_foods': total_foods,
            'dqn_foods': len(dqn_foods),
            'ql_foods': len(ql_foods),
            'matches': matches,
            'differences': len(differences),
            'sync_rate': sync_rate,
            'max_food': max_food,
            'filename': filename,
            'total_foods': total_foods,
            'dqn_foods': len(dqn_foods),
            'ql_foods': len(ql_foods),
            'matches': matches,
            'differences': len(differences),
            'sync_rate': sync_rate,
            'max_food': max_food
        }

    def create_comprehensive_visualizations(self):
        """Crear visualizaciones comprehensivas de todas las comidas"""
        print("\n📊 CREANDO VISUALIZACIONES COMPREHENSIVAS...")
        
        # Configurar figura grande
        fig = plt.figure(figsize=(28, 24))
        fig.suptitle('ANÁLISIS COMPLETO UNIFICADO - DQN vs Q-Learning', 
                     fontsize=24, fontweight='bold')
        
        food_numbers = sorted(self.food_analysis.keys())
        
        # 1. Tasa de éxito por número de comida con énfasis en comida #1
        ax1 = plt.subplot(4, 4, 1)
        ql_success = [self.food_analysis[fn]['ql_success_rate'] for fn in food_numbers]
        dqn_success = [self.food_analysis[fn]['dqn_success_rate'] for fn in food_numbers]
        
        x = np.arange(len(food_numbers))
        width = 0.35
        
        # Colores especiales para comida #1
        ql_colors = ['gold' if fn == 1 else 'orange' for fn in food_numbers]
        dqn_colors = ['red' if fn == 1 else 'skyblue' for fn in food_numbers]
        
        bars1 = ax1.bar(x - width/2, ql_success, width, label='Q-Learning', color=ql_colors, alpha=0.8)
        bars2 = ax1.bar(x + width/2, dqn_success, width, label='DQN', color=dqn_colors, alpha=0.8)
        
        # Destacar comida #1
        if 1 in food_numbers:
            idx_1 = food_numbers.index(1)
            bars1[idx_1].set_edgecolor('black')
            bars1[idx_1].set_linewidth(3)
            bars2[idx_1].set_edgecolor('black')
            bars2[idx_1].set_linewidth(3)
        
        ax1.set_title('Tasa de Éxito por Número de Comida\n(Comida #1 destacada)', fontweight='bold')
        ax1.set_xlabel('Número de Comida')
        ax1.set_ylabel('Tasa de Éxito (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'#{fn}' for fn in food_numbers])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sincronización por comida con énfasis en comida #1
        ax2 = plt.subplot(4, 4, 2)
        sync_rates = [self.food_analysis[fn]['sincronizacion_perfecta'] / self.food_analysis[fn]['episodios_aparece'] * 100 
                      for fn in food_numbers]
        
        colors = []
        for i, (fn, rate) in enumerate(zip(food_numbers, sync_rates)):
            if fn == 1:
                colors.append('gold' if rate == 100 else 'orange')
            else:
                colors.append('green' if rate == 100 else 'orange' if rate >= 80 else 'red')
        
        bars = ax2.bar(food_numbers, sync_rates, color=colors, alpha=0.7)
        
        # Destacar comida #1
        if 1 in food_numbers:
            idx_1 = food_numbers.index(1)
            bars[idx_1].set_edgecolor('black')
            bars[idx_1].set_linewidth(3)
        
        ax2.set_title('Tasa de Sincronización por Comida\n(Comida #1 destacada)', fontweight='bold')
        ax2.set_xlabel('Número de Comida')
        ax2.set_ylabel('Sincronización (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Mapa de calor especial para comida #1
        ax3 = plt.subplot(4, 4, 3)
        
        first_food_data = self.df[self.df['Comida_Numero'] == 1]
        positions = [pos for pos in first_food_data['QL_Posicion'] if pos != 'N/A']
        
        if positions:
            # Crear mapa de calor de posiciones para comida #1
            board_size = 16
            position_heatmap = np.zeros((board_size, board_size))
            
            for pos in positions:
                if isinstance(pos, tuple) and len(pos) == 2:
                    x, y = pos
                    if 0 <= x < board_size and 0 <= y < board_size:
                        position_heatmap[y, x] += 1
            
            im = ax3.imshow(position_heatmap, cmap='Reds', alpha=0.8)
            ax3.set_title('Mapa de Calor - COMIDA #1\n(VERIFICADO)', fontweight='bold')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            
            # Marcar la posición más común
            if len(positions) > 0:
                most_common_pos = Counter(positions).most_common(1)[0][0]
                ax3.scatter(most_common_pos[0], most_common_pos[1], c='blue', s=200, 
                           marker='*', edgecolors='white', linewidth=3)
                ax3.text(most_common_pos[0], most_common_pos[1]-1, 'MÁS COMÚN', 
                        ha='center', va='top', fontweight='bold', color='blue')
            
            plt.colorbar(im, ax=ax3)
            
            # Agregar estadísticas
            sync_rate_1 = first_food_data['Posiciones_Coinciden'].sum() / len(first_food_data) * 100
            ax3.text(0.02, 0.98, f'Sincronización: {sync_rate_1:.0f}%\nEpisodios: {len(first_food_data)}', 
                   transform=ax3.transAxes, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                   verticalalignment='top')
        else:
            ax3.text(0.5, 0.5, 'COMIDA #1\nNO ENCONTRADA', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=16, fontweight='bold', color='red')
            ax3.set_title('ERROR: Comida #1', fontweight='bold')
        
        # 4. Tiempo promedio para comer
        ax4 = plt.subplot(4, 4, 4)
        ql_times = [self.food_analysis[fn]['ql_avg_time_to_eat'] if self.food_analysis[fn]['ql_avg_time_to_eat'] != 'N/A' else 0 
                    for fn in food_numbers]
        dqn_times = [self.food_analysis[fn]['dqn_avg_time_to_eat'] if self.food_analysis[fn]['dqn_avg_time_to_eat'] != 'N/A' else 0 
                     for fn in food_numbers]
        
        ax4.plot(food_numbers, ql_times, 'o-', label='Q-Learning', color='orange', linewidth=2, markersize=6)
        ax4.plot(food_numbers, dqn_times, 's-', label='DQN', color='skyblue', linewidth=2, markersize=6)
        ax4.set_title('Tiempo Promedio para Comer', fontweight='bold')
        ax4.set_xlabel('Número de Comida')
        ax4.set_ylabel('Pasos Promedio')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Mapa de calor - Tasa de éxito por episodio y comida
        ax5 = plt.subplot(4, 4, 5)
        
        episodes = sorted(self.df['Episodio'].unique())
        max_food = min(15, max(food_numbers))  # Limitar a primeras 15 comidas
        
        success_matrix = np.zeros((len(episodes), max_food))
        
        for i, ep in enumerate(episodes):
            ep_data = self.df[self.df['Episodio'] == ep]
            for j, food_num in enumerate(range(1, max_food + 1)):
                food_data = ep_data[ep_data['Comida_Numero'] == food_num]
                if not food_data.empty:
                    both_ate = food_data['Ambos_La_Comieron'].iloc[0]
                    success_matrix[i, j] = 1 if both_ate else 0
        
        im = ax5.imshow(success_matrix, cmap='RdYlGn', aspect='auto', alpha=0.8)
        ax5.set_title('Éxito por Episodio y Comida\n(Verde: Ambos comen)', fontweight='bold')
        ax5.set_xlabel('Número de Comida')
        ax5.set_ylabel('Episodio')
        ax5.set_xticks(range(max_food))
        ax5.set_xticklabels(range(1, max_food + 1))
        ax5.set_yticks(range(len(episodes)))
        ax5.set_yticklabels(episodes)
        plt.colorbar(im, ax=ax5)
        
        # 6. Distribución de apariciones
        ax6 = plt.subplot(4, 4, 6)
        appearances = [self.food_analysis[fn]['episodios_aparece'] for fn in food_numbers]
        colors_appear = ['gold' if fn == 1 else 'purple' for fn in food_numbers]
        bars = ax6.bar(food_numbers, appearances, color=colors_appear, alpha=0.7)
        
        # Destacar comida #1
        if 1 in food_numbers:
            idx_1 = food_numbers.index(1)
            bars[idx_1].set_edgecolor('black')
            bars[idx_1].set_linewidth(3)
        
        ax6.set_title('Episodios donde Aparece cada Comida\n(Comida #1 destacada)', fontweight='bold')
        ax6.set_xlabel('Número de Comida')
        ax6.set_ylabel('Número de Episodios')
        ax6.grid(True, alpha=0.3)
        
        # 7. Análisis de dificultad por comida
        ax7 = plt.subplot(4, 4, 7)
        difficulty = [100 - (self.food_analysis[fn]['ql_success_rate'] + self.food_analysis[fn]['dqn_success_rate']) / 2 
                      for fn in food_numbers]
        
        ax7.plot(food_numbers, difficulty, 'o-', color='red', linewidth=2, markersize=8, alpha=0.7)
        ax7.fill_between(food_numbers, difficulty, alpha=0.3, color='red')
        ax7.set_title('Dificultad por Número de Comida', fontweight='bold')
        ax7.set_xlabel('Número de Comida')
        ax7.set_ylabel('Dificultad (%)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Comparación de eficiencia por comida
        ax8 = plt.subplot(4, 4, 8)
        efficiency_comparison = []
        for fn in food_numbers[:10]:  # Primeras 10 comidas
            ql_time = self.food_analysis[fn]['ql_avg_time_to_eat']
            dqn_time = self.food_analysis[fn]['dqn_avg_time_to_eat']
            
            if ql_time != 'N/A' and dqn_time != 'N/A' and ql_time > 0 and dqn_time > 0:
                efficiency = (dqn_time - ql_time) / max(ql_time, dqn_time) * 100
                efficiency_comparison.append(efficiency)
            else:
                efficiency_comparison.append(0)
        
        colors_eff = ['green' if eff < 0 else 'red' if eff > 0 else 'gray' for eff in efficiency_comparison]
        ax8.bar(range(len(efficiency_comparison)), efficiency_comparison, color=colors_eff, alpha=0.7)
        ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax8.set_title('Eficiencia Relativa\n(Verde: QL más rápido)', fontweight='bold')
        ax8.set_xlabel('Número de Comida')
        ax8.set_ylabel('Diferencia Relativa (%)')
        ax8.set_xticks(range(len(efficiency_comparison)))
        ax8.set_xticklabels([f'#{i+1}' for i in range(len(efficiency_comparison))])
        ax8.grid(True, alpha=0.3)
        
        # 9. Tendencias temporales
        ax9 = plt.subplot(4, 4, 9)
        trend_x = food_numbers[:15]  # Primeras 15 comidas
        trend_y = [self.food_analysis[fn]['ambos_exito_rate'] for fn in trend_x]
        
        # Ajustar línea de tendencia
        if len(trend_x) > 1:
            z = np.polyfit(trend_x, trend_y, 1)
            p = np.poly1d(z)
            
            ax9.scatter(trend_x, trend_y, alpha=0.7, s=80)
            ax9.plot(trend_x, p(trend_x), "r--", alpha=0.8, linewidth=2)
            
            ax9.set_title('Tendencia: Éxito vs Número de Comida', fontweight='bold')
            ax9.set_xlabel('Número de Comida')
            ax9.set_ylabel('Tasa de Éxito Conjunta (%)')
            ax9.grid(True, alpha=0.3)
        
        # 10. Matriz de correlación de éxito
        ax10 = plt.subplot(4, 4, 10)
        episodes = sorted(self.df['Episodio'].unique())
        success_data = []
        for ep in episodes:
            ep_data = self.df[self.df['Episodio'] == ep]
            ep_success = []
            for fn in range(1, min(11, max(food_numbers) + 1)):  # Primeras 10 comidas
                food_data = ep_data[ep_data['Comida_Numero'] == fn]
                if not food_data.empty:
                    both_ate = food_data['Ambos_La_Comieron'].iloc[0]
                    ep_success.append(1 if both_ate else 0)
                else:
                    ep_success.append(0)
            success_data.append(ep_success)
        
        if success_data:
            corr_matrix = np.corrcoef(np.array(success_data).T)
            im = ax10.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax10.set_title('Correlación de Éxito entre Comidas', fontweight='bold')
            ax10.set_xlabel('Número de Comida')
            ax10.set_ylabel('Número de Comida')
            food_labels = [f'#{i+1}' for i in range(len(corr_matrix))]
            ax10.set_xticks(range(len(food_labels)))
            ax10.set_yticks(range(len(food_labels)))
            ax10.set_xticklabels(food_labels)
            ax10.set_yticklabels(food_labels)
            plt.colorbar(im, ax=ax10)
        
        # 11. Comparación de variabilidad
        ax11 = plt.subplot(4, 4, 11)
        ql_variability = []
        dqn_variability = []
        
        for fn in food_numbers[:10]:
            food_data = self.df[self.df['Comida_Numero'] == fn]
            ql_times = [t for t in food_data['QL_Pasos_Para_Comer'] 
                       if t != 'N/A' and isinstance(t, (int, float))]
            dqn_times = [t for t in food_data['DQN_Pasos_Para_Comer'] 
                        if t != 'N/A' and isinstance(t, (int, float))]
            
            ql_var = np.std(ql_times) if len(ql_times) > 1 else 0
            dqn_var = np.std(dqn_times) if len(dqn_times) > 1 else 0
            
            ql_variability.append(ql_var)
            dqn_variability.append(dqn_var)
        
        x_var = range(len(ql_variability))
        width = 0.35
        
        ax11.bar([i - width/2 for i in x_var], ql_variability, width, 
                label='Q-Learning', color='orange', alpha=0.8)
        ax11.bar([i + width/2 for i in x_var], dqn_variability, width, 
                label='DQN', color='skyblue', alpha=0.8)
        
        ax11.set_title('Variabilidad en Tiempo para Comer', fontweight='bold')
        ax11.set_xlabel('Número de Comida')
        ax11.set_ylabel('Desviación Estándar (pasos)')
        ax11.set_xticks(x_var)
        ax11.set_xticklabels([f'#{i+1}' for i in x_var])
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        
        # 12. Estadísticas de sincronización por episodio
        ax12 = plt.subplot(4, 4, 12)
        episodes = sorted(self.episode_stats, key=lambda x: x['episode'])
        ep_numbers = [ep['episode'] for ep in episodes]
        sync_rates_ep = [ep['sync_rate'] for ep in episodes]
        
        colors_sync_ep = ['green' if rate == 100 else 'orange' if rate >= 80 else 'red' for rate in sync_rates_ep]
        bars = ax12.bar(ep_numbers, sync_rates_ep, color=colors_sync_ep, alpha=0.7)
        ax12.set_title('Sincronización por Episodio', fontweight='bold')
        ax12.set_xlabel('Episodio')
        ax12.set_ylabel('Sincronización (%)')
        ax12.grid(True, alpha=0.3)
        ax12.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Perfecto')
        
        # 13-16. Mapas de calor para comidas específicas
        important_foods = [2, 3, 4, 5]  # Comidas importantes después de #1
        for idx, food_num in enumerate(important_foods):
            ax = plt.subplot(4, 4, 13 + idx)
            
            food_data = self.df[self.df['Comida_Numero'] == food_num]
            
            if not food_data.empty:
                # Crear mapa de calor
                board_size = 16
                heatmap = np.zeros((board_size, board_size))
                
                for _, food_row in food_data.iterrows():
                    pos = food_row['QL_Posicion']
                    if isinstance(pos, tuple) and len(pos) == 2:
                        x, y = pos
                        if 0 <= x < board_size and 0 <= y < board_size:
                            heatmap[y, x] += 1
                
                # Visualizar
                if np.sum(heatmap) > 0:
                    im = ax.imshow(heatmap, cmap='Blues', alpha=0.8)
                    ax.set_title(f'Comida #{food_num}', fontweight='bold')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    
                    # Añadir información de éxito
                    success_rate = (food_data['Ambos_La_Comieron'].sum() / len(food_data) * 100) if len(food_data) > 0 else 0
                    ax.text(0.02, 0.98, f'Éxito: {success_rate:.0f}%', 
                           transform=ax.transAxes, fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                           verticalalignment='top')
                else:
                    ax.text(0.5, 0.5, f'Sin datos\nComida #{food_num}', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'No encontrada\nComida #{food_num}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, fontweight='bold', color='red')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('analisis_completo_unificado_visualizaciones.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("✓ Visualizaciones comprehensivas guardadas en 'analisis_completo_unificado_visualizaciones.png'")

    def analyze_food_sequences_validation(self):
        """Análisis específico de validación de secuencias con énfasis en comida #1"""
        print("\n🔍 ANÁLISIS DE VALIDACIÓN DE SECUENCIAS...")
        
        food_1_issues = []
        
        for ep_num, episode_data in self.all_food_info.items():
            ql_sequence = episode_data['ql_sequence']
            dqn_sequence = episode_data['dqn_sequence']
            
            # Verificar que las secuencias sean idénticas
            sequences_identical = ql_sequence == dqn_sequence
            
            # Verificar que las posiciones observadas coincidan con la secuencia
            ql_foods = episode_data['ql_foods']
            dqn_foods = episode_data['dqn_foods']
            
            positions_match = True
            mismatches = []
            
            #  VERIFICACIÓN ESPECIAL DE COMIDA #1
            food_1_status = "OK"
            if ql_sequence and len(ql_sequence) > 0:
                expected_pos_1 = ql_sequence[0]
                
                ql_observed_1 = ql_foods.get(1, {}).get('position')
                dqn_observed_1 = dqn_foods.get(1, {}).get('position')
                
                if not ql_observed_1:
                    food_1_status = "QL_MISSING"
                    food_1_issues.append(f"Ep {ep_num}: Q-Learning sin comida #1")
                elif not dqn_observed_1:
                    food_1_status = "DQN_MISSING"
                    food_1_issues.append(f"Ep {ep_num}: DQN sin comida #1")
                elif ql_observed_1 != expected_pos_1:
                    food_1_status = "QL_WRONG_POS"
                    food_1_issues.append(f"Ep {ep_num}: QL comida #1 en {ql_observed_1}, esperado {expected_pos_1}")
                elif dqn_observed_1 != expected_pos_1:
                    food_1_status = "DQN_WRONG_POS"
                    food_1_issues.append(f"Ep {ep_num}: DQN comida #1 en {dqn_observed_1}, esperado {expected_pos_1}")
                elif ql_observed_1 != dqn_observed_1:
                    food_1_status = "DESYNC"
                    food_1_issues.append(f"Ep {ep_num}: comida #1 desincronizada - QL:{ql_observed_1} vs DQN:{dqn_observed_1}")
            
            # Verificar el resto de comidas
            for food_num in range(1, min(len(ql_sequence) + 1, 21)):  # Hasta comida 20
                if food_num <= len(ql_sequence):
                    expected_pos = ql_sequence[food_num - 1]
                    
                    ql_observed = ql_foods.get(food_num, {}).get('position')
                    dqn_observed = dqn_foods.get(food_num, {}).get('position')
                    
                    if ql_observed and ql_observed != expected_pos:
                        positions_match = False
                        mismatches.append(f"QL comida #{food_num}: esperado {expected_pos}, observado {ql_observed}")
                    
                    if dqn_observed and dqn_observed != expected_pos:
                        positions_match = False
                        mismatches.append(f"DQN comida #{food_num}: esperado {expected_pos}, observado {dqn_observed}")
            
            self.validation_results[ep_num] = {
                'sequences_identical': sequences_identical,
                'positions_match': positions_match,
                'mismatches': mismatches,
                'sequence_length': len(ql_sequence),
                'ql_max_food_reached': max(ql_foods.keys()) if ql_foods else 0,
                'dqn_max_food_reached': max(dqn_foods.keys()) if dqn_foods else 0,
                'food_1_status': food_1_status,
                'ql_has_food_1': 1 in ql_foods,
                'dqn_has_food_1': 1 in dqn_foods
            }
        
        # Resumen de validación con énfasis en comida #1
        total_episodes = len(self.validation_results)
        perfect_sequences = sum(1 for v in self.validation_results.values() if v['sequences_identical'])
        perfect_positions = sum(1 for v in self.validation_results.values() if v['positions_match'])
        
        # Estadísticas específicas de comida #1
        episodes_with_ql_food_1 = sum(1 for v in self.validation_results.values() if v['ql_has_food_1'])
        episodes_with_dqn_food_1 = sum(1 for v in self.validation_results.values() if v['dqn_has_food_1'])
        episodes_with_both_food_1 = sum(1 for v in self.validation_results.values() 
                                       if v['ql_has_food_1'] and v['dqn_has_food_1'])
        perfect_food_1 = sum(1 for v in self.validation_results.values() if v['food_1_status'] == "OK")
        
        print(f"\nRESULTADOS DE VALIDACIÓN:")
        print(f"  Total de episodios: {total_episodes}")
        print(f"  Secuencias idénticas: {perfect_sequences}/{total_episodes} ({perfect_sequences/total_episodes*100:.1f}%)")
        print(f"  Posiciones correctas: {perfect_positions}/{total_episodes} ({perfect_positions/total_episodes*100:.1f}%)")
        
        print(f"\n ESTADO ESPECÍFICO DE COMIDA #1:")
        print(f"  Episodios con comida #1 en Q-Learning: {episodes_with_ql_food_1}/{total_episodes}")
        print(f"  Episodios con comida #1 en DQN: {episodes_with_dqn_food_1}/{total_episodes}")
        print(f"  Episodios con comida #1 en ambos: {episodes_with_both_food_1}/{total_episodes}")
        print(f"  Episodios con comida #1 perfecta: {perfect_food_1}/{total_episodes}")
        
        if perfect_food_1 == total_episodes:
            print(f"   COMIDA #1 COMPLETAMENTE CORREGIDA")
        else:
            print(f"   COMIDA #1 AÚN TIENE PROBLEMAS:")
            for issue in food_1_issues:
                print(f"    • {issue}")

    def print_detailed_food_analysis(self):
        """Imprimir análisis detallado de cada comida con énfasis en comida #1"""
        print("\n" + "="*100)
        print("ANÁLISIS DETALLADO POR NÚMERO DE COMIDA (UNIFICADO)")
        print("="*100)
        
        #  ANÁLISIS ESPECIAL DE COMIDA #1
        if 1 in self.food_analysis:
            analysis_1 = self.food_analysis[1]
            print(f"\n COMIDA #1 (ANÁLISIS ESPECIAL)")
            print("=" * 60)
            print(f"    ENCONTRADA en {analysis_1['episodios_aparece']} episodios")
            print(f"    Posiciones únicas: {analysis_1['posiciones_unicas']}")
            print(f"    Sincronización perfecta: {analysis_1['sincronizacion_perfecta']}/{analysis_1['episodios_aparece']} episodios")
            print(f"    Tasa de sincronización: {(analysis_1['sincronizacion_perfecta']/analysis_1['episodios_aparece']*100):.1f}%")
            print(f"    Tasa de éxito Q-Learning: {analysis_1['ql_success_rate']:.1f}%")
            print(f"    Tasa de éxito DQN: {analysis_1['dqn_success_rate']:.1f}%")
            print(f"    Ambos la comen: {analysis_1['ambos_exito_rate']:.1f}%")
            
            if analysis_1['ql_avg_time_to_eat'] != 'N/A':
                print(f"   Tiempo promedio Q-Learning: {analysis_1['ql_avg_time_to_eat']:.1f} pasos")
            if analysis_1['dqn_avg_time_to_eat'] != 'N/A':
                print(f"    Tiempo promedio DQN: {analysis_1['dqn_avg_time_to_eat']:.1f} pasos")
            
            print(f"   Posiciones más comunes:")
            for pos, count in analysis_1['posiciones_mas_comunes']:
                print(f"     {pos}: {count} veces")
        else:
            print(f"\n COMIDA #1 NO ENCONTRADA EN EL ANÁLISIS")
        
        # Análisis del resto de comidas
        for food_num, analysis in self.food_analysis.items():
            if food_num == 1:
                continue  # Ya se analizó arriba
                
            print(f"\n COMIDA #{food_num}")
            print("-" * 60)
            print(f"   Aparece en {analysis['episodios_aparece']} episodios")
            print(f"   Posiciones únicas: {analysis['posiciones_unicas']}")
            print(f"   Sincronización perfecta: {analysis['sincronizacion_perfecta']}/{analysis['episodios_aparece']} episodios")
            print(f"   Tasa de éxito Q-Learning: {analysis['ql_success_rate']:.1f}%")
            print(f"   Tasa de éxito DQN: {analysis['dqn_success_rate']:.1f}%")
            print(f"   Ambos la comen: {analysis['ambos_exito_rate']:.1f}%")
            
            if analysis['ql_avg_time_to_eat'] != 'N/A':
                print(f"   Tiempo promedio Q-Learning: {analysis['ql_avg_time_to_eat']:.1f} pasos")
            if analysis['dqn_avg_time_to_eat'] != 'N/A':
                print(f"   Tiempo promedio DQN: {analysis['dqn_avg_time_to_eat']:.1f} pasos")
            
            print(f"   Posiciones más comunes:")
            for pos, count in analysis['posiciones_mas_comunes']:
                print(f"     {pos}: {count} veces")

    def create_individual_episode_maps(self):
        """Crear mapas individuales para cada episodio - FUNCIONALIDAD PRINCIPAL"""
        print("\n CREANDO MAPAS INDIVIDUALES POR EPISODIO...")
        print("   Esta es una de las funcionalidades principales del análisis unificado")
        
        ql_foods, dqn_foods = self.extract_individual_episode_foods()
        
        # Obtener episodios comunes
        common_episodes = sorted(set(ql_foods.keys()).intersection(set(dqn_foods.keys())))
        print(f" Procesando {len(common_episodes)} episodios: {common_episodes}")
        
        print(f"\n GENERANDO MAPAS DETALLADOS:")
        print(f"   Cada mapa mostrará:")
        print(f"   ✓ Todas las comidas numeradas con colores únicos")
        print(f"   ✓ Posiciones exactas (x,y) para cada comida")
        print(f"   ✓ Comparación lado a lado DQN vs Q-Learning")
        print(f"   ✓ Comida #1 destacada con borde dorado")
        print(f"   ✓ Estadísticas de sincronización")
        print(f"   ✓ Leyenda de colores por comida")
        
        # Procesar cada episodio individualmente
        for episode_num in common_episodes:
            dqn_episode_foods = dqn_foods[episode_num]
            ql_episode_foods = ql_foods[episode_num]
            
            print(f"\n    Procesando Episodio {episode_num}:")
            print(f"      DQN: {len(dqn_episode_foods)} comidas detectadas")
            print(f"      Q-Learning: {len(ql_episode_foods)} comidas detectadas")
            
            # Mostrar preview de las primeras comidas
            all_foods = sorted(set(dqn_episode_foods.keys()).union(set(ql_episode_foods.keys())))
            if all_foods:
                print(f"      Comidas encontradas: #{min(all_foods)} a #{max(all_foods)}")
                
                # Verificar comida #1 específicamente
                if 1 in all_foods:
                    dqn_pos_1 = dqn_episode_foods.get(1, 'N/A')
                    ql_pos_1 = ql_episode_foods.get(1, 'N/A')
                    sync_1 = " SYNC" if dqn_pos_1 == ql_pos_1 else " DIFF"
                    print(f"       Comida #1: DQN={dqn_pos_1}, QL={ql_pos_1} {sync_1}")
            
            # Crear mapa completo individual
            stats = self.create_individual_episode_food_map(episode_num, dqn_episode_foods, ql_episode_foods)
            self.episode_stats.append(stats)
            
            # Mostrar información en consola
            print(f"    Resultado: {stats['total_foods']} comidas, Sync: {stats['sync_rate']:.1f}%")
        
        print(f"\n MAPAS INDIVIDUALES COMPLETADOS:")
        print(f"   • {len(common_episodes)} archivos PNG generados")
        print(f"   • Nomenclatura: episodio_XX_mapa_completo_unificado.png")
        print(f"   • Resolución: 250 DPI para máxima calidad")
        print(f"   • Cada imagen es autocontenida con toda la información necesaria")

    def create_episode_overview_grid(self):
        """Crear vista general de todos los episodios en una sola imagen"""
        
        print("\n Creando vista general de todos los episodios...")
        
        n_episodes = len(self.episode_stats)
        cols = 5
        rows = (n_episodes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(25, 5 * rows))
        fig.suptitle('VISTA GENERAL UNIFICADA - MAPAS DE COMIDAS POR EPISODIO', 
                     fontsize=20, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        board_size = 16
        
        for idx, stats in enumerate(self.episode_stats):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            episode_num = stats['episode']
            
            # Configurar subplot
            ax.set_xlim(-0.5, board_size - 0.5)
            ax.set_ylim(-0.5, board_size - 0.5)
            ax.set_aspect('equal')
            ax.set_title(f'Ep {episode_num}\n{stats["total_foods"]} comidas\nSync: {stats["sync_rate"]:.0f}%', 
                        fontweight='bold', fontsize=12)
            
            # Fondo según sincronización
            if stats['sync_rate'] == 100:
                bg_color = 'lightgreen'
            elif stats['sync_rate'] >= 80:
                bg_color = 'lightyellow'
            else:
                bg_color = 'lightcoral'
            
            ax.add_patch(patches.Rectangle((-0.5, -0.5), board_size, board_size, 
                                         facecolor=bg_color, alpha=0.3))
            
            # Solo mostrar algunos números para no saturar
            foods_to_show = min(stats['total_foods'], 8)
            ax.text(0.5, 0.5, f"Ver imagen\nindividual\npara detalles", 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Ocultar subplots vacíos
        for idx in range(len(self.episode_stats), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('vista_general_unificada_todos_episodios.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print("✓ Vista general guardada: vista_general_unificada_todos_episodios.png")
    def create_summary_statistics(self):
        """Crear resumen estadístico de todos los episodios"""
        
        print("\n Creando estadísticas resumidas...")
        
        # Calcular estadísticas
        total_episodes = len(self.episode_stats)
        perfect_sync_episodes = sum(1 for s in self.episode_stats if s['sync_rate'] == 100)
        avg_sync_rate = np.mean([s['sync_rate'] for s in self.episode_stats])
        avg_foods_per_episode = np.mean([s['total_foods'] for s in self.episode_stats])
        max_foods_episode = max(self.episode_stats, key=lambda s: s['total_foods'])
        min_foods_episode = min(self.episode_stats, key=lambda s: s['total_foods'])
        
        # Crear figura de estadísticas
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ESTADÍSTICAS GLOBALES UNIFICADAS DE MAPAS DE COMIDAS', fontsize=16, fontweight='bold')
        
        # 1. Sincronización por episodio
        episodes = [s['episode'] for s in self.episode_stats]
        sync_rates = [s['sync_rate'] for s in self.episode_stats]
        colors_sync = ['green' if rate == 100 else 'orange' if rate >= 80 else 'red' for rate in sync_rates]
        
        bars1 = ax1.bar(episodes, sync_rates, color=colors_sync, alpha=0.7)
        ax1.set_title('Tasa de Sincronización por Episodio', fontweight='bold')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Sincronización (%)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Sincronización Perfecta')
        
        # Agregar valores en las barras
        for bar, rate in zip(bars1, sync_rates):
            ax1.text(bar.get_x() + bar.get_width()/2., rate + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. Cantidad de comidas por episodio
        total_foods = [s['total_foods'] for s in self.episode_stats]
        dqn_foods = [s['dqn_foods'] for s in self.episode_stats]
        ql_foods = [s['ql_foods'] for s in self.episode_stats]
        
        x = np.arange(len(episodes))
        width = 0.35
        
        bars2 = ax2.bar(x - width/2, dqn_foods, width, label='DQN', color='skyblue', alpha=0.8)
        bars3 = ax2.bar(x + width/2, ql_foods, width, label='Q-Learning', color='orange', alpha=0.8)
        
        ax2.set_title('Cantidad de Comidas por Episodio', fontweight='bold')
        ax2.set_xlabel('Episodio')
        ax2.set_ylabel('Número de Comidas')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Ep {ep}' for ep in episodes])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribución de sincronización (CORREGIDO - sin alpha)
        sync_categories = ['Perfecta (100%)', 'Alta (80-99%)', 'Baja (<80%)']
        perfect_count = sum(1 for rate in sync_rates if rate == 100)
        high_count = sum(1 for rate in sync_rates if 80 <= rate < 100)
        low_count = sum(1 for rate in sync_rates if rate < 80)
        
        counts = [perfect_count, high_count, low_count]
        colors_pie = ['green', 'orange', 'red']
        
        # CORRECCIÓN: Remover el parámetro alpha de pie()
        ax3.pie(counts, labels=sync_categories, colors=colors_pie, autopct='%1.1f%%', 
               startangle=90)
        ax3.set_title('Distribución de Sincronización', fontweight='bold')
        
        # 4. Tabla de estadísticas
        ax4.axis('off')
        
        stats_text = f"""RESUMEN ESTADÍSTICO GLOBAL UNIFICADO:

EPISODIOS ANALIZADOS: {total_episodes}

SINCRONIZACIÓN:
  • Episodios perfectos: {perfect_sync_episodes}/{total_episodes}
  • Tasa promedio: {avg_sync_rate:.1f}%
  • Estado general: {'EXCELENTE' if avg_sync_rate >= 95 else 'BUENO' if avg_sync_rate >= 80 else 'PARCIAL'}

COMIDAS:
  • Promedio por episodio: {avg_foods_per_episode:.1f}
  • Episodio con más comidas: {max_foods_episode['episode']} ({max_foods_episode['total_foods']} comidas)
  • Episodio con menos comidas: {min_foods_episode['episode']} ({min_foods_episode['total_foods']} comidas)

CONTROL EXPERIMENTAL:
  • Tasa de validación: {perfect_sync_episodes/total_episodes*100:.1f}%
  • Estado: {'PERFECTO' if perfect_sync_episodes == total_episodes else 'PARCIAL'}
  • Análisis unificado: COMPLETO
"""
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', 
                          facecolor='lightgreen' if avg_sync_rate >= 95 else 'lightyellow', 
                          alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('estadisticas_unificadas_mapas_comidas.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print("Estadísticas guardadas: estadisticas_unificadas_mapas_comidas.png")
        
        return {
            'total_episodes': total_episodes,
            'perfect_sync_episodes': perfect_sync_episodes,
            'avg_sync_rate': avg_sync_rate,
            'avg_foods_per_episode': avg_foods_per_episode
        }
    def create_individual_food_reports(self):
        """Crear reportes individuales para cada comida"""
        print("\nCREANDO REPORTES INDIVIDUALES POR COMIDA...")
        
        with open('reporte_detallado_unificado_por_comida.txt', 'w', encoding='utf-8') as f:
            f.write("REPORTE DETALLADO UNIFICADO POR NÚMERO DE COMIDA\n")
            f.write("=" * 80 + "\n\n")
            
            for food_num in sorted(self.food_analysis.keys()):
                analysis = self.food_analysis[food_num]
                food_data = self.df[self.df['Comida_Numero'] == food_num]
                
                f.write(f"COMIDA #{food_num}")
                if food_num == 1:
                    f.write("  (COMIDA INICIAL - VERIFICADA)")
                f.write("\n")
                f.write("-" * 40 + "\n")
                
                f.write(f"INFORMACIÓN GENERAL:\n")
                f.write(f"  • Aparece en {analysis['episodios_aparece']} episodios\n")
                f.write(f"  • Posiciones únicas utilizadas: {analysis['posiciones_unicas']}\n")
                f.write(f"  • Sincronización perfecta: {analysis['sincronizacion_perfecta']}/{analysis['episodios_aparece']} episodios\n")
                f.write(f"  • Tasa de sincronización: {(analysis['sincronizacion_perfecta']/analysis['episodios_aparece']*100):.1f}%\n\n")
                
                f.write(f"RENDIMIENTO:\n")
                f.write(f"  • Q-Learning la come: {analysis['ql_success_rate']:.1f}% de las veces\n")
                f.write(f"  • DQN la come: {analysis['dqn_success_rate']:.1f}% de las veces\n")
                f.write(f"  • Ambos la comen: {analysis['ambos_exito_rate']:.1f}% de las veces\n\n")
                
                if analysis['ql_avg_time_to_eat'] != 'N/A':
                    f.write(f"TIEMPO PARA COMER:\n")
                    f.write(f"  • Q-Learning promedio: {analysis['ql_avg_time_to_eat']:.1f} pasos\n")
                    if analysis['dqn_avg_time_to_eat'] != 'N/A':
                        f.write(f"  • DQN promedio: {analysis['dqn_avg_time_to_eat']:.1f} pasos\n")
                        diff = analysis['dqn_avg_time_to_eat'] - analysis['ql_avg_time_to_eat']
                        faster = "Q-Learning" if diff > 0 else "DQN"
                        f.write(f"  • Más rápido: {faster} ({abs(diff):.1f} pasos de diferencia)\n")
                    f.write("\n")
                
                f.write(f"POSICIONES MÁS FRECUENTES:\n")
                for pos, count in analysis['posiciones_mas_comunes']:
                    f.write(f"  • {pos}: aparece {count} veces\n")
                
                f.write(f"\nDETALLE POR EPISODIO:\n")
                f.write(f"Ep | Posición    | QL Come | DQN Come | QL Pasos | DQN Pasos | Sync\n")
                f.write("-" * 70 + "\n")
                
                for _, row in food_data.iterrows():
                    ep = row['Episodio']
                    pos = str(row['QL_Posicion'])[:10]
                    ql_ate = "✓" if row['QL_Fue_Comida'] else "✗"
                    dqn_ate = "✓" if row['DQN_Fue_Comida'] else "✗"
                    ql_steps = str(row['QL_Pasos_Para_Comer']) if row['QL_Pasos_Para_Comer'] != 'N/A' else "-"
                    dqn_steps = str(row['DQN_Pasos_Para_Comer']) if row['DQN_Pasos_Para_Comer'] != 'N/A' else "-"
                    sync = "✓" if row['Posiciones_Coinciden'] else "✗"
                    
                    f.write(f"{ep:2d} | {pos:11s} | {ql_ate:7s} | {dqn_ate:8s} | {ql_steps:8s} | {dqn_steps:9s} | {sync:4s}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print("✓ Reporte detallado guardado en 'reporte_detallado_unificado_por_comida.txt'")

    def generate_comprehensive_summary(self):
        """Generar resumen comprehensivo final unificado"""
        print("\n GENERANDO RESUMEN COMPREHENSIVO UNIFICADO...")
        
        with open('resumen_comprehensivo_unificado.txt', 'w', encoding='utf-8') as f:
            f.write("RESUMEN COMPREHENSIVO UNIFICADO - ANÁLISIS COMPLETO DE TODAS LAS COMIDAS\n")
            f.write("=" * 80 + "\n\n")
            
            # Estadísticas generales
            total_records = len(self.df)
            total_episodes = len(self.df['Episodio'].unique())
            total_foods = len(self.df['Comida_Numero'].unique())
            
            f.write("ESTADÍSTICAS GENERALES:\n")
            f.write(f"  • Total de registros analizados: {total_records}\n")
            f.write(f"  • Episodios: {total_episodes}\n")
            f.write(f"  • Números de comida diferentes: {total_foods}\n")
            f.write(f"  • Rango de comidas: #{min(self.df['Comida_Numero'])} a #{max(self.df['Comida_Numero'])}\n\n")
            
            # ANÁLISIS ESPECÍFICO DE COMIDA #1
            food_1_data = self.df[self.df['Comida_Numero'] == 1]
            f.write("ANÁLISIS ESPECÍFICO DE COMIDA #1:\n")
            if len(food_1_data) > 0:
                sync_count_1 = food_1_data['Posiciones_Coinciden'].sum()
                f.write(f"   Comida #1 encontrada en {len(food_1_data)} episodios\n")
                f.write(f"   Sincronización perfecta: {sync_count_1}/{len(food_1_data)} episodios\n")
                f.write(f"   Tasa de sincronización: {sync_count_1/len(food_1_data)*100:.1f}%\n")
                
                if 1 in self.food_analysis:
                    analysis_1 = self.food_analysis[1]
                    f.write(f"   Tasa de éxito Q-Learning: {analysis_1['ql_success_rate']:.1f}%\n")
                    f.write(f"   Tasa de éxito DQN: {analysis_1['dqn_success_rate']:.1f}%\n")
                    f.write(f"   Ambos la comen: {analysis_1['ambos_exito_rate']:.1f}%\n")
                
                f.write(f"   ESTADO: COMIDA #1 VERIFICADA EN ANÁLISIS UNIFICADO\n\n")
            else:
                f.write(f"   ERROR: Comida #1 aún no se encuentra\n\n")
            
            # Validación experimental
            perfect_sync = sum(1 for v in self.validation_results.values() 
                              if v['sequences_identical'] and v['positions_match'])
            perfect_food_1 = sum(1 for v in self.validation_results.values() if v['food_1_status'] == "OK")
            
            f.write("VALIDACIÓN EXPERIMENTAL:\n")
            f.write(f"  • Episodios con validación perfecta: {perfect_sync}/{total_episodes}\n")
            f.write(f"  • Tasa de validación general: {perfect_sync/total_episodes*100:.1f}%\n")
            f.write(f"  • Episodios con comida #1 perfecta: {perfect_food_1}/{total_episodes}\n")
            f.write(f"  • Tasa de validación comida #1: {perfect_food_1/total_episodes*100:.1f}%\n")
            f.write(f"  • Control experimental: {'EXITOSO' if perfect_sync == total_episodes else 'PARCIAL'}\n")
            f.write(f"  • Control comida #1: {'PERFECTO' if perfect_food_1 == total_episodes else 'PARCIAL'}\n\n")
            
            # Análisis de rendimiento
            avg_ql_success = np.mean([self.food_analysis[fn]['ql_success_rate'] for fn in self.food_analysis.keys()])
            avg_dqn_success = np.mean([self.food_analysis[fn]['dqn_success_rate'] for fn in self.food_analysis.keys()])
            
            f.write("RENDIMIENTO PROMEDIO:\n")
            f.write(f"  • Q-Learning: {avg_ql_success:.2f}% de éxito\n")
            f.write(f"  • DQN: {avg_dqn_success:.2f}% de éxito\n")
            f.write(f"  • Mejor algoritmo: {'Q-Learning' if avg_ql_success > avg_dqn_success else 'DQN' if avg_dqn_success > avg_ql_success else 'Empate'}\n")
            f.write(f"  • Diferencia: {abs(avg_ql_success - avg_dqn_success):.2f} puntos porcentuales\n\n")
            
            # Top comidas por dificultad
            easiest_foods = sorted(self.food_analysis.keys(), 
                                  key=lambda fn: self.food_analysis[fn]['ambos_exito_rate'], 
                                  reverse=True)[:5]
            hardest_foods = sorted(self.food_analysis.keys(), 
                                  key=lambda fn: self.food_analysis[fn]['ambos_exito_rate'])[:5]
            
            f.write("COMIDAS MÁS FÁCILES (mayor tasa de éxito conjunto):\n")
            for i, food_num in enumerate(easiest_foods, 1):
                rate = self.food_analysis[food_num]['ambos_exito_rate']
                special = " (INICIAL)" if food_num == 1 else ""
                f.write(f"  {i}. Comida #{food_num}: {rate:.1f}% de éxito{special}\n")
            
            f.write("\nCOMIDAS MÁS DIFÍCILES (menor tasa de éxito conjunto):\n")
            for i, food_num in enumerate(hardest_foods, 1):
                rate = self.food_analysis[food_num]['ambos_exito_rate']
                special = "  (INICIAL)" if food_num == 1 else ""
                f.write(f"  {i}. Comida #{food_num}: {rate:.1f}% de éxito{special}\n")
            
            # Estadísticas de mapas individuales
            if self.episode_stats:
                summary_stats = self.create_summary_statistics()
                f.write(f"\nESTADÍSTICAS DE MAPAS INDIVIDUALES:\n")
                f.write(f"  • Episodios procesados: {summary_stats['total_episodes']}\n")
                f.write(f"  • Sincronización perfecta: {summary_stats['perfect_sync_episodes']}\n")
                f.write(f"  • Tasa promedio de sincronización: {summary_stats['avg_sync_rate']:.1f}%\n")
                f.write(f"  • Promedio de comidas por episodio: {summary_stats['avg_foods_per_episode']:.1f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("CONCLUSIONES UNIFICADAS:\n\n")
            
            # Conclusión específica sobre comida #1
            if perfect_food_1 == total_episodes:
                f.write(" COMIDA #1 COMPLETAMENTE VERIFICADA:\n")
                f.write("   • La comida #1 aparece correctamente en todos los episodios\n")
                f.write("   • Las posiciones están perfectamente sincronizadas\n")
                f.write("   • El análisis unificado confirma la corrección\n\n")
            else:
                f.write(" COMIDA #1 PARCIALMENTE VERIFICADA:\n")
                f.write("   • Algunos episodios aún tienen problemas con comida #1\n")
                f.write("   • Revisar logs de verificación para detalles específicos\n\n")
            
            if perfect_sync == total_episodes:
                f.write(" CONTROL EXPERIMENTAL PERFECTO:\n")
                f.write("   • Todas las secuencias están perfectamente sincronizadas\n")
                f.write("   • Las comparaciones entre algoritmos son completamente válidas\n")
                f.write("   • Los resultados son científicamente confiables\n\n")
            else:
                f.write(" CONTROL EXPERIMENTAL PARCIAL:\n")
                f.write("   • Algunas secuencias tienen problemas de sincronización\n")
                f.write("   • Interpretar resultados con precaución\n")
                f.write("   • Revisar implementación del controlador\n\n")
            
            if avg_ql_success > avg_dqn_success + 2:
                f.write(" Q-LEARNING SUPERIOR:\n")
                f.write(f"   • Q-Learning supera consistentemente a DQN\n")
                f.write(f"   • Ventaja promedio: {avg_ql_success - avg_dqn_success:.1f} puntos\n")
            elif avg_dqn_success > avg_ql_success + 2:
                f.write(" DQN SUPERIOR:\n")
                f.write(f"   • DQN supera consistentemente a Q-Learning\n")
                f.write(f"   • Ventaja promedio: {avg_dqn_success - avg_ql_success:.1f} puntos\n")
            else:
                f.write(" RENDIMIENTO SIMILAR:\n")
                f.write("   • Ambos algoritmos tienen rendimiento comparable\n")
                f.write("   • Diferencias no son estadísticamente significativas\n")
        
        print("✓ Resumen comprehensivo unificado guardado en 'resumen_comprehensivo_unificado.txt'")

    def run_complete_analysis(self):
        """Ejecutar el análisis completo unificado"""
        print(" Iniciando análisis completo unificado...")
        
        # 1. Cargar datos
        if not self.load_all_data():
            print(" No se pudieron cargar los datos necesarios")
            return
        
        # 2. Verificar presencia de comida #1
        food_1_status = self.verify_food_1_presence()
        
        # 3. Extraer información completa
        self.extract_complete_food_info()
        
        if not self.all_food_info:
            print(" No se pudo extraer información de comidas")
            return
        
        print(f"✓ Información extraída para {len(self.all_food_info)} episodios")
        
        # 4. Crear tabla completa
        self.create_complete_food_table()
        
        # 5. Analizar por número de comida
        self.analyze_food_by_number()
        
        # 6. Imprimir análisis detallado con énfasis en comida #1
        self.print_detailed_food_analysis()
        
        # 7. Crear mapas individuales por episodio
        self.create_individual_episode_maps()
        
        # 8. Crear vista general de episodios
        self.create_episode_overview_grid()
        
        # 9. Crear estadísticas resumidas
        summary_stats = self.create_summary_statistics()
        
        # 10. Crear visualizaciones comprehensivas
        self.create_comprehensive_visualizations()
        
        # 11. Validar secuencias con énfasis en comida #1
        self.analyze_food_sequences_validation()
        
        # 12. Crear reportes individuales
        self.create_individual_food_reports()
        
        # 13. Generar resumen final unificado
        self.generate_comprehensive_summary()
        
        # 14. Mostrar resumen final
        self.display_final_summary(food_1_status, summary_stats)

    def display_final_summary(self, food_1_status, summary_stats):
        """Mostrar resumen final del análisis unificado"""
        print("\n" + "="*80)
        print(" ANÁLISIS COMPLETO UNIFICADO FINALIZADO")
        print("="*80)
        
        print(f"\n ARCHIVOS GENERADOS:")
        files_generated = [
            'analisis_completo_unificado_todas_las_comidas.csv',
            'analisis_completo_unificado_visualizaciones.png',
            'reporte_detallado_unificado_por_comida.txt',
            'vista_general_unificada_todos_episodios.png',
            'estadisticas_unificadas_mapas_comidas.png',
            'resumen_comprehensivo_unificado.txt'
        ]
        
        for i, filename in enumerate(files_generated, 1):
            print(f"  {i}. {filename}")
        
        
        print(f"\n    ARCHIVOS GENERADOS:")
        for stats in self.episode_stats:
            ep_num = stats['episode']
            total_foods = stats['total_foods']
            sync_rate = stats['sync_rate']
            status = " correcto" if sync_rate == 100 else "peligro" if sync_rate >= 80 else "error"
            print(f"      • episodio_{ep_num:02d}_mapa_completo_unificado.png")
            print(f"        └─ {total_foods} comidas, Sincronización: {sync_rate:.0f}% {status}")
        
    
        print(f"\n ESTADÍSTICAS FINALES:")
        total_foods = len(self.food_analysis)
        total_episodes = len(self.all_food_info)
        
        print(f"   • {total_foods} números de comida analizados")
        print(f"   • {total_episodes} episodios procesados")
        print(f"   • {len(self.df)} registros individuales")
        
        if summary_stats:
            print(f"   • Sincronización promedio: {summary_stats['avg_sync_rate']:.1f}%")
            print(f"   • Episodios perfectos: {summary_stats['perfect_sync_episodes']}")
        
        # Verificación final de comida #1
        food_1_data = self.df[self.df['Comida_Numero'] == 1]
        if len(food_1_data) > 0:
            sync_rate = food_1_data['Posiciones_Coinciden'].sum() / len(food_1_data) * 100
            print(f"\n RESULTADO FINAL COMIDA #1:")
            print(f"   • Encontrada en {len(food_1_data)} episodios")
            print(f"   • Sincronización: {sync_rate:.1f}%")
            print(f"   • Estado: {'VERIFICADA' if sync_rate >= 95 else ' PARCIAL'}")
            
            if 1 in self.food_analysis:
                analysis_1 = self.food_analysis[1]
                print(f"   • Éxito Q-Learning: {analysis_1['ql_success_rate']:.1f}%")
                print(f"   • Éxito DQN: {analysis_1['dqn_success_rate']:.1f}%")
                print(f"   • Éxito conjunto: {analysis_1['ambos_exito_rate']:.1f}%")
        else:
            print(f"\n COMIDA #1 AÚN NO ENCONTRADA")
            print(f"   • Verificar que se ejecutó correctamente el análisis")
        
        # Algoritmo ganador
        if self.food_analysis:
            avg_ql = np.mean([self.food_analysis[fn]['ql_success_rate'] for fn in self.food_analysis.keys()])
            avg_dqn = np.mean([self.food_analysis[fn]['dqn_success_rate'] for fn in self.food_analysis.keys()])
            
            print(f"\n ALGORITMO GANADOR:")
            if avg_ql > avg_dqn + 2:
                print(f"    Q-Learning superior ({avg_ql:.1f}% vs {avg_dqn:.1f}%)")
            elif avg_dqn > avg_ql + 2:
                print(f"    DQN superior ({avg_dqn:.1f}% vs {avg_ql:.1f}%)")
            else:
                print(f"    Rendimiento similar (QL: {avg_ql:.1f}%, DQN: {avg_dqn:.1f}%)")
        
        print(f"\n ANÁLISIS UNIFICADO COMPLETADO EXITOSAMENTE")


def main():
    """Función principal para ejecutar el análisis unificado"""
    analyzer = UnifiedFoodAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()