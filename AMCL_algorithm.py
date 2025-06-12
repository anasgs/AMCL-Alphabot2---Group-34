# python3 AMCL_v6.py test_18_simples.bag

import rospy
import rosbag
import random
import numpy as np
import tf
import copy
import time
import os
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Int32MultiArray, Int32, Float32MultiArray
from sensor_msgs.msg import CompressedImage

import argparse
import traceback

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
import cv2

# Parâmetros otimizados
MIN_PARTICLES = 250
MAX_PARTICLES = 500
PROCESSING_SKIP = 1

# --- MAPA PEQUENO CORREDOR ---
RESOLUTION = 0.3  # metros por célula
RECTANGLE_WIDTH_M = 6.0   # área navegável
RECTANGLE_HEIGHT_M = 1.8  # área navegável
WALL_THICKNESS = 1  # células de espessura da parede

# Conversões para células
inner_cols = int(RECTANGLE_WIDTH_M / RESOLUTION)  # 20 células
inner_rows = int(RECTANGLE_HEIGHT_M / RESOLUTION)  # 6 células

# Dimensões totais incluindo as paredes
total_cols = inner_cols + 2 * WALL_THICKNESS  # 22 células
total_rows = inner_rows + 2 * WALL_THICKNESS  # 8 células

print(f"Mapa: {RECTANGLE_WIDTH_M}m x {RECTANGLE_HEIGHT_M}m")
print(f"Células navegáveis: {inner_cols} x {inner_rows}")
print(f"Dimensões totais: {total_cols} x {total_rows} células")

# Criar grelha: 1 = parede (obstáculo), 0 = área livre
grid = np.ones((total_rows, total_cols), dtype=int)

# Área navegável no centro
start_row = WALL_THICKNESS  # linha 1
end_row = start_row + inner_rows  # linha 7
start_col = WALL_THICKNESS   # coluna 1
end_col = start_col + inner_cols  # coluna 10

grid[start_row:end_row, start_col:end_col] = 0

GRID_LAYOUT = grid
GRID_ROWS, GRID_COLS = grid.shape

VALID_POSITIONS = [(r, c) for r in range(start_row, end_row) 
                   for c in range(start_col, end_col)]
VALID_POSITIONS_ARRAY = np.array(VALID_POSITIONS)

# Posições dos ArUcos PARA VISUALIZAÇÃO (nas paredes)
MAP_MARKERS_REAL = {
    0: (start_row+3, end_col-1),           # ArUco 0: segunda linha, última coluna navegável
    1: (start_row, end_col-1),             # ArUco 1: primeira linha, última coluna navegável
    2: (end_row - 1, start_col+8),         # ArUco 2: última linha, coluna 9
    3: (end_row-1, start_col+1),           # ArUco 3: última linha, coluna 2
    4: (start_row, start_col + int(1.0 / RESOLUTION)+9),  # ArUco 4: primeira linha, coluna 13
    5: (start_row, start_col+2),           # ArUco 5: primeira linha, coluna 3
    8: (end_row-1, start_col + int(1.0 / RESOLUTION)+9), # ArUco 8: última linha, coluna 13
    10: (start_row, start_col + int(1.0 / RESOLUTION)+3) # ArUco 10: primeira linha, coluna 7
}
# Posições dos ArUcos PARA VISUALIZAÇÃO (nas paredes)
MAP_MARKERS_DISPLAY = {
    0: (start_row+3, end_col),             # ArUco 0: segunda linha, parede direita
    1: (start_row-1, end_col-1),           # ArUco 1: parede superior, última coluna
    2: (end_row, start_col+8),             # ArUco 2: parede inferior, coluna 9
    3: (end_row, start_col+1),             # ArUco 3: parede inferior, coluna 2
    4: (start_row-1, start_col + int(1.0 / RESOLUTION)+9), # ArUco 4: parede superior, coluna 13
    5: (start_row-1, start_col+2),         # ArUco 5: parede superior, coluna 3
    8: (end_row, start_col + int(1.0 / RESOLUTION)+9),    # ArUco 8: parede inferior, coluna 13
    10: (start_row-1, start_col + int(1.0 / RESOLUTION)+3) # ArUco 10: parede superior, coluna 7
}

ARUCO_POSITIONS = np.array(list(MAP_MARKERS_REAL.values()))

ARUCO_ORIENTATIONS = {
    0: (-1, 0),   # ArUco 0: horizontal, aponta para esquerda
    1: (0, 1),   # ArUco 1: vertical, aponta para baixo  
    2: (-1, 0),   # ArUco 2: horizontal, aponta para esquerda
    3: (1, 0),    # ArUco 3: horizontal, aponta para direita
    4: (0, 1),   # ArUco 4: vertical, aponta para baixo
    5: (0, 1),   # ArUco 5: vertical, aponta para baixo
    8: (0, -1),    # ArUco 8: vertical, aponta para cima
    10: (-1, 0),  # ArUco 10: horizontal, aponta para esquerda
}

# TRAJETÓRIA GROUND TRUTH ***
GROUND_TRUTH_CHECKPOINTS = [
    (4, 19),  # Checkpoint 1 - início
    (5, 13),  # Checkpoint 2
    (6, 5),   # Checkpoint 3
    (2, 3),   # Checkpoint 4
    (2, 13),  # Checkpoint 5
    (4, 19)   # Checkpoint 6 - final (mesmo que início)
]

class Particle:
    def __init__(self, row, col, theta=0.0, weight=1.0):
        self.row = row
        self.col = col
        self.row_continuous = float(row)
        self.col_continuous = float(col)
        self.theta = theta
        self.weight = weight

    def move(self, linear_vel, angular_vel, dt):
        """Movimento baseado em velocidades reais do cmd_vel """
        self.move_2d(linear_vel, 0.0, angular_vel, dt)

    def move_2d(self, linear_vel_x, linear_vel_y, angular_vel_z, dt):
        """Movimento 2D completo com velocidades X, Y e rotação Z"""
        # Aplicar escala de velocidade
        scaled_linear_x = linear_vel_x 
        scaled_linear_y = linear_vel_y 
        scaled_angular_z = angular_vel_z 
        
        # Magnitude para cálculo de ruído
        linear_magnitude = np.sqrt(scaled_linear_x**2 + scaled_linear_y**2)
        
        # Ruído proporcional
        sigma_pos = max(0.01, 0.1 * linear_magnitude + 0.02) 
        sigma_theta = max(0.01, 0.05 * abs(scaled_angular_z) + 0.02)
        
        # Movimento angular
        self.theta += scaled_angular_z * dt + np.random.normal(0, sigma_theta)
        # Wrapping eficiente
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        # Movimento linear 2D no referencial global
        # As velocidades linear_vel_x e linear_vel_y são no referencial do robô
        # Transformar para referencial global usando a orientação atual
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        
        # Transformação do referencial do robô para o global
        global_vel_x = scaled_linear_x * cos_theta - scaled_linear_y * sin_theta
        global_vel_y = scaled_linear_x * sin_theta + scaled_linear_y * cos_theta
        
        # Adicionar ruído
        dx = global_vel_x * dt + np.random.normal(0, sigma_pos * RESOLUTION * 0.1)
        dy = global_vel_y * dt + np.random.normal(0, sigma_pos * RESOLUTION * 0.1)
        
        # Atualização direta das coordenadas contínuas
        self.col_continuous = np.clip(self.col_continuous + dx / RESOLUTION, start_col, end_col - 1)
        self.row_continuous = np.clip(self.row_continuous + dy / RESOLUTION, start_row, end_row - 1)
        
        # Coordenadas discretas
        self.col = int(round(self.col_continuous))
        self.row = int(round(self.row_continuous))

class MCLProcessor:
    def __init__(self, bag_path, output_bag_path=None):
        print("=== MCL com Distâncias de Marcadores ===")

        self.bag_path = bag_path
        self.output_bag_path = output_bag_path or bag_path.replace('.bag', '_mcl_with_distances.bag')
        
        # Calibração do sensor de distância baseada em dados reais
        self.setup_distance_calibration()
        
        # Inicialização das partículas concentradas perto da posição inicial
        self.N = MIN_PARTICLES
        self.particles = self.init_particles_uniform()
        
        # Controlo de movimento
        self.last_cmd_vel = None
        self.last_cmd_time = None
        
        # Controlo temporal
        self.last_aruco_time = None
        
        self.current_aruco_poses = {}  # {marker_id: (x, y, z, qx, qy, qz, qw)}
        self.pose_history = []  # Lista de (timestamp, {marker_id: pose})
        self.last_pose_time = None
        
        # Armazenamento de distâncias dos marcadores
        self.current_marker_distances = {}  # {marker_id: distance}
        self.distance_history = []  # Lista de (timestamp, {marker_id: distance})
        self.last_distance_time = None
        
        # Contadores
        self.frame_count = 0
        self.processed_frames = 0
        self.aruco_detections = 0
        self.distance_detections = 0  # Contador de detecções de distância
        self.pose_detections = 0
        self.start_time = None
        
        # Resultados
        self.pose_estimates = []
        self.particle_counts = []
        self.detected_markers = []
        self.confidence_history = []
        self.entropy_history = []  
        self.aruco_timeline = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 8: [], 10: []}
        self.particle_history = []
        self.velocity_history = []
        
        self._last_confidence = 0.0
        self._last_entropy = 0.0
    
    def setup_distance_calibration(self):
        """Configura calibração do sensor baseada em dados reais coletados"""
    
        # Dados de calibração coletados: robô vs realidade
        robot_measurements = [2.5, 7.02, 6.70, 6.15, 5.9, 5.47, 5, 4.1, 2.8, 2.2, 1.4]
        real_distances = [2.21, 6.20, 5.9, 5.6, 5.3, 5, 4.4, 3.8, 2.6, 2, 1.5]
        
        # Análise estatística dos erros
        errors = [r - real for r, real in zip(robot_measurements, real_distances)]
        self.calibration_stats = {
            'mean_error': sum(errors) / len(errors),
            'std_error': (sum([(e - sum(errors)/len(errors))**2 for e in errors]) / (len(errors)-1))**0.5,
            'relative_error': sum([abs(e)/real for e, real in zip(errors, real_distances)]) / len(errors) * 100
        }
        
        # Modelo de correção linear: real = slope * measured + intercept
        # Regressão linear simples
        n = len(robot_measurements)
        sum_x = sum(robot_measurements)
        sum_y = sum(real_distances)
        sum_xy = sum([x*y for x, y in zip(robot_measurements, real_distances)])
        sum_x2 = sum([x*x for x in robot_measurements])
        
        self.calibration_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        self.calibration_intercept = (sum_y - self.calibration_slope * sum_x) / n
        
        # Calcular R² e RMSE
        corrected = [self.calibration_slope * m + self.calibration_intercept for m in robot_measurements]
        ss_res = sum([(real - corr)**2 for real, corr in zip(real_distances, corrected)])
        ss_tot = sum([(real - sum(real_distances)/len(real_distances))**2 for real in real_distances])
        self.calibration_r2 = 1 - (ss_res / ss_tot)
        self.calibration_rmse = (ss_res / len(real_distances))**0.5
        
        self.distance_measurement_sigma = max(0.15, self.calibration_rmse * 1.5)  # Mínimo 15cm
       
    def correct_distance_measurement(self, measured_distance):
        """Corrige distância medida usando modelo de calibração"""
        if measured_distance <= 0:
            return measured_distance
            
        # Aplicar correção linear
        corrected = self.calibration_slope * measured_distance + self.calibration_intercept
        
        # Garantir que a correção é razoável (não pode ser negativa ou muito diferente)
        corrected = max(0.1, corrected)
        
        return corrected   
    
    def quaternion_to_euler(self, qx, qy, qz, qw):
        """Converte quaternion para ângulos de Euler (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def calculate_robot_orientation_from_aruco_pose(self, aruco_id, aruco_pose, robot_to_aruco_distance):
        """Calcula orientação do robô com base na pose do ArUco detectado"""
        if aruco_id not in ARUCO_ORIENTATIONS:
            return None
            
        x, y, z, qx, qy, qz, qw = aruco_pose
        
        # Converter quaternion para ângulo yaw
        _, _, aruco_yaw_in_camera = self.quaternion_to_euler(qx, qy, qz, qw)
        
        # Orientação do ArUco no mapa mundial
        aruco_orientation = ARUCO_ORIENTATIONS[aruco_id]
        aruco_world_yaw = np.arctan2(aruco_orientation[1], aruco_orientation[0])
        
        # MÉTODO 1: Baseado na geometria relativa
        camera_to_aruco_angle = np.arctan2(x, z)
        estimated_robot_theta = aruco_world_yaw + np.pi - camera_to_aruco_angle
        
        # MÉTODO 2: Usando a orientação observada do ArUco
        orientation_difference = aruco_yaw_in_camera - aruco_world_yaw
        estimated_robot_theta_v2 = -orientation_difference
        
        # MÉTODO 3: Combinação híbrida
        weight1, weight2 = 0.7, 0.3
        combined_theta = weight1 * estimated_robot_theta + weight2 * estimated_robot_theta_v2
        
        # Normalizar ângulo
        combined_theta = np.arctan2(np.sin(combined_theta), np.cos(combined_theta))
        
        return combined_theta

    def process_aruco_poses(self, poses_msg, timestamp):
        """Processa poses dos ArUcos do tópico /aruco/marker_poses"""
        try:
            poses_data = str(poses_msg).strip()
            if not poses_data:
                return
                
            lines = poses_data.split('\n')
            current_poses = {}
            
            for line in lines:
                if not line.strip():
                    continue
                    
                try:
                    # Parse: timestamp,marker_id,detection_timestamp,frame_id,x,y,z,qx,qy,qz,qw
                    parts = line.split(',')
                    if len(parts) >= 11:
                        marker_id = int(parts[1])
                        
                        # Extrair pose (posição + quaternion)
                        x = float(parts[4])
                        y = float(parts[5])
                        z = float(parts[6])
                        qx = float(parts[7])
                        qy = float(parts[8])
                        qz = float(parts[9])
                        qw = float(parts[10])
                        
                        if marker_id in MAP_MARKERS_REAL:
                            current_poses[marker_id] = (x, y, z, qx, qy, qz, qw)
                            
                except (ValueError, IndexError) as e:
                    print(f"❌ Erro ao processar linha de pose: {line[:50]}... - {e}")
                    continue
            
            if current_poses:
                self.current_aruco_poses = current_poses
                self.pose_history.append((timestamp, current_poses.copy()))
                self.pose_detections += 1
                self.last_pose_time = timestamp
                
                print(f"\n🎯 Poses de ArUcos detectadas: {list(current_poses.keys())}")
                
        except Exception as e:
            print(f"❌ Erro ao processar poses de ArUcos: {e}")
            traceback.print_exc()
       
    def init_particles_uniform(self):
        """Inicializa partículas espalhadas por toda a área navegável"""
        particles = []
        
        indices = np.random.choice(len(VALID_POSITIONS), self.N, replace=True)
        positions = VALID_POSITIONS_ARRAY[indices]
        thetas = np.random.uniform(-np.pi, np.pi, self.N)
        
        for i in range(self.N):
            row, col = positions[i]
            particles.append(Particle(row, col, thetas[i]))
        
        self.particles = particles
        self.normalize_weights()
        particles = self.particles
        
        print(f"Inicializadas {len(particles)} partículas espalhadas")
        return particles

    def is_free_cell(self, row, col):
        """Verifica se a célula está livre (dentro da área navegável)"""
        return (start_row <= row < end_row and start_col <= col < end_col)

    def grid_to_world(self, row, col):
        """Converte células da grelha para coordenadas do mundo"""
        x = (col + 0.5) * RESOLUTION
        y = (row + 0.5) * RESOLUTION
        return x, y

    def calculate_expected_distance(self, particle_row, particle_col, marker_row, marker_col):
        """Calcula a distância esperada entre uma partícula e um marcador (em metros)"""
        # Convertendo para coordenadas do mundo
        p_x, p_y = self.grid_to_world(particle_row, particle_col)
        m_x, m_y = self.grid_to_world(marker_row, marker_col)
        
        # Distância euclidiana em metros
        distance = np.sqrt((p_x - m_x)**2 + (p_y - m_y)**2)
        return distance

    def can_particle_see_aruco(self, particle_row, particle_col, aruco_id):
        """Verifica se uma partícula está no campo de visão do ArUco"""
        if aruco_id not in ARUCO_ORIENTATIONS:
            return True
        
        aruco_pos = MAP_MARKERS_REAL[aruco_id]
        aruco_row, aruco_col = aruco_pos
        aruco_orientation = ARUCO_ORIENTATIONS[aruco_id]
        
        # Diferença de posição
        dx = particle_col - aruco_col
        dy = particle_row - aruco_row
        
        tolerance = 2  # ±2 células de tolerância
        
        if aruco_orientation == (0, 1):     # Aponta para baixo
            return dy > 0 and abs(dx) <= tolerance  # Corredor vertical abaixo
        
        elif aruco_orientation == (0, -1):  # Aponta para cima
            return dy < 0 and abs(dx) <= tolerance  # Corredor vertical acima
        
        elif aruco_orientation == (-1, 0):  # Aponta para esquerda
            return dx < 0 and abs(dy) <= tolerance  # Corredor horizontal à esquerda
        
        elif aruco_orientation == (1, 0):   # Aponta para direita
            return dx > 0 and abs(dy) <= tolerance  # Corredor horizontal à direita
        
        return True

    def validate_and_extract_distances(self, distances_msg):
        """Valida e extrai distâncias de diferentes formatos de mensagem"""
        distances_data = []
        
        # Tentar diferentes atributos da mensagem
        raw_data = None
        if hasattr(distances_msg, 'data'):
            raw_data = distances_msg.data
        elif hasattr(distances_msg, 'distances'):
            raw_data = distances_msg.distances
        elif hasattr(distances_msg, 'values'):
            raw_data = distances_msg.values
        elif hasattr(distances_msg, 'range'):
            raw_data = distances_msg.range
        else:
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+', str(distances_msg))
            if numbers:
                raw_data = [float(n) for n in numbers]
                
        if raw_data is None:
            return []
        
        # Converter para lista se necessário
        if isinstance(raw_data, (list, tuple)):
            distances_data = list(raw_data)
        elif hasattr(raw_data, '__iter__') and not isinstance(raw_data, str):
            distances_data = list(raw_data)
        else:
            distances_data = [raw_data]
        
        # Filtrar e validar distâncias
        valid_distances = []
        for i, d in enumerate(distances_data):
            if isinstance(d, (int, float)):
                if 0.1 <= d <= 10.0:
                    valid_distances.append(float(d))
                    print(f"     ✅ Distância {i}: {d:.3f}m (válida)")
                else:
                    print(f"     ⚠️ Distância {i}: {d:.3f}m (fora do range 0.1-10.0m)")
            elif d == 0 or d is None:
                print(f"     ⚠️ Distância {i}: {d} (vazia/nula, ignorada)")
            else:
                print(f"     ❌ Distância {i}: {d} (tipo inválido: {type(d)})")
        
        print(f"   Total de distâncias válidas: {len(valid_distances)}")
        return valid_distances

    def get_aruco_detection_order(self, timestamp):
        """Determina a ordem dos ArUcos detectados para mapear com as distâncias"""
        # Procurar detecções recentes de ArUco (últimos 3 segundos)
        for det_time, aruco_ids in reversed(self.detected_markers):
            time_diff = (timestamp - det_time).to_sec()
            if time_diff <= 3.0:  # 3 segundos de tolerância
                # Ordenar IDs para consistência
                return sorted(aruco_ids)
        return []

    def process_marker_distances(self, distances_msg, timestamp):
        """Processa distâncias dos marcadores do tópico /marker_distances"""
        # Extrair e validar distâncias da mensagem
        valid_distances = self.validate_and_extract_distances(distances_msg)
        
        if not valid_distances:
            # Dados vazios - limpar distâncias atuais se existirem
            if self.current_marker_distances:
                self.current_marker_distances = {}
            return
        
        # Correção de calibração às distâncias
        corrected_distances = []
        for raw_distance in valid_distances:
            corrected = self.correct_distance_measurement(raw_distance)
            corrected_distances.append(corrected)
        
        # Obter ordem dos ArUcos detectados recentemente
        recent_aruco_ids = self.get_aruco_detection_order(timestamp)
        
        # Mapear distâncias para IDs de ArUco baseado na ordem de detecção
        current_distances = {}
        mapped_count = 0
        
        for i, distance in enumerate(valid_distances):
            if i < len(recent_aruco_ids):
                marker_id = recent_aruco_ids[i]
                if marker_id in MAP_MARKERS_REAL:
                    current_distances[marker_id] = float(distance)
                    mapped_count += 1
                else:
                    print(f"⚠️ ArUco {marker_id} não está no mapa")
            else:
                print(f"⚠️ Mais distâncias ({len(valid_distances)}) que ArUcos detectados ({len(recent_aruco_ids)})")
                break
        
        if current_distances:
            self.current_marker_distances = current_distances
            self.distance_history.append((timestamp, current_distances.copy()))
            self.distance_detections += 1
            self.last_distance_time = timestamp
            
            # Atualizar pesos das partículas com base nas distâncias
            self.update_particle_weights_with_distances(current_distances)
        else:
            print(f"⚠️ Nenhuma distância pôde ser mapeada para ArUcos válidos")

    def update_particle_weights_with_distances(self, measured_distances):
        """Atualiza pesos das partículas com base nas distâncias medidas usando zonas de distância"""
        if not measured_distances or not self.particles:
            return
        
        print(f"   🔄 Atualizando pesos com {len(measured_distances)} distâncias usando zonas...")
        
        # Extrair posições das partículas
        particle_positions = np.array([[p.row, p.col] for p in self.particles])
        weights = np.array([p.weight for p in self.particles])
        
        # Criar zonas de distância para cada marcador
        zone_weights = np.ones_like(weights)
        
        for marker_id, measured_dist in measured_distances.items():
            if marker_id not in MAP_MARKERS_REAL:
                continue
                
            marker_pos = np.array(MAP_MARKERS_REAL[marker_id])
            marker_row, marker_col = marker_pos
            
            print(f"     📍 Processando ArUco {marker_id}: distância={measured_dist:.2f}m")
            
            # Calcular distâncias esperadas para todas as partículas (vetorizado)
            expected_distances = np.array([
                self.calculate_expected_distance(p_row, p_col, marker_row, marker_col)
                for p_row, p_col in particle_positions
            ])
            
            # ZONA DE ALTA CONFIANÇA: Partículas muito próximas da distância correta
            distance_errors = np.abs(expected_distances - measured_dist)
           
            # Múltiplas zonas com diferentes pesos
            # Zona 1: ±15cm = peso muito alto (10x)
            zone1_mask = distance_errors <= 0.15
            zone1_weight = 10.0
            
            # Zona 2: ±30cm = peso alto (5x)
            zone2_mask = (distance_errors > 0.15) & (distance_errors <= 0.30)
            zone2_weight = 5.0
            
            # Zona 3: ±50cm = peso médio (2x)
            zone3_mask = (distance_errors > 0.30) & (distance_errors <= 0.50)
            zone3_weight = 2.0
            
            # Zona 4: ±80cm = peso baixo (1.2x)
            zone4_mask = (distance_errors > 0.50) & (distance_errors <= 0.80)
            zone4_weight = 1.2
            
            # Zona 5: >80cm = penalização (0.1x)
            zone5_mask = distance_errors > 0.80
            zone5_weight = 0.1
            
            # Aplicar pesos por zona
            marker_zone_weights = np.ones_like(weights)
            marker_zone_weights[zone1_mask] *= zone1_weight
            marker_zone_weights[zone2_mask] *= zone2_weight
            marker_zone_weights[zone3_mask] *= zone3_weight
            marker_zone_weights[zone4_mask] *= zone4_weight
            marker_zone_weights[zone5_mask] *= zone5_weight
            
            # Dar peso extra se a partícula está numa posição geometricamente consistente
            # Calcular se a partícula pode "ver" o marcador sem obstáculos
            visibility_weights = self.calculate_visibility_weights(particle_positions, marker_pos)
            marker_zone_weights *= visibility_weights
            
            # Acumular pesos desta zona
            zone_weights *= marker_zone_weights
            
            # Todas as zonas se multiplicam
            final_weights = weights * zone_weights
            
            # Evitar que pesos fiquem muito extremos
            if np.max(final_weights) > 0:
                # Aplicar uma compressão logarítmica suave para evitar dominância excessiva
                log_weights = np.log1p(final_weights / np.max(final_weights))
                final_weights = np.exp(log_weights) * np.mean(weights)
            
            # Atualizar pesos das partículas
            for i, p in enumerate(self.particles):
                p.weight = final_weights[i]
            
            # Normalizar pesos
            self.normalize_weights()
          
    def calculate_visibility_weights(self, particle_positions, marker_pos):
        """Calcula pesos baseados na visibilidade geométrica do marcador"""
        weights = np.ones(len(particle_positions))
        
        for i, (p_row, p_col) in enumerate(particle_positions):
            # Verificar se há linha de visão clara entre partícula e marcador
            if self.has_line_of_sight(p_row, p_col, marker_pos[0], marker_pos[1]):
                weights[i] = 1.5  # Bonus de 50% para visibilidade clara
            else:
                weights[i] = 0.8  # Penalização leve para visibilidade obstruída
        
        return weights

    def has_line_of_sight(self, from_row, from_col, to_row, to_col):
        """Verifica se há linha de visão clara entre dois pontos"""
        # Algoritmo simples de ray-casting
        # Para o nosso mapa pequeno, verificamos se há paredes no caminho
        
        # Usar algoritmo de Bresenham simplificado
        steps = max(abs(to_row - from_row), abs(to_col - from_col))
        if steps == 0:
            return True
            
        for step in range(1, int(steps)):
            t = step / steps
            check_row = int(from_row + t * (to_row - from_row))
            check_col = int(from_col + t * (to_col - from_col))
            
            # Verificar se o ponto está numa parede
            if (0 <= check_row < GRID_ROWS and 0 <= check_col < GRID_COLS):
                if grid[check_row, check_col] == 1:  # Parede
                    return False
            else:
                return False  # Fora dos limites
        
        return True

    def process_aruco_detections(self, marker_ids_msg, timestamp):
        """Processa detecções de ArUcos do tópico /aruco/marker_ids"""
        detected_ids = marker_ids_msg.data
        
        if not detected_ids:
            return
        
        self.aruco_detections += 1
        
        # Correção de IDs
        unique_ids = list(set(detected_ids))
        valid_ids = [id_val for id_val in unique_ids if id_val in MAP_MARKERS_REAL]
        
        if not valid_ids:
            return
        
        print(f"\n🎯 ArUcos detectados: {detected_ids} → Válidos: {valid_ids}")
        
        # Estimar orientação do robô usando poses dos ArUcos 
        estimated_robot_orientations = []
        
        for aruco_id in valid_ids:
            if (aruco_id in self.current_aruco_poses and 
                self.last_pose_time is not None):
                
                # Verificar se a pose é recente (últimos 2 segundos)
                time_diff = (timestamp - self.last_pose_time).to_sec()
                if time_diff <= 2.0:
                    aruco_pose = self.current_aruco_poses[aruco_id]
                    
                    # Estimar orientação do robô baseada na pose do ArUco
                    estimated_theta = self.calculate_robot_orientation_from_aruco_pose(
                        aruco_id, aruco_pose, np.linalg.norm(aruco_pose[:3]))
                    
                    if estimated_theta is not None:
                        estimated_robot_orientations.append(estimated_theta)
                        print(f"     🧭 ArUco {aruco_id}: orientação estimada = {np.degrees(estimated_theta):.1f}°")
        
        # Calcular orientação média se temos múltiplas estimativas
        final_estimated_orientation = None
        if estimated_robot_orientations:
            # Usar média circular para ângulos
            sin_sum = sum(np.sin(theta) for theta in estimated_robot_orientations)
            cos_sum = sum(np.cos(theta) for theta in estimated_robot_orientations)
            final_estimated_orientation = np.arctan2(sin_sum, cos_sum)
            
            print(f"   🎯 Orientação final estimada: {np.degrees(final_estimated_orientation):.1f}°")
            
            # APLICAR A ORIENTAÇÃO ESTIMADA ÀS PARTÍCULAS 
            # Ajustar orientações das partículas com maior peso para convergirem
            for particle in self.particles:
                if particle.weight > np.mean([p.weight for p in self.particles]):  # Apenas partículas acima da média
                    # Suavemente ajustar orientação da partícula em direção à estimativa
                    orientation_diff = final_estimated_orientation - particle.theta
                    # Normalizar diferença para [-π, π]
                    while orientation_diff > np.pi:
                        orientation_diff -= 2 * np.pi
                    while orientation_diff < -np.pi:
                        orientation_diff += 2 * np.pi
                    
                    # Aplicar correção suave (apenas 20% da diferença)
                    particle.theta += 0.2 * orientation_diff
                    # Normalizar ângulo final
                    particle.theta = np.arctan2(np.sin(particle.theta), np.cos(particle.theta))

        # Combinar detecção visual + distâncias (sem poses nos pesos)
        if valid_ids:
            # Extrair posições das partículas
            particle_positions = np.array([[p.row, p.col] for p in self.particles])
            particle_orientations = np.array([p.theta for p in self.particles])
            weights = np.array([p.weight for p in self.particles])
            
            # Para cada ArUco válido
            for aruco_id in valid_ids:
                marker_pos = np.array(MAP_MARKERS_REAL[aruco_id])
                
                # Cálculo vetorizado de distâncias
                distances = np.linalg.norm(particle_positions - marker_pos, axis=1)
                in_range = distances <= 4.0

                # Verificar campo de visão do ArUco 
                visibility_mask = np.array([
                    self.can_particle_see_aruco(p_row, p_col, aruco_id) 
                    for p_row, p_col in particle_positions
                ])

                prob_see = np.exp(-distances / 2.0)
                # Combinar distância + visibilidade
                position_weights = np.where(
                    in_range & visibility_mask,  # Deve estar perto E no campo de visão
                    1.0 + prob_see * 10.0, 
                    0.1  # Penalização forte se não pode ser visto
                )
                
                # --- ORIENTAÇÃO (baseada na direção esperada para ver o ArUco) ---
                dx = marker_pos[1] - particle_positions[:, 1]  # col
                dy = marker_pos[0] - particle_positions[:, 0]  # row
                expected_theta = np.arctan2(dy, dx)
                angle_diff = np.abs(particle_orientations - expected_theta)
                angle_diff = np.where(angle_diff > np.pi, 2 * np.pi - angle_diff, angle_diff)
                orientation_weights = np.exp(-angle_diff**2 / (2 * (np.pi / 4)**2))  # σ = 45°

                # Adicionar pesos de distância se disponíveis
                distance_weights = np.ones_like(position_weights)
                if (aruco_id in self.current_marker_distances and 
                    self.last_distance_time is not None):
                    
                    # Verificar se a distância é recente (últimos 2 segundos)
                    time_diff = (timestamp - self.last_distance_time).to_sec()
                    if time_diff <= 2.0:
                        measured_dist = self.current_marker_distances[aruco_id]
                        marker_row, marker_col = marker_pos
                        
                        expected_distances = np.array([
                            self.calculate_expected_distance(p_row, p_col, marker_row, marker_col)
                            for p_row, p_col in particle_positions
                        ])
                        
                        distance_errors = np.abs(expected_distances - measured_dist)
                        distance_sigma = 0.3  # 30cm
                        distance_weights = np.exp(-(distance_errors**2) / (2 * distance_sigma**2))
                        
                        print(f"     📏 Usando distância para ArUco {aruco_id}: {measured_dist:.2f}m")

                # Combina pesos (posição + orientação + distância, SEM poses)
                combined_weights = position_weights * orientation_weights * distance_weights
                weights *= combined_weights
            
            # Atualizar pesos das partículas
            for i, p in enumerate(self.particles):
                p.weight = weights[i]
                    
        # Normaliza e reamostra
        self.normalize_weights()
        confidence = self.calculate_confidence()
        entropy = self.calculate_entropy()
        N_eff = self.calculate_effective_particles()
        
        self.confidence_history.append((timestamp, confidence))
        self.entropy_history.append((timestamp, entropy))
        
        print(f"   📊 Confiança: {confidence:.3f} | Entropia: {entropy:.3f} | N_eff: {N_eff:.1f}")
        
        should_resample = False
        reason = ""
        
        if confidence > 0.5:
            should_resample = True
            reason = "alta confiança → otimização"
        elif confidence > 0.1: 
            should_resample = True
            reason = "confiança suficiente → adaptação dinâmica"
        else:
            print("   🆘 Confiança crítica → recovery + reamostragem")
            self.recovery_strategy()
            should_resample = True
            reason = "pós-recovery"
        
        if should_resample:
            print(f"   🔄 Reamostragem: {reason}")
            self.resample(timestamp)
        
        self.last_aruco_time = timestamp
        self.detected_markers.append((timestamp, valid_ids))
    def calculate_effective_particles(self):
        """Calcula número efetivo de partículas"""
        if not self.particles:
            return 0
        
        weights = [p.weight for p in self.particles]
        total_weight = sum(weights)
        if total_weight == 0:
            return len(self.particles)
        
        # Normalizar pesos
        normalized_weights = [w/total_weight for w in weights]
        
        # N_eff = 1 / Σ(w_i²)
        sum_weights_sq = sum(w**2 for w in normalized_weights)
        return 1.0 / sum_weights_sq if sum_weights_sq > 0 else len(self.particles)

    def recovery_strategy(self):
        """Reinicializa as partículas de menor peso"""
        print("🔄 ESTRATÉGIA DE RECUPERAÇÃO")
        
        if not self.particles:
            return
        
        # Sorting usando numpy para encontrar as 20% piores partículas
        weights = np.array([p.weight for p in self.particles])
        worst_indices = np.argpartition(weights, len(weights) // 5)[:len(weights) // 5]
        
        # Sampling vetorizado de novas posições
        num_reinit = len(worst_indices)
        new_positions = self.smart_particle_sampling(num_reinit)
        new_thetas = np.random.uniform(-np.pi, np.pi, num_reinit)
        
        # Reinicializar partículas
        for i, idx in enumerate(worst_indices):
            row, col = new_positions[i]
            self.particles[idx] = Particle(row, col, new_thetas[i], weight=0.1)
        
        self.normalize_weights()
        print(f"   ✅ Recovery concluído. Partículas reiniciadas: {num_reinit}")
    
    def add_particles_smart(self, num_add):
        """Adiciona partículas perto da distribuição atual"""
        new_positions = self.smart_particle_sampling(num_add)
        new_thetas = np.random.uniform(-np.pi, np.pi, num_add)
        
        for i in range(num_add):
            row, col = new_positions[i]
            new_particle = Particle(row, col, new_thetas[i], weight=0.1)
            self.particles.append(new_particle)
        
        self.N = len(self.particles)
        print(f"  📈 +{num_add} partículas inteligentes (total: {self.N})")
            
    def process_cmd_vel(self, cmd_vel_msg, timestamp):
        """Processa comandos de velocidade"""
        self.last_cmd_vel = cmd_vel_msg
        self.last_cmd_time = timestamp
        
        linear_vel_x = cmd_vel_msg.twist.linear.x
        linear_vel_y = cmd_vel_msg.twist.linear.y 
        angular_vel_z = cmd_vel_msg.twist.angular.z
         
        linear_vel_magnitude = np.sqrt(linear_vel_x**2 + linear_vel_y**2)
        self.velocity_history.append((timestamp, linear_vel_magnitude, angular_vel_z, 
                                    linear_vel_x, linear_vel_y)) 
         
    def move_particles(self, dt, current_timestamp=None):
        """Move partículas baseado no último comando de velocidade"""
        if self.last_cmd_vel is None:
            return
        
        linear_vel_x = self.last_cmd_vel.twist.linear.x
        linear_vel_y = self.last_cmd_vel.twist.linear.y
        angular_vel_z = self.last_cmd_vel.twist.angular.z
        
        # Verificação de adaptação de partículas
        if current_timestamp:
            time_since_aruco = float('inf')
            if self.last_aruco_time is not None:
                time_since_aruco = (current_timestamp - self.last_aruco_time).to_sec()
            
            # Adiciona partículas se necessário
            if (time_since_aruco > 5.0 and len(self.particles) < MAX_PARTICLES and 
                np.random.random() < 0.1):
                
                new_count = min(5, MAX_PARTICLES - len(self.particles))
                self.add_particles_smart(new_count)
                
        for p in self.particles:
            old_row, old_col = p.row, p.col
            old_row_continuous = p.row_continuous
            old_col_continuous = p.col_continuous
            p.move_2d(linear_vel_x, linear_vel_y, angular_vel_z, dt)
            
            # Verificação de colisão
            if not self.is_free_cell(p.row, p.col):
                p.weight *= 0.01  # Penalização forte para colisões
                # Reverte para posição anterior se possível
                if self.is_free_cell(old_row, old_col):
                    p.row, p.col = old_row, old_col
                    p.row_continuous = old_row_continuous
                    p.col_continuous = old_col_continuous

        self.normalize_weights()
            
    def update_aruco_timeline(self, timestamp):
        """Atualiza timeline de detecção de ArUcos para cada frame"""
        current_time = timestamp.to_sec()
        
        # Verifica se há detecções recentes (últimos 0.5 segundos)
        recent_detections = set()
        for det_time, aruco_ids in self.detected_markers:
            if abs(current_time - det_time.to_sec()) <= 0.5:
                recent_detections.update(aruco_ids)
        
        # Atualiza timeline para cada ArUco
        for aruco_id in [0, 1, 2, 3, 4, 5, 8, 10]:
            detected = aruco_id / 2.0 if aruco_id in recent_detections else 0
            self.aruco_timeline[aruco_id].append((timestamp, detected))
        
    def calculate_entropy(self):
        """Calcula entropia atual da distribuição de pesos"""
        if not self.particles:
            return 0.0
        weights = np.array([p.weight for p in self.particles])
        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0.0
        
        # Normaliza pesos
        normalized_weights = weights / total_weight
        
        # Calcula entropia
        mask = normalized_weights > 1e-12
        entropy = -np.sum(normalized_weights[mask] * np.log(normalized_weights[mask]))
        
        return entropy
           
    def record_particle_state(self, timestamp):
        """Records current particle state for visualization"""
        if not self.particles:
            return
        
        particle_data = [(p.row, p.col, p.weight) for p in self.particles]
        x, y, theta = self.get_pose_estimate()
        self.particle_history.append((timestamp, particle_data, (x, y, theta)))
        
    def calculate_confidence(self):
        """Confiança baseada na concentração de pesos (alta concentração = alta confiança)"""
        if not self.particles:
            return 0.0
        
        entropy = self.calculate_entropy()
        max_entropy = np.log(len(self.particles))
        
        if max_entropy > 0:
            confidence = 1.0 - (entropy / max_entropy)
        else:
            confidence = 1.0
            
        return max(0.0, min(1.0, confidence))
    
    def normalize_weights(self):
        """Normaliza pesos das partículas"""
        if not self.particles:
            return
            
        weights = np.array([p.weight for p in self.particles])
        total = np.sum(weights)
        
        if total == 0:
            uniform_weight = 1.0 / len(self.particles)
            for p in self.particles:
                p.weight = uniform_weight
        else:
            normalized_weights = weights / total
            for i, p in enumerate(self.particles):
                p.weight = normalized_weights[i]

    def low_variance_sampler(self, particles, weights):
        """Reamostragem com baixa variância"""
        M = len(particles)
        if M == 0:
            return []
        
        r = np.random.uniform(0, 1.0 / M)
        c = np.cumsum(weights)
        u = r + np.arange(M) / M
        
        # Encontrar índices usando searchsorted
        indices = np.searchsorted(c, u)
        indices = np.clip(indices, 0, M - 1)
        
        # Criar novas partículas
        new_particles = []
        for i in indices:
            new_particle = copy.deepcopy(particles[i])
            new_particles.append(new_particle)
            
        return new_particles
    
    def resample(self, timestamp):
        """Reamostragem que adapta dinamicamente o número de partículas"""
        weights = [p.weight for p in self.particles]
        if not weights:
            return
            
        # Calcula número efetivo de partículas
        sum_weights_sq = sum(w**2 for w in weights)
        N_eff = 1.0 / sum_weights_sq if sum_weights_sq > 0 else self.N
        
        threshold = self.N / 4
        
        if N_eff < threshold:
            print(f"   🔄 Reamostragem: N_eff={N_eff:.1f} < {threshold}")
            self.particles = self.low_variance_sampler(self.particles, weights)
            
            confidence = self.calculate_confidence()
            
            # Verifica há quanto tempo viu ArUcos
            time_since_aruco = float('inf')
            if self.last_aruco_time is not None:
                time_since_aruco = (timestamp - self.last_aruco_time).to_sec()

            if confidence > 0.3:  # alta confiança + ArUco recente
                new_N = max(int(self.N * 0.85), MIN_PARTICLES)  # reduz partículas significativamente
                print(f"  📉 Alta confiança + ArUco recente → reduz partículas")
            elif confidence < 0.2 or time_since_aruco > 5.0:  # baixa confiança ou sem ArUcos há tempo
                new_N = min(int(self.N * 1.2), MAX_PARTICLES)  # aumenta partículas
                print(f"  📈 Baixa confiança/sem ArUcos → aumenta partículas")
            else:
                new_N = self.N  # mantém o mesmo número
                
            if new_N != self.N:
                if new_N > self.N:
                    # Adição vetorizada de partículas
                    num_add = new_N - self.N
                    self.add_particles_smart(num_add)
                else:
                    # Remoção das piores partículas
                    particle_weights = np.array([p.weight for p in self.particles])
                    keep_indices = np.argpartition(particle_weights, -new_N)[-new_N:]
                    self.particles = [self.particles[i] for i in keep_indices]
                
                self.N = new_N
    
    def smart_particle_sampling(self, num_particles):
        """Cria partículas baseadas na distribuição atual E distâncias conhecidas"""
        if not self.particles or len(self.particles) < 5:
            # Fallback: distribuição uniforme apenas se há poucas partículas
            indices = np.random.choice(len(VALID_POSITIONS), num_particles, replace=True)
            return VALID_POSITIONS_ARRAY[indices]
        
        # Se temos distâncias recentes, usar para criar partículas inteligentes
        if (self.current_marker_distances and 
            self.last_distance_time and 
            len(self.current_marker_distances) >= 2):
            
            print(f"  🧠 Sampling inteligente usando {len(self.current_marker_distances)} distâncias")
            return self.distance_based_particle_sampling(num_particles)
        
        # Método original baseado na distribuição atual
        return self.distribution_based_sampling(num_particles)

    def distance_based_particle_sampling(self, num_particles):
        """Cria partículas baseadas em interseções de círculos de distância"""
        new_positions = []
        
        # Converter distâncias para lista de marcadores e distâncias
        markers_data = [(marker_id, distance, MAP_MARKERS_REAL[marker_id]) 
                       for marker_id, distance in self.current_marker_distances.items() 
                       if marker_id in MAP_MARKERS_REAL]
        
        if len(markers_data) < 2:
            # Fallback se não há dados suficientes
            return self.distribution_based_sampling(num_particles)
        
        # Para cada nova partícula, tentar encontrar posições que satisfaçam as distâncias
        attempts_per_particle = 50
        successful_samples = 0
        tolerance = self.distance_measurement_sigma * 1.5
        
        for _ in range(num_particles):
            best_pos = None
            best_error = float('inf')
            
            for attempt in range(attempts_per_particle):
                # Sampling aleatório na área navegável
                random_idx = np.random.randint(len(VALID_POSITIONS))
                candidate_pos = VALID_POSITIONS[random_idx]
                candidate_row, candidate_col = candidate_pos
                
                # Calcular erro total das distâncias para esta posição
                total_error = 0
                for marker_id, target_distance, (marker_row, marker_col) in markers_data:
                    expected_distance = self.calculate_expected_distance(
                        candidate_row, candidate_col, marker_row, marker_col)
                    error = abs(expected_distance - target_distance)
                    total_error += error
                
                # Manter a melhor posição encontrada
                if total_error < best_error:
                    best_error = total_error
                    best_pos = candidate_pos
                
                # Se encontramos uma posição muito boa, parar
                if total_error < tolerance:  # Erro total < 30cm
                    successful_samples += 1
                    break
            
            if best_pos is not None:
                new_positions.append(best_pos)
            else:
                # Fallback: posição aleatória válida
                random_idx = np.random.randint(len(VALID_POSITIONS))
                new_positions.append(VALID_POSITIONS[random_idx])
        
        return np.array(new_positions)

    def distribution_based_sampling(self, num_particles):
        """Sampling baseado na distribuição atual de partículas"""
        # Calcular centro de massa das partículas atuais
        current_rows = np.array([p.row for p in self.particles])
        current_cols = np.array([p.col for p in self.particles])
        weights = np.array([p.weight for p in self.particles])
        
        # Centro ponderado das partículas existentes
        if np.sum(weights) > 1e-10:
            center_row = np.average(current_rows, weights=weights)
            center_col = np.average(current_cols, weights=weights)
        else:
            center_row = np.mean(current_rows)
            center_col = np.mean(current_cols)
        
        # Gerar partículas numa gaussiana ao redor do centro atual
        new_positions = []
        sigma = 1.5  # Desvio padrão em células (ajustável)
        
        for _ in range(num_particles):
            attempts = 0
            while attempts < 20:
                # Gaussiana ao redor do centro atual
                new_row = np.clip(np.random.normal(center_row, sigma), start_row, end_row - 1)
                new_col = np.clip(np.random.normal(center_col, sigma), start_col, end_col - 1)
                
                new_row = int(round(new_row))
                new_col = int(round(new_col))
                
                if self.is_free_cell(new_row, new_col):
                    new_positions.append((new_row, new_col))
                    break
                attempts += 1
            
            # Fallback: posição válida próxima
            if attempts >= 20:
                # Encontrar posição válida mais próxima do centro
                min_dist = float('inf')
                best_pos = None
                for valid_pos in VALID_POSITIONS:
                    dist = (valid_pos[0] - center_row)**2 + (valid_pos[1] - center_col)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_pos = valid_pos
                new_positions.append(best_pos)
        
        return np.array(new_positions)
    
    def get_pose_estimate(self):
        """Estimativa de pose c/ base nas 80% melhores partículas"""
        if not self.particles:
            return 0, 0, 0
        
        self.normalize_weights()
        
        # Usar top 80% das partículas para mais estabilidade
        sorted_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)
        top_30_percent = max(1, len(sorted_particles) * 8 // 10)
        top_particles = sorted_particles[:top_30_percent]
        
        total_weight = sum(p.weight for p in top_particles)
        
        if total_weight < 1e-10:
            # Fallback: média simples das top 80%
            avg_row = sum(getattr(p, 'row_continuous', p.row) for p in top_particles) / len(top_particles)
            avg_col = sum(getattr(p, 'col_continuous', p.col) for p in top_particles) / len(top_particles)
            avg_sin = sum(np.sin(p.theta) for p in top_particles) / len(top_particles)
            avg_cos = sum(np.cos(p.theta) for p in top_particles) / len(top_particles)
            avg_theta = np.arctan2(avg_sin, avg_cos)
        else:
            # Média ponderada das top 80% partículas
            avg_row = sum(getattr(p, 'row_continuous', p.row) * p.weight for p in top_particles) / total_weight
            avg_col = sum(getattr(p, 'col_continuous', p.col) * p.weight for p in top_particles) / total_weight
            
            # Ângulo médio circular
            avg_sin = sum(np.sin(p.theta) * p.weight for p in top_particles) / total_weight
            avg_cos = sum(np.cos(p.theta) * p.weight for p in top_particles) / total_weight
            avg_theta = np.arctan2(avg_sin, avg_cos)
        
        x, y = self.grid_to_world(avg_row, avg_col)
        return x, y, avg_theta
    
    def print_progress(self, current_time):
        """Mostra progresso"""
        if self.start_time is None:
            self.start_time = time.time()
            
        elapsed = time.time() - self.start_time
        fps = self.processed_frames / elapsed if elapsed > 0 else 0
        
        confidence = self.confidence_history[-1][1] if self.confidence_history else 0
        
        print(f"\r⏳ Frame {self.processed_frames} | "
              f"FPS: {fps:.1f} | Partículas: {self.N} | "
              f"ArUcos: {self.aruco_detections} | Distâncias: {self.distance_detections} | "
              f"Confiança: {confidence:.3f}", end="", flush=True)
        
    def process_bag(self):
        """Processa o bag file apenas para análise (sem gravação de bag de saída)"""
        print(f"\n📁 Processando: {self.bag_path}")
        print("🎯 Modo análise: apenas vídeo e PNG serão gerados")
        
        if not os.path.exists(self.bag_path):
            print(f"❌ Ficheiro não encontrado: {self.bag_path}")
            return
        
        # Verificar tópicos disponíveis no bag
        try:
            with rosbag.Bag(self.bag_path, 'r') as bag:
                bag_info = bag.get_type_and_topic_info()
                available_topics = list(bag_info.topics.keys())
                print(f"\n🔍 Tópicos disponíveis no bag:")
                for topic in sorted(available_topics):
                    topic_info = bag_info.topics[topic]
                    print(f"   • {topic} ({topic_info.message_count} msgs, tipo: {topic_info.msg_type})")
                
                # Verificar especificamente o tópico de distâncias
                distance_topics = [t for t in available_topics if 'distance' in t.lower() or 'marker' in t.lower()]
                print(f"\n📏 Tópicos relacionados com distâncias/marcadores:")
                for topic in distance_topics:
                    topic_info = bag_info.topics[topic]
                    print(f"   • {topic} ({topic_info.message_count} msgs)")
        except Exception as e:
            print(f"❌ Erro ao analisar bag: {e}")
            return
        
        try:
            with rosbag.Bag(self.bag_path, 'r') as bag:
                print("\n🚀 Iniciando processamento MCL com distâncias...")
                
                prev_time = None
                messages_processed = 0
                distance_topic_found = False
                
                for topic, msg, t in bag.read_messages():
                    messages_processed += 1
                    
                    # Monitorar tópicos de distância
                    if 'distance' in topic.lower() or (topic == '/marker_distances'):
                        try:
                            self.process_marker_distances(msg, t)
                        except Exception as e:
                            print(f"❌ Erro ao processar distâncias do tópico {topic}: {e}")
                            traceback.print_exc()
                    
                    # Processar poses dos ArUcos
                    elif '/aruco/marker_poses' in topic or 'pose' in topic.lower():
                        try:
                            self.process_aruco_poses(msg, t)
                            pose_topic_found = True
                        except Exception as e:
                            print(f"❌ Erro ao processar poses do tópico {topic}: {e}")
                            traceback.print_exc()
                    
                    # Processa ArUcos
                    elif topic == '/aruco/marker_ids':
                        try:
                            self.process_aruco_detections(msg, t)
                        except Exception as e:
                            print(f"❌ Erro ao processar ArUcos: {e}")
                    
                    # Processa comandos de velocidade
                    elif topic == '/cmd_vel':
                        try:
                            self.process_cmd_vel(msg, t)
                        except Exception as e:
                            print(f"❌ Erro ao processar cmd_vel: {e}")
                    
                    # Processa imagens (para contagem)
                    elif topic == '/raspicam_node/image/compressed':
                        try:
                            self.frame_count += 1
                            self.processed_frames += 1
                            
                            # Movimento das partículas
                            if prev_time is not None:
                                dt = (t - prev_time).to_sec()
                                if 0 < dt < 1.0:
                                    self.move_particles(dt)
                            
                            # Estimativa de pose
                            x, y, theta = self.get_pose_estimate()
                            self.pose_estimates.append((t, x, y, theta))
                            self.particle_counts.append((t, self.N))
                            self.update_aruco_timeline(t)
                            
                            # Record particle state every 5 frames for video
                            if self.processed_frames % 5 == 0:
                                self.record_particle_state(t)
                            
                            prev_time = t
                            
                            # Progresso
                            if self.processed_frames % 50 == 0:
                                self.print_progress(t.to_sec())
                                
                        except Exception as e:
                            print(f"❌ Erro ao processar frame {self.processed_frames}: {e}")
                
        except Exception as e:
            print(f"\n❌ ERRO ao abrir bag de entrada: {e}")
            traceback.print_exc()
            return
        
        print(f"\n\n✅ Processamento concluído!")
        print(f"📊 Estatísticas:")
        print(f"   - Frames processados: {self.processed_frames}")
        print(f"   - Detecções ArUco: {self.aruco_detections}")
        print(f"   - Detecções de distância: {self.distance_detections}")
        print(f"   - Estimativas de pose: {len(self.pose_estimates)}")
        print(f"   - Histórico de distâncias: {len(self.distance_history)} entradas")
        
        if self.confidence_history:
            confidences = [c[1] for c in self.confidence_history]
            print(f"   - Confiança média: {np.mean(confidences):.3f}")
            print(f"   - Confiança final: {confidences[-1]:.3f}")

    def create_continuous_probability_map(self, particles):
        """Cria mapa de probabilidade contínuo a partir de partículas (row,col)"""
        # Inicializa mapa de probabilidade
        prob_map = np.zeros((GRID_ROWS, GRID_COLS))
        
        if not particles:
            return prob_map
        
        rows = np.array([p[0] for p in particles])
        cols = np.array([p[1] for p in particles])
        weights = np.array([p[2] for p in particles])
        
        sigma = 0.8
        sigma_sq_2 = 2 * sigma**2
        
        # OTIMIZAÇÃO: Meshgrid para coordenadas
        r_grid, c_grid = np.meshgrid(range(GRID_ROWS), range(GRID_COLS), indexing='ij')
        
        # Para cada partícula, adicionar contribuição gaussiana
        for i in range(len(particles)):
            dr = r_grid - rows[i]
            dc = c_grid - cols[i]
            dist_sq = dr**2 + dc**2
            contrib = weights[i] * np.exp(-dist_sq / sigma_sq_2)
            prob_map += contrib
        
        # Normalização
        if prob_map.max() > 0:
            prob_map = prob_map / prob_map.max()
        
        return prob_map
    
    def create_visualization_video(self, output_video="mcl_distances_visualization.mp4"):
        """Cria vídeo mostrando evolução das partículas e estimativas de pose"""
        try:
            print(f"\n🎬 Criando vídeo de visualização: {output_video}")
            
            if len(self.particle_history) < 2:
                print(f"❌ Dados insuficientes para vídeo: {len(self.particle_history)} estados gravados")
                print("💡 Dica: Execute o processamento novamente para gravar estados das partículas")
                return
            
            print(f"📊 Estados de partículas disponíveis: {len(self.particle_history)}")
            
            # Configuração da figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.patch.set_facecolor('white')
            
            self._setup_visualization_axes(ax1, ax2)
            
            # Elementos dinâmicos
            prob_image = ax1.imshow(np.zeros((GRID_ROWS, GRID_COLS)), 
                                  cmap='Blues', origin='upper', alpha=0.9, vmin=0, vmax=1)
            
            cbar = plt.colorbar(prob_image, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label('Probabilidade', fontsize=12)
            
            pose_arrow = None
            trajectory_line, = ax2.plot([], [], 'navy', linewidth=3, alpha=0.8)
            
            # Em vez de um ponto, colorir a célula inteira
            estimated_cell = None  # Para armazenar o patch da célula estimada
            
            info_text = fig.text(0.01, 0.99, '', fontsize=11, verticalalignment='top',
                               bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', alpha=0.9),
                               fontfamily='monospace')
            
            trajectory_x, trajectory_y = [], []
            
            def animate(frame):
                nonlocal pose_arrow, trajectory_x, trajectory_y, estimated_cell
                
                timestamp, particles, (est_x, est_y, est_theta) = self.particle_history[frame]
                
                # Mapa de probabilidade
                prob_map = self.create_continuous_probability_map(particles)
                wall_mask = (grid == 1)
                prob_map[wall_mask] = 0
                prob_image.set_array(prob_map)
                
                # Trajetória
                cell_row = int(round(est_y / RESOLUTION - 0.5))  # Conversão direta
                cell_col = int(round(est_x / RESOLUTION - 0.5))  # Conversão direta

                cell_row = np.clip(cell_row, start_row, end_row - 1)
                cell_col = np.clip(cell_col, start_col, end_col - 1)
                
                trajectory_x.append(cell_col)
                trajectory_y.append(cell_row)
                trajectory_line.set_data(trajectory_x, trajectory_y)
                
                # Remover célula anterior
                if estimated_cell:
                    estimated_cell.remove()
                
                from matplotlib.patches import Rectangle
                estimated_cell = Rectangle((cell_col - 0.5, cell_row - 0.5), 1, 1,
                                        facecolor='red', alpha=0.7, edgecolor='white', 
                                        linewidth=2, zorder=10)
                ax2.add_patch(estimated_cell)
                
                # Seta de orientação no centro da célula
                if pose_arrow:
                    pose_arrow.remove()
                
                arrow_length = 0.8
                dx = arrow_length * np.cos(est_theta)
                dy = arrow_length * np.sin(est_theta)
                
                pose_arrow = FancyArrowPatch((cell_col, cell_row), 
                                        (cell_col + dx, cell_row + dy),
                                        color='white', linewidth=3, 
                                        arrowstyle='->', mutation_scale=20, zorder=11)
                ax2.add_patch(pose_arrow)
                
                # Info text com informações de distância
                confidence = 0
                if frame < len(self.confidence_history):
                    confidence = self.confidence_history[frame][1]
                
                time_sec = timestamp.to_sec() - self.particle_history[0][0].to_sec()
                
                # Procurar distâncias próximas do timestamp atual
                current_distances = {}
                for dist_time, distances in self.distance_history:
                    if abs((dist_time - timestamp).to_sec()) <= 1.0:  # 1 segundo de tolerância
                        current_distances = distances
                        break
                
                distance_info = ""
                if current_distances:
                    distance_info = f"\nDistâncias ativas:\n"
                    for marker_id, dist in current_distances.items():
                        distance_info += f"  A{marker_id}: {dist:.2f}m\n"
                else:
                    distance_info = "\nSem distâncias ativas"
                
                info_text.set_text(
                    f'Frame:      {frame * 5:4d}\n'
                    f'Tempo:      {time_sec:6.1f}s\n'
                    f'Partículas: {len(particles):3d}\n'
                    f'Confiança:  {confidence:5.3f}\n'
                    f'Célula: ({cell_row}, {cell_col})\n'
                    f'Pose: ({est_x:.2f}, {est_y:.2f})\n'
                    f'Ângulo:     {np.degrees(est_theta):6.1f}°'
                    f'{distance_info}'
                )
                
                return prob_image, trajectory_line, info_text
            
            print("🎭 Criando animação...")
            anim = animation.FuncAnimation(fig, animate, frames=len(self.particle_history),
                                         interval=120, blit=False, repeat=True)
            
            # Adicionar legenda para melhor compreensão
            ax2.legend(['Trajetória', 'Posição Estimada (Célula)'], 
                      loc='upper right', fontsize=10,
                      bbox_to_anchor=(1.0, 1.0))
            
            plt.tight_layout()
            
            print("💾 Guardando vídeo...")
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=12, metadata=dict(artist='MCL com Distâncias'), bitrate=3000)
            anim.save(output_video, writer=writer, dpi=100)
            
            print(f"✅ Vídeo criado: {output_video}")
            plt.close(fig)
            
        except Exception as e:
            print(f"❌ Erro ao criar vídeo: {e}")
            traceback.print_exc()

    def _setup_visualization_axes(self, ax1, ax2):
        """Setup dos eixos com grid visível em ambos os painéis"""
        for ax, title in [(ax1, 'Mapa de Probabilidade'), 
                         (ax2, 'Trajetória e Pose Estimada')]:
            
            # Paredes
            wall_mask = (grid == 1).astype(float)
            ax.imshow(wall_mask, cmap='gray', origin='upper', vmin=0, vmax=1, alpha=0.6)
            
            # Grid mais visível em ambos os painéis
            for i in range(GRID_ROWS + 1):
                ax.axhline(i - 0.5, color='black', linewidth=0.8, alpha=0.7)
            for j in range(GRID_COLS + 1):
                ax.axvline(j - 0.5, color='black', linewidth=0.8, alpha=0.7)
            
            # Adicionar números das células para referência
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    # Só mostrar números em células livres
                    if grid[i, j] == 0:  # célula livre
                        ax.text(j, i, f'{i},{j}', ha='center', va='center', 
                               fontsize=7, color='gray', alpha=0.6, 
                               bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.3))
            
            # ArUcos em destaque
            for marker_id, (row, col) in MAP_MARKERS_DISPLAY.items():
                ax.plot(col, row, 's', markersize=16, color='red', markeredgecolor='darkred', linewidth=3)
                ax.text(col, row, f'{marker_id}', color='white', ha='center', va='center', 
                       fontweight='bold', fontsize=12)
            
            ax.set_xlim(-0.5, GRID_COLS - 0.5)
            ax.set_ylim(GRID_ROWS - 0.5, -0.5)
            ax.set_aspect('equal')
            ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
            ax.set_xlabel('Colunas (X)', fontsize=12)
            ax.set_ylabel('Linhas (Y)', fontsize=12)
            
            # Adicionar ticks para mostrar coordenadas das células
            ax.set_xticks(range(GRID_COLS))
            ax.set_yticks(range(GRID_ROWS))
            ax.set_xticklabels(range(GRID_COLS), fontsize=10)
            ax.set_yticklabels(range(GRID_ROWS), fontsize=10)
            
            # Grid menor nos ticks
            ax.tick_params(axis='both', which='major', labelsize=9, 
                          color='gray', length=4, width=1)

    def create_static_visualization(self, output_prefix="mcl_analysis"):
        """Creates focused static visualizations with English titles"""
        print(f"\n📊 Creating focused visual analysis...")
        
        # 1. CONFIDENCE EVOLUTION
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        if self.confidence_history:
            times = np.array([t.to_sec() for t, conf in self.confidence_history])
            times = times - times[0]  # Start from 0
            confidences = np.array([conf for t, conf in self.confidence_history])
            
            ax.plot(times, confidences, 'g-', linewidth=3, label='Confidence')
            ax.fill_between(times, confidences, alpha=0.3, color='green')
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Confidence', fontsize=12)
            #ax.set_title('MCL Confidence Evolution', fontweight='bold', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            ax.set_ylim(0, 0.6)
            
            # Add particle adaptation thresholds
            ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='> 0.3: Reduce number of particles')
            ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='< 0.2: Increase number of particles')
            ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='< 0.1: Implement recovery strategy')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig("confidence.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ confidence.png")

        # 2. FINAL TRAJECTORY 
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.patch.set_facecolor('white')

        if self.pose_estimates:
            # Convert to grid-based coordinates
            trajectory_points = []
            for t, x, y, theta in self.pose_estimates:
                grid_col = x / RESOLUTION - 0.5
                grid_row = y / RESOLUTION - 0.5
                trajectory_points.append((grid_row, grid_col))  # Note the order: row (Y), col (X)

            trajectory_points = np.array(trajectory_points)
            y_coords, x_coords = trajectory_points.T

            # Plot actual estimated trajectory
            ax.plot(x_coords, y_coords, 'b-', alpha=0.8, linewidth=3, label='Estimated Trajectory')

            # Draw trajectory start and end as filled squares at grid positions
            start_grid = GROUND_TRUTH_CHECKPOINTS[0]
            end_grid = GROUND_TRUTH_CHECKPOINTS[-1]
            ax.add_patch(plt.Rectangle((start_grid[1] - 0.5, start_grid[0] - 0.5), 1, 1, facecolor='green', edgecolor='darkgreen', linewidth=2, label='Start'))
            ax.add_patch(plt.Rectangle((end_grid[1] - 0.5, end_grid[0] - 0.5), 1, 1, facecolor='red', edgecolor='darkred', linewidth=2, label='End'))

            # Interpolate ground truth path smoothly with boundary constraints
            checkpoints_np = np.array(GROUND_TRUTH_CHECKPOINTS)
            cp_rows, cp_cols = checkpoints_np.T
            
            # Create parameter values for interpolation (distance-based parameterization)
            distances = np.zeros(len(GROUND_TRUTH_CHECKPOINTS))
            for i in range(1, len(GROUND_TRUTH_CHECKPOINTS)):
                dist = np.sqrt((cp_rows[i] - cp_rows[i-1])**2 + (cp_cols[i] - cp_cols[i-1])**2)
                distances[i] = distances[i-1] + dist
            
            # Normalize to [0, 1]
            t_checkpoints = distances / distances[-1] if distances[-1] > 0 else np.linspace(0, 1, len(GROUND_TRUTH_CHECKPOINTS))
            
            # Create smooth interpolation with many points
            from scipy.interpolate import interp1d
            
            # Use quadratic interpolation for smoother but more constrained curves
            t_smooth = np.linspace(0, 1, 200)  # 200 points for smooth curve
            
            # Interpolate rows and columns separately
            f_rows = interp1d(t_checkpoints, cp_rows, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            f_cols = interp1d(t_checkpoints, cp_cols, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            
            smooth_rows = f_rows(t_smooth)
            smooth_cols = f_cols(t_smooth)
            
            # Clamp interpolated points to stay within navigable boundaries
            # Define navigable area bounds (from your code: start_row to end_row, start_col to end_col)
            min_row, max_row = start_row, end_row - 1
            min_col, max_col = start_col, end_col - 1
            
            smooth_rows = np.clip(smooth_rows, min_row, max_row)
            smooth_cols = np.clip(smooth_cols, min_col, max_col)
            
            # Additional check: remove points that would be in walls
            valid_mask = np.ones(len(smooth_rows), dtype=bool)
            for i in range(len(smooth_rows)):
                row_idx = int(round(smooth_rows[i]))
                col_idx = int(round(smooth_cols[i]))
                if (row_idx < 0 or row_idx >= GRID_ROWS or 
                    col_idx < 0 or col_idx >= GRID_COLS or 
                    grid[row_idx, col_idx] == 1):  # Wall check
                    valid_mask[i] = False
            
            smooth_rows = smooth_rows[valid_mask]
            smooth_cols = smooth_cols[valid_mask]
            
            # Plot smooth ground truth path
            ax.plot(smooth_cols, smooth_rows, 'k--', linewidth=2, label='Ground Truth Path', alpha=0.6)

            # Mark individual checkpoints
            for i, (cp_row, cp_col) in enumerate(GROUND_TRUTH_CHECKPOINTS):
                if i == 0:  # Start checkpoint
                    ax.plot(cp_col, cp_row, 'o', markersize=8, color='green', markeredgecolor='darkgreen', markeredgewidth=2)
                elif i == len(GROUND_TRUTH_CHECKPOINTS) - 1:  # End checkpoint
                    ax.plot(cp_col, cp_row, 'o', markersize=8, color='red', markeredgecolor='darkred', markeredgewidth=2)
                else:  # Intermediate checkpoints
                    ax.plot(cp_col, cp_row, 'o', markersize=6, color='orange', markeredgecolor='darkorange', markeredgewidth=1)
                    ax.text(cp_col + 0.3, cp_row + 0.3, f'CP{i+1}', fontsize=8, color='black', fontweight='bold')

        # Walls and ArUcos
        wall_mask = (grid == 1)
        ax.imshow(wall_mask, cmap='gray', origin='upper', alpha=0.3)

        # Grid lines
        for i in range(GRID_ROWS + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        for j in range(GRID_COLS + 1):
            ax.axvline(j - 0.5, color='black', linewidth=0.5, alpha=0.3)

        # ArUco markers
        for marker_id, (row, col) in MAP_MARKERS_DISPLAY.items():
            ax.plot(col, row, 's', markersize=12, color='red')
            ax.text(col, row, f'A{marker_id}', color='white', ha='center', va='center', fontweight='bold')

        ax.set_xlim(-0.5, GRID_COLS - 0.5)
        ax.set_ylim(GRID_ROWS - 0.5, -0.5)
        ax.set_xticks(range(GRID_COLS))
        ax.set_xticklabels(range(1, GRID_COLS + 1))
        ax.set_yticks(range(GRID_ROWS))
        ax.set_yticklabels(range(1, GRID_ROWS + 1))
        ax.set_aspect('equal')
        ax.set_xlabel('Grid Columns (X)', fontsize=12)
        ax.set_ylabel('Grid Rows (Y)', fontsize=12)
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig("final_trajectory.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ final_trajectory.png saved")
        
        # 3. PARTICLE COUNT EVOLUTION
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        if self.particle_counts:
            times = np.array([t.to_sec() for t, count in self.particle_counts])
            times = times - times[0]  # Start from 0
            counts = np.array([count for t, count in self.particle_counts])
            
            ax.plot(times, counts, 'b-', linewidth=3)
            ax.fill_between(times, counts, alpha=0.3, color='blue')
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Number of Particles', fontsize=12)
            ax.set_title('Dynamic Particle Adaptation', fontweight='bold', fontsize=16)
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits based on actual min/max particle counts
            min_particles = max(int(counts.min() * 0.95), 0)  # 5% margin below minimum
            max_particles = int(counts.max() * 1.05)  # 5% margin above maximum
            ax.set_ylim(min_particles, max_particles)
            
            # Add min/max lines
            ax.axhline(y=MIN_PARTICLES, color='red', linestyle='--', alpha=0.7, label=f'Min: {MIN_PARTICLES}')
            ax.axhline(y=MAX_PARTICLES, color='orange', linestyle='--', alpha=0.7, label=f'Max: {MAX_PARTICLES}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_particles.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {output_prefix}_particles.png")

        # 4. VELOCITIES
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.patch.set_facecolor('white')
        
        if self.velocity_history:
            times = np.array([t.to_sec() for t, lin_vel, ang_vel, lin_x, lin_y in self.velocity_history])
            times = times - times[0]  # Start from 0
            linear_velocities = np.array([lin_vel for t, lin_vel, ang_vel, lin_x, lin_y in self.velocity_history])
            angular_velocities = np.array([ang_vel for t, lin_vel, ang_vel, lin_x, lin_y in self.velocity_history])
            
            # Linear Velocity
            ax1.plot(times, linear_velocities, 'purple', linewidth=2)
            ax1.fill_between(times, linear_velocities, alpha=0.3, color='purple')
            ax1.set_xlabel('Time (s)', fontsize=12)
            ax1.set_ylabel('Linear Velocity (m/s)', fontsize=12)
            ax1.set_title('Linear Velocity', fontweight='bold', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            if len(linear_velocities) > 0:
                mean_vel = np.mean(linear_velocities)
                ax1.axhline(y=mean_vel, color='orange', linestyle='--', alpha=0.7, 
                        label=f'Mean: {mean_vel:.3f} m/s')
                ax1.legend()
            
            # Angular Velocity
            ax2.plot(times, angular_velocities, 'teal', linewidth=2)
            ax2.fill_between(times, angular_velocities, alpha=0.3, color='teal')
            ax2.set_xlabel('Time (s)', fontsize=12)
            ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
            ax2.set_title('Angular Velocity', fontweight='bold', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            if len(angular_velocities) > 0:
                mean_vel = np.mean(np.abs(angular_velocities))
                ax2.axhline(y=mean_vel, color='orange', linestyle='--', alpha=0.7, 
                        label=f'Mean |ω|: {mean_vel:.3f} rad/s')
                ax2.axhline(y=-mean_vel, color='orange', linestyle='--', alpha=0.7)
                ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_velocities.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {output_prefix}_velocities.png")

        # 5. ARUCO DISTANCES
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.patch.set_facecolor('white')
        
        if self.distance_history:
            ax.set_title('ArUco Distance Measurements Timeline', fontweight='bold', fontsize=16)
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Distance (m)', fontsize=12)
            
            colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 
                    4: 'purple', 5: 'brown', 8: 'pink', 10: 'gray'}
            
            # Organize data by marker
            marker_data = {}
            start_time = self.distance_history[0][0].to_sec()
            
            for timestamp, distances in self.distance_history:
                time_sec = timestamp.to_sec() - start_time
                for marker_id, distance in distances.items():
                    if marker_id not in marker_data:
                        marker_data[marker_id] = {'times': [], 'distances': []}
                    marker_data[marker_id]['times'].append(time_sec)
                    marker_data[marker_id]['distances'].append(distance)
            
            # Plot each marker
            for marker_id, data in marker_data.items():
                if data['times']:
                    ax.plot(data['times'], data['distances'], 
                        color=colors.get(marker_id, 'black'), 
                        linewidth=3, marker='o', markersize=6,
                        label=f'ArUco {marker_id}', alpha=0.8)
            
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            if marker_data:
                ax.set_ylim(0, max([max(data['distances']) for data in marker_data.values()]) * 1.1)
        else:
            ax.text(0.5, 0.5, 'No distance data available', 
                transform=ax.transAxes, ha='center', va='center', fontsize=16)
            ax.set_title('ArUco Distance Measurements Timeline', fontweight='bold', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_distances.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {output_prefix}_distances.png")

        # 6. PARTICLE EVOLUTION 
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor('white')

        if len(self.particle_history) >= 3:
            target_times = [45.0, 85.0, 125.0]  # seconds
            titles = ['45 seconds', '85 seconds', '125 seconds']
            
            # Get start time for relative timing
            start_time = self.particle_history[0][0].to_sec()
            
            frame_indices = []
            actual_times = []
            
            for target_time in target_times:
                best_frame_idx = 0
                min_time_diff = float('inf')
                
                for idx, (timestamp, particles, pose) in enumerate(self.particle_history):
                    relative_time = timestamp.to_sec() - start_time
                    time_diff = abs(relative_time - target_time)
                    
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_frame_idx = idx
                
                frame_indices.append(best_frame_idx)
                actual_time = self.particle_history[best_frame_idx][0].to_sec() - start_time
                actual_times.append(actual_time)
            
            # Update titles with actual times found
            titles = [f'{target_time}s' 
                    for target_time, actual_time in zip(target_times, actual_times)]
            
            # Track global min/max for consistent colorbar scaling
            global_min = float('inf')
            global_max = float('-inf')
            prob_maps = []
            
            # First pass: calculate all probability maps and find global min/max
            for frame_idx in frame_indices:
                timestamp, particles, (est_x, est_y, est_theta) = self.particle_history[frame_idx]
                
                # Create probability map
                prob_map = self.create_continuous_probability_map(particles)
                wall_mask = (grid == 1)
                prob_map[wall_mask] = 0
                
                # Find min/max of non-zero values
                non_zero_values = prob_map[prob_map > 0]
                if len(non_zero_values) > 0:
                    global_min = min(global_min, np.min(non_zero_values))
                    global_max = max(global_max, np.max(non_zero_values))
                
                prob_maps.append(prob_map)
            
            # Handle case where all values are zero
            if global_min == float('inf'):
                global_min = 0
                global_max = 1
            
            for i, (frame_idx, title) in enumerate(zip(frame_indices, titles)):
                ax = axes[i]
                timestamp, particles, (est_x, est_y, est_theta) = self.particle_history[frame_idx]
                prob_map = prob_maps[i]
                
                # Show probability map with consistent scaling
                im = ax.imshow(prob_map, cmap='Blues', origin='upper', alpha=0.9, 
                            vmin=global_min, vmax=global_max)
                
                # Walls
                wall_mask = (grid == 1)
                ax.imshow(wall_mask, cmap='gray', origin='upper', alpha=0.3)
                
                # Grid
                for j in range(GRID_ROWS + 1):
                    ax.axhline(j - 0.5, color='black', linewidth=0.5, alpha=0.3)
                for j in range(GRID_COLS + 1):
                    ax.axvline(j - 0.5, color='black', linewidth=0.5, alpha=0.3)
                
                # ArUcos
                for marker_id, (row, col) in MAP_MARKERS_DISPLAY.items():
                    ax.plot(col, row, 's', markersize=10, color='red')
                    ax.text(col, row, f'{marker_id}', color='white', ha='center', va='center', 
                            fontweight='bold', fontsize=9)
                
                # Estimated position
                est_col = est_x / RESOLUTION - 0.5
                est_row = est_y / RESOLUTION - 0.5
                ax.plot(est_col, est_row, 'r*', markersize=20, markeredgecolor='white', 
                        markeredgewidth=2, label='Estimated Position')
        
                ax.set_title(f'{title}', 
                            fontweight='bold', fontsize=12)
                ax.set_xlim(-0.5, GRID_COLS - 0.5)
                ax.set_ylim(GRID_ROWS - 0.5, -0.5)
                ax.set_xticks(range(GRID_COLS))
                ax.set_xticklabels(range(1, GRID_COLS + 1))
                ax.set_yticks(range(GRID_ROWS))
                ax.set_yticklabels(range(1, GRID_ROWS + 1))
                ax.set_aspect('equal')
                
                if i == 0:
                    ax.legend(loc='upper right', fontsize=9)
                    ax.set_ylabel('Grid Rows (Y)', fontsize=10)
                
                ax.set_xlabel('Grid Columns (X)', fontsize=10)
                
                # Add colorbar to last subplot with Min/Max labels
                if i == 2:
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.6)
                    cbar.set_label('Particle Density', fontsize=10)
                    
                    # Set just "Minimum" and "Maximum" labels
                    cbar.set_ticks([global_min, global_max])
                    cbar.set_ticklabels(['Minimum', 'Maximum'])

        plt.suptitle('Particle Distribution Evolution', fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.savefig("grid-map.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ grid-map.png")

        print(f"\n✅ All focused analyses created with prefix '{output_prefix}_'")
            
def main():
    parser = argparse.ArgumentParser(description='MCL com Distâncias Reais de ArUcos - Localização Robótica')
    parser.add_argument('bag_file', help='Ficheiro bag de entrada com dados do robô')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.bag_file):
        print(f"❌ Ficheiro não encontrado: {args.bag_file}")
        return 1
    
    try:
        # Verificar tamanho do ficheiro de entrada
        input_size = os.path.getsize(args.bag_file) / (1024 * 1024)  # MB
        print(f"📁 Ficheiro de entrada: {input_size:.1f}MB")
        
        # Criar processor sem bag de saída
        processor = MCLProcessor(args.bag_file, output_bag_path=None)
        
        # Processamento
        print("🚀 Iniciando processamento de localização MCL...")
        processor.process_bag()
        
        if len(processor.pose_estimates) > 10:
            # Criar apenas visualizações
            print("\n🎨 Criando análises de localização...")
            
            try:
                print("📊 Gerando análise estática...")
                processor.create_static_visualization("mcl_distances_analysis.png")
                print("✅ PNG criado: mcl_distances_analysis.png")
            except Exception as e:
                print(f"❌ Erro na análise estática: {e}")
                traceback.print_exc()
            
            try:
                if len(processor.particle_history) > 5:
                    print("🎬 Gerando vídeo de evolução...")
                    processor.create_visualization_video("mcl_distances_evolution.mp4")
                    print("✅ Vídeo criado: mcl_distances_evolution.mp4")
                else:
                    print("⚠️ Dados insuficientes para vídeo (precisa de mais estados de partículas)")
            except Exception as e:
                print(f"❌ Erro no vídeo: {e}")
                traceback.print_exc()
            
            print("\n🎉 Análise de localização MCL concluída!")
            print("📁 Ficheiros gerados:")
            if os.path.exists("mcl_distances_analysis.png"):
                print(f"   • mcl_distances_analysis.png")
            if os.path.exists("mcl_distances_evolution.mp4"):
                print(f"   • mcl_distances_evolution.mp4")
            
            # Resumo final
            print(f"\n📊 Resumo da localização:")
            print(f"   • Frames processados: {processor.processed_frames}")
            print(f"   • Detecções ArUco: {processor.aruco_detections}")
            print(f"   • Medições de distância: {processor.distance_detections}")
            
            if processor.distance_detections > 0:
                print(f"\n📏 Dados de distância do robô:")
                unique_markers = set(mid for _, distances in processor.distance_history for mid in distances.keys())
                print(f"   • ArUcos com distâncias medidas: {sorted(list(unique_markers))}")
                
                # Estatística das medições reais
                all_distances = [dist for _, distances in processor.distance_history for dist in distances.values()]
                if all_distances:
                    print(f"   • Distância média medida: {np.mean(all_distances):.3f}m")
                    print(f"   • Faixa de medições: {min(all_distances):.3f}m - {max(all_distances):.3f}m")
                    print(f"   • Total de medições: {len(all_distances)}")
                    
                    # Calcular precisão da localização
                    if processor.confidence_history:
                        confidences = [c[1] for c in processor.confidence_history]
                        avg_confidence = np.mean(confidences)
                        final_confidence = confidences[-1]
                        
                        print(f"\n📈 Qualidade da localização:")
                        print(f"   • Confiança média: {avg_confidence:.3f}")
                        print(f"   • Confiança final: {final_confidence:.3f}")
                        
                print(f"\n⚠️ Nenhuma medição de distância foi encontrada no bag file")
                print(f"   • Certifique-se de que o tópico '/marker_distances' existe")
                print(f"   • Verifique se o robô estava a medir distâncias aos ArUcos")
                print(f"   • A localização foi baseada apenas em detecções visuais")
                
            return 0
        else:
            print("\n❌ Dados insuficientes para análise de localização")
            print("   • Precisa de mais de 10 estimativas de pose")
            print("   • Verifique se o bag contém dados de imagem e ArUcos")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Interrompido pelo utilizador")
        return 1
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())