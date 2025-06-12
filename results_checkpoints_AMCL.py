# python3 results_AMCL.py test_26.bag test_27.bag

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
import sys 

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

# Posições dos ArUcos PARA CÁLCULOS (nas paredes)
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

# Orientações dos ArUcos no mapa (CORRIGIDAS)
ARUCO_ORIENTATIONS = {
    0: (-1, 0),   # ArUco 0: horizontal, aponta para esquerda
    1: (0, 1),    # ArUco 1: vertical, aponta para baixo  
    2: (-1, 0),   # ArUco 2: horizontal, aponta para esquerda
    3: (1, 0),    # ArUco 3: horizontal, aponta para direita
    4: (0, 1),    # ArUco 4: vertical, aponta para baixo
    5: (0, 1),    # ArUco 5: vertical, aponta para baixo
    8: (0, -1),   # ArUco 8: vertical, aponta para cima
    10: (-1, 0),  # ArUco 10: horizontal, aponta para esquerda
}

ARUCO_POSITIONS = np.array(list(MAP_MARKERS_REAL.values()))

# TRAJETÓRIA GROUND TRUTH ***
GROUND_TRUTH_CHECKPOINTS = [
    (4, 19),  # Checkpoint 1 - início
    (5, 13),  # Checkpoint 2
    (6, 5),   # Checkpoint 3
    (2, 3),   # Checkpoint 4
    (2, 13),  # Checkpoint 5
    (4, 19)   # Checkpoint 6 - final (mesmo que início)
]

class TrajectoryEvaluator:
    """Classe para avaliar trajetórias e calcular erros"""
    
    def __init__(self, ground_truth_checkpoints):
        self.checkpoints = ground_truth_checkpoints
        self.interpolated_trajectory = self.interpolate_trajectory()
        
    def interpolate_trajectory(self):
        """Interpola trajetória entre checkpoints"""
        interpolated = []
        
        for i in range(len(self.checkpoints) - 1):
            start = np.array(self.checkpoints[i])
            end = np.array(self.checkpoints[i + 1])
            
            # Calcular número de pontos baseado na distância
            distance = np.linalg.norm(end - start)
            num_points = max(int(distance * 2), 2)  # 2 pontos por célula de distância
            
            # Interpolação linear
            for j in range(num_points):
                t = j / (num_points - 1) if num_points > 1 else 0
                point = start + t * (end - start)
                interpolated.append(tuple(point))
                
        return interpolated
    
    def grid_to_world(self, row, col):
        """Converte células da grelha para coordenadas do mundo"""
        x = (col + 0.5) * RESOLUTION
        y = (row + 0.5) * RESOLUTION
        return x, y
    
    def world_to_grid(self, x, y):
        """Converte coordenadas do mundo para células da grelha"""
        col = int(x / RESOLUTION - 0.5)
        row = int(y / RESOLUTION - 0.5)
        return row, col
    
    def find_closest_ground_truth_point(self, estimated_x, estimated_y):
        """Encontra o ponto mais próximo na trajetória ground truth"""
        # Convert world coordinates to grid coordinates properly
        estimated_col = estimated_x / RESOLUTION - 0.5  # Convert to continuous grid coordinates
        estimated_row = estimated_y / RESOLUTION - 0.5  # Convert to continuous grid coordinates
        
        min_distance = float('inf')
        closest_point = None
        
        for gt_row, gt_col in self.interpolated_trajectory:
            # Calculate distance in grid cells using continuous coordinates
            distance = np.sqrt((estimated_row - gt_row)**2 + (estimated_col - gt_col)**2)
            if distance < min_distance:
                min_distance = distance
                closest_point = (gt_row, gt_col)
        
        return closest_point, min_distance
    
    def calculate_trajectory_errors(self, estimated_poses):
        """Calcula erros de trajetória com melhor mapeamento checkpoint-pose (Method 2)"""
        errors = {
            'position_errors': [],
            'distances_to_trajectory': [],
            'mean_error': 0,
            'max_error': 0,
            'rmse': 0,
            'checkpoint_errors': []
        }
        
        if not estimated_poses:
            return errors
        
        # Calcular erros ponto a ponto
        for timestamp, est_x, est_y, est_theta in estimated_poses:
            closest_gt, distance_error = self.find_closest_ground_truth_point(est_x, est_y)
            
            if closest_gt:
                # Converter para metros
                distance_error_meters = distance_error * RESOLUTION
                errors['position_errors'].append(distance_error_meters)
                errors['distances_to_trajectory'].append((timestamp, distance_error_meters))
        
        if errors['position_errors']:
            errors['mean_error'] = np.mean(errors['position_errors'])
            errors['max_error'] = np.max(errors['position_errors'])
            errors['rmse'] = np.sqrt(np.mean([e**2 for e in errors['position_errors']]))
        
        # Find the actual closest estimated pose to each checkpoint
        for i, (checkpoint_row, checkpoint_col) in enumerate(self.checkpoints):
            min_distance_to_checkpoint = float('inf')
            best_error = float('inf')
            
            # Check all estimated poses to find the one closest to this checkpoint
            for timestamp, est_x, est_y, est_theta in estimated_poses:
                # Convert estimated world coordinates to grid coordinates
                est_col = est_x / RESOLUTION - 0.5
                est_row = est_y / RESOLUTION - 0.5
                
                # Calculate distance to this checkpoint in grid coordinates
                distance_to_checkpoint = np.sqrt(
                    (est_row - checkpoint_row)**2 + (est_col - checkpoint_col)**2
                )
                
                if distance_to_checkpoint < min_distance_to_checkpoint:
                    min_distance_to_checkpoint = distance_to_checkpoint
                    best_error = distance_to_checkpoint * RESOLUTION  # Convert to meters
            
            if best_error != float('inf'):
                errors['checkpoint_errors'].append((i + 1, best_error))
        
        return errors

class Particle:
    def __init__(self, row, col, theta=0.0, weight=1.0):
        self.row = row
        self.col = col
        self.row_continuous = float(row)
        self.col_continuous = float(col)
        self.theta = theta
        self.weight = weight

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
    def __init__(self, bag_path, evaluator):
        print(f"=== MCL Trajectory Evaluation: {os.path.basename(bag_path)} ===")

        self.bag_path = bag_path
        self.evaluator = evaluator
        
        # Calibração do sensor de distância
        self.setup_distance_calibration()
        
        # Inicialização das partículas
        self.N = MIN_PARTICLES
        self.particles = self.init_particles_uniform()
        
        # Controlo de movimento
        self.last_cmd_vel = None
        self.last_cmd_time = None
        
        # Controlo temporal
        self.last_aruco_time = None
        
        # Armazenamento de dados dos sensores
        self.current_marker_distances = {}
        self.distance_history = []
        self.last_distance_time = None
        
        self.current_aruco_poses = {}
        self.pose_history = []
        self.last_pose_time = None
        
        # Contadores
        self.frame_count = 0
        self.processed_frames = 0
        self.aruco_detections = 0
        self.distance_detections = 0
        self.pose_detections = 0
        self.start_time = None
        
        # Resultados
        self.pose_estimates = []
        self.particle_counts = []
        self.detected_markers = []
        self.confidence_history = []
        self.entropy_history = []
        self.velocity_history = []
        
    def setup_distance_calibration(self):
        """Configura calibração do sensor"""
        # Dados de calibração coletados
        robot_measurements = [2.5, 7.02, 6.70, 6.15, 5.9, 5.47, 5, 4.1, 2.8, 2.2, 1.4]
        real_distances = [2.21, 6.20, 5.9, 5.6, 5.3, 5, 4.4, 3.8, 2.6, 2, 1.5]
        
        # Regressão linear simples
        n = len(robot_measurements)
        sum_x = sum(robot_measurements)
        sum_y = sum(real_distances)
        sum_xy = sum([x*y for x, y in zip(robot_measurements, real_distances)])
        sum_x2 = sum([x*x for x in robot_measurements])
        
        self.calibration_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        self.calibration_intercept = (sum_y - self.calibration_slope * sum_x) / n
        
        # Calcular RMSE
        corrected = [self.calibration_slope * m + self.calibration_intercept for m in robot_measurements]
        ss_res = sum([(real - corr)**2 for real, corr in zip(real_distances, corrected)])
        self.calibration_rmse = (ss_res / len(real_distances))**0.5
        
        self.distance_measurement_sigma = max(0.15, self.calibration_rmse * 1.5)

    def correct_distance_measurement(self, measured_distance):
        """Corrige distância medida usando modelo de calibração"""
        if measured_distance <= 0:
            return measured_distance
        corrected = self.calibration_slope * measured_distance + self.calibration_intercept
        return max(0.1, corrected)

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
        return particles
    
    def update_particle_weights_with_distances(self, measured_distances):
        """Atualiza pesos das partículas com base nas distâncias medidas usando zonas de distância"""
        if not measured_distances or not self.particles:
            return
        
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
        
    def is_free_cell(self, row, col):
        """Verifica se a célula está livre"""
        return (start_row <= row < end_row and start_col <= col < end_col)

    def grid_to_world(self, row, col):
        """Converte células da grelha para coordenadas do mundo"""
        x = (col + 0.5) * RESOLUTION
        y = (row + 0.5) * RESOLUTION
        return x, y

    def calculate_expected_distance(self, particle_row, particle_col, marker_row, marker_col):
        """Calcula a distância esperada entre uma partícula e um marcador"""
        p_x, p_y = self.grid_to_world(particle_row, particle_col)
        m_x, m_y = self.grid_to_world(marker_row, marker_col)
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
            return dy > 0 and abs(dx) <= tolerance
        elif aruco_orientation == (0, -1):  # Aponta para cima
            return dy < 0 and abs(dx) <= tolerance
        elif aruco_orientation == (-1, 0):  # Aponta para esquerda
            return dx < 0 and abs(dy) <= tolerance
        elif aruco_orientation == (1, 0):   # Aponta para direita
            return dx > 0 and abs(dy) <= tolerance
        
        return True

    # Funções para processar poses dos ArUcos
    def quaternion_to_euler(self, qx, qy, qz, qw):
        """Converte quaternion para ângulos de Euler"""
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return 0, 0, yaw  # Só precisamos do yaw

    def calculate_robot_orientation_from_aruco_pose(self, aruco_id, aruco_pose, robot_to_aruco_distance):
        """Calcula orientação do robô com base na pose do ArUco"""
        if aruco_id not in ARUCO_ORIENTATIONS:
            return None
            
        x, y, z, qx, qy, qz, qw = aruco_pose
        
        # Converter quaternion para ângulo yaw
        _, _, aruco_yaw_in_camera = self.quaternion_to_euler(qx, qy, qz, qw)
        
        # Orientação do ArUco no mapa mundial
        aruco_orientation = ARUCO_ORIENTATIONS[aruco_id]
        aruco_world_yaw = np.arctan2(aruco_orientation[1], aruco_orientation[0])
        
        # Método 1: Baseado na geometria relativa
        camera_to_aruco_angle = np.arctan2(x, z)
        estimated_robot_theta = aruco_world_yaw + np.pi - camera_to_aruco_angle
        
        # Método 2: Usando a orientação observada do ArUco
        orientation_difference = aruco_yaw_in_camera - aruco_world_yaw
        estimated_robot_theta_v2 = -orientation_difference
        
        # Combinação híbrida
        weight1, weight2 = 0.7, 0.3
        combined_theta = weight1 * estimated_robot_theta + weight2 * estimated_robot_theta_v2
        
        # Normalizar ângulo
        combined_theta = np.arctan2(np.sin(combined_theta), np.cos(combined_theta))
        
        return combined_theta

    def process_aruco_poses(self, poses_msg, timestamp):
        """Processa poses dos ArUcos"""
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
                    parts = line.split(',')
                    if len(parts) >= 11:
                        marker_id = int(parts[1])
                        
                        x = float(parts[4])
                        y = float(parts[5])
                        z = float(parts[6])
                        qx = float(parts[7])
                        qy = float(parts[8])
                        qz = float(parts[9])
                        qw = float(parts[10])
                        
                        if marker_id in MAP_MARKERS_REAL:
                            current_poses[marker_id] = (x, y, z, qx, qy, qz, qw)
                            
                except (ValueError, IndexError):
                    continue
            
            if current_poses:
                self.current_aruco_poses = current_poses
                self.pose_history.append((timestamp, current_poses.copy()))
                self.pose_detections += 1
                self.last_pose_time = timestamp
                
        except Exception as e:
            print(f"❌ Erro ao processar poses: {e}")

    def validate_and_extract_distances(self, distances_msg):
        """Valida e extrai distâncias"""
        raw_data = None
        if hasattr(distances_msg, 'data'):
            raw_data = distances_msg.data
        elif hasattr(distances_msg, 'distances'):
            raw_data = distances_msg.distances
        else:
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+', str(distances_msg))
            if numbers:
                raw_data = [float(n) for n in numbers]
                
        if raw_data is None:
            return []
        
        # Converter para lista
        if isinstance(raw_data, (list, tuple)):
            distances_data = list(raw_data)
        else:
            distances_data = [raw_data]
        
        # Filtrar distâncias válidas
        valid_distances = []
        for d in distances_data:
            if isinstance(d, (int, float)) and 0.1 <= d <= 10.0:
                valid_distances.append(float(d))
        
        return valid_distances

    def get_aruco_detection_order(self, timestamp):
        """Determina a ordem dos ArUcos detectados"""
        for det_time, aruco_ids in reversed(self.detected_markers):
            time_diff = (timestamp - det_time).to_sec()
            if time_diff <= 3.0:
                return sorted(aruco_ids)
        return []

    def process_marker_distances(self, distances_msg, timestamp):
        """Processa distâncias dos marcadores"""
        valid_distances = self.validate_and_extract_distances(distances_msg)
        
        if not valid_distances:
            self.current_marker_distances = {}
            return
        
        # Correção de calibração
        corrected_distances = [self.correct_distance_measurement(d) for d in valid_distances]
        
        # Mapear para ArUcos
        recent_aruco_ids = self.get_aruco_detection_order(timestamp)
        current_distances = {}
        
        for i, distance in enumerate(valid_distances):
            if i < len(recent_aruco_ids):
                marker_id = recent_aruco_ids[i]
                if marker_id in MAP_MARKERS_REAL:
                    current_distances[marker_id] = float(distance)
        
        if current_distances:
            self.current_marker_distances = current_distances
            self.distance_history.append((timestamp, current_distances.copy()))
            self.distance_detections += 1
            self.last_distance_time = timestamp
            self.update_particle_weights_with_distances(current_distances)

    def process_aruco_detections(self, marker_ids_msg, timestamp):
        """Processa detecções de ArUcos"""
        detected_ids = marker_ids_msg.data
        
        if not detected_ids:
            return
        
        self.aruco_detections += 1
        
        unique_ids = list(set(detected_ids))
        valid_ids = [id_val for id_val in unique_ids if id_val in MAP_MARKERS_REAL]
        
        if not valid_ids:
            return
        
        # Estimar orientação usando poses (se disponível)
        estimated_robot_orientations = []
        
        for aruco_id in valid_ids:
            if (aruco_id in self.current_aruco_poses and 
                self.last_pose_time is not None):
                
                time_diff = (timestamp - self.last_pose_time).to_sec()
                if time_diff <= 2.0:
                    aruco_pose = self.current_aruco_poses[aruco_id]
                    
                    estimated_theta = self.calculate_robot_orientation_from_aruco_pose(
                        aruco_id, aruco_pose, np.linalg.norm(aruco_pose[:3]))
                    
                    if estimated_theta is not None:
                        estimated_robot_orientations.append(estimated_theta)
        
        # Calcular orientação média e ajustar partículas
        if estimated_robot_orientations:
            sin_sum = sum(np.sin(theta) for theta in estimated_robot_orientations)
            cos_sum = sum(np.cos(theta) for theta in estimated_robot_orientations)
            final_estimated_orientation = np.arctan2(sin_sum, cos_sum)
            
            # Ajustar orientações das partículas de maior peso
            mean_weight = np.mean([p.weight for p in self.particles])
            for particle in self.particles:
                if particle.weight > mean_weight:
                    orientation_diff = final_estimated_orientation - particle.theta
                    while orientation_diff > np.pi:
                        orientation_diff -= 2 * np.pi
                    while orientation_diff < -np.pi:
                        orientation_diff += 2 * np.pi
                    
                    particle.theta += 0.2 * orientation_diff
                    particle.theta = np.arctan2(np.sin(particle.theta), np.cos(particle.theta))

        # Atualizar pesos das partículas
        if valid_ids:
            particle_positions = np.array([[p.row, p.col] for p in self.particles])
            particle_orientations = np.array([p.theta for p in self.particles])
            weights = np.array([p.weight for p in self.particles])
            
            for aruco_id in valid_ids:
                marker_pos = np.array(MAP_MARKERS_REAL[aruco_id])
                
                # Cálculo de distâncias
                distances = np.linalg.norm(particle_positions - marker_pos, axis=1)
                in_range = distances <= 4.0
                
                # Verificar campo de visão 
                visibility_mask = np.array([
                    self.can_particle_see_aruco(p_row, p_col, aruco_id) 
                    for p_row, p_col in particle_positions
                ])
                
                prob_see = np.exp(-distances / 2.0)
                position_weights = np.where(
                    in_range & visibility_mask,
                    1.0 + prob_see * 10.0, 
                    0.1
                )
                
                # Orientação
                dx = marker_pos[1] - particle_positions[:, 1]
                dy = marker_pos[0] - particle_positions[:, 0]
                expected_theta = np.arctan2(dy, dx)
                angle_diff = np.abs(particle_orientations - expected_theta)
                angle_diff = np.where(angle_diff > np.pi, 2 * np.pi - angle_diff, angle_diff)
                orientation_weights = np.exp(-angle_diff**2 / (2 * (np.pi / 4)**2))

                # Distâncias medidas
                distance_weights = np.ones_like(position_weights)
                if (aruco_id in self.current_marker_distances and 
                    self.last_distance_time is not None):
                    
                    time_diff = (timestamp - self.last_distance_time).to_sec()
                    if time_diff <= 2.0:
                        measured_dist = self.current_marker_distances[aruco_id]
                        marker_row, marker_col = marker_pos
                        
                        expected_distances = np.array([
                            self.calculate_expected_distance(p_row, p_col, marker_row, marker_col)
                            for p_row, p_col in particle_positions
                        ])
                        
                        distance_errors = np.abs(expected_distances - measured_dist)
                        distance_sigma = 0.3
                        distance_weights = np.exp(-(distance_errors**2) / (2 * distance_sigma**2))

                # Combinar pesos
                combined_weights = position_weights * orientation_weights * distance_weights
                weights *= combined_weights
            
            # Atualizar pesos das partículas
            for i, p in enumerate(self.particles):
                p.weight = weights[i]
                
        # Normalizar e reamostragem
        self.normalize_weights()
        confidence = self.calculate_confidence()
        
        self.confidence_history.append((timestamp, confidence))
        
        # Reamostragem adaptativa
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
            print(f"Reamostragem: {reason}")
            self.resample(timestamp)
        
        self.last_aruco_time = timestamp
        self.detected_markers.append((timestamp, valid_ids))

    def calculate_confidence(self):
        """Calcula confiança baseada na entropia"""
        if not self.particles:
            return 0.0
        
        weights = np.array([p.weight for p in self.particles])
        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0.0
        
        normalized_weights = weights / total_weight
        mask = normalized_weights > 1e-12
        entropy = -np.sum(normalized_weights[mask] * np.log(normalized_weights[mask]))
        
        max_entropy = np.log(len(self.particles))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
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

    def recovery_strategy(self):
        """Reinicializa partículas de menor peso"""
        if not self.particles:
            return
        
        weights = np.array([p.weight for p in self.particles])
        worst_indices = np.argpartition(weights, len(weights) // 5)[:len(weights) // 5]
        
        # Criar novas partículas
        for idx in worst_indices:
            new_idx = np.random.randint(len(VALID_POSITIONS))
            row, col = VALID_POSITIONS[new_idx]
            new_theta = np.random.uniform(-np.pi, np.pi)
            self.particles[idx] = Particle(row, col, new_theta, weight=0.1)
        
        self.normalize_weights()

    def low_variance_sampler(self, particles, weights):
        """Reamostragem com baixa variância"""
        M = len(particles)
        if M == 0:
            return []
        
        r = np.random.uniform(0, 1.0 / M)
        c = np.cumsum(weights)
        u = r + np.arange(M) / M
        
        indices = np.searchsorted(c, u)
        indices = np.clip(indices, 0, M - 1)
        
        new_particles = []
        for i in indices:
            new_particle = copy.deepcopy(particles[i])
            new_particles.append(new_particle)
            
        return new_particles

    def resample(self, timestamp):
        """Reamostragem adaptativa"""
        weights = [p.weight for p in self.particles]
        if not weights:
            return
            
        sum_weights_sq = sum(w**2 for w in weights)
        N_eff = 1.0 / sum_weights_sq if sum_weights_sq > 0 else self.N
        
        threshold = self.N / 4
        
        if N_eff < threshold:
            self.particles = self.low_variance_sampler(self.particles, weights)
            
            confidence = self.calculate_confidence()
            time_since_aruco = float('inf')
            if self.last_aruco_time is not None:
                time_since_aruco = (timestamp - self.last_aruco_time).to_sec()

            # Adaptação dinâmica do número de partículas
            if confidence > 0.3:
                new_N = max(int(self.N * 0.85), MIN_PARTICLES)
            elif confidence < 0.2 or time_since_aruco > 5.0:
                new_N = min(int(self.N * 1.2), MAX_PARTICLES)
            else:
                new_N = self.N
                
            if new_N != self.N:
                if new_N > self.N:
                    # Adicionar partículas
                    num_add = new_N - self.N
                    for _ in range(num_add):
                        new_idx = np.random.randint(len(VALID_POSITIONS))
                        row, col = VALID_POSITIONS[new_idx]
                        new_theta = np.random.uniform(-np.pi, np.pi)
                        self.particles.append(Particle(row, col, new_theta, weight=0.1))
                else:
                    # Remover partículas
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
                if total_error < tolerance:
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

    # 6. UPDATE recovery_strategy() to use smart sampling:
    def recovery_strategy(self):
        """Reinicializa as partículas de menor peso"""
        if not self.particles:
            return
        
        weights = np.array([p.weight for p in self.particles])
        worst_indices = np.argpartition(weights, len(weights) // 5)[:len(weights) // 5]
        
        # Sampling vetorizado de novas posições using smart sampling
        num_reinit = len(worst_indices)
        new_positions = self.smart_particle_sampling(num_reinit)
        new_thetas = np.random.uniform(-np.pi, np.pi, num_reinit)
        
        # Reinicializar partículas
        for i, idx in enumerate(worst_indices):
            row, col = new_positions[i]
            self.particles[idx] = Particle(row, col, new_thetas[i], weight=0.1)
        
        self.normalize_weights()
    
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
        
        for p in self.particles:
            old_row, old_col = p.row, p.col
            old_row_continuous = p.row_continuous
            old_col_continuous = p.col_continuous
            p.move_2d(linear_vel_x, linear_vel_y, angular_vel_z, dt)
            
            # Verificar colisões
            if not self.is_free_cell(p.row, p.col):
                p.weight *= 0.01
                if self.is_free_cell(old_row, old_col):
                    p.row, p.col = old_row, old_col
                    p.row_continuous = old_row_continuous
                    p.col_continuous = old_col_continuous

        self.normalize_weights()

    def get_pose_estimate(self):
        """Estimativa de pose baseada nas melhores partículas"""
        if not self.particles:
            return 0, 0, 0
        
        self.normalize_weights()
        
        # Usar top 80% das partículas
        sorted_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)
        top_percent = max(1, len(sorted_particles) * 8 // 10)
        top_particles = sorted_particles[:top_percent]
        
        total_weight = sum(p.weight for p in top_particles)
        
        if total_weight < 1e-10:
            # Média simples
            avg_row = sum(getattr(p, 'row_continuous', p.row) for p in top_particles) / len(top_particles)
            avg_col = sum(getattr(p, 'col_continuous', p.col) for p in top_particles) / len(top_particles)
            avg_sin = sum(np.sin(p.theta) for p in top_particles) / len(top_particles)
            avg_cos = sum(np.cos(p.theta) for p in top_particles) / len(top_particles)
            avg_theta = np.arctan2(avg_sin, avg_cos)
        else:
            # Média ponderada
            avg_row = sum(getattr(p, 'row_continuous', p.row) * p.weight for p in top_particles) / total_weight
            avg_col = sum(getattr(p, 'col_continuous', p.col) * p.weight for p in top_particles) / total_weight
            
            avg_sin = sum(np.sin(p.theta) * p.weight for p in top_particles) / total_weight
            avg_cos = sum(np.cos(p.theta) * p.weight for p in top_particles) / total_weight
            avg_theta = np.arctan2(avg_sin, avg_cos)
        
        x, y = self.grid_to_world(avg_row, avg_col)
        return x, y, avg_theta

    def process_bag(self):
        """Processa o bag file"""
        print(f"\n📁 Processando: {self.bag_path}")
        
        if not os.path.exists(self.bag_path):
            print(f"❌ Ficheiro não encontrado: {self.bag_path}")
            return False
        
        try:
            with rosbag.Bag(self.bag_path, 'r') as bag:
                print("🚀 Iniciando processamento AMCL...")
                
                prev_time = None
                
                for topic, msg, t in bag.read_messages():
                    
                    # Processar poses dos ArUcos
                    if '/aruco/marker_poses' in topic or 'pose' in topic.lower():
                        try:
                            self.process_aruco_poses(msg, t)
                        except Exception as e:
                            pass
                    
                    # Processar distâncias
                    elif 'distance' in topic.lower() or (topic == '/marker_distances'):
                        try:
                            self.process_marker_distances(msg, t)
                        except Exception as e:
                            pass
                    
                    # Processar ArUcos
                    elif topic == '/aruco/marker_ids':
                        try:
                            self.process_aruco_detections(msg, t)
                        except Exception as e:
                            pass
                    
                    # Processar velocidades
                    elif topic == '/cmd_vel':
                        try:
                            self.process_cmd_vel(msg, t)
                        except Exception as e:
                            pass
                    
                    # Processar imagens
                    elif topic == '/raspicam_node/image/compressed':
                        try:
                            self.frame_count += 1
                            self.processed_frames += 1
                            
                            # Movimento das partículas
                            if prev_time is not None:
                                dt = (t - prev_time).to_sec()
                                if 0 < dt < 1.0:
                                    self.move_particles(dt, t)
                            
                            # Estimativa de pose
                            x, y, theta = self.get_pose_estimate()
                            self.pose_estimates.append((t, x, y, theta))
                            self.particle_counts.append((t, self.N))
                            
                            prev_time = t
                            
                            # Progresso
                            if self.processed_frames % 100 == 0:
                                print(f"\r⏳ Frame {self.processed_frames} | Partículas: {self.N}", end="", flush=True)
                                
                        except Exception as e:
                            pass
                
        except Exception as e:
            print(f"\n❌ ERRO ao processar bag: {e}")
            return False
        
        print(f"\n✅ Processamento concluído!")
        print(f"   - Frames processados: {self.processed_frames}")
        print(f"   - Detecções ArUco: {self.aruco_detections}")
        print(f"   - Detecções de distância: {self.distance_detections}")
        print(f"   - Detecções de pose: {self.pose_detections}")
        print(f"   - Estimativas de pose: {len(self.pose_estimates)}")
        
        return True

# ---- Resultados ----

def create_separate_performance_visualizations(results_26, results_27, evaluator):
    """Cria visualizações separadas da performance do AMCL"""
    
    # Combinar dados dos dois testes
    combined_pose_estimates = results_26['pose_estimates'] + results_27['pose_estimates']
    combined_confidence = results_26['confidence_history'] + results_27['confidence_history']
    combined_errors = {
        'position_errors': results_26['errors']['position_errors'] + results_27['errors']['position_errors'],
        'distances_to_trajectory': results_26['errors']['distances_to_trajectory'] + results_27['errors']['distances_to_trajectory'],
        'checkpoint_errors': results_26['errors']['checkpoint_errors'] + results_27['errors']['checkpoint_errors']
    }
    
    # Calcular estatísticas consolidadas
    if combined_errors['position_errors']:
        combined_mean_error = np.mean(combined_errors['position_errors'])
        combined_max_error = np.max(combined_errors['position_errors'])
        combined_rmse = np.sqrt(np.mean([e**2 for e in combined_errors['position_errors']]))
        combined_std = np.std(combined_errors['position_errors'])
    else:
        combined_mean_error = combined_max_error = combined_rmse = combined_std = 0

    # 1. AMCL PERFORMANCE - TRAJECTORIES
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        # Setup do mapa
        wall_mask = (grid == 1).astype(float)
        ax.imshow(wall_mask, cmap='gray', origin='upper', vmin=0, vmax=1, alpha=0.6)
        
        # Grid
        for i in range(GRID_ROWS + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        for j in range(GRID_COLS + 1):
            ax.axvline(j - 0.5, color='black', linewidth=0.5, alpha=0.3)
        
        # ArUcos
        for marker_id, (row, col) in MAP_MARKERS_DISPLAY.items():
            ax.plot(col, row, 's', markersize=10, color='red')
            ax.text(col, row, f'{marker_id}', color='white', ha='center', va='center', fontweight='bold')
        
        # Ground truth trajectory
        gt_trajectory = evaluator.interpolated_trajectory
        gt_rows, gt_cols = zip(*gt_trajectory)
        ax.plot(gt_cols, gt_rows, 'g-', linewidth=4, alpha=0.8, label='Ground Truth')
        
        # Checkpoints
        for i, (cp_row, cp_col) in enumerate(evaluator.checkpoints):
            ax.plot(cp_col, cp_row, 'go', markersize=12, markeredgecolor='darkgreen', markeredgewidth=2)
            ax.text(cp_col + 0.3, cp_row + 0.3, f'CP{i+1}', fontsize=10, fontweight='bold')
        
        # Test 1 trajectory
        if results_26['pose_estimates']:
            trajectory_26 = []
            for t, x, y, theta in results_26['pose_estimates']:
                grid_col = x / RESOLUTION - 0.5
                grid_row = y / RESOLUTION - 0.5
                trajectory_26.append((grid_col, grid_row))
            
            trajectory_26 = np.array(trajectory_26)
            if len(trajectory_26) > 0:
                x_coords, y_coords = trajectory_26.T
                ax.plot(x_coords, y_coords, 'b-', alpha=0.8, linewidth=3, label='Test 1')
        
        # Test 2 trajectory  
        if results_27['pose_estimates']:
            trajectory_27 = []
            for t, x, y, theta in results_27['pose_estimates']:
                grid_col = x / RESOLUTION - 0.5
                grid_row = y / RESOLUTION - 0.5
                trajectory_27.append((grid_col, grid_row))
            
            trajectory_27 = np.array(trajectory_27)
            if len(trajectory_27) > 0:
                x_coords, y_coords = trajectory_27.T
                ax.plot(x_coords, y_coords, 'r-', alpha=0.8, linewidth=3, label='Test 2')
        
        ax.set_xlim(-0.5, GRID_COLS - 0.5)
        ax.set_ylim(GRID_ROWS - 0.5, -0.5)
        ax.set_xticks(range(GRID_COLS))
        ax.set_xticklabels(range(1, GRID_COLS + 1))
        ax.set_yticks(range(GRID_ROWS))
        ax.set_yticklabels(range(1, GRID_ROWS + 1))
        ax.set_aspect('equal')
        ax.set_title('AMCL Performance', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.set_xlabel('Grid Columns', fontsize=12)
        ax.set_ylabel('Grid Rows', fontsize=12)
        
        plt.tight_layout()
        plt.savefig("amcl_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ amcl_performance.png saved")
        
    except Exception as e:
        print(f"❌ Erro ao criar trajectories plot: {e}")

    # 2. AMCL LOCALIZATION PERFORMANCE EVALUATION
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        
        # 2.1 ERRO DE POSIÇÃO AO LONGO DO TEMPO
        ax = axes[0, 0]
        
        if combined_errors['distances_to_trajectory']:
            # Separar por teste para cores diferentes
            test_26_data = [(t.to_sec() - results_26['pose_estimates'][0][0].to_sec(), e) 
                           for t, e in results_26['errors']['distances_to_trajectory']]
            test_27_data = [(t.to_sec() - results_27['pose_estimates'][0][0].to_sec(), e) 
                           for t, e in results_27['errors']['distances_to_trajectory']]
            
            if test_26_data:
                times_26, errors_26 = zip(*test_26_data)
                ax.plot(times_26, errors_26, 'b-', linewidth=1.5, alpha=0.7, label='Test 1')
            
            if test_27_data:
                times_27, errors_27 = zip(*test_27_data)
                ax.plot(times_27, errors_27, 'r-', linewidth=1.5, alpha=0.7, label='Test 2')
            
            # Linha de erro médio
            ax.axhline(y=combined_mean_error, color='orange', linestyle='--', linewidth=2, 
                      label=f'Mean Error: {combined_mean_error:.3f}m')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Position Error (m)', fontsize=12)
        ax.set_title('Position Error Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 2.2 ESTATÍSTICAS DE ERRO CONSOLIDADAS
        ax = axes[0, 1]
        
        metrics = ['Mean', 'Max', 'RMSE', 'Std Dev']
        values = [combined_mean_error, combined_max_error, combined_rmse, combined_std]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{value:.3f}m', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Error (m)', fontsize=12)
        ax.set_title('Combined Error Statistics', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2.3 DISTRIBUIÇÃO DE ERROS (HISTOGRAMA)
        ax = axes[1, 0]
        
        if combined_errors['position_errors']:
            ax.hist(combined_errors['position_errors'], bins=30, alpha=0.7, color='steelblue', 
                   edgecolor='black', linewidth=0.5)
            ax.axvline(combined_mean_error, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {combined_mean_error:.3f}m')
            ax.axvline(combined_mean_error + combined_std, color='orange', linestyle='--', linewidth=2, 
                      label=f'Mean + σ: {combined_mean_error + combined_std:.3f}m')
        
        ax.set_xlabel('Position Error (m)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 2.4 CONFIANÇA AO LONGO DO TEMPO
        ax = axes[1, 1]
        
        if combined_confidence:
            # Separar dados de confiança por teste
            conf_26_data = [(t.to_sec() - results_26['confidence_history'][0][0].to_sec(), conf) 
                           for t, conf in results_26['confidence_history']]
            conf_27_data = [(t.to_sec() - results_27['confidence_history'][0][0].to_sec(), conf) 
                           for t, conf in results_27['confidence_history']]
            
            if conf_26_data:
                times_26, conf_26 = zip(*conf_26_data)
                ax.plot(times_26, conf_26, 'b-', linewidth=1.5, alpha=0.7, label='Test 1')
            
            if conf_27_data:
                times_27, conf_27 = zip(*conf_27_data)
                ax.plot(times_27, conf_27, 'r-', linewidth=1.5, alpha=0.7, label='Test 2')
            
            # Linhas de threshold
            ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='High Confidence (0.5)')
            ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Low Confidence (0.2)')
            ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Critical (0.1)')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title('AMCL Confidence Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.suptitle('AMCL Localization Performance Evaluation', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig("amcl_localization_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ amcl_localization_performance.png saved")
        
    except Exception as e:
        print(f"❌ Erro ao criar localization performance plot: {e}")

    # 3. ERROR BY CHECKPOINTS
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        if combined_errors['checkpoint_errors']:
            # Agrupar erros por checkpoint
            checkpoint_stats = {}
            for cp_num, error in combined_errors['checkpoint_errors']:
                if cp_num not in checkpoint_stats:
                    checkpoint_stats[cp_num] = []
                checkpoint_stats[cp_num].append(error)
            
            # Calcular média e desvio padrão para cada checkpoint
            checkpoints = sorted(checkpoint_stats.keys())
            means = [np.mean(checkpoint_stats[cp]) for cp in checkpoints]
            stds = [np.std(checkpoint_stats[cp]) if len(checkpoint_stats[cp]) > 1 else 0 for cp in checkpoints]
            
            # Calculate the maximum height including error bars
            max_heights = [mean + std for mean, std in zip(means, stds)]
            max_height = max(max_heights) if max_heights else max(means)
            
            # Set y-axis limit with some padding
            y_limit = max_height * 1.15  # 15% padding above the highest point
            ax.set_ylim(0, y_limit)
            
            bars = ax.bar([f'CP{cp}' for cp in checkpoints], means, yerr=stds, 
                        capsize=5, color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)
            
            # Adicionar valores - positioned smartly to stay within the plot
            for bar, mean, std in zip(bars, means, stds):
                bar_height = bar.get_height()
                error_bar_top = bar_height + std
                
                # Position text inside the plot area
                if error_bar_top < y_limit * 0.9:  # If there's space above error bar
                    text_y = error_bar_top + y_limit * 0.02  # Small offset above error bar
                    va = 'bottom'
                else:  # If error bar is too high, put text inside the bar
                    text_y = bar_height * 0.5  # Middle of the bar
                    va = 'center'
                
                # Choose text color based on position
                text_color = 'black' if va == 'bottom' else 'white'
                
                ax.text(bar.get_x() + bar.get_width()/2., text_y,
                    f'{mean:.3f}m', ha='center', va=va, 
                    fontsize=11, fontweight='bold', color=text_color)
        
        ax.set_ylabel('Position Error (m)', fontsize=12)
        ax.set_xlabel('Checkpoint', fontsize=12)
        ax.set_title('Error by Checkpoint', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig("error_by_checkpoint.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ error_by_checkpoint.png saved")
        
    except Exception as e:
        print(f"❌ Erro ao criar checkpoint errors plot: {e}")

def main():
    parser = argparse.ArgumentParser(description='AMCL Performance Evaluation - Combined Analysis')
    parser.add_argument('test_26', help='Test 26 bag file')
    parser.add_argument('test_27', help='Test 27 bag file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_26):
        print(f"❌ Ficheiro não encontrado: {args.test_26}")
        return 1
    
    if not os.path.exists(args.test_27):
        print(f"❌ Ficheiro não encontrado: {args.test_27}")
        return 1
    
    try:
        # Criar avaliador de trajetória
        evaluator = TrajectoryEvaluator(GROUND_TRUTH_CHECKPOINTS)
        print(f"🎯 Ground Truth: {len(evaluator.checkpoints)} checkpoints")
        print(f"📈 Trajetória interpolada: {len(evaluator.interpolated_trajectory)} pontos")
        
        # Processar Test 26
        print(f"\n" + "="*60)
        print(f"🔍 PROCESSANDO TEST 26")
        print(f"="*60)
        
        processor_26 = MCLProcessor(args.test_26, evaluator)
        success_26 = processor_26.process_bag()
        
        if success_26:
            errors_26 = evaluator.calculate_trajectory_errors(processor_26.pose_estimates)
            results_26 = {
                'pose_estimates': processor_26.pose_estimates,
                'confidence_history': processor_26.confidence_history,
                'errors': errors_26,
                'bag_name': 'test_26'
            }
        else:
            print("❌ Falha no processamento do Test 26")
            return 1
        
        # Processar Test 27
        print(f"\n" + "="*60)
        print(f"🔍 PROCESSANDO TEST 27")
        print(f"="*60)
        
        processor_27 = MCLProcessor(args.test_27, evaluator)
        success_27 = processor_27.process_bag()
        
        if success_27:
            errors_27 = evaluator.calculate_trajectory_errors(processor_27.pose_estimates)
            results_27 = {
                'pose_estimates': processor_27.pose_estimates,
                'confidence_history': processor_27.confidence_history,
                'errors': errors_27,
                'bag_name': 'test_27'
            }
        else:
            print("❌ Falha no processamento do Test 27")
            return 1
        
        # Análise consolidada
        print(f"\n" + "="*60)
        print(f"📊 ANÁLISE CONSOLIDADA DA PERFORMANCE DO AMCL")
        print(f"="*60)
        
        # Combinar estatísticas
        combined_position_errors = errors_26['position_errors'] + errors_27['position_errors']
        combined_checkpoint_errors = errors_26['checkpoint_errors'] + errors_27['checkpoint_errors']
        
        if combined_position_errors:
            combined_mean = np.mean(combined_position_errors)
            combined_max = np.max(combined_position_errors)
            combined_rmse = np.sqrt(np.mean([e**2 for e in combined_position_errors]))
            combined_std = np.std(combined_position_errors)
            combined_median = np.median(combined_position_errors)
            
            # Percentis
            p25 = np.percentile(combined_position_errors, 25)
            p75 = np.percentile(combined_position_errors, 75)
            p95 = np.percentile(combined_position_errors, 95)
        
        print(f"\n🎯 ESTATÍSTICAS CONSOLIDADAS:")
        print(f"   • Total de estimativas de pose: {len(processor_26.pose_estimates) + len(processor_27.pose_estimates)}")
        print(f"   • Total de detecções ArUco: {processor_26.aruco_detections + processor_27.aruco_detections}")
        print(f"   • Total de medições de distância: {processor_26.distance_detections + processor_27.distance_detections}")
        
        print(f"\n📏 PRECISÃO DE LOCALIZAÇÃO:")
        print(f"   • Erro médio: {combined_mean:.3f}m")
        print(f"   • Erro mediano: {combined_median:.3f}m")
        print(f"   • Erro máximo: {combined_max:.3f}m")
        print(f"   • RMSE: {combined_rmse:.3f}m")
        print(f"   • Desvio padrão: {combined_std:.3f}m")
        
        print(f"\n📊 DISTRIBUIÇÃO DE ERROS:")
        print(f"   • 25% dos erros ≤ {p25:.3f}m")
        print(f"   • 75% dos erros ≤ {p75:.3f}m") 
        print(f"   • 95% dos erros ≤ {p95:.3f}m")
        
        # Análise de confiança
        all_confidences = [conf for _, conf in results_26['confidence_history']] + \
                         [conf for _, conf in results_27['confidence_history']]
        
        if all_confidences:
            mean_confidence = np.mean(all_confidences)
            high_conf_ratio = len([c for c in all_confidences if c > 0.5]) / len(all_confidences)
            low_conf_ratio = len([c for c in all_confidences if c < 0.2]) / len(all_confidences)
            
            print(f"\n🎲 CONFIANÇA DO ALGORITMO:")
            print(f"   • Confiança média: {mean_confidence:.3f}")
            print(f"   • % tempo com alta confiança (>0.5): {high_conf_ratio*100:.1f}%")
            print(f"   • % tempo com baixa confiança (<0.2): {low_conf_ratio*100:.1f}%")
        
        # Análise por checkpoint
        if combined_checkpoint_errors:
            print(f"\n🎯 PERFORMANCE NOS CHECKPOINTS:")
            
            # Agrupar por checkpoint
            checkpoint_stats = {}
            for cp_num, error in combined_checkpoint_errors:
                if cp_num not in checkpoint_stats:
                    checkpoint_stats[cp_num] = []
                checkpoint_stats[cp_num].append(error)
            
            for cp_num in sorted(checkpoint_stats.keys()):
                errors = checkpoint_stats[cp_num]
                mean_err = np.mean(errors)
                std_err = np.std(errors) if len(errors) > 1 else 0
                print(f"   • Checkpoint {cp_num}: {mean_err:.3f}m ± {std_err:.3f}m (n={len(errors)})")
        
        # Criar visualizações separadas
        print(f"\n🎨 Criando visualizações separadas...")
        create_separate_performance_visualizations(results_26, results_27, evaluator)
        
        # Avaliação da qualidade
        print(f"\n⭐ AVALIAÇÃO DA QUALIDADE DO AMCL:")
        if combined_mean < 0.1:
            quality = "EXCELENTE"
        elif combined_mean < 0.2:
            quality = "BOA"
        elif combined_mean < 0.5:
            quality = "ACEITÁVEL"
        else:
            quality = "PRECISA MELHORIAS"
        
        print(f"   • Classificação geral: {quality}")
        print(f"   • Precisão: {combined_mean:.3f}m (quanto menor, melhor)")
        print(f"   • Consistência: {combined_std:.3f}m (quanto menor, melhor)")
        print(f"   • Robustez: {p95:.3f}m (95% dos erros abaixo deste valor)")
        
        print(f"\n🎉 Análise de performance concluída!")
        print(f"📁 Ficheiros gerados:")
        print(f"   • amcl_performance.png")
        print(f"   • amcl_localization_performance.png")
        print(f"   • error_by_checkpoint.png")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrompido pelo utilizador")
        return 1
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())