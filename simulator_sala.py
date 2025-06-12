import tkinter as tk
from tkinter import ttk
import numpy as np
import random
from tkinter import simpledialog
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import pandas as pd


# Tarefas:
# Pensar em ajustar o field of view da camara
# 6. Fazer resultados:
# - Percurso: (1,0) -> (1, 35) -> (23, 35) -> (23, 41) -> (1,41) -> (1, penultima coluna) -> (51, penultima coluna) -> (51, 0) -> (1,0)
# - Fazer vídeo do percurso.
# - Ao lado fazer percurso do robô estimado. (mostrar como o nº de partículas diminui quando a incerteza diminui)
# - Erro absoluto médio (MAE): c/ distância euclideana
# - Comparar distância euclideana e manhattan: teste t de student
# - Convergência do filtro de partículas: Em quantos passos o sistema converge para uma posição próxima da verdadeira.

# MAPA E GRID - ROOM
RESOLUTION = 0.3  # metros por célula

# Dimensões do grid em células
GRID_WIDTH = 20 # células
GRID_HEIGHT = 17  # células

# Inicializa grid - tudo começa como área livre = 0
grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

# Adiciona bordas da sala (paredes externas)
grid[0, :] = 1  # Parede superior
grid[-1, :] = 1  # Parede inferior
grid[:, 0] = 1  # Parede esquerda
grid[:, -1] = 1  # Parede direita

# Obstáculo 1: corners (15,14),(18,14),(15,15),(18,15) - já em coordenadas de grid
# x vai de 15 a 17 (18 exclusive), y vai de 14 a 14 (15 exclusive)
grid[1:2, 16:19] = 1

# Obstáculo 2: corners (0,6), (0,18),(1,6),(1,18) - já em coordenadas de grid  
# x vai de 0 a 17 (18 exclusive), y vai de 1 a 5 (6 exclusive)
grid[15, 7:19] = 1

grid[6:9, 6:17] = 1

map_width_cells = GRID_WIDTH
map_height_cells = GRID_HEIGHT

GRID_LAYOUT = grid
GRID_ROWS, GRID_COLS = grid.shape
CELL_SIZE = 20  # pixels

# Define landmarks (ArUco markers positions em coordenadas de grid)
landmarks = [
    (1, 16),  # Landmark 1
    (2,0),
    (11,0),
    (15,7),
    (15,17),
    (8,19),
    (7,6),
    (6,8),
    (6,15),
    (8,8),
    (8,15)
]

class Particle:
    def __init__(self, row, col, weight=1.0):
        self.row = row
        self.col = col
        self.weight = weight

class SimulatorGUI:
    def __init__(self, root):
        self.visibility_history = []
        self.root = root
        self.root.title("AMCL Simulator")

        self.localization_mse_history = []

        self.grid_map = GRID_LAYOUT.copy()
        self.landmarks = landmarks
        self.robot_pos = (3,2)
        self.particles = []
        self.robot_dir = 'N'  # direção inicial: North

        self.num_particles = tk.IntVar(value=200)
        self.sigma_motion = tk.DoubleVar(value=0.2)
        self.sigma_sensor = tk.DoubleVar(value=1.0)
        self.resample_strategy = tk.StringVar(value="Systematic")

        self.min_particles = 200
        self.max_particles = 1000
        self.effective_sample_threshold = 0.5

        self.landmark_visible = False
        self.steps_without_detection = 0

        self.setup_ui()
        self.reset_particles()
        self.bind_keys()


    def set_corridor_map(self):
        global GRID_LAYOUT, GRID_ROWS, GRID_COLS
        # Redefine para o mapa do corredor (o atual)
        self.grid_map = GRID_LAYOUT.copy()
        self.robot_pos = (3, 2)
        self.reset_particles()
        self.draw_grid()
        print("[MAPA] Mapa do corredor carregado.")

    def set_room_map(self):
        global GRID_LAYOUT, GRID_ROWS, GRID_COLS
        # Exemplo de mapa de sala (você pode ajustar depois)
        room_grid = np.ones_like(GRID_LAYOUT)
        # Exemplo: área livre no centro
        room_grid[10:30, 10:30] = 0
        self.grid_map = room_grid
        self.robot_pos = (15, 15)
        self.reset_particles()
        self.draw_grid()
        print("[MAPA] Mapa da sala carregado (exemplo).")
        

    def bind_keys(self):
        self.root.bind('<Up>', lambda event: self.move_robot('N'))
        self.root.bind('<Down>', lambda event: self.move_robot('S'))
        self.root.bind('<Left>', lambda event: self.move_robot('W'))
        self.root.bind('<Right>', lambda event: self.move_robot('E'))

    def setup_ui(self):
        # Define a cor de fundo desejada
        bg_color = "#aab6e4"  # Um tom claro e moderno
        bg_color_2 = "#3f5297"

        # Estilo ttk para o painel de controlo
        style = ttk.Style()
        style.configure("Custom.TFrame", background=bg_color_2)
        style.configure("Custom.TLabel", background=bg_color_2)
        style.configure("Custom.TButton", background=bg_color_2)

        # Muda o fundo da janela principal
        self.root.configure(bg=bg_color)

        control_frame = ttk.Frame(self.root, padding=10, relief="groove")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        map_label = ttk.Label(control_frame, text="Map", font=("Arial", 12, "bold"))
        map_label.pack(pady=(5, 0))

        btn_corridor = ttk.Button(control_frame, text="5th Floor Corridor", command=self.set_corridor_map)
        btn_corridor.pack(fill=tk.X, pady=(0, 2))

        btn_room = ttk.Button(control_frame, text="Laboratory Room", command=self.set_room_map)
        btn_room.pack(fill=tk.X, pady=(0, 10))

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=8)

        ttk.Label(control_frame, text="Particles (Adaptive):", font=("Arial", 10)).pack()
        ttk.Label(control_frame, textvariable=self.num_particles, font=("Arial", 10, "bold")).pack()

        ttk.Label(control_frame, text="Motion Noise σ:").pack(pady=(8, 0))
        tk.Scale(control_frame, from_=0.0, to=3.0, resolution=0.01, variable=self.sigma_motion,
                 orient=tk.HORIZONTAL, length=140).pack(fill=tk.X)

        ttk.Label(control_frame, text="Sensor Noise σ:").pack(pady=(8, 0))
        tk.Scale(control_frame, from_=0.1, to=5.0, resolution=0.1, variable=self.sigma_sensor,
                 orient=tk.HORIZONTAL, length=140).pack(fill=tk.X)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=8)

        ttk.Label(control_frame, text="Move Robot:", font=("Arial", 10)).pack(pady=(10, 0))
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=2)

        # Botões de movimento em grid (setas)
        ttk.Button(btn_frame, text="↑", width=3, command=lambda: self.move_robot('N')).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(btn_frame, text="←", width=3, command=lambda: self.move_robot('W')).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(btn_frame, text="→", width=3, command=lambda: self.move_robot('E')).grid(row=1, column=2, padx=2, pady=2)
        ttk.Button(btn_frame, text="↓", width=3, command=lambda: self.move_robot('S')).grid(row=2, column=1, padx=2, pady=2)

        ttk.Button(control_frame, text="Auto Path", command=self.start_auto_path).pack(fill=tk.X, pady=(8, 0))
        ttk.Button(control_frame, text="Reset", command=self.reset_particles).pack(fill=tk.X, pady=(2, 0))
        ttk.Button(control_frame, text="Random Kidnapping", command=self.kidnap_robot_button).pack(fill=tk.X, pady=(2, 0))
        ttk.Button(control_frame, text="Show Violin Plot", command=self.show_violin_plot).pack(fill=tk.X, pady=(8, 0))

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=8)

        # Legenda das partículas
                # Legenda das partículas
        legend_canvas = tk.Canvas(control_frame, width=180, height=50, bg="white", highlightthickness=0)
        legend_canvas.pack(pady=(0, 10))

        for i in range(100):
            weight_level = i / 100
            r = int(255 * (1 - weight_level))
            g = int(255 * (1 - weight_level))
            b = int(255 * (1 - weight_level) + 139 * weight_level)
            color = f"#{r:02x}{g:02x}{b:02x}"
            legend_canvas.create_line(50 + i, 25, 50 + i, 35, fill=color)

        legend_canvas.create_text(10, 5, anchor="nw", text="Particle's Weight Scale:", fill="black", font=("Arial", 9, "bold"))
        legend_canvas.create_text(8, 22, anchor="nw", text="Low", fill="black", font=("Arial", 8))
        legend_canvas.create_text(155, 22, anchor="nw", text="High", fill="black", font=("Arial", 8))

        # Legenda dos ArUcos (Landmarks)
        aruco_legend = tk.Canvas(control_frame, width=180, height=30, bg="white", highlightthickness=0)
        aruco_legend.pack(pady=(0, 10))
        # Bola igual ao ArUco do mapa
        aruco_legend.create_oval(15, 8, 28, 21, fill="#b8860b", outline="")
        aruco_legend.create_text(35, 14, anchor="w", text="ArUco (Landmark)", fill="black", font=("Arial", 10, "bold"))
        
        # Canvas do mapa
        self.canvas = tk.Canvas(self.root, width=GRID_COLS * CELL_SIZE, height=GRID_ROWS * CELL_SIZE, bg="white")
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)
        self.draw_grid()
        self.canvas.bind("<Button-1>", self.kidnap_robot_click)
        
    def draw_grid(self):
        self.canvas.delete("all")

        # 1. Desenhar o mapa
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="gray")

        # 2. Acumular pesos por célula
        cell_weights = np.zeros((GRID_ROWS, GRID_COLS))
        for p in self.particles:
            if 0 <= p.row < GRID_ROWS and 0 <= p.col < GRID_COLS:
                cell_weights[p.row, p.col] += p.weight

        max_weight = np.max(cell_weights) if np.max(cell_weights) > 0 else 1e-6

        # 3. Pintar cada célula
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE

                if self.grid_map[r, c] == 1:
                    fill_color = "gray"  # obstáculo
                else:
                    norm = cell_weights[r, c] / max_weight
                    # Interpolar entre branco (peso 0) e azul escuro (peso alto)
                    r_col = int(255 * (1 - norm))  # de 255 a 0
                    g_col = int(255 * (1 - norm))  # de 255 a 0
                    b_col = 255  # azul constante
                    fill_color = f"#{r_col:02x}{g_col:02x}{b_col:02x}"

                self.canvas.create_rectangle(x0, y0, x1, y1, outline="gray", fill=fill_color)

        # 4. Desenhar os ArUcos (landmarks)
        # Destacar os landmarks visíveis (detetados)
        visible_set = set(getattr(self, "last_visible_landmarks", []))
        for (r, c) in self.landmarks:
            cx, cy = c * CELL_SIZE, r * CELL_SIZE
            if (r, c) in visible_set:
                color = "red"  # Detetado
            else:
                color = "#b8860b"  # Normal
            self.canvas.create_oval(cx + 2, cy + 2, cx + CELL_SIZE - 2, cy + CELL_SIZE - 2, fill=color)

        # 5. Desenhar o robô por cima de tudo
        rr, rc = self.robot_pos
        cx = rc * CELL_SIZE + CELL_SIZE // 2
        cy = rr * CELL_SIZE + CELL_SIZE // 2
        half = CELL_SIZE // 2 - 2

        # Triângulo orientado
        if self.robot_dir == 'N':
            points = [cx, cy - half, cx - half, cy + half, cx + half, cy + half]
        elif self.robot_dir == 'S':
            points = [cx, cy + half, cx - half, cy - half, cx + half, cy - half]
        elif self.robot_dir == 'E':
            points = [cx + half, cy, cx - half, cy - half, cx - half, cy + half]
        elif self.robot_dir == 'W':
            points = [cx - half, cy, cx + half, cy - half, cx + half, cy + half]

        self.canvas.create_polygon(points, fill="red")
        
    def move_robot(self, direction):
        self.robot_dir = direction
        dir_map = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}
        dr, dc = dir_map[direction]
        rr, rc = self.robot_pos
        nr, nc = rr + dr, rc + dc
        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and self.grid_map[nr, nc] == 0:
            self.robot_pos = (nr, nc)
            self.step()
        else:
            print("[INFO] Movimento bloqueado por obstáculo ou borda.")
    
    def reset_particles(self, *_):
        self.particles.clear()
        for _ in range(self.num_particles.get()):
            while True:
                r = random.randint(0, GRID_ROWS - 1)
                c = random.randint(0, GRID_COLS - 1)
                if self.grid_map[r, c] == 0:
                    self.particles.append(Particle(r, c))
                    break
        self.draw_grid()
    
    def kidnap_robot_click(self, event):
        """Sequestro com clique no mapa (com evento)."""
        self.kidnap_robot(event)

    def kidnap_robot_button(self):
        """Sequestro com botão (aleatório)."""
        self.kidnap_robot(event=None)

    def kidnap_robot(self, event=None):
        # Se evento for mouse click, pegar coordenadas do clique
        if event:
            col = event.x // CELL_SIZE
            row = event.y // CELL_SIZE
        else:
            while True:
                row = random.randint(0, GRID_ROWS - 1)
                col = random.randint(0, GRID_COLS - 1)
                if self.grid_map[row, col] == 0:
                    break

        self.robot_pos = (row, col)
        print(f"[Kidnap] Robot moved to ({row}, {col})")

        # Se o evento for clique, pedir direção
        if event is not None:
            direction = simpledialog.askstring("Nova direção", "Introduza a direção (N, S, E, W):")
            if direction and direction.upper() in {'N', 'S', 'E', 'W'}:
                self.robot_dir = direction.upper()
                print(f"[Kidnap] Direção definida para {self.robot_dir}")
        else:
            self.robot_dir = random.choice(['N', 'S', 'E', 'W'])
            print(f"[Kidnap] Direção aleatória: {self.robot_dir}")
            
        self.draw_grid()
    
    def compute_effective_sample_size(self):
        weights = np.array([p.weight for p in self.particles])
        if weights.sum() == 0:
            return 0
        weights = weights / weights.sum()  # Normalize weights
        return 1.0 / np.sum(np.square(weights))

    def apply_gaussian_motion(self, row, col, dr, dc, sigma):
        """Aplica movimento com ruído gaussiano para a partícula.
        
        Args:
            row, col: Posição atual da partícula
            dr, dc: Direção de movimento determinística
            sigma: Desvio padrão do ruído gaussiano
            
        Returns:
            Tupla (new_row, new_col) com a posição após o movimento com ruído
        """
        # Aplicar ruído gaussiano ao movimento
        noise_r = np.random.normal(0, sigma)
        noise_c = np.random.normal(0, sigma)
        
        # Calcular a nova posição com movimento determinístico + ruído gaussiano
        new_row = int(round(row + dr + noise_r))
        new_col = int(round(col + dc + noise_c))
        
        # Garantir que a posição está dentro dos limites do mapa
        new_row = max(0, min(new_row, GRID_ROWS - 1))
        new_col = max(0, min(new_col, GRID_COLS - 1))
        
        # Verificar se a nova posição é válida (não é obstáculo)
        if self.grid_map[new_row, new_col] == 0:
            return new_row, new_col
        else:
            # Se for obstáculo, manter a posição original
            return row, col
    
    def step(self):
        rr, rc = self.robot_pos
        dir_map = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}
        dr, dc = dir_map[self.robot_dir]

        errors = []

        # Encontrar o primeiro landmark visível na direção atual
        visible_landmarks = []
        for lr, lc in self.landmarks:
            d_row, d_col = lr - rr, lc - rc
            max_range = 20  # Forward detection range
            side_range = 4  # visão periférica

            # North
            if self.robot_dir == 'N' and d_row < 0 and abs(d_row) <= max_range and abs(d_col) <= side_range:
                # Se está mesmo em cima (apenas 1 célula acima), só permite se d_col == 0
                if abs(d_row) == 1 and d_col != 0:
                    continue
                blocked = any(self.grid_map[rr - k, rc + d_col] == 1 for k in range(1, abs(d_row)))
                if not blocked:
                    visible_landmarks.append((lr, lc))

            # South
            elif self.robot_dir == 'S' and d_row > 0 and d_row <= max_range and abs(d_col) <= side_range:
                if d_row == 1 and d_col != 0:
                    continue
                blocked = any(self.grid_map[rr + k, rc + d_col] == 1 for k in range(1, d_row))
                if not blocked:
                    visible_landmarks.append((lr, lc))

            # West
            elif self.robot_dir == 'W' and d_col < 0 and abs(d_col) <= max_range and abs(d_row) <= side_range:
                if abs(d_col) == 1 and d_row != 0:
                    continue
                blocked = any(self.grid_map[rr + d_row, rc - k] == 1 for k in range(1, abs(d_col)))
                if not blocked:
                    visible_landmarks.append((lr, lc))

            # East
            elif self.robot_dir == 'E' and d_col > 0 and d_col <= max_range and abs(d_row) <= side_range:
                if d_col == 1 and d_row != 0:
                    continue
                blocked = any(self.grid_map[rr + d_row, rc + k] == 1 for k in range(1, d_col))
                if not blocked:
                    visible_landmarks.append((lr, lc))

        # Determinar o estado da visibilidade e ajustar incerteza e número de partículas
        landmark_visible = len(visible_landmarks) > 0
        self.landmark_visible = landmark_visible  

        # Ajustar o sigma de movimento com base na visibilidade de landmarks
        alpha = 0.1 # fator de suavização
        base_motion_sigma = self.sigma_motion.get()
        
        if landmark_visible:
            # Landmark visível: REDUZIR incerteza e número de partículas
            print("[VISIBILIDADE] Landmark detectado! Reduzindo incerteza e número de partículas")
            # Reduzir o sigma de movimento (menos incerteza)
            adjusted_motion_sigma = base_motion_sigma * 0.5  # Reduz em 50%
            
            # Reduzir o número de partículas gradualmente (convergência)
            current_particles = self.num_particles.get()
            target_particles = max(self.min_particles, int(current_particles * 0.8))  # Reduz em 20%
            new_particles = int((1 - alpha - 0.2) * current_particles + alpha * target_particles)
            new_particles = max(self.min_particles, min(self.max_particles, new_particles))  # Garantir limites
            self.num_particles.set(new_particles)
            
            # Medição da distância com ruído (Euclideana)
            #true_dist = min(np.hypot(rr - lr, rc - lc) for lr, lc in visible_landmarks)
            #measured_dist = true_dist + np.random.normal(0, self.sigma_sensor.get())

             # Medição da distância com ruído (Manhattan)
            true_dist = min(abs(rr - lr) + abs(rc - lc) for lr, lc in visible_landmarks)
            measured_dist = true_dist + np.random.normal(0, self.sigma_sensor.get())
        else:
            # Sem landmark: AUMENTAR incerteza e número de partículas
            print("[VISIBILIDADE] Nenhum landmark detectado! Aumentando incerteza e número de partículas")
            # Aumentar o sigma de movimento (mais incerteza)
            adjusted_motion_sigma = base_motion_sigma * 1.2  # Aumenta em 30%
            
            # Aumentar o número de partículas (divergência)
            current_particles = self.num_particles.get()
            target_particles = min(self.max_particles, int(current_particles * 1.2))  # Aumenta em 20%
            new_particles = int((1 - alpha) * current_particles + alpha * target_particles)
            new_particles = max(self.min_particles, min(self.max_particles, new_particles))  # Garantir limites
            self.num_particles.set(new_particles)
            
            measured_dist = None  # Sem leitura
        
        # Atualizar posição de cada partícula usando ruído gaussiano com sigma ajustado
        for p in self.particles:
            # Aplicar o modelo de movimento com ruído gaussiano ajustado pela visibilidade
            p.row, p.col = self.apply_gaussian_motion(
                p.row, p.col, dr, dc, adjusted_motion_sigma
            )
            
        # As partículas próximas do ArUco recebem pesos maiores.
        for p in self.particles:
            if measured_dist is None:
                p.weight = 1.0  # Sem leitura: peso neutro
            else:
                # Euclideana
                #pdist = min(np.hypot(p.row - lr, p.col - lc) for lr, lc in visible_landmarks) if visible_landmarks else float('inf')
                #Manhattan
                pdist = min(abs(p.row - lr) + abs(p.col - lc) for lr, lc in visible_landmarks) if visible_landmarks else float('inf')

                p.weight = np.exp(-0.5 * ((measured_dist - pdist) / self.sigma_sensor.get()) ** 2)
        
        # Normalização do peso
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight
                
        # Calcular neff para avaliar a qualidade da distribuição
        neff = self.compute_effective_sample_size()
        print(f"[AMOSTRAGEM] Neff: {neff:.2f}, Partículas: {self.num_particles.get()}")
        
        # Ajuste adaptativo baseado em neff
        if neff < self.effective_sample_threshold * self.num_particles.get():
            # neff baixo - distribuição pobre, injetar partículas
            print("[AMOSTRAGEM] neff baixo, injetando partículas aleatórias")
            n_inject = int(0.2 * self.num_particles.get())  # 20% de injeção
            injected_particles = []
            for _ in range(n_inject):
                while True:
                    r = random.randint(0, GRID_ROWS - 1)
                    c = random.randint(0, GRID_COLS - 1)
                    if self.grid_map[r, c] == 0:
                        injected_particles.append(Particle(r, c))
                        break
        else:
            # neff alto - boa distribuição
            injected_particles = []

        # Resampling - manter ou substituir partículas com base nos pesos
        if total_weight > 0:
            weights = [p.weight for p in self.particles]
            indices = self.systematic_resample(weights, self.num_particles.get() - len(injected_particles))
            self.particles = [Particle(self.particles[i].row, self.particles[i].col) for i in indices]
        
        # Detecção de sequestro (kidnapping) baseada em erro médio
        errors = []
        if measured_dist is not None:
            # Estimar distância média prevista pelas partículas
            expected_dists = []
            for p in self.particles:
                # Euclideana
                #pdist = min(np.hypot(p.row - lr, p.col - lc) for lr, lc in visible_landmarks) if visible_landmarks else float('inf')
                #Manhattan
                pdist = min(abs(p.row - lr) + abs(p.col - lc) for lr, lc in visible_landmarks) if visible_landmarks else float('inf')

                
                expected_dists.append(pdist)

            mean_expected = np.mean(expected_dists)
            #Euclideana
            #error = abs(measured_dist - mean_expected)
            #Manhattan        
            error = abs(p.row - rr) + abs(p.col - rc)

            errors.append(error)
            
            # Se erro for maior que threshold, suspeita de sequestro (kidnapping)
            if error > 3.0:  
                print("[KIDNAPPING] Alto erro sensorial detectado! Injetando partículas aleatórias")
                num_to_inject = int(0.3 * self.num_particles.get())  # injeta 30% adicionais
                for _ in range(num_to_inject):
                    while True:
                        r = random.randint(0, GRID_ROWS - 1)
                        c = random.randint(0, GRID_COLS - 1)
                        if self.grid_map[r, c] == 0:
                            self.particles.append(Particle(r, c, weight=1.0))
                            break
                # Rebalancear o total
                if len(self.particles) > self.max_particles:
                    self.particles = random.sample(self.particles, self.max_particles)
                self.num_particles.set(len(self.particles))        

        # Calcular a média das partículas (posição estimada)
        if self.particles:
            #Posição média das partículas (centroide da nuvem)
            #mean_row = np.mean([p.row for p in self.particles])
            #mean_col = np.mean([p.col for p in self.particles])

            #Média ponderada das partículas (centroide ponderado)
            rows = np.array([p.row for p in self.particles])
            cols = np.array([p.col for p in self.particles])
            weights = np.array([p.weight for p in self.particles])
            mean_row = np.average(rows, weights=weights)
            mean_col = np.average(cols, weights=weights)

            #parte do código comum independente do método de estiamtiva da posição
            rr, rc = self.robot_pos
            mse = (mean_row - rr) ** 2 + (mean_col - rc) ** 2
            self.localization_mse_history.append(mse)
        else:
            self.localization_mse_history.append(0)

        self.visibility_history.append(landmark_visible)
        self.last_visible_landmarks = visible_landmarks  # Adicione isto antes de self.draw_grid()
        # Atualizar visualização
        self.draw_grid()

    def systematic_resample(self, weights, N):
        positions = (np.arange(N) + random.random()) / N
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes
    
    def show_violin_plot(self):
        if not self.particles:
            print("No particles to plot.")
            return

        # Save current state
        original_particles = [Particle(p.row, p.col, p.weight) for p in self.particles]
        original_motion = self.sigma_motion.get()
        original_num_particles = self.num_particles.get()
        original_robot_pos = self.robot_pos

        errors_dict = {}
        all_errors = []

        for sigma, label in [(1.0, "Motion Noise σ = 1.0m"), (3.0, "Motion Noise σ = 3.0m")]:
            self.sigma_motion.set(sigma)
            self.reset_particles()
            self.step()
            rr, rc = self.robot_pos
            errors = [abs(p.row - rr) + abs(p.col - rc) for p in self.particles]
            errors_dict[label] = errors
            all_errors.extend(errors)

        # Restore state
        self.particles = original_particles
        self.sigma_motion.set(original_motion)
        self.num_particles.set(original_num_particles)
        self.robot_pos = original_robot_pos

        # Prepare DataFrame
        df = pd.DataFrame({
            "error": sum(errors_dict.values(), []),
            "group": [k for k, v in errors_dict.items() for _ in v]
        })

        plt.figure(figsize=(7, 7))
        palette = {
            "Motion Noise σ = 1.0m": "#d8b4d3",
            "Motion Noise σ = 3.0m": "#c4e0db"
        }
        ax = sns.violinplot(
            x="group", y="error", hue="group", data=df,
            palette=palette, inner=None, legend=False
        )

        # Add boxplot inside each violin
                # Add boxplot inside each violin
        box_plot = ax.boxplot(
            [errors_dict["Motion Noise σ = 1.0m"], errors_dict["Motion Noise σ = 3.0m"]],
            positions=[0, 1],
            widths=0.15,
            patch_artist=True,
            whis=[0, 100],  # Inclui todos os valores, cobre o violino todo
            showfliers=False
        )
        for patch, color in zip(box_plot['boxes'], ["#d8b4d3", "#c4e0db"]):
            patch.set_facecolor('white')
            patch.set_edgecolor(color)
            patch.set_linewidth(2)
            patch.set_alpha(0.8)
        for element in ['whiskers', 'caps', 'medians']:
            for item in box_plot[element]:
                item.set_color('black')
                item.set_linewidth(1.5)

        # Adiciona o valor da média acima da linha do mean em cada boxplot
        means = [
            np.mean(errors_dict["Motion Noise σ = 1.0m"]),
            np.mean(errors_dict["Motion Noise σ = 3.0m"])
        ]
                # Adiciona o valor da média acima da linha do mean em cada boxplot
        means = [
            np.mean(errors_dict["Motion Noise σ = 1.0m"]),
            np.mean(errors_dict["Motion Noise σ = 3.0m"])
        ]
        y_min = min(all_errors)
        y_max = max(all_errors)
        y_range = y_max - y_min
        offset = max(0.25, y_range * 0.04)  # Offset dinâmico para evitar sobreposição

        for i, mean in enumerate(means):
            ax.text(
                i, mean + offset,
                f"{mean:.2f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black'
            )

        y_min = min(all_errors)
        y_max = max(all_errors)
        y_range = y_max - y_min
        padding = max(y_range * 0.1, 0.2)
        y_min = min(all_errors) - padding
        y_max = max(all_errors) + padding

        ax.set_xticklabels(['σ = 1.0 m', 'σ = 3.0 m'])
        ax.set_ylim(bottom=0, top=y_max + max(1, y_range * 0.15))

        plt.ylabel("Particle Error (grid indices)")
        plt.xlabel("Motion Noise Standard Deviation")
        plt.title("Violin Plot of Particle Errors\n(Motion Noise Comparison)")
        plt.tight_layout()
        plt.show()

    def generate_room_border_path(self):
        """Gera um caminho ao redor da zona livre da sala, evitando obstáculos e paredes."""
        path = []
        min_row = 1
        max_row = GRID_ROWS - 2
        min_col = 1
        max_col = GRID_COLS - 2

        # Percorre toda a borda interna, só adicionando células livres
        # Top border (left to right)
        for c in range(min_col, max_col + 1):
            if self.grid_map[min_row, c] == 0:
                path.append((min_row, c))
        # Right border (top to bottom)
        for r in range(min_row + 1, max_row + 1):
            if self.grid_map[r, max_col] == 0:
                path.append((r, max_col))
        # Bottom border (right to left)
        for c in range(max_col - 1, min_col - 1, -1):
            if self.grid_map[max_row, c] == 0:
                path.append((max_row, c))
        # Left border (bottom to top)
        for r in range(max_row - 1, min_row, -1):
            if self.grid_map[r, min_col] == 0:
                path.append((r, min_col))

        # Remove waypoints consecutivos que não são vizinhos livres (para evitar "saltos" sobre obstáculos)
        filtered_path = []
        last = None
        for pos in path:
            if last is None or (abs(pos[0] - last[0]) + abs(pos[1] - last[1]) == 1):
                filtered_path.append(pos)
                last = pos
        return filtered_path
    
    def follow_path_step(self):
        if not hasattr(self, "auto_path") or self.auto_path_index >= len(self.auto_path):
            print("Percurso completo.")
            return

        target = self.auto_path[self.auto_path_index]
        rr, rc = self.robot_pos
        tr, tc = target

        # Decide direction to move
        if rr < tr:
            self.robot_dir = 'S'
            self.move_robot('S')
        elif rr > tr:
            self.robot_dir = 'N'
            self.move_robot('N')
        elif rc < tc:
            self.robot_dir = 'E'
            self.move_robot('E')
        elif rc > tc:
            self.robot_dir = 'W'
            self.move_robot('W')
        else:
            # Chegou ao waypoint, avança para o próximo
            self.auto_path_index += 1

        # Só agenda o próximo passo se ainda não terminou
        if self.auto_path_index < len(self.auto_path):
            self.root.after(120, self.follow_path_step)

    def start_auto_path(self):
        self.auto_path = self.generate_room_border_path()
        self.auto_path_index = 0
        self.follow_path_step()


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulatorGUI(root)
    root.mainloop()
