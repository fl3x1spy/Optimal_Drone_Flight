import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from dataclasses import dataclass
from typing import List, Tuple, Dict
from scipy.interpolate import splprep, splev

@dataclass
class Config:
    FLIGHT_DURATION_SEC: float = 10.0
    TARGET_FPS: int = 60
    INTERVAL_MS: int = 16
    PATH_RESOLUTION: int = 2000
    SURFACE_RESOLUTION: int = 12
    TRAIL_LENGTH: int = 20
    FIG_SIZE: Tuple[int, int] = (12, 9)
    X_LIM: Tuple[int, int] = (0, 60)
    Y_LIM: Tuple[int, int] = (0, 50)
    Z_LIM: Tuple[int, int] = (0, 30)
    BG_COLOR: str = '#87CEEB'
    TANK_RADIUS: float = 4
    TANK_HEIGHT: float = 8
    STACK_RADIUS: float = 1.2
    STACK_HEIGHT: float = 25
    TOWER_RADIUS: float = 1
    TOWER_HEIGHT: float = 18
    SMOKE_PARTICLES: int = 8
    
    @property
    def TOTAL_FRAMES(self) -> int:
        return int(self.FLIGHT_DURATION_SEC * self.TARGET_FPS)

TANK_POSITIONS = np.array([
    [15, 15, 0], [35, 12, 0], [45, 25, 0], [25, 35, 0],
    [8, 25, 0], [55, 20, 0]
])

TOWER_POSITIONS = np.array([
    [25, 18, 0], [42, 28, 0], [20, 42, 0]
])

SMOKESTACKS = np.array([
    [30, 8, 0], [48, 15, 0], [38, 35, 0]
])

PIPE_NETWORK = [
    [[10, 10, 3], [18, 12, 3], [25, 10, 3], [32, 11, 3], [38, 13, 3], [42, 15, 3], [48, 17, 3]],
    [[12, 18, 2.8], [16, 20, 2.8], [20, 22, 2.8], [24, 20, 2.8], [22, 18, 2.8], [18, 16, 2.8]],
    [[38, 22, 2.8], [42, 24, 2.8], [46, 26, 2.8], [50, 28, 2.8], [48, 30, 2.8], [44, 32, 2.8]],
    [[22, 8, 3.5], [26, 9, 3.5], [28, 11, 3.5], [30, 13, 3.5]],
    [[44, 10, 3.5], [47, 12, 3.5], [49, 14, 3.5]],
    [[15, 28, 2.5], [18, 30, 2.5], [22, 32, 2.5], [25, 34, 2.5]],
    [[35, 18, 3], [39, 20, 3], [43, 22, 3], [47, 24, 3]],
    [[28, 15, 2.7], [32, 17, 2.7], [36, 19, 2.7]],
    [[40, 8, 3.2], [44, 10, 3.2], [47, 12, 3.2]]
]

DEFECTS: Dict[str, List[float]] = {
    'утечка_магистраль': [28.2, 10.8, 3.1],
    'трещина_дымовуха': [48, 15, 12],
    'дефект_вышка': [42, 28, 8],
    'коррозия_резервуар': [25.3, 35.2, 2.8]
}

class FastPathGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.path_points = self._precompute_path()
        
    def _generate_waypoints(self) -> np.ndarray:
        base = np.array([0.0, 0.0, 0.0])
        waypoints = [base]
        all_objects = list(TANK_POSITIONS) + list(TOWER_POSITIONS) + list(SMOKESTACKS)
        remaining = all_objects.copy()
        current = base[:2]
        
        while remaining:
            distances = [np.linalg.norm(np.array(obj[:2]) - current) for obj in remaining]
            idx = np.argmin(distances)
            next_obj = remaining.pop(idx)
            
            if any(np.allclose(next_obj, t) for t in TANK_POSITIONS):
                height = 12.0
            elif any(np.allclose(next_obj, t) for t in TOWER_POSITIONS):
                height = 22.0
            else:
                height = 28.0
                
            waypoints.append([next_obj[0], next_obj[1], height])
            current = next_obj[:2]
        
        waypoints.append(base)
        return np.array(waypoints)
    
    def _precompute_path(self) -> np.ndarray:
        waypoints = self._generate_waypoints()
        distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)
        t_params = distances / distances[-1] if distances[-1] > 0 else np.linspace(0, 1, len(waypoints))
        
        try:
            tck, u = splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], 
                            u=t_params, s=1.0, k=3)
            frames = np.arange(self.config.TOTAL_FRAMES)
            progress = frames / self.config.TOTAL_FRAMES
            x, y, z = splev(progress, tck)
            return np.column_stack([x, y, z])
        except:
            from scipy.interpolate import interp1d
            f = interp1d(t_params, waypoints, axis=0, kind='cubic')
            progress = np.linspace(0, 1, self.config.TOTAL_FRAMES)
            return f(progress)
    
    def get_position(self, frame: int) -> np.ndarray:
        idx = frame % self.config.TOTAL_FRAMES
        return self.path_points[idx]

class FastSmokeSystem:
    def __init__(self, stacks: np.ndarray, config: Config):
        self.stacks = stacks
        self.config = config
        self.n_particles = config.SMOKE_PARTICLES * len(stacks)
        self.base_positions = np.repeat(stacks, config.SMOKE_PARTICLES, axis=0)
        self.delays = np.tile(np.linspace(0, 20, config.SMOKE_PARTICLES), len(stacks))
        self.speeds = np.random.uniform(1.5, 2.5, self.n_particles)
        self.phases = np.random.uniform(0, 2*np.pi, self.n_particles)
        self.sizes = np.random.uniform(0.6, 1.2, self.n_particles)
        
    def update(self, time: float) -> Tuple[np.ndarray, np.ndarray]:
        heights = ((time * self.speeds + self.delays) % 25)
        drift = heights * 0.15
        x = self.base_positions[:, 0] + np.sin(time * 0.5 + self.phases) * drift
        y = self.base_positions[:, 1] + np.cos(time * 0.3 + self.phases) * drift * 0.5
        z = self.base_positions[:, 2] + 26 + heights
        sizes = self.sizes * (1 + heights * 0.08) * 80
        return np.column_stack([x, y, z]), sizes

class FastRenderer:
    def __init__(self, ax, light_source, config: Config):
        self.ax = ax
        self.ls = light_source
        self.config = config
        self.res = config.SURFACE_RESOLUTION
        
    def draw_tank(self, cx: float, cy: float, cz: float):
        r, h = self.config.TANK_RADIUS, self.config.TANK_HEIGHT
        theta = np.linspace(0, 2*np.pi, self.res)
        z = np.linspace(cz, cz+h, self.res//2)
        theta, z = np.meshgrid(theta, z)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        rgb = self.ls.shade(z, cmap=cm.Greys, blend_mode='soft')
        self.ax.plot_surface(x, y, z, facecolors=rgb, alpha=0.9, shade=False, antialiased=False)
        
        phi = np.linspace(0, np.pi/2, self.res//2)
        theta = np.linspace(0, 2*np.pi, self.res)
        phi, theta = np.meshgrid(phi, theta)
        x = cx + r * np.sin(phi) * np.cos(theta)
        y = cy + r * np.sin(phi) * np.sin(theta)
        z = cz + h + r * np.cos(phi)
        rgb = self.ls.shade(z, cmap=cm.Greys, blend_mode='soft')
        self.ax.plot_surface(x, y, z, facecolors=rgb, alpha=0.9, shade=False)
        
    def draw_smokestack(self, cx: float, cy: float, cz: float):
        r, h = self.config.STACK_RADIUS, self.config.STACK_HEIGHT
        theta = np.linspace(0, 2*np.pi, self.res)
        z = np.linspace(cz, cz+h, self.res)
        theta, z = np.meshgrid(theta, z)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        rgb = self.ls.shade(z, cmap=cm.copper, blend_mode='soft')
        self.ax.plot_surface(x, y, z, facecolors=rgb, alpha=0.9, shade=False)
        
        phi = np.linspace(0, 2*np.pi, self.res)
        x_ring = cx + (r+0.2) * np.cos(phi)
        y_ring = cy + (r+0.2) * np.sin(phi)
        self.ax.plot(x_ring, y_ring, np.full_like(phi, cz + 12), color='#8B4513', lw=1.5)
            
    def draw_tower(self, cx: float, cy: float, cz: float):
        r, h = self.config.TOWER_RADIUS, self.config.TOWER_HEIGHT
        angles = np.linspace(0, 2*np.pi, 5)[:-1]
        
        for a in angles:
            x0, y0 = cx + r * np.cos(a), cy + r * np.sin(a)
            self.ax.plot([x0, x0], [y0, y0], [cz, cz+h], color='#4682B4', lw=1.5)
        
        for z in [cz, cz+h/2, cz+h]:
            theta = np.linspace(0, 2*np.pi, self.res)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            self.ax.plot(x, y, z, color='#87CEEB', lw=1)
            
    def draw_pipes(self, pipes: List):
        for segment in pipes:
            coords = np.array(segment)
            self.ax.plot(coords[:,0], coords[:,1], coords[:,2], color='#B22222', lw=6, alpha=0.9, solid_capstyle='round')
            
    def draw_ground(self):
        x = np.linspace(0, 60, 10)
        y = np.linspace(0, 50, 10)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        rgb = self.ls.shade(z, cmap=cm.YlGn, blend_mode='soft')
        self.ax.plot_surface(x, y, z, facecolors=rgb, alpha=0.8, shade=False)

class FastDrone:
    def __init__(self, ax, config: Config):
        self.ax = ax
        self.config = config
        self.body = None
        self.trail_buf = np.zeros((config.TRAIL_LENGTH, 3))
        self.trail_idx = 0
        self.trail_fill = 0
        
    def create(self):
        self.body = self.ax.scatter([], [], [], c='#FFD700', s=500, edgecolors='#FF8C00', lw=2, zorder=25)
        self.trail, = self.ax.plot([], [], [], 'cyan', alpha=0.3, lw=1.5)
        return self.body, self.trail
    
    def update(self, pos: np.ndarray):
        self.body._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
        self.trail_buf[self.trail_idx] = pos
        self.trail_idx = (self.trail_idx + 1) % self.config.TRAIL_LENGTH
        if self.trail_fill < self.config.TRAIL_LENGTH:
            self.trail_fill += 1
        
        if self.trail_fill > 1:
            indices = [(self.trail_idx - i - 1) % self.config.TRAIL_LENGTH for i in range(self.trail_fill)]
            points = self.trail_buf[indices]
            self.trail.set_data(points[:, 0], points[:, 1])
            self.trail.set_3d_properties(points[:, 2])

def main():
    FLIGHT_TIME = 5.0
    
    config = Config(
        FLIGHT_DURATION_SEC=FLIGHT_TIME,
        TARGET_FPS=60,
        SURFACE_RESOLUTION=12,
        SMOKE_PARTICLES=8,
        TRAIL_LENGTH=20
    )
    
    fig = plt.figure(figsize=config.FIG_SIZE, facecolor=config.BG_COLOR)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
    ax.set_xlim(*config.X_LIM)
    ax.set_ylim(*config.Y_LIM)
    ax.set_zlim(*config.Z_LIM)
    ax.set_facecolor(config.BG_COLOR)
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('white')
        axis.pane.set_alpha(0.1)
    ax.grid(False)
    
    ls = LightSource(azdeg=60, altdeg=40)
    renderer = FastRenderer(ax, ls, config)
    
    path_gen = FastPathGenerator(config)
    smoke = FastSmokeSystem(SMOKESTACKS, config)
    
    renderer.draw_ground()
    for pos in TANK_POSITIONS:
        renderer.draw_tank(pos[0], pos[1], pos[2])
    for pos in SMOKESTACKS:
        renderer.draw_smokestack(pos[0], pos[1], pos[2])
    for pos in TOWER_POSITIONS:
        renderer.draw_tower(pos[0], pos[1], pos[2])
    renderer.draw_pipes(PIPE_NETWORK)
    
    ax.plot(path_gen.path_points[:,0], path_gen.path_points[:,1], path_gen.path_points[:,2], 'purple', lw=3, alpha=0.3)
    
    base = ax.scatter(0, 0, 0, c='lime', s=300, edgecolors='green', lw=2, zorder=20)
    
    drone = FastDrone(ax, config)
    drone_vis, trail_vis = drone.create()
    
    smoke_scatter = ax.scatter([], [], [], c='white', s=30, alpha=0.3, zorder=15)
    
    alerts = ax.scatter([], [], [], c='red', s=400, marker='X', lw=2, zorder=30)
    
    status = ax.text2D(0.02, 0.98, '', fontsize=10, transform=ax.transAxes,
                      bbox=dict(fc='lightblue', alpha=0.8, boxstyle='round'), verticalalignment='top')
    fps_text = ax.text2D(0.98, 0.98, 'FPS: --', fontsize=10, transform=ax.transAxes, 
                        horizontalalignment='right', bbox=dict(fc='white', alpha=0.8, boxstyle='round'),
                        verticalalignment='top', family='monospace')
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markeredgecolor='green', 
               markersize=10, label='База', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700', markeredgecolor='#FF8C00', 
               markersize=10, label='Дрон', markeredgewidth=2),
        Line2D([0], [0], color='purple', lw=2, label='Траектория', alpha=0.7),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
               markersize=8, label='Дым', alpha=0.6),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markeredgecolor='darkred',
               markersize=10, label='Дефект', markeredgewidth=2),
        Line2D([0], [0], color='#B22222', lw=4, label='Трубопровод'),
        Line2D([0], [0], color='#4682B4', lw=3, label='Вышка'),
        Line2D([0], [0], color='#8B4513', lw=3, label='Дымовая труба'),
        Line2D([0], [0], color='gray', lw=3, label='Резервуар')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
    
    ax.set_xlabel('X (м)', fontweight='bold')
    ax.set_ylabel('Y (м)', fontweight='bold')
    ax.set_zlabel('Z (м)', fontweight='bold')
    ax.set_title(f'Инспекция промышленных объектов | Полет: {FLIGHT_TIME}с', fontsize=11, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    import time
    last_time = [time.time()]
    frame_count = [0]
    
    def animate(frame):
        current_time = time.time()
        frame_count[0] += 1
        
        if frame_count[0] % 30 == 0:
            elapsed = current_time - last_time[0]
            fps = 30 / elapsed if elapsed > 0 else 0
            fps_text.set_text(f'FPS: {fps:.1f}')
            last_time[0] = current_time
        
        time_sec = frame / config.TARGET_FPS
        pos = path_gen.get_position(frame)
        drone.update(pos)
        
        smoke_pos, smoke_sizes = smoke.update(time_sec)
        smoke_scatter._offsets3d = (smoke_pos[:,0], smoke_pos[:,1], smoke_pos[:,2])
        smoke_scatter.set_sizes(smoke_sizes)
        
        near = []
        for name, dpos in DEFECTS.items():
            dx = pos[0] - dpos[0]
            dy = pos[1] - dpos[1]
            dz = pos[2] - dpos[2]
            if dx*dx + dy*dy + dz*dz < 9.0:
                near.append(dpos)
        
        if near:
            arr = np.array(near)
            alerts._offsets3d = (arr[:,0], arr[:,1], arr[:,2])
            status.set_text(f'⚠ {len(near)} дефектов')
            status.get_bbox_patch().set_facecolor('#FF6B6B')
        else:
            alerts._offsets3d = ([np.nan], [np.nan], [np.nan])
            progress = (frame / config.TOTAL_FRAMES) * 100
            status.set_text(f'{progress:.0f}%')
            status.get_bbox_patch().set_facecolor('lightblue')
        
        #ax.view_init(elev=25, azim=frame * 0.3)
        
        return [drone_vis, trail_vis, smoke_scatter, alerts, status, fps_text]
    
    anim = FuncAnimation(fig, animate, frames=config.TOTAL_FRAMES,
                        interval=config.INTERVAL_MS, repeat=True, blit=False)
    
    plt.show()
    return anim

if __name__ == "__main__":
    anim = main()