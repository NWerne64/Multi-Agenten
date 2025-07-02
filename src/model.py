# src/model.py
import mesa
from mesa.datacollection import DataCollector
import numpy as np
from src.config import (
    GRID_WIDTH, GRID_HEIGHT, NUM_AGENTS,
    NUM_WOOD_PATCHES, NUM_STONE_PATCHES,
    MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE,
    BASE_SIZE,
    BLACKBOARD_SIZE_X, BLACKBOARD_SIZE_Y,
    UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME,
    INITIAL_EXPLORATION_ANCHORS,
    RESOURCE_GOALS,
    VISION_RANGE_RADIUS,
    BLACKBOARD_SYNC_INTERVAL,
    ANCHOR_REACHED_THRESHOLD_DISTANCE,
    SUPERVISOR_INITIAL_EXPLORATION_HOTSPOTS,
    MIN_EXPLORE_TARGET_SEPARATION,
    MIN_UNKNOWN_RATIO_FOR_CONTINUED_EXPLORATION
)
from src.supervisor_agent import SupervisorAgent
# KORRIGIERTER IMPORT: Benenne den Standard-Worker beim Import um
from src.worker_agent import WorkerAgent as DefaultWorkerAgent
from src.agent import ResourceCollectorAgent
from src.supervisor_agent_corridor_only import SupervisorAgent as SupervisorAgentCorridorOnly
from src.worker_agent_touring import WorkerAgent as TouringWorkerAgent


class AoELiteModel(mesa.Model):
    def __init__(self,
                 width=GRID_WIDTH, height=GRID_HEIGHT,
                 num_agents_val=NUM_AGENTS,
                 strategy="decentralized",
                 supervisor_type="mixed_strategy",
                 supervisor_home_pos_ratio=(0.5, 0.5),
                 agent_vision_radius=VISION_RANGE_RADIUS,
                 blackboard_sync_interval=BLACKBOARD_SYNC_INTERVAL,
                 num_wood_patches=NUM_WOOD_PATCHES,
                 num_stone_patches=NUM_STONE_PATCHES,
                 anchor_reached_threshold=ANCHOR_REACHED_THRESHOLD_DISTANCE,
                 supervisor_initial_exploration_hotspots_cfg=SUPERVISOR_INITIAL_EXPLORATION_HOTSPOTS,
                 min_explore_target_separation_cfg=MIN_EXPLORE_TARGET_SEPARATION,
                 min_unknown_ratio_for_continued_exploration_cfg=MIN_UNKNOWN_RATIO_FOR_CONTINUED_EXPLORATION,
                 ):
        super().__init__()

        self.SUPERVISOR_INITIAL_EXPLORATION_HOTSPOTS_val = supervisor_initial_exploration_hotspots_cfg
        self.MIN_EXPLORE_TARGET_SEPARATION_val = min_explore_target_separation_cfg
        self.MIN_UNKNOWN_RATIO_FOR_CONTINUED_EXPLORATION_val = min_unknown_ratio_for_continued_exploration_cfg

        self.grid_width_val = width
        self.grid_height_val = height
        self.num_agents_val = num_agents_val
        self.strategy = strategy
        self.supervisor_type_val = supervisor_type
        self.agent_vision_radius_val = agent_vision_radius
        self.blackboard_sync_interval_val = blackboard_sync_interval
        self.num_wood_patches_val = num_wood_patches
        self.num_stone_patches_val = num_stone_patches
        self.resource_goals = RESOURCE_GOALS.copy()
        self.anchor_reached_threshold_val = anchor_reached_threshold
        self.grid = mesa.space.MultiGrid(self.grid_width_val, self.grid_height_val, torus=False)
        self.base_coords_list = []
        self.base_deposit_point = None
        self._place_base(self.grid_width_val, self.grid_height_val)

        if self.base_coords_list:
            base_anchor_x = self.base_coords_list[0][0]
            base_anchor_y = self.base_coords_list[0][1]
            sup_x = base_anchor_x - 2
            sup_y = base_anchor_y
            self.supervisor_home_pos = (max(0, min(sup_x, self.grid_width_val - 1)),
                                        max(0, min(sup_y, self.grid_height_val - 1)))
        else:
            abs_sh_x = int(self.grid_width_val * supervisor_home_pos_ratio[0])
            abs_sh_y = int(self.grid_height_val * supervisor_home_pos_ratio[1])
            self.supervisor_home_pos = (max(0, min(abs_sh_x, self.grid_width_val - 1)),
                                        max(0, min(abs_sh_y, self.grid_height_val - 1)))
            print(f"WARNUNG: Basis nicht gefunden, Supervisor startet bei Fallback-Position {self.supervisor_home_pos}")

        self.supervisor_agent_instance = None
        self.blackboard_coords_list = []
        self.blackboard_map = np.full((self.grid_width_val, self.grid_height_val), UNKNOWN, dtype=int)
        self.resource_claims = {}
        if self.strategy == "decentralized":
            self._place_blackboard_object(self.grid_width_val, self.grid_height_val)

        self.occupied_for_initial_resource_placement = set(self.base_coords_list) | set(self.blackboard_coords_list)
        if self.strategy == "supervisor":
            self.occupied_for_initial_resource_placement.add(self.supervisor_home_pos)

        self.resources_on_grid = {}
        self._place_all_resources(self.grid_width_val, self.grid_height_val)
        self.base_resources_collected = {'wood': 0, 'stone': 0}

        # Agenten erstellen
        if self.strategy == "supervisor":
            if self.supervisor_type_val == "corridor_only":
                print("INFO: Lade SupervisorAgentCorridorOnly und TouringWorkerAgent")
                supervisor_class_to_use = SupervisorAgentCorridorOnly
                worker_class_to_use = TouringWorkerAgent
            else:
                print("INFO: Lade standardmäßigen SupervisorAgent und DefaultWorkerAgent")
                supervisor_class_to_use = SupervisorAgent
                worker_class_to_use = DefaultWorkerAgent

            self.supervisor_agent_instance = supervisor_class_to_use(
                model=self,
                home_pos=self.supervisor_home_pos,
                role_id_for_display="supervisor_0"
            )
            self.grid.place_agent(self.supervisor_agent_instance, self.supervisor_home_pos)
            print(f"Supervisor-Agent ({self.supervisor_type_val}) erstellt bei {self.supervisor_home_pos}")

            for i in range(self.num_agents_val):
                spawn_pos = self._get_valid_spawn_pos(self.supervisor_home_pos)
                worker = worker_class_to_use(
                    model=self,
                    supervisor_home_pos=self.supervisor_home_pos,
                    display_id_num=i
                )
                self.grid.place_agent(worker, spawn_pos)
                print(f"Worker-Agent {i} (Typ: {worker_class_to_use.__name__}) erstellt bei {spawn_pos}")
        else:
            self.exploration_anchor_points = []
            if INITIAL_EXPLORATION_ANCHORS:
                for rel_x, rel_y in INITIAL_EXPLORATION_ANCHORS:
                    abs_x = int(self.grid_width_val * rel_x)
                    abs_y = int(self.grid_height_val * rel_y)
                    abs_x = max(0, min(abs_x, self.grid_width_val - 1))
                    abs_y = max(0, min(abs_y, self.grid_height_val - 1))
                    self.exploration_anchor_points.append((abs_x, abs_y))

            for i in range(self.num_agents_val):
                initial_anchor = None
                spawn_pos = None
                if self.base_coords_list:
                    spawn_pos = self._get_valid_spawn_pos(self.random.choice(self.base_coords_list))
                else:
                    print(f"Warnung: Dezentraler Agent {i} startet zufällig, da keine Basis gefunden wurde.")
                    spawn_pos = self._get_valid_spawn_pos()

                if i < len(self.exploration_anchor_points):
                    initial_anchor = self.exploration_anchor_points[i]

                agent = ResourceCollectorAgent(model=self, initial_anchor_point=initial_anchor,
                                               creation_id_for_display=i)
                if spawn_pos:
                    self.grid.place_agent(agent, spawn_pos)
                    print(f"Dezentraler Agent {i} erstellt bei {spawn_pos} mit Anker {initial_anchor}")
                else:
                    print(f"FEHLER: Konnte keinen Spawnpunkt für dezentralen Agent {i} finden.")

        self.completion_step = -1
        self.datacollector = DataCollector(
            model_reporters={
                "CompletionSteps": "completion_step",
                "CollectedWood": lambda m: m.base_resources_collected.get('wood', 0),
                "CollectedStone": lambda m: m.base_resources_collected.get('stone', 0),
            }
        )
        print(f"Modell initialisiert. Strategie: {self.strategy}, Agenten: {self.num_agents_val}")


    def _get_valid_spawn_pos(self, preferred_pos=None):
        if preferred_pos:
            if self.grid.is_cell_empty(preferred_pos):
                return preferred_pos
            # Suche in größerem Radius, falls direkt belegt
            for radius_check in range(1, 5):
                neighbors = self.grid.get_neighborhood(preferred_pos, moore=True, include_center=False,
                                                       radius=radius_check)
                shuffled_neighbors = list(neighbors)
                self.random.shuffle(shuffled_neighbors)
                for n_pos in shuffled_neighbors:
                    if self.grid.is_cell_empty(n_pos):
                        return n_pos
        # Fallback auf zufällige Position
        for _attempt in range(self.grid_width_val * self.grid_height_val):  # Mehr Versuche
            x = self.random.randrange(self.grid_width_val)
            y = self.random.randrange(self.grid_height_val)
            if self.grid.is_cell_empty((x, y)):
                return (x, y)
        print(
            f"EXTREM-WARNUNG: Konnte absolut keine freie Spawn-Position finden (bevorzugt: {preferred_pos}). Nutze (0,0).")
        # Notfall-Platzierung, könnte zu Problemen führen, wenn (0,0) auch nicht ideal ist.
        if self.grid.is_cell_empty((0, 0)): return (0, 0)
        return (self.random.randrange(self.grid_width_val),
                self.random.randrange(self.grid_height_val))  # Letzter verzweifelter Versuch

    def submit_report_to_supervisor(self, worker_id, report_type, data):
        # ... (Logik bleibt gleich)
        if self.strategy == "supervisor" and self.supervisor_agent_instance:
            self.supervisor_agent_instance.receive_report_from_worker(worker_id, report_type, data)

    def request_task_from_supervisor(self, worker_id):
        # ... (Logik bleibt gleich)
        if self.strategy == "supervisor" and self.supervisor_agent_instance:
            return self.supervisor_agent_instance.request_task_from_worker(worker_id)
        return None

    def _is_cell_occupied_for_initial_placement(self, pos):
        # ... (Logik bleibt gleich)
        return pos in self.occupied_for_initial_resource_placement

    def _place_base(self, grid_width, grid_height):
        # ... (Logik bleibt gleich)
        center_x = grid_width // 2;
        center_y = grid_height // 2;
        self.base_coords_list = []
        start_bx = center_x - BASE_SIZE // 2;
        start_by = center_y - BASE_SIZE // 2
        for j_offset in range(BASE_SIZE):
            for i_offset in range(BASE_SIZE):
                bx, by = start_bx + i_offset, start_by + j_offset
                if 0 <= bx < grid_width and 0 <= by < grid_height:  # Sicherstellen, dass Basis innerhalb des Grids liegt
                    self.base_coords_list.append((bx, by))
        if self.base_coords_list:
            self.base_deposit_point = self.base_coords_list[0]
        else:
            print("WARNUNG: Basis konnte nicht platziert werden (möglicherweise zu klein für BASE_SIZE).")

    def _place_blackboard_object(self, grid_width, grid_height):
        # ... (Logik bleibt gleich)
        if not self.base_coords_list: return
        max_base_x = max(c[0] for c in self.base_coords_list)
        min_base_y_for_bb_alignment = min(
            c[1] for c in self.base_coords_list if c[0] == min(b[0] for b in self.base_coords_list))
        start_x = max_base_x + 2;
        start_y = min_base_y_for_bb_alignment;
        self.blackboard_coords_list = []
        for j in range(BLACKBOARD_SIZE_Y):
            for i in range(BLACKBOARD_SIZE_X):
                bb_x = start_x + i;
                bb_y = start_y + j
                if 0 <= bb_x < grid_width and 0 <= bb_y < grid_height:
                    self.blackboard_coords_list.append((bb_x, bb_y))
        if not self.blackboard_coords_list and self.strategy == "decentralized":
            print("Warnung: Physisches Blackboard konnte nicht platziert werden (dezentrale Strategie).")

    def _place_all_resources(self, grid_width, grid_height):
        # ... (Logik bleibt gleich)
        self._place_resource_type_clusters('wood', self.num_wood_patches_val, grid_width, grid_height)
        self._place_resource_type_clusters('stone', self.num_stone_patches_val, grid_width, grid_height)

    def _place_resource_type_clusters(self, resource_type, total_patches_to_place, grid_width, grid_height):
        # ... (Logik bleibt gleich)
        patches_placed_this_type = 0;
        max_placement_attempts = 500;
        current_attempts = 0
        while patches_placed_this_type < total_patches_to_place and current_attempts < max_placement_attempts:
            current_attempts += 1
            remaining_to_place_for_type = total_patches_to_place - patches_placed_this_type
            if remaining_to_place_for_type == 0: break
            current_cluster_size = self.random.randint(MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE)
            current_cluster_size = min(current_cluster_size, remaining_to_place_for_type)
            start_pos = None;
            for _ in range(grid_width * grid_height // 10):  # Max Versuche für Startposition
                potential_x, potential_y = self.random.randrange(grid_width), self.random.randrange(grid_height)
                if (potential_x, potential_y) not in self.occupied_for_initial_resource_placement:
                    start_pos = (potential_x, potential_y);
                    break
            if start_pos is None: continue  # Nächster Versuch, einen Cluster zu starten

            cluster_generated_count = 0;
            actual_cluster_cells = []
            self.resources_on_grid[start_pos] = {'type': resource_type}
            self.occupied_for_initial_resource_placement.add(start_pos);
            actual_cluster_cells.append(start_pos)
            patches_placed_this_type += 1;
            cluster_generated_count += 1

            idx = 0
            while cluster_generated_count < current_cluster_size and idx < len(actual_cluster_cells):
                if patches_placed_this_type >= total_patches_to_place: break
                parent_cell = actual_cluster_cells[idx];
                idx += 1
                neighbors = list(self.grid.get_neighborhood(parent_cell, moore=True, include_center=False, radius=1))
                self.random.shuffle(neighbors)
                for neighbor_pos in neighbors:
                    if cluster_generated_count >= current_cluster_size: break
                    if patches_placed_this_type >= total_patches_to_place: break
                    if (0 <= neighbor_pos[0] < grid_width and 0 <= neighbor_pos[1] < grid_height and \
                            neighbor_pos not in self.occupied_for_initial_resource_placement):
                        self.resources_on_grid[neighbor_pos] = {'type': resource_type}
                        self.occupied_for_initial_resource_placement.add(neighbor_pos);
                        actual_cluster_cells.append(neighbor_pos)
                        patches_placed_this_type += 1;
                        cluster_generated_count += 1
        if patches_placed_this_type < total_patches_to_place:
            print(
                f"Warnung: Nur {patches_placed_this_type}/{total_patches_to_place} {resource_type}-Patches platziert nach {current_attempts} Versuchen.")

    def update_blackboard_cell(self, pos, agent_reported_state):
        # ... (Logik bleibt gleich)
        if self.strategy == "decentralized":
            x, y = pos
            if not (0 <= x < self.grid_width_val and 0 <= y < self.grid_height_val): return
            current_bb_state = self.blackboard_map[x, y]
            if agent_reported_state == RESOURCE_COLLECTED_BY_ME or agent_reported_state == EMPTY_EXPLORED:
                self.blackboard_map[x, y] = EMPTY_EXPLORED
                if pos in self.resource_claims: del self.resource_claims[pos]
                return
            if current_bb_state == EMPTY_EXPLORED: return
            if agent_reported_state == WOOD_SEEN or agent_reported_state == STONE_SEEN:
                if current_bb_state == UNKNOWN or current_bb_state == BASE_KNOWN:
                    self.blackboard_map[x, y] = agent_reported_state;
                    return
            if agent_reported_state == BASE_KNOWN:
                if current_bb_state == UNKNOWN: self.blackboard_map[x, y] = agent_reported_state; return

    def add_claim(self, pos, agent_id):  # Für dezentrale Strategie
        # ... (Logik bleibt gleich)
        if self.strategy == "decentralized":
            if pos not in self.resources_on_grid:
                if pos in self.resource_claims: del self.resource_claims[pos]
                return False
            current_claimant = self.resource_claims.get(pos)
            if current_claimant is None or current_claimant == agent_id:
                self.resource_claims[pos] = agent_id;
                return True
            return False
        return False  # Nicht relevant für Supervisor-Strategie

    def remove_claim(self, pos, agent_id):  # Für dezentrale Strategie
        # ... (Logik bleibt gleich)
        if self.strategy == "decentralized":
            current_claimant = self.resource_claims.get(pos)
            if current_claimant == agent_id:
                del self.resource_claims[pos];
                return True
            return False
        return False

    def get_claimant(self, pos):  # Für dezentrale Strategie
        # ... (Logik bleibt gleich)
        if self.strategy == "decentralized":
            return self.resource_claims.get(pos)
        return None

    def step(self):
        # Wenn die Simulation nicht mehr läuft, nichts tun.
        if not self.running:
            return

        # 1. Agenten aktivieren (Mesa 3.0 Stil)
        if self.strategy == "supervisor":
            # Den Supervisor manuell zuerst aktivieren, damit er planen kann
            if self.supervisor_agent_instance:
                self.supervisor_agent_instance.step()

            # --- KORRIGIERTE ZEILE ---
            # Wähle alle Agenten aus, die entweder ein DefaultWorkerAgent ODER ein TouringWorkerAgent sind
            worker_agents = self.agents.select(
                lambda agent: isinstance(agent, (DefaultWorkerAgent, TouringWorkerAgent)))

            # Aktiviere alle gefundenen Worker in zufälliger Reihenfolge
            if worker_agents:
                worker_agents.shuffle_do("step")
        else:  # Dezentrale Strategie
            self.agents.shuffle_do("step")

        # 2. Prüfen, ob die Ziele erreicht wurden
        goals_met_count = 0
        for resource_type, target_amount in self.resource_goals.items():
            if self.base_resources_collected.get(resource_type, 0) >= target_amount:
                goals_met_count += 1

        # Die `self.schedule.steps` wurden durch `self.steps` ersetzt, welches von Mesa 3.0 automatisch verwaltet wird.
        if goals_met_count == len(self.resource_goals) and self.completion_step == -1:
            self.completion_step = self.steps
            self.running = False
            print(f"----- ZIELE ERREICHT in {self.completion_step} Schritten! (Strategie: {self.strategy}) -----")
            print(f"Gesammelte Ressourcen: {self.base_resources_collected}")

        # 3. Daten für den aktuellen Schritt sammeln
        self.datacollector.collect(self)