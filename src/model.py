# src/model.py
import mesa
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
    ANCHOR_REACHED_THRESHOLD_DISTANCE
)
from src.supervisor_agent import SupervisorAgent
from src.worker_agent import WorkerAgent
from src.agent import ResourceCollectorAgent


class AoELiteModel(mesa.Model):
    def __init__(self,
                 width=GRID_WIDTH, height=GRID_HEIGHT,
                 num_agents_val=NUM_AGENTS,
                 strategy="decentralized",
                 supervisor_home_pos_ratio=(0.5, 0.5),  # Fallback, wenn Basis nicht platziert
                 agent_vision_radius=VISION_RANGE_RADIUS,
                 blackboard_sync_interval=BLACKBOARD_SYNC_INTERVAL,
                 num_wood_patches=NUM_WOOD_PATCHES,
                 num_stone_patches=NUM_STONE_PATCHES,
                 anchor_reached_threshold=ANCHOR_REACHED_THRESHOLD_DISTANCE
                 ):
        super().__init__()

        self.grid_width_val = width
        self.grid_height_val = height
        self.num_agents_val = num_agents_val
        self.strategy = strategy

        self.agent_vision_radius_val = agent_vision_radius
        self.blackboard_sync_interval_val = blackboard_sync_interval
        self.num_wood_patches_val = num_wood_patches
        self.num_stone_patches_val = num_stone_patches
        self.resource_goals = RESOURCE_GOALS.copy()
        self.anchor_reached_threshold_val = anchor_reached_threshold

        self.UNKNOWN_val = UNKNOWN
        self.EMPTY_EXPLORED_val = EMPTY_EXPLORED
        self.WOOD_SEEN_val = WOOD_SEEN
        self.STONE_SEEN_val = STONE_SEEN
        self.BASE_KNOWN_val = BASE_KNOWN
        self.RESOURCE_COLLECTED_BY_ME_val = RESOURCE_COLLECTED_BY_ME

        self.grid = mesa.space.MultiGrid(self.grid_width_val, self.grid_height_val, torus=False)

        self.base_coords_list = []
        self.base_deposit_point = None
        self._place_base(self.grid_width_val, self.grid_height_val)

        # Supervisor Home Position definieren
        if self.base_coords_list:
            base_anchor_x = self.base_coords_list[0][0]
            base_anchor_y = self.base_coords_list[0][1]
            sup_x = base_anchor_x - 2
            sup_y = base_anchor_y
            self.supervisor_home_pos = (max(0, min(sup_x, self.grid_width_val - 1)),
                                        max(0, min(sup_y, self.grid_height_val - 1)))
        else:  # Fallback, falls Basis nicht platziert wurde
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
            self.supervisor_agent_instance = SupervisorAgent(model=self, home_pos=self.supervisor_home_pos,
                                                             role_id_for_display="supervisor_0")
            self.grid.place_agent(self.supervisor_agent_instance, self.supervisor_home_pos)

            for i in range(self.num_agents_val):
                spawn_pos = self._get_valid_spawn_pos(self.supervisor_home_pos)
                worker = WorkerAgent(model=self, supervisor_home_pos=self.supervisor_home_pos, display_id_num=i)
                self.grid.place_agent(worker, spawn_pos)
        else:
            self.exploration_anchor_points = []
            for rel_x, rel_y in INITIAL_EXPLORATION_ANCHORS:
                abs_x = int(self.grid_width_val * rel_x);
                abs_y = int(self.grid_height_val * rel_y)
                abs_x = max(0, min(abs_x, self.grid_width_val - 1));
                abs_y = max(0, min(abs_y, self.grid_height_val - 1))
                self.exploration_anchor_points.append((abs_x, abs_y))

            for i in range(self.num_agents_val):
                initial_anchor = None;
                spawn_pos = None
                if i < len(self.base_coords_list) and i < len(self.exploration_anchor_points):
                    spawn_pos = self._get_valid_spawn_pos(self.base_coords_list[i])
                    initial_anchor = self.exploration_anchor_points[i]
                else:
                    print(f"Warnung: Dezentraler Agent {i} startet zufällig ohne Anker/festen Basispunkt.")
                    spawn_pos = self._get_valid_spawn_pos()
                    initial_anchor = None

                agent = ResourceCollectorAgent(model=self, initial_anchor_point=initial_anchor,
                                               creation_id_for_display=i)
                if spawn_pos:
                    self.grid.place_agent(agent, spawn_pos)
                else:
                    print(f"FEHLER: Konnte keinen Spawnpunkt für dezentralen Agent {i} finden.")

        self.simulation_running = True
        self.completion_step = -1

    def _get_valid_spawn_pos(self, preferred_pos=None):
        if preferred_pos:
            if self.grid.is_cell_empty(preferred_pos):
                return preferred_pos
            neighbors = self.grid.get_neighborhood(preferred_pos, moore=True, include_center=False, radius=3)
            shuffled_neighbors = list(neighbors)
            self.random.shuffle(shuffled_neighbors)
            for n_pos in shuffled_neighbors:
                if self.grid.is_cell_empty(n_pos):
                    return n_pos
        for _attempt in range(self.grid_width_val * self.grid_height_val):
            x = self.random.randrange(self.grid_width_val)
            y = self.random.randrange(self.grid_height_val)
            if self.grid.is_cell_empty((x, y)):
                return (x, y)
        print(
            f"WARNUNG: Konnte keine freie Spawn-Position finden (bevorzugt: {preferred_pos}). Nutze (0,0) als Notfall-Fallback.")
        if self.grid.is_cell_empty((0, 0)): return (0, 0)
        # Wenn (0,0) auch belegt ist, könnte man hier einen Fehler werfen oder eine andere Strategie fahren
        # raise Exception("Konnte keine freie Spawn-Position finden.")
        return (0, 0)

    def submit_report_to_supervisor(self, worker_id, report_type, data):
        if self.strategy == "supervisor" and self.supervisor_agent_instance:
            self.supervisor_agent_instance.receive_report_from_worker(worker_id, report_type, data)

    def request_task_from_supervisor(self, worker_id):
        if self.strategy == "supervisor" and self.supervisor_agent_instance:
            return self.supervisor_agent_instance.request_task_from_worker(worker_id)
        return None

    def _is_cell_occupied_for_initial_placement(self, pos):
        return pos in self.occupied_for_initial_resource_placement

    def _place_base(self, grid_width, grid_height):
        center_x = grid_width // 2;
        center_y = grid_height // 2;
        self.base_coords_list = []
        start_bx = center_x - BASE_SIZE // 2;
        start_by = center_y - BASE_SIZE // 2
        for j_offset in range(BASE_SIZE):
            for i_offset in range(BASE_SIZE):
                self.base_coords_list.append((start_bx + i_offset, start_by + j_offset))
        if self.base_coords_list: self.base_deposit_point = self.base_coords_list[0]

    def _place_blackboard_object(self, grid_width, grid_height):
        if not self.base_coords_list:
            # print("Warnung: Basis nicht platziert, Blackboard-Position kann nicht bestimmt werden.") # Ist ok, wenn Supervisor-Strategie
            return
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
        if not self.blackboard_coords_list: print(
            "Warnung: Physisches Blackboard konnte nicht platziert werden (dezentrale Strategie).")

    def _place_all_resources(self, grid_width, grid_height):
        self._place_resource_type_clusters('wood', self.num_wood_patches_val, grid_width, grid_height)
        self._place_resource_type_clusters('stone', self.num_stone_patches_val, grid_width, grid_height)

    def _place_resource_type_clusters(self, resource_type, total_patches_to_place, grid_width, grid_height):
        patches_placed_this_type = 0;
        max_placement_attempts = 500
        current_attempts = 0
        while patches_placed_this_type < total_patches_to_place and current_attempts < max_placement_attempts:
            current_attempts += 1
            remaining_to_place_for_type = total_patches_to_place - patches_placed_this_type
            if remaining_to_place_for_type == 0: break
            current_cluster_size = self.random.randint(MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE)
            current_cluster_size = min(current_cluster_size, remaining_to_place_for_type)
            start_pos = None
            for _ in range(grid_width * grid_height // 10):
                potential_x, potential_y = self.random.randrange(grid_width), self.random.randrange(grid_height)
                if (potential_x, potential_y) not in self.occupied_for_initial_resource_placement:
                    start_pos = (potential_x, potential_y);
                    break
            if start_pos is None:
                continue
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
        if self.strategy == "decentralized":
            x, y = pos
            if not (0 <= x < self.grid_width_val and 0 <= y < self.grid_height_val): return
            current_bb_state = self.blackboard_map[x, y]
            if agent_reported_state == RESOURCE_COLLECTED_BY_ME or agent_reported_state == EMPTY_EXPLORED:
                self.blackboard_map[x, y] = EMPTY_EXPLORED
                if pos in self.resource_claims:
                    del self.resource_claims[pos]
                return
            if current_bb_state == EMPTY_EXPLORED: return
            if agent_reported_state == WOOD_SEEN or agent_reported_state == STONE_SEEN:
                if current_bb_state == UNKNOWN or current_bb_state == BASE_KNOWN:
                    self.blackboard_map[x, y] = agent_reported_state;
                    return
            if agent_reported_state == BASE_KNOWN:
                if current_bb_state == UNKNOWN: self.blackboard_map[x, y] = agent_reported_state; return

    def add_claim(self, pos, agent_id):
        if self.strategy == "decentralized":
            if pos not in self.resources_on_grid:
                if pos in self.resource_claims: del self.resource_claims[pos]
                return False
            current_claimant = self.resource_claims.get(pos)
            if current_claimant is None or current_claimant == agent_id:
                self.resource_claims[pos] = agent_id
                return True
            return False
        return False

    def remove_claim(self, pos, agent_id):
        if self.strategy == "decentralized":
            current_claimant = self.resource_claims.get(pos)
            if current_claimant == agent_id:
                del self.resource_claims[pos]
                return True
            return False
        return False

    def get_claimant(self, pos):
        if self.strategy == "decentralized":
            return self.resource_claims.get(pos)
        return None

    def step(self):
        if not self.simulation_running:
            return

        if self.strategy == "supervisor":
            if self.supervisor_agent_instance:
                self.supervisor_agent_instance.step()

            worker_agents = self.agents.select(lambda agent: isinstance(agent, WorkerAgent))
            if worker_agents:
                worker_agents.shuffle_do("step")
        else:
            # Filtert ResourceCollectorAgents für dezentrale Strategie
            # shuffle_do wirkt auf alle Agenten im self.agents Set.
            # Wenn nur ResourceCollectorAgents im Set sind (bei dezentraler Strategie), ist das ok.
            self.agents.shuffle_do("step")

        goals_met_count = 0
        for resource_type, target_amount in self.resource_goals.items():
            if self.base_resources_collected.get(resource_type, 0) >= target_amount:
                goals_met_count += 1

        if goals_met_count == len(self.resource_goals) and self.simulation_running:
            self.simulation_running = False
            self.completion_step = self.steps
            print(f"----- ZIELE ERREICHT in {self.completion_step} Schritten! (Strategie: {self.strategy}) -----")
            print(f"Gesammelte Ressourcen: {self.base_resources_collected}")