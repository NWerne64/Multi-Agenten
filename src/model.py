# src/model.py
import mesa
import numpy as np
from src.config import (  # Importiere Konstanten aus config.py
    GRID_WIDTH, GRID_HEIGHT, NUM_AGENTS,
    NUM_WOOD_PATCHES, NUM_STONE_PATCHES,
    MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE,
    BASE_SIZE,
    BLACKBOARD_SIZE_X, BLACKBOARD_SIZE_Y,
    # Hier müssen alle verwendeten Zellzustände importiert werden:
    UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME,
    INITIAL_EXPLORATION_ANCHORS
)
from src.agent import ResourceCollectorAgent


class AoELiteModel(mesa.Model):
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT, num_agents=NUM_AGENTS):
        super().__init__()
        self.grid = mesa.space.MultiGrid(width, height, torus=True)

        self.base_coords_list = []
        self.base_deposit_point = None
        self._place_base(width, height)

        self.blackboard_coords_list = []
        self._place_blackboard_object(width, height)

        self.occupied_for_initial_resource_placement = set(self.base_coords_list) | set(self.blackboard_coords_list)

        self.resources_on_grid = {}
        self._place_all_resources(width, height)

        self.base_resources_collected = {'wood': 0, 'stone': 0}
        self.blackboard_map = np.full((width, height), UNKNOWN, dtype=int)

        self.exploration_anchor_points = []
        for rel_x, rel_y in INITIAL_EXPLORATION_ANCHORS:
            abs_x = int(self.grid.width * rel_x)
            abs_y = int(self.grid.height * rel_y)
            abs_x = max(0, min(abs_x, self.grid.width - 1))
            abs_y = max(0, min(abs_y, self.grid.height - 1))
            self.exploration_anchor_points.append((abs_x, abs_y))

        for i in range(num_agents):
            initial_anchor = None
            spawn_pos = None
            if i < len(self.base_coords_list) and i < len(self.exploration_anchor_points):
                spawn_pos = self.base_coords_list[i]
                initial_anchor = self.exploration_anchor_points[i]
            else:
                print(f"Warnung: Agent {i} hat keinen spezifischen Startplatz/Ankerpunkt und startet zufällig.")
                while True:
                    spawn_pos_cand = (self.random.randrange(width), self.random.randrange(height))
                    if not self._is_cell_occupied_for_initial_placement(spawn_pos_cand) and \
                            not any(isinstance(agent, ResourceCollectorAgent) for agent in
                                    self.grid.get_cell_list_contents([spawn_pos_cand])):
                        spawn_pos = spawn_pos_cand
                        break
            agent = ResourceCollectorAgent(creation_id=i, model=self, initial_anchor_point=initial_anchor)
            if spawn_pos:
                self.grid.place_agent(agent, spawn_pos)
            else:
                print(f"FEHLER: Konnte keinen Spawnpunkt für Agent {i} finden.")

    def _is_cell_occupied_for_initial_placement(self, pos):
        return pos in self.occupied_for_initial_resource_placement

    def _place_base(self, grid_width, grid_height):
        center_x = grid_width // 2;
        center_y = grid_height // 2
        self.base_coords_list = []
        start_bx = center_x - BASE_SIZE // 2;
        start_by = center_y - BASE_SIZE // 2
        for j_offset in range(BASE_SIZE):
            for i_offset in range(BASE_SIZE):
                self.base_coords_list.append((start_bx + i_offset, start_by + j_offset))
        if self.base_coords_list:
            self.base_deposit_point = self.base_coords_list[0]

    def _place_blackboard_object(self, grid_width, grid_height):
        if not self.base_coords_list: return
        max_base_x = max(c[0] for c in self.base_coords_list)
        min_base_y_for_bb_alignment = min(
            c[1] for c in self.base_coords_list if c[0] == min(b[0] for b in self.base_coords_list))
        start_x = max_base_x + 2;
        start_y = min_base_y_for_bb_alignment
        self.blackboard_coords_list = []
        for j in range(BLACKBOARD_SIZE_Y):
            for i in range(BLACKBOARD_SIZE_X):
                bb_x = start_x + i;
                bb_y = start_y + j
                if 0 <= bb_x < grid_width and 0 <= bb_y < grid_height:
                    self.blackboard_coords_list.append((bb_x, bb_y))
        if not self.blackboard_coords_list: print("Warnung: Physisches Blackboard konnte nicht platziert werden.")

    def _place_all_resources(self, grid_width, grid_height):
        self._place_resource_type_clusters('wood', NUM_WOOD_PATCHES, grid_width, grid_height)
        self._place_resource_type_clusters('stone', NUM_STONE_PATCHES, grid_width, grid_height)

    def _place_resource_type_clusters(self, resource_type, total_patches_to_place, grid_width, grid_height):
        patches_placed_this_type = 0;
        max_placement_attempts = 500
        while patches_placed_this_type < total_patches_to_place and max_placement_attempts > 0:
            max_placement_attempts -= 1;
            remaining_to_place_for_type = total_patches_to_place - patches_placed_this_type
            if remaining_to_place_for_type == 0: break
            current_cluster_size = self.random.randint(MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE)
            current_cluster_size = min(current_cluster_size, remaining_to_place_for_type)
            start_pos = None
            for _ in range(100):
                potential_x, potential_y = self.random.randrange(grid_width), self.random.randrange(grid_height)
                if (potential_x, potential_y) not in self.occupied_for_initial_resource_placement:
                    start_pos = (potential_x, potential_y);
                    break
            if start_pos is None: continue
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
                f"Warnung: Nur {patches_placed_this_type}/{total_patches_to_place} {resource_type}-Patches platziert.")

    def update_blackboard_cell(self, pos, agent_reported_state):
        """Aktualisiert eine Zelle im Blackboard basierend auf der Meldung eines Agenten."""
        x, y = pos
        if not (0 <= x < self.grid.width and 0 <= y < self.grid.height):
            return

        current_bb_state = self.blackboard_map[x, y]

        # Konfliktlösungsregeln:
        if agent_reported_state == RESOURCE_COLLECTED_BY_ME or agent_reported_state == EMPTY_EXPLORED:
            self.blackboard_map[x, y] = agent_reported_state
        elif agent_reported_state == WOOD_SEEN or agent_reported_state == STONE_SEEN:
            if current_bb_state == UNKNOWN or current_bb_state == EMPTY_EXPLORED:
                self.blackboard_map[x, y] = agent_reported_state
        elif agent_reported_state == BASE_KNOWN:
            if current_bb_state == UNKNOWN:
                self.blackboard_map[x, y] = agent_reported_state
        # UNKNOWN vom Agenten überschreibt nichts Sinnvolles auf dem Blackboard.

    def step(self):
        self.agents.shuffle_do("step")