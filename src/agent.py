# src/agent.py
import mesa
import numpy as np
from src.config import (
    VISION_RANGE_RADIUS, UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN,
    RESOURCE_COLLECTED_BY_ME, BLACKBOARD_SYNC_INTERVAL,
    ANCHOR_REACHED_THRESHOLD_DISTANCE  # Importiere neue Konstante
)


class ResourceCollectorAgent(mesa.Agent):
    def __init__(self, creation_id, model, initial_anchor_point=None):  # Neuer Parameter
        super().__init__(model)
        self.inventory_slot = None
        # self.state wird unten basierend auf initial_anchor_point gesetzt
        self.target_pos = None
        self.agent_display_id = creation_id + 1

        self.known_map = np.full(
            (self.model.grid.width, self.model.grid.height),
            UNKNOWN, dtype=int
        )

        self.time_since_last_blackboard_visit = 0
        self.blackboard_visit_priority = 0

        # Initialisierung für Ankerpunkt-Exploration
        self.initial_anchor_point = initial_anchor_point
        self.has_reached_initial_anchor = (initial_anchor_point is None)

        if not self.has_reached_initial_anchor:
            self.state = "INITIAL_EXPLORATION"
            self.target_pos = self.initial_anchor_point
        else:
            self.state = "SEEKING_RESOURCE"  # Normaler Start, falls kein Anker

    # --- Die Methoden _manhattan_distance, _is_cell_in_current_vision, update_perception,
    # --- _sync_with_blackboard, _find_frontier_target, _move_towards,
    # --- _collect_resource, _deposit_resource bleiben exakt wie in der letzten Version.
    # --- Ich füge sie hier nicht erneut ein, um die Antwort übersichtlich zu halten.
    # --- Die einzige Änderung betrifft die step()-Methode und die Initialisierung.

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_cell_in_current_vision(self, pos_to_check):
        min_x = self.pos[0] - VISION_RANGE_RADIUS;
        max_x = self.pos[0] + VISION_RANGE_RADIUS
        min_y = self.pos[1] - VISION_RANGE_RADIUS;
        max_y = self.pos[1] + VISION_RANGE_RADIUS
        return min_x <= pos_to_check[0] <= max_x and min_y <= pos_to_check[1] <= max_y

    def update_perception(self):
        visible_cells_coords = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=VISION_RANGE_RADIUS)
        for cell_pos in visible_cells_coords:
            cx, cy = cell_pos;
            newly_observed_state = EMPTY_EXPLORED
            if self.known_map[cx, cy] == RESOURCE_COLLECTED_BY_ME: continue
            if cell_pos in self.model.resources_on_grid:
                resource_type = self.model.resources_on_grid[cell_pos]['type']
                newly_observed_state = WOOD_SEEN if resource_type == 'wood' else STONE_SEEN
            elif cell_pos in self.model.base_coords_list:
                newly_observed_state = BASE_KNOWN
            if self.known_map[cx, cy] != RESOURCE_COLLECTED_BY_ME: self.known_map[cx, cy] = newly_observed_state

    def _sync_with_blackboard(self):
        for r_idx in range(self.model.grid.width):
            for c_idx in range(self.model.grid.height):
                agent_knowledge_for_cell = self.known_map[r_idx, c_idx]
                if agent_knowledge_for_cell != UNKNOWN:
                    self.model.update_blackboard_cell((r_idx, c_idx), agent_knowledge_for_cell)
        for r_idx in range(self.model.grid.width):
            for c_idx in range(self.model.grid.height):
                pos = (r_idx, c_idx);
                blackboard_cell_knowledge = self.model.blackboard_map[r_idx, c_idx]
                if blackboard_cell_knowledge == RESOURCE_COLLECTED_BY_ME or blackboard_cell_knowledge == EMPTY_EXPLORED:
                    self.known_map[r_idx, c_idx] = blackboard_cell_knowledge
                elif not self._is_cell_in_current_vision(pos):
                    if self.known_map[r_idx, c_idx] == UNKNOWN and blackboard_cell_knowledge != UNKNOWN:
                        self.known_map[r_idx, c_idx] = blackboard_cell_knowledge
        self.time_since_last_blackboard_visit = 0;
        self.blackboard_visit_priority = 0
        self.state = "SEEKING_RESOURCE";
        self.target_pos = None

    def _find_frontier_target(self):
        frontier_cells_with_dist = [];
        known_passable_indices = np.where(
            (self.known_map == EMPTY_EXPLORED) | (self.known_map == BASE_KNOWN) |
            (self.known_map == RESOURCE_COLLECTED_BY_ME))
        candidate_frontier_parents = []
        if known_passable_indices[0].size > 0:
            for i in range(known_passable_indices[0].size):
                candidate_frontier_parents.append((known_passable_indices[0][i], known_passable_indices[1][i]))
        self.random.shuffle(candidate_frontier_parents)
        for r_idx, c_idx in candidate_frontier_parents:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r_idx + dr, c_idx + dc
                if 0 <= nr < self.model.grid.width and 0 <= nc < self.model.grid.height:
                    if self.known_map[nr, nc] == UNKNOWN:
                        frontier_cells_with_dist.append(
                            {'pos': (r_idx, c_idx), 'dist': self._manhattan_distance(self.pos, (r_idx, c_idx))})
                        break
            if len(frontier_cells_with_dist) > 100: break
        if not frontier_cells_with_dist: return None
        frontier_cells_with_dist.sort(key=lambda f: f['dist'])
        return frontier_cells_with_dist[0]['pos']

    def _decide_target_and_state(self):
        if self.inventory_slot is not None:
            self.state = "MOVING_TO_BASE";
            self.target_pos = self.model.base_deposit_point;
            return
        potential_resource_targets = []
        resource_indices = np.where((self.known_map == WOOD_SEEN) | (self.known_map == STONE_SEEN))
        if resource_indices[0].size > 0:
            for i in range(resource_indices[0].size):
                pos = (resource_indices[0][i], resource_indices[1][i]);
                dist = self._manhattan_distance(self.pos, pos)
                resource_type_id = self.known_map[pos[0], pos[1]]
                resource_type = 'wood' if resource_type_id == WOOD_SEEN else 'stone'
                potential_resource_targets.append({'pos': pos, 'dist': dist, 'type': resource_type})
            if potential_resource_targets:
                potential_resource_targets.sort(key=lambda t: t['dist'])
                self.target_pos = potential_resource_targets[0]['pos']
                self.state = "MOVING_TO_RESOURCE";
                return
        frontier_target = self._find_frontier_target()
        if frontier_target:
            self.target_pos = frontier_target;
            self.state = "MOVING_TO_FRONTIER"
        else:
            self.blackboard_visit_priority = 1;
            self.state = "SEEKING_RESOURCE";
            self.target_pos = None

    def _move_towards(self, target_pos):
        if self.pos == target_pos: return
        possible_steps_raw = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1)
        if not possible_steps_raw: return
        possible_steps = list(possible_steps_raw);
        self.random.shuffle(possible_steps)
        best_step = None;
        min_dist = float('inf')
        for step_option in possible_steps:
            dist = self._manhattan_distance(step_option, target_pos)
            if dist < min_dist: min_dist = dist; best_step = step_option
        if best_step is not None: self.model.grid.move_agent(self, best_step)

    def _collect_resource(self):
        current_known_status = self.known_map[self.pos[0], self.pos[1]]
        if self.inventory_slot is None and (current_known_status == WOOD_SEEN or current_known_status == STONE_SEEN) and \
                self.pos in self.model.resources_on_grid:
            resource_data = self.model.resources_on_grid.pop(self.pos)
            self.inventory_slot = {'type': resource_data['type']}
            self.known_map[self.pos[0], self.pos[1]] = RESOURCE_COLLECTED_BY_ME
            self.state = "MOVING_TO_BASE";
            self.target_pos = self.model.base_deposit_point
        else:
            if self.known_map[self.pos[0], self.pos[1]] not in [RESOURCE_COLLECTED_BY_ME, BASE_KNOWN]:
                self.known_map[self.pos[0], self.pos[1]] = EMPTY_EXPLORED
            self.state = "SEEKING_RESOURCE";
            self.target_pos = None

    def _deposit_resource(self):
        if self.inventory_slot is not None and self.pos in self.model.base_coords_list:
            resource_type = self.inventory_slot['type']
            self.model.base_resources_collected[resource_type] += 1
            self.inventory_slot = None;
            self.blackboard_visit_priority = 1
            self.state = "SEEKING_RESOURCE";
            self.target_pos = None
        elif self.inventory_slot is not None:
            self.state = "MOVING_TO_BASE";
            self.target_pos = self.model.base_deposit_point
        else:
            self.state = "SEEKING_RESOURCE";
            self.target_pos = None

    def step(self):
        self.update_perception()
        self.time_since_last_blackboard_visit += 1

        # 1. Behandle den initialen Explorationszustand zuerst
        if self.state == "INITIAL_EXPLORATION":
            if self.target_pos is None or self.pos == self.target_pos or \
                    self._manhattan_distance(self.pos, self.target_pos) < ANCHOR_REACHED_THRESHOLD_DISTANCE:
                self.has_reached_initial_anchor = True
                self.initial_anchor_point = None
                self.state = "SEEKING_RESOURCE"
                self.target_pos = None
                # print(f"Agent {self.unique_id} ({self.agent_display_id}): Reached/near initial anchor. Switching to SEEKING_RESOURCE.")
                # Rufe _decide_target_and_state direkt auf, um im selben Step ein neues Ziel zu finden
                self._decide_target_and_state()
            elif self.target_pos:
                self._move_towards(self.target_pos)
                return  # Bewegung zum Anker ist die Aktion für diesen Step
            else:  # Sollte nicht passieren, wenn target_pos korrekt gesetzt wurde
                self.has_reached_initial_anchor = True
                self.state = "SEEKING_RESOURCE"
                self._decide_target_and_state()  # Gehe sicher, dass ein neuer Zustand/Ziel gesetzt wird

        # 2. Entscheidung für Blackboard-Besuch (nur wenn nicht mehr in INITIAL_EXPLORATION und nicht schon auf dem Weg oder beim Syncen)
        if self.state not in ["INITIAL_EXPLORATION", "MOVING_TO_BLACKBOARD", "SYNCING_WITH_BLACKBOARD"]:
            needs_to_visit_blackboard = False
            if self.blackboard_visit_priority == 1:
                needs_to_visit_blackboard = True
            elif self.time_since_last_blackboard_visit >= BLACKBOARD_SYNC_INTERVAL:
                needs_to_visit_blackboard = True
            # Die "Ratlosigkeit" (keine Ressourcen, keine Frontiers) setzt blackboard_visit_priority in _decide_target_and_state

            if needs_to_visit_blackboard and self.model.blackboard_coords_list:
                self.state = "MOVING_TO_BLACKBOARD"
                self.target_pos = self.random.choice(self.model.blackboard_coords_list)

        # 3. Führe Aktionen basierend auf dem aktuellen Zustand aus
        # Die Reihenfolge der if/elif ist wichtig. INITIAL_EXPLORATION wurde oben behandelt.

        if self.state == "SEEKING_RESOURCE":
            # _decide_target_and_state wurde möglicherweise schon oben aufgerufen (nach INITIAL_EXPLORATION)
            # oder muss hier aufgerufen werden, wenn der Zustand von woanders hierher kam.
            # Um Dopplung zu vermeiden und sicherzustellen, dass es immer aktuell ist:
            if self.target_pos is None and self.inventory_slot is None:  # Nur wenn noch kein Ziel/Aktion für diesen Step klar ist
                self._decide_target_and_state()

            # Nach _decide_target_and_state kann der Status gewechselt haben
            if self.state == "MOVING_TO_RESOURCE" or self.state == "MOVING_TO_FRONTIER" or \
                    (
                            self.state == "MOVING_TO_BLACKBOARD" and self.target_pos != self.model.base_deposit_point):  # Nicht zur Basis, falls das Ziel Blackboard ist
                if self.target_pos: self._move_towards(self.target_pos)
            elif self.state == "MOVING_TO_BASE":  # Falls _decide_target_and_state dies setzt (Inventar voll)
                if self.target_pos: self._move_towards(self.target_pos)

        elif self.state == "MOVING_TO_RESOURCE":
            if self.pos == self.target_pos:
                self._collect_resource()
            elif self.target_pos and \
                    (self.known_map[self.target_pos[0], self.target_pos[1]] == WOOD_SEEN or \
                     self.known_map[self.target_pos[0], self.target_pos[1]] == STONE_SEEN):
                self._move_towards(self.target_pos)
            else:
                self.state = "SEEKING_RESOURCE";
                self.target_pos = None

        elif self.state == "MOVING_TO_FRONTIER":
            if self.pos == self.target_pos:
                self.state = "SEEKING_RESOURCE";
                self.target_pos = None
                self._decide_target_and_state()
            elif self.target_pos:
                self._move_towards(self.target_pos)
            else:
                self.state = "SEEKING_RESOURCE"

        elif self.state == "MOVING_TO_BLACKBOARD":
            if self.pos in self.model.blackboard_coords_list:
                self.state = "SYNCING_WITH_BLACKBOARD"
                self._sync_with_blackboard()
            elif self.target_pos:
                self._move_towards(self.target_pos)
            else:
                self.state = "SEEKING_RESOURCE"

        elif self.state == "SYNCING_WITH_BLACKBOARD":
            # Zustand wird in _sync_with_blackboard() auf SEEKING_RESOURCE gesetzt.
            # Falls der Agent hier noch ist, ist etwas schiefgelaufen oder _sync braucht >1 Step (tut es nicht).
            if self.state == "SYNCING_WITH_BLACKBOARD":  # Zur Sicherheit
                self.state = "SEEKING_RESOURCE"
                self._decide_target_and_state()


        elif self.state == "COLLECTING_RESOURCE":
            self._collect_resource()
        elif self.state == "MOVING_TO_BASE":
            if self.pos in self.model.base_coords_list:
                self._deposit_resource()
            elif self.target_pos:
                self._move_towards(self.model.base_deposit_point)
            else:
                self.state = "SEEKING_RESOURCE"
        elif self.state == "DEPOSITING_RESOURCE":
            self._deposit_resource()
        elif self.state == "IDLE_FULLY_EXPLORED":
            pass