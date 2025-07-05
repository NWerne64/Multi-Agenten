# src/agent.py
import mesa
import numpy as np
from src.config import (
    UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN,
    RESOURCE_COLLECTED_BY_ME
)


class ResourceCollectorAgent(mesa.Agent):
    # creation_id_for_display ist für deine spezifische Anzeige-ID (z.B. fortlaufende Nummer)
    def __init__(self, model, initial_anchor_point=None, creation_id_for_display=0):
        super().__init__(model=model)  # Mesa.Agent kümmert sich um self.unique_id (Integer)
        self.display_id = creation_id_for_display + 1  # Speichere deine Anzeige-ID separat

        self.inventory_slot = None
        self.target_pos = None

        self.known_map = np.full(
            (self.model.grid_width_val, self.model.grid_height_val),
            UNKNOWN, dtype=int
        )
        self.time_since_last_blackboard_visit = self.random.randrange(
            self.model.blackboard_sync_interval_val // 2,
            self.model.blackboard_sync_interval_val + 1
        )
        self.blackboard_visit_priority = 0
        self.just_synced_with_blackboard = False
        self.vision_radius = self.model.agent_vision_radius_val
        self.anchor_reached_threshold = self.model.anchor_reached_threshold_val

        self.initial_anchor_point = initial_anchor_point
        self.has_reached_initial_anchor = (initial_anchor_point is None)

        self.claimed_resource_pos = None
        self.last_successful_frontier_target = None

        if not self.has_reached_initial_anchor:
            self.state = "INITIAL_EXPLORATION"
            self.target_pos = self.initial_anchor_point
        else:
            self.state = "SEEKING_RESOURCE"

    def _manhattan_distance(self, pos1, pos2):
        if pos1 is None or pos2 is None: return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_cell_in_current_vision(self, pos_to_check):
        if pos_to_check is None or self.pos is None: return False
        min_x = self.pos[0] - self.vision_radius
        max_x = self.pos[0] + self.vision_radius
        min_y = self.pos[1] - self.vision_radius
        max_y = self.pos[1] + self.vision_radius
        return min_x <= pos_to_check[0] <= max_x and \
            min_y <= pos_to_check[1] <= max_y

    def update_perception(self):
        if self.pos is None: return
        visible_cells_coords = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=self.vision_radius
        )
        for cell_pos in visible_cells_coords:
            cx, cy = cell_pos
            current_known_agent_status = self.known_map[cx, cy]

            if current_known_agent_status == RESOURCE_COLLECTED_BY_ME:
                continue

            newly_observed_state = EMPTY_EXPLORED
            if cell_pos in self.model.resources_on_grid:
                resource_type = self.model.resources_on_grid[cell_pos]['type']
                newly_observed_state = WOOD_SEEN if resource_type == 'wood' else STONE_SEEN
            elif cell_pos in self.model.base_coords_list:
                newly_observed_state = BASE_KNOWN

            if self.known_map[cx, cy] != RESOURCE_COLLECTED_BY_ME:
                self.known_map[cx, cy] = newly_observed_state

    def _sync_with_blackboard(self):
        # Schritt 1: Agent schreibt seine bekannten Informationen auf das Blackboard des Modells
        for r_idx in range(self.model.grid_width_val):
            for c_idx in range(self.model.grid_height_val):
                agent_knowledge_for_cell = self.known_map[r_idx, c_idx]
                if agent_knowledge_for_cell != UNKNOWN:
                    self.model.update_blackboard_cell((r_idx, c_idx), agent_knowledge_for_cell)

        # NEUE ZEILE: Kommunikationszähler des Modells inkrementieren
        # Dies zählt jede Synchronisation mit dem Blackboard als ein Kommunikationsereignis.
        self.model.communication_counter += 1

        # Schritt 2: Agent aktualisiert seine eigene Karte mit Informationen vom Blackboard des Modells
        for r_idx in range(self.model.grid_width_val):
            for c_idx in range(self.model.grid_height_val):
                pos = (r_idx, c_idx)
                # Überspringe Zellen im aktuellen Sichtfeld, da diese direkt wahrgenommen werden
                if self._is_cell_in_current_vision(pos):
                    continue

                agent_current_cell_knowledge = self.known_map[r_idx, c_idx]
                blackboard_cell_knowledge = self.model.blackboard_map[r_idx, c_idx]

                # Wenn das Blackboard anzeigt, dass eine Zelle leer und erkundet ist,
                # übernimmt der Agent diese Information.
                if blackboard_cell_knowledge == EMPTY_EXPLORED:
                    self.known_map[r_idx, c_idx] = EMPTY_EXPLORED
                # Wenn der Agent eine Zelle nicht kennt, aber das Blackboard gültige Informationen hat,
                # übernimmt der Agent diese Information (außer UNKNOWN oder EMPTY_EXPLORED, die bereits oben behandelt wurden).
                elif agent_current_cell_knowledge == UNKNOWN and \
                        blackboard_cell_knowledge not in [UNKNOWN, EMPTY_EXPLORED]:
                    self.known_map[r_idx, c_idx] = blackboard_cell_knowledge
                # Wenn das Blackboard die Basis kennt und der Agent sie noch nicht kennt,
                # aktualisiert der Agent seine Karte.
                elif blackboard_cell_knowledge == BASE_KNOWN and agent_current_cell_knowledge != BASE_KNOWN:
                    self.known_map[r_idx, c_idx] = BASE_KNOWN

        self.time_since_last_blackboard_visit = 0
        self.blackboard_visit_priority = 0
        self.state = "SEEKING_RESOURCE"  # Setze den Status zurück auf Ressourcensuche
        self.target_pos = None  # Setze das Ziel zurück
        self.just_synced_with_blackboard = True

    def _find_frontier_target(self):
        candidate_exploration_targets = []
        NEIGHBOR_OFFSETS = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]

        SCORE_WEIGHT_UNKNOWN_COUNT = 20
        SCORE_WEIGHT_DISTANCE = 1
        LOCAL_EXPLORATION_BONUS = 500
        # Dynamischer Suchradius basierend auf Sichtweite für lokale Suche
        LOCAL_SEARCH_RADIUS = self.vision_radius + 2
        MAX_PARENT_CELLS_GLOBAL = 100

        processed_local_parent_cells = set()

        if self.last_successful_frontier_target is not None:
            lsft_x, lsft_y = self.last_successful_frontier_target
            min_r = max(0, lsft_x - LOCAL_SEARCH_RADIUS)
            max_r = min(self.model.grid_width_val - 1, lsft_x + LOCAL_SEARCH_RADIUS)
            min_c = max(0, lsft_y - LOCAL_SEARCH_RADIUS)
            max_c = min(self.model.grid_height_val - 1, lsft_y + LOCAL_SEARCH_RADIUS)

            local_parent_candidates_coords = []
            for r_idx in range(min_r, max_r + 1):
                for c_idx in range(min_c, max_c + 1):
                    if self._manhattan_distance((r_idx, c_idx),
                                                self.last_successful_frontier_target) <= LOCAL_SEARCH_RADIUS:
                        if self.known_map[r_idx, c_idx] in [EMPTY_EXPLORED, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME]:
                            local_parent_candidates_coords.append((r_idx, c_idx))

            self.random.shuffle(local_parent_candidates_coords)

            for parent_pos in local_parent_candidates_coords:
                processed_local_parent_cells.add(parent_pos)
                px, py = parent_pos
                num_unknown_neighbors = 0
                actual_unknown_targets = []
                for dx, dy in NEIGHBOR_OFFSETS:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < self.model.grid_width_val and 0 <= ny < self.model.grid_height_val and \
                            self.known_map[nx, ny] == UNKNOWN:
                        num_unknown_neighbors += 1
                        actual_unknown_targets.append((nx, ny))

                if num_unknown_neighbors > 0:
                    dist_to_parent = self._manhattan_distance(self.pos, parent_pos)
                    score = (num_unknown_neighbors * SCORE_WEIGHT_UNKNOWN_COUNT) - \
                            (dist_to_parent * SCORE_WEIGHT_DISTANCE) + LOCAL_EXPLORATION_BONUS
                    for unknown_target in actual_unknown_targets:
                        candidate_exploration_targets.append({'target_pos': unknown_target, 'score': score})

        known_passable_rows, known_passable_cols = np.where(
            (self.known_map == EMPTY_EXPLORED) |
            (self.known_map == BASE_KNOWN) |
            (self.known_map == RESOURCE_COLLECTED_BY_ME)
        )

        if known_passable_rows.size > 0:
            all_global_parent_coords = list(zip(known_passable_rows, known_passable_cols))
            self.random.shuffle(all_global_parent_coords)

            global_parents_evaluated_count = 0
            for parent_pos in all_global_parent_coords:
                if parent_pos in processed_local_parent_cells:
                    continue
                if global_parents_evaluated_count >= MAX_PARENT_CELLS_GLOBAL:
                    break

                px, py = parent_pos
                num_unknown_neighbors = 0
                actual_unknown_targets = []
                for dx, dy in NEIGHBOR_OFFSETS:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < self.model.grid_width_val and 0 <= ny < self.model.grid_height_val and \
                            self.known_map[nx, ny] == UNKNOWN:
                        num_unknown_neighbors += 1
                        actual_unknown_targets.append((nx, ny))

                if num_unknown_neighbors > 0:
                    dist_to_parent = self._manhattan_distance(self.pos, parent_pos)
                    score = (num_unknown_neighbors * SCORE_WEIGHT_UNKNOWN_COUNT) - \
                            (dist_to_parent * SCORE_WEIGHT_DISTANCE)
                    for unknown_target in actual_unknown_targets:
                        candidate_exploration_targets.append({'target_pos': unknown_target, 'score': score})
                global_parents_evaluated_count += 1

        if not candidate_exploration_targets:
            self.last_successful_frontier_target = None
            return None

        candidate_exploration_targets.sort(key=lambda x: x['score'], reverse=True)
        best_score = candidate_exploration_targets[0]['score']
        top_scoring_targets = [cand for cand in candidate_exploration_targets if cand['score'] == best_score]

        chosen_candidate = self.random.choice(top_scoring_targets)
        chosen_target_pos = chosen_candidate['target_pos']

        if self.last_successful_frontier_target is not None:
            dist_new_to_old_focus = self._manhattan_distance(chosen_target_pos, self.last_successful_frontier_target)
            # Heuristik zum Zurücksetzen des Fokus, abhängig von Suchradius und Sichtweite
            RESET_FOCUS_THRESHOLD = LOCAL_SEARCH_RADIUS + self.vision_radius
            if dist_new_to_old_focus > RESET_FOCUS_THRESHOLD:
                self.last_successful_frontier_target = None

        return chosen_target_pos

    def _release_my_claim(self):
        if self.claimed_resource_pos:
            # Ruft die Modellmethode auf, die nur für dezentrale Strategie aktiv ist
            self.model.remove_claim(self.claimed_resource_pos, self.unique_id)
            self.claimed_resource_pos = None

    def _decide_target_and_state(self):
        # KORREKTUR: "self.model.running" statt "self.model.simulation_running"
        if not self.model.running:
            self.state = "IDLE_AWAITING_SIM_END";
            self._release_my_claim();
            self.last_successful_frontier_target = None;
            self.target_pos = None;
            return

        if self.inventory_slot is not None:
            self._release_my_claim();
            self.last_successful_frontier_target = None;
            self.state = "MOVING_TO_BASE";
            self.target_pos = self.model.base_deposit_point;
            return

        needed_resources_map = {};
        goals_still_open = False
        for res_type, target_amount in self.model.resource_goals.items():
            collected_amount = self.model.base_resources_collected.get(res_type, 0)
            needed = target_amount - collected_amount
            if needed > 0: needed_resources_map[res_type] = needed; goals_still_open = True

        if not goals_still_open:
            self._release_my_claim();
            self.last_successful_frontier_target = None;
            self.state = "IDLE_AWAITING_SIM_END";
            self.target_pos = None;
            return

        prioritized_needed_resources = sorted(needed_resources_map.items(), key=lambda item: item[1], reverse=True)

        for res_type_needed, _ in prioritized_needed_resources:
            resource_state_to_look_for = WOOD_SEEN if res_type_needed == 'wood' else STONE_SEEN
            known_patches_indices = np.where(self.known_map == resource_state_to_look_for)
            potential_targets_for_this_type = []
            if known_patches_indices[0].size > 0:
                for i in range(known_patches_indices[0].size):
                    pos = (known_patches_indices[0][i], known_patches_indices[1][i])
                    claimant = self.model.get_claimant(pos)  # Wichtig für dezentrale Strategie
                    if claimant is None or claimant == self.unique_id:
                        potential_targets_for_this_type.append(
                            {'pos': pos, 'dist': self._manhattan_distance(self.pos, pos)})

            if potential_targets_for_this_type:
                potential_targets_for_this_type.sort(key=lambda t: t['dist'])
                for target_candidate in potential_targets_for_this_type:
                    chosen_target_pos = target_candidate['pos']
                    self._release_my_claim()
                    if self.model.add_claim(chosen_target_pos, self.unique_id):  # Wichtig für dezentral
                        self.last_successful_frontier_target = None;
                        self.target_pos = chosen_target_pos;
                        self.claimed_resource_pos = chosen_target_pos;
                        self.state = "MOVING_TO_RESOURCE";
                        return

        self._release_my_claim()
        frontier_target = self._find_frontier_target()
        if frontier_target:
            self.target_pos = frontier_target;
            self.state = "MOVING_TO_FRONTIER"
        else:
            self.last_successful_frontier_target = None
            if self.model.strategy == "decentralized" and not self.just_synced_with_blackboard: self.blackboard_visit_priority = 1
            self.state = "IDLE_AWAITING_INFO";
            self.target_pos = None

    def _move_towards(self, target_pos):
        if self.pos is None or target_pos is None or self.pos == target_pos: return
        possible_steps_raw = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1)
        if not possible_steps_raw: return

        possible_steps = list(possible_steps_raw);
        self.random.shuffle(possible_steps)
        best_step = None;
        min_dist = self._manhattan_distance(self.pos, target_pos)
        for step_option in possible_steps:
            dist = self._manhattan_distance(step_option, target_pos)
            if dist < min_dist: min_dist = dist; best_step = step_option

        if best_step is not None:
            self.model.grid.move_agent(self, best_step)

    def _collect_resource(self):
        if self.pos is None: return
        if self.inventory_slot is None and self.claimed_resource_pos == self.pos and self.pos in self.model.resources_on_grid:
            resource_data = self.model.resources_on_grid.pop(self.pos)
            self.inventory_slot = {'type': resource_data['type']}
            self.known_map[self.pos[0], self.pos[1]] = RESOURCE_COLLECTED_BY_ME
            self._release_my_claim();
            self.last_successful_frontier_target = None;
            self.state = "MOVING_TO_BASE";
            self.target_pos = self.model.base_deposit_point
        else:
            self._release_my_claim()
            if self.pos is not None and self.known_map[self.pos[0], self.pos[1]] not in [RESOURCE_COLLECTED_BY_ME,
                                                                                         EMPTY_EXPLORED, BASE_KNOWN]:
                self.known_map[self.pos[0], self.pos[1]] = EMPTY_EXPLORED
            if self.inventory_slot is not None:
                self.last_successful_frontier_target = None;
                self.state = "MOVING_TO_BASE";
                self.target_pos = self.model.base_deposit_point
            else:
                self.state = "SEEKING_RESOURCE";
                self.target_pos = None

    def _deposit_resource(self):
        if self.pos is None: return
        self._release_my_claim()
        if self.inventory_slot is not None and self.pos in self.model.base_coords_list:
            resource_type = self.inventory_slot['type']
            self.model.base_resources_collected[resource_type] += 1
            self.inventory_slot = None;
            if self.model.strategy == "decentralized": self.blackboard_visit_priority = 1
            self.last_successful_frontier_target = None;
            self.state = "SEEKING_RESOURCE";
            self.target_pos = None
        elif self.inventory_slot is not None:
            self.state = "MOVING_TO_BASE";
            self.target_pos = self.model.base_deposit_point
        else:
            self.state = "SEEKING_RESOURCE";
            self.target_pos = None

    def step(self):
        # KORREKTUR: "self.model.running" statt "self.model.simulation_running"
        if not self.model.running:
            self.state = "IDLE_AWAITING_SIM_END";
            return

        self.update_perception()  # Verwendet self.vision_radius
        self.time_since_last_blackboard_visit += 1

        if self.state != "SYNCING_WITH_BLACKBOARD" and self.just_synced_with_blackboard:
            self.just_synced_with_blackboard = False

        # FIX: Tippfehler "dezentralized" zu "decentralized" korrigiert
        if self.model.strategy == "decentralized" and self.state == "INITIAL_EXPLORATION":
            if self.target_pos and \
                    (self.pos == self.target_pos or self._manhattan_distance(self.pos,
                                                                             self.target_pos) < self.anchor_reached_threshold):  # Nutzt Instanzvariable
                self.has_reached_initial_anchor = True;
                self.initial_anchor_point = None;
                self.last_successful_frontier_target = self.pos;
                self.state = "SEEKING_RESOURCE";
                self.target_pos = None
            elif self.target_pos:
                self._move_towards(self.target_pos);
                return
            else:
                self.has_reached_initial_anchor = True;
                self.state = "SEEKING_RESOURCE";
                self.target_pos = None

        # FIX: Tippfehler "dezentralized" zu "decentralized" korrigiert
        if self.model.strategy == "decentralized" and \
                self.state not in ["INITIAL_EXPLORATION", "SYNCING_WITH_BLACKBOARD", "MOVING_TO_BLACKBOARD"] and \
                self.inventory_slot is None:
            trigger_bb_visit_now = False
            if self.blackboard_visit_priority == 1 and not self.just_synced_with_blackboard:
                trigger_bb_visit_now = True
            elif self.time_since_last_blackboard_visit >= self.model.blackboard_sync_interval_val:  # Nutzt Modellparameter
                trigger_bb_visit_now = True
            if trigger_bb_visit_now and self.model.blackboard_coords_list:
                self.last_successful_frontier_target = None
                if self.pos in self.model.blackboard_coords_list:
                    self.state = "SYNCING_WITH_BLACKBOARD"
                else:
                    self.state = "MOVING_TO_BLACKBOARD";
                    self.target_pos = self.random.choice(
                        self.model.blackboard_coords_list)

        if self.state == "SYNCING_WITH_BLACKBOARD":
            self._sync_with_blackboard();
            # HINWEIS: An dieser Stelle sollte der Zähler zurückgesetzt werden, damit der Agent nach dem nächsten Intervall wiederkommt.
            # z.B. self.time_since_last_blackboard_visit = 0
            self.last_successful_frontier_target = None;
            return

        if self.state == "SEEKING_RESOURCE":
            if self.inventory_slot is None and self.target_pos is None: self._decide_target_and_state()
            if self.target_pos and self.state in ["MOVING_TO_RESOURCE", "MOVING_TO_FRONTIER", "MOVING_TO_BLACKBOARD",
                                                  "MOVING_TO_BASE"]:
                self._move_towards(self.target_pos)

        elif self.state == "MOVING_TO_RESOURCE":
            # FIX: Tippfehler "dezentralized" zu "decentralized" korrigiert
            if self.pos == self.target_pos:
                self._collect_resource()
            elif self.target_pos and self.model.strategy == "decentralized" and \
                    (self.known_map[self.target_pos[0], self.target_pos[1]] == WOOD_SEEN or \
                     self.known_map[self.target_pos[0], self.target_pos[1]] == STONE_SEEN):
                self._move_towards(self.target_pos)
            else:
                self._release_my_claim();
                self.state = "SEEKING_RESOURCE";
                self.target_pos = None

        elif self.state == "MOVING_TO_FRONTIER":
            if self.pos == self.target_pos:
                self.last_successful_frontier_target = self.pos;
                self.state = "SEEKING_RESOURCE";
                self.target_pos = None
            elif self.target_pos:
                self._move_towards(self.target_pos)
            else:
                self.state = "SEEKING_RESOURCE"

        elif self.state == "MOVING_TO_BLACKBOARD":
            if self.pos in self.model.blackboard_coords_list:
                self.state = "SYNCING_WITH_BLACKBOARD"
            elif self.target_pos:
                self._move_towards(self.target_pos)
            else:
                self.state = "SEEKING_RESOURCE";
                self.last_successful_frontier_target = None

        elif self.state == "MOVING_TO_BASE":
            if self.pos in self.model.base_coords_list:
                self._deposit_resource()
            elif self.target_pos:
                self._move_towards(self.target_pos)
            else:
                self.state = "SEEKING_RESOURCE";
                self.last_successful_frontier_target = None

        elif self.state == "IDLE_AWAITING_INFO":
            # FIX: Tippfehler "dezentralized" zu "decentralized" korrigiert
            if self.model.strategy == "decentralized" and \
                    not self.just_synced_with_blackboard and \
                    self.model.steps % (
                    self.model.blackboard_sync_interval_val // 3) == 0:  # model.steps statt current_step
                self.blackboard_visit_priority = 1;
                self.state = "SEEKING_RESOURCE"

        elif self.state == "IDLE_AWAITING_SIM_END":
            pass