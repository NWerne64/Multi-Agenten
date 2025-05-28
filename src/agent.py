# src/agent.py
import mesa
import numpy as np
from src.config import (
    VISION_RANGE_RADIUS, UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN,
    RESOURCE_COLLECTED_BY_ME, BLACKBOARD_SYNC_INTERVAL,
    ANCHOR_REACHED_THRESHOLD_DISTANCE
)


class ResourceCollectorAgent(mesa.Agent):
    def __init__(self, creation_id, model, initial_anchor_point=None):
        super().__init__(model)
        self.inventory_slot = None
        self.target_pos = None
        self.agent_display_id = creation_id + 1

        self.known_map = np.full(
            (self.model.grid.width, self.model.grid.height),
            UNKNOWN, dtype=int
        )

        self.time_since_last_blackboard_visit = self.random.randrange(
            BLACKBOARD_SYNC_INTERVAL // 2, BLACKBOARD_SYNC_INTERVAL + 1
        )
        self.blackboard_visit_priority = 0
        self.just_synced_with_blackboard = False

        self.initial_anchor_point = initial_anchor_point
        self.has_reached_initial_anchor = (initial_anchor_point is None)

        self.claimed_resource_pos = None

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
        min_x = self.pos[0] - VISION_RANGE_RADIUS
        max_x = self.pos[0] + VISION_RANGE_RADIUS
        min_y = self.pos[1] - VISION_RANGE_RADIUS
        max_y = self.pos[1] + VISION_RANGE_RADIUS
        return min_x <= pos_to_check[0] <= max_x and \
            min_y <= pos_to_check[1] <= max_y

    def update_perception(self):
        if self.pos is None: return
        visible_cells_coords = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=VISION_RANGE_RADIUS
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
        # Phase 1: Schreiben aufs Blackboard
        for r_idx in range(self.model.grid.width):
            for c_idx in range(self.model.grid.height):
                agent_knowledge_for_cell = self.known_map[r_idx, c_idx]
                if agent_knowledge_for_cell != UNKNOWN:
                    self.model.update_blackboard_cell((r_idx, c_idx), agent_knowledge_for_cell)

        # Phase 2: Lesen vom Blackboard und Mergen
        for r_idx in range(self.model.grid.width):
            for c_idx in range(self.model.grid.height):
                pos = (r_idx, c_idx)
                if self._is_cell_in_current_vision(pos):
                    continue

                agent_current_cell_knowledge = self.known_map[r_idx, c_idx]
                blackboard_cell_knowledge = self.model.blackboard_map[r_idx, c_idx]

                if blackboard_cell_knowledge == EMPTY_EXPLORED:
                    self.known_map[r_idx, c_idx] = EMPTY_EXPLORED
                elif agent_current_cell_knowledge == UNKNOWN and \
                        blackboard_cell_knowledge not in [UNKNOWN, EMPTY_EXPLORED]:
                    self.known_map[r_idx, c_idx] = blackboard_cell_knowledge
                elif blackboard_cell_knowledge == BASE_KNOWN and agent_current_cell_knowledge != BASE_KNOWN:
                    self.known_map[r_idx, c_idx] = BASE_KNOWN

        self.time_since_last_blackboard_visit = 0
        self.blackboard_visit_priority = 0
        self.state = "SEEKING_RESOURCE"
        self.target_pos = None
        self.just_synced_with_blackboard = True

    def _find_frontier_target(self):
        frontier_cells_with_dist = []
        known_passable_indices = np.where(
            (self.known_map == EMPTY_EXPLORED) |
            (self.known_map == BASE_KNOWN) |
            (self.known_map == RESOURCE_COLLECTED_BY_ME)
        )

        candidate_frontier_parents = []
        if known_passable_indices[0].size > 0:
            for i in range(known_passable_indices[0].size):
                candidate_frontier_parents.append(
                    (known_passable_indices[0][i], known_passable_indices[1][i])
                )

        if not candidate_frontier_parents:
            return None

        self.random.shuffle(candidate_frontier_parents)

        # Äußere Schleife
        for r_idx, c_idx in candidate_frontier_parents[:50]:  # Limitiere geprüfte Eltern für Performance
            # Innere Schleife
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r_idx + dr, c_idx + dc

                if 0 <= nr < self.model.grid.width and 0 <= nc < self.model.grid.height:
                    if self.known_map[nr, nc] == UNKNOWN:
                        frontier_cells_with_dist.append({
                            'pos': (r_idx, c_idx),
                            'dist': self._manhattan_distance(self.pos, (r_idx, c_idx))
                        })
                        break  # Breche innere Schleife ab (für dieses r_idx, c_idx)

            # Dieses 'if' und 'break' gehören zur äußeren Schleife (für r_idx, c_idx)
            if len(frontier_cells_with_dist) > 20:  # Genug Frontier-Ziele für Auswahl gefunden
                break  # Breche äußere Schleife ab

        if not frontier_cells_with_dist:
            return None

        frontier_cells_with_dist.sort(key=lambda f: f['dist'])
        return frontier_cells_with_dist[0]['pos']

    def _release_my_claim(self):
        if self.claimed_resource_pos:
            self.model.remove_claim(self.claimed_resource_pos, self.unique_id)
            self.claimed_resource_pos = None

    def _decide_target_and_state(self):
        if not self.model.simulation_running:
            self.state = "IDLE_AWAITING_SIM_END";
            self._release_my_claim();
            self.target_pos = None;
            return

        if self.inventory_slot is not None:
            self._release_my_claim()
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
                    claimant = self.model.get_claimant(pos)
                    if claimant is None or claimant == self.unique_id:
                        potential_targets_for_this_type.append(
                            {'pos': pos, 'dist': self._manhattan_distance(self.pos, pos)})

            if potential_targets_for_this_type:
                potential_targets_for_this_type.sort(key=lambda t: t['dist'])
                for target_candidate in potential_targets_for_this_type:
                    chosen_target_pos = target_candidate['pos']
                    self._release_my_claim()
                    if self.model.add_claim(chosen_target_pos, self.unique_id):
                        self.target_pos = chosen_target_pos
                        self.claimed_resource_pos = chosen_target_pos
                        self.state = "MOVING_TO_RESOURCE"
                        return

        self._release_my_claim()
        frontier_target = self._find_frontier_target()
        if frontier_target:
            self.target_pos = frontier_target;
            self.state = "MOVING_TO_FRONTIER"
        else:
            if not self.just_synced_with_blackboard: self.blackboard_visit_priority = 1
            self.state = "IDLE_AWAITING_INFO";
            self.target_pos = None

    def _move_towards(self, target_pos):
        if self.pos is None or target_pos is None or self.pos == target_pos: return
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
        if self.pos is None: return
        # Claim sollte hier noch existieren, wenn der Agent das korrekte Ziel erreicht hat
        # current_known_status_on_map = self.known_map[self.pos[0], self.pos[1]] # Kann veraltet sein

        # Prüfe direkt mit der "echten" Welt und ob der Claim noch passt
        if self.inventory_slot is None and \
                self.claimed_resource_pos == self.pos and \
                self.pos in self.model.resources_on_grid:

            resource_data = self.model.resources_on_grid.pop(self.pos)  # Entfernt aus der "echten" Welt
            self.inventory_slot = {'type': resource_data['type']}
            self.known_map[self.pos[0], self.pos[1]] = RESOURCE_COLLECTED_BY_ME  # Update EIGENE known_map
            self._release_my_claim()  # Claim nach erfolgreichem Sammeln freigeben
            self.state = "MOVING_TO_BASE";
            self.target_pos = self.model.base_deposit_point
        else:
            # Sammeln fehlgeschlagen (Ressource weg, falscher Claim, etc.)
            self._release_my_claim()  # Wichtig: Immer Claim freigeben bei Fehlschlag/Zielwechsel
            if self.pos is not None and self.known_map[self.pos[0], self.pos[1]] not in [RESOURCE_COLLECTED_BY_ME,
                                                                                         EMPTY_EXPLORED, BASE_KNOWN]:
                # Wenn es nicht schon als definitiv leer bekannt ist, markiere es als leer gesehen
                self.known_map[self.pos[0], self.pos[1]] = EMPTY_EXPLORED

            if self.inventory_slot is not None:  # Sollte nicht passieren, wenn Logik stimmt
                self.state = "MOVING_TO_BASE";
                self.target_pos = self.model.base_deposit_point
            else:
                self.state = "SEEKING_RESOURCE"; self.target_pos = None

    def _deposit_resource(self):
        if self.pos is None: return
        self._release_my_claim()  # Sicherstellen, dass kein Claim aktiv ist beim Abliefern

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
            self.state = "SEEKING_RESOURCE"; self.target_pos = None

    def step(self):
        if not self.model.simulation_running:
            self.state = "IDLE_AWAITING_SIM_END"
            if self.state == "IDLE_AWAITING_SIM_END": return

        self.update_perception()
        self.time_since_last_blackboard_visit += 1

        if self.state != "SYNCING_WITH_BLACKBOARD" and self.just_synced_with_blackboard:
            self.just_synced_with_blackboard = False

        # 1. INITIAL_EXPLORATION: Hat Vorrang, wenn aktiv.
        if self.state == "INITIAL_EXPLORATION":
            if self.target_pos and \
                    (self.pos == self.target_pos or self._manhattan_distance(self.pos,
                                                                             self.target_pos) < ANCHOR_REACHED_THRESHOLD_DISTANCE):
                self.has_reached_initial_anchor = True
                self.initial_anchor_point = None
                self.state = "SEEKING_RESOURCE"
                self.target_pos = None
                # Fällt durch zur normalen Entscheidungsfindung weiter unten
            elif self.target_pos:
                self._move_towards(self.target_pos)
                return
            else:
                self.has_reached_initial_anchor = True
                self.state = "SEEKING_RESOURCE"
                self.target_pos = None

        # 2. Entscheidung für Blackboard-Besuch (nur wenn "frei" und nicht gerade andere BB-Aktion)
        # Diese Prüfung erfolgt, BEVOR die normale Zustandslogik für SEEKING etc. greift,
        # damit ein dringender Blackboard-Besuch Vorrang hat.
        if self.state not in ["INITIAL_EXPLORATION", "SYNCING_WITH_BLACKBOARD", "MOVING_TO_BLACKBOARD"] and \
                self.inventory_slot is None:

            trigger_bb_visit_now = False
            if self.blackboard_visit_priority == 1 and not self.just_synced_with_blackboard:
                trigger_bb_visit_now = True
            elif self.time_since_last_blackboard_visit >= BLACKBOARD_SYNC_INTERVAL:
                trigger_bb_visit_now = True

            if trigger_bb_visit_now and self.model.blackboard_coords_list:
                if self.pos in self.model.blackboard_coords_list:
                    self.state = "SYNCING_WITH_BLACKBOARD"
                else:
                    self.state = "MOVING_TO_BLACKBOARD"
                    self.target_pos = self.random.choice(self.model.blackboard_coords_list)

        # 3. Haupt-Zustandslogik
        # Die Reihenfolge ist wichtig. SYNCING_WITH_BLACKBOARD sollte früh behandelt werden, da es mit 'return' endet.
        if self.state == "SYNCING_WITH_BLACKBOARD":
            self._sync_with_blackboard()
            return  # WICHTIG: Beende Step nach Sync

        # Die anderen Zustände
        if self.state == "SEEKING_RESOURCE":
            if self.inventory_slot is None and self.target_pos is None:
                self._decide_target_and_state()

            if self.target_pos:  # Nur bewegen, wenn ein Ziel gesetzt wurde
                if self.state == "MOVING_TO_RESOURCE" or self.state == "MOVING_TO_FRONTIER" or \
                        self.state == "MOVING_TO_BLACKBOARD":
                    self._move_towards(self.target_pos)
                elif self.state == "MOVING_TO_BASE":  # Sollte durch _decide abgedeckt sein
                    self._move_towards(self.target_pos)

        elif self.state == "MOVING_TO_RESOURCE":
            if self.pos == self.target_pos:
                self._collect_resource()
            elif self.target_pos and (self.known_map[self.target_pos[0], self.target_pos[1]] == WOOD_SEEN or \
                                      self.known_map[self.target_pos[0], self.target_pos[1]] == STONE_SEEN):
                self._move_towards(self.target_pos)
            else:  # Ziel existiert nicht mehr oder ist ungültig
                self._release_my_claim()  # Wichtig: Alten Claim freigeben
                self.state = "SEEKING_RESOURCE";
                self.target_pos = None

        elif self.state == "MOVING_TO_FRONTIER":
            if self.pos == self.target_pos:
                self.state = "SEEKING_RESOURCE";
                self.target_pos = None
                # Kein _decide_target_and_state hier, das passiert im nächsten SEEKING_RESOURCE Block oben
            elif self.target_pos:
                self._move_towards(self.target_pos)
            else:
                self.state = "SEEKING_RESOURCE"

        elif self.state == "MOVING_TO_BLACKBOARD":
            if self.pos in self.model.blackboard_coords_list:
                self.state = "SYNCING_WITH_BLACKBOARD"
                # Der Sync wird im nächsten Step im SYNCING_WITH_BLACKBOARD Block oben behandelt
            elif self.target_pos:
                self._move_towards(self.target_pos)
            else:
                self.state = "SEEKING_RESOURCE"

        elif self.state == "COLLECTING_RESOURCE":
            self._collect_resource()

        elif self.state == "MOVING_TO_BASE":
            if self.pos in self.model.base_coords_list:
                self._deposit_resource()
            elif self.target_pos:
                self._move_towards(self.model.base_deposit_point)
            else:
                self.state = "SEEKING_RESOURCE"  # Sollte target_pos haben

        elif self.state == "DEPOSITING_RESOURCE":
            self._deposit_resource()

        elif self.state == "IDLE_AWAITING_INFO":
            if not self.just_synced_with_blackboard and \
                    self.model.current_step % (BLACKBOARD_SYNC_INTERVAL // 3) == 0:  # Häufiger prüfen
                self.blackboard_visit_priority = 1
                self.state = "SEEKING_RESOURCE"

        elif self.state == "IDLE_AWAITING_SIM_END":
            pass