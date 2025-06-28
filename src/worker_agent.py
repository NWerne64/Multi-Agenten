# src/worker_agent.py
import mesa
import numpy as np
from src.config import (
    UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME,
)


class WorkerAgent(mesa.Agent):
    NEIGHBOR_OFFSETS = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]

    def __init__(self, model, supervisor_home_pos, display_id_num):
        super().__init__(model=model)
        self.display_id = display_id_num

        self.current_task = None
        self.state = "IDLE"
        self.newly_observed_map_segment = {}
        self.inventory_slot = None
        self.supervisor_home_pos = supervisor_home_pos
        self.vision_radius = self.model.agent_vision_radius_val  #
        self.worker_internal_map = np.full((model.grid_width_val, model.grid_height_val), UNKNOWN, dtype=int)  #

        # Für 'explore_area' Aufgaben
        self.current_path_to_explore_index = 0  #
        self.local_exploration_steps_done = 0  #
        self.local_exploration_max_steps = self.model.random.randint(3, 6)  #
        self.is_current_task_initial_hotspot = False  #

        # Attribute für Kaskaden-Exploration (derzeit nicht aktiv genutzt)
        self.chained_sortie_steps_taken = 0
        self.chained_frontiers_visited_count = 0
        self.current_chained_target_pos = None
        self.is_on_chained_exploration_sortie = False

        # Attribute für Stripe Exploration (jetzt Korridor Exploration)
        self.stripe_steps_taken = 0  # Wird jetzt als Zähler für Korridor-Schritte wiederverwendet
        self.current_stripe_direction = None  # Nicht mehr primär genutzt für Korridore
        self.DEFAULT_STRIPE_LENGTH_val = getattr(self.model, 'DEFAULT_STRIPE_LENGTH_val', 20)  #

        # NEU: Für 'explore_corridor' Aufgaben
        self.current_corridor_path = []
        self.current_corridor_path_index = 0

    def _manhattan_distance(self, pos1, pos2):
        if pos1 is None or pos2 is None: return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self):
        self._update_own_perception()
        self._execute_fsm()

    def _update_own_perception(self):
        # Unverändert
        if self.pos is None: return
        visible_cells_coords = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=self.vision_radius
        )
        for cell_pos in visible_cells_coords:
            cx, cy = cell_pos
            current_cell_state_on_worker_map = self.worker_internal_map[cx, cy]
            actual_grid_content_type = EMPTY_EXPLORED
            if cell_pos in self.model.resources_on_grid:
                resource_type = self.model.resources_on_grid[cell_pos]['type']
                actual_grid_content_type = WOOD_SEEN if resource_type == 'wood' else STONE_SEEN
            elif cell_pos in self.model.base_coords_list:
                actual_grid_content_type = BASE_KNOWN

            if current_cell_state_on_worker_map != actual_grid_content_type or \
                    cell_pos not in self.newly_observed_map_segment or \
                    self.newly_observed_map_segment[cell_pos] != actual_grid_content_type:
                if current_cell_state_on_worker_map == RESOURCE_COLLECTED_BY_ME and \
                        actual_grid_content_type in [WOOD_SEEN, STONE_SEEN]:
                    pass
                self.worker_internal_map[cx, cy] = actual_grid_content_type
                self.newly_observed_map_segment[cell_pos] = actual_grid_content_type

    def _move_towards(self, target_pos):
        # Unverändert
        if self.pos is None or target_pos is None or self.pos == target_pos: return
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1)
        if not possible_steps:
            # print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Cannot move from {self.pos}, no possible steps to reach {target_pos}.")
            return

        best_step = None;
        min_dist = self._manhattan_distance(self.pos, target_pos)
        shuffled_steps = list(possible_steps);
        self.random.shuffle(shuffled_steps)

        if len(shuffled_steps) == 1 and self._manhattan_distance(shuffled_steps[0], target_pos) >= min_dist:
            best_step = shuffled_steps[0]
        else:
            for step_option in shuffled_steps:
                dist = self._manhattan_distance(step_option, target_pos)
                if dist < min_dist: min_dist = dist; best_step = step_option

        if best_step:
            self.model.grid.move_agent(self, best_step)

    def _execute_fsm(self):
        """
        Haupt-Zustandsmaschine (FSM) des Workers.
        Beinhaltet jetzt die Logik zur Abarbeitung von Touren.
        """
        previous_state = self.state

        if self.state == "IDLE":
            self.state = "MOVING_TO_SUPERVISOR"

        elif self.state == "MOVING_TO_SUPERVISOR":
            if self.pos == self.supervisor_home_pos:
                self.state = "AWAITING_TASK"
                self._report_and_request_new_task()
            else:
                self._move_towards(self.supervisor_home_pos)

        elif self.state == "AWAITING_TASK":
            if not self.current_task:
                # Polling für neue Aufgaben, falls der Worker wartet
                if self.pos == self.supervisor_home_pos and self.model.steps % 10 == self.unique_id % 10:
                    self._report_and_request_new_task()

        # --- NEUER ZUSTAND FÜR DIE TOUR-ABARBEITUNG ---
        elif self.state == "EXECUTING_TOUR":
            if self.current_task is None or self.current_task.get('type') != 'execute_tour':
                self._task_failed_or_issue("invalid_tour_task_data")
                return

            # Prüfen, ob die Tour abgeschlossen ist
            if self.current_tour_step_index >= len(self.current_tour_steps):
                self._task_completed()
                return

            # Das aktuelle Teil-Ziel aus der Tour holen
            current_sub_task = self.current_tour_steps[self.current_tour_step_index]
            target_pos = current_sub_task.get('target_pos')

            if target_pos is None:  # Sicherheitsprüfung
                self._task_failed_or_issue(f"tour_step_{self.current_tour_step_index}_missing_target")
                return

            # Prüfen, ob das Teil-Ziel erreicht wurde
            if self.pos == target_pos:
                # Hier wird die Aktion für das Teil-Ziel ausgeführt, in diesem Fall "erkunden"
                # Eine einfache Form der Erkundung: Umgebung wahrnehmen
                self._update_own_perception()

                # Zum nächsten Schritt der Tour übergehen
                self.current_tour_step_index += 1
            else:
                # Zum Ziel des aktuellen Tour-Schritts bewegen
                self._move_towards(target_pos)

        # --- Bestehende Logik für andere Aufgabentypen (weitgehend unverändert) ---
        elif self.state == "MOVING_TO_COLLECT_TARGET":
            if self.current_task is None: self.state = "IDLE"; return
            target = self.current_task['target_pos']
            if self.pos == target:
                self.state = "COLLECTING_AT_TARGET"
            else:
                self._move_towards(target)

        elif self.state == "COLLECTING_AT_TARGET":
            if self.current_task is None: self.state = "IDLE"; return
            target_pos = self.current_task['target_pos']
            if self.inventory_slot is None and target_pos in self.model.resources_on_grid:
                expected_type = self.current_task.get('resource_type')
                actual_type_data = self.model.resources_on_grid.get(target_pos)
                if actual_type_data and (not expected_type or expected_type == actual_type_data['type']):
                    resource_data = self.model.resources_on_grid.pop(target_pos)
                    self.inventory_slot = {'type': resource_data['type']}
                    self.worker_internal_map[target_pos[0], target_pos[1]] = RESOURCE_COLLECTED_BY_ME
                    self.newly_observed_map_segment[target_pos] = RESOURCE_COLLECTED_BY_ME
                    self.state = "MOVING_TO_BASE_FOR_TASK_DELIVERY"
                else:
                    reason = "wrong_res_type" if actual_type_data else "res_vanished"
                    if target_pos not in self.model.resources_on_grid:
                        self.worker_internal_map[target_pos[0], target_pos[1]] = EMPTY_EXPLORED
                        self.newly_observed_map_segment[target_pos] = EMPTY_EXPLORED
                    self._task_failed_or_issue(reason)
            else:
                reason = "inv_full_or_not_found_at_target"
                if target_pos not in self.model.resources_on_grid:
                    self.worker_internal_map[target_pos[0], target_pos[1]] = EMPTY_EXPLORED
                    self.newly_observed_map_segment[target_pos] = EMPTY_EXPLORED
                self._task_failed_or_issue(reason)

        elif self.state == "MOVING_TO_BASE_FOR_TASK_DELIVERY":
            if self.current_task is None and self.inventory_slot is None: self.state = "IDLE"; return
            if self.pos == self.model.base_deposit_point:
                if self.inventory_slot:
                    res_type = self.inventory_slot['type']
                    self.model.base_resources_collected[res_type] += 1
                    self.inventory_slot = None
                self._task_completed()
            else:
                self._move_towards(self.model.base_deposit_point)

        # Die spezifischen Zustände für explore_area und explore_corridor bleiben für den Fall,
        # dass solche Aufgaben doch einzeln vergeben werden (z.B. Hotspots oder Korridore).
        # Unsere neue Touren-Logik umgeht diese für die normale Erkundung.
        elif self.state == "MOVING_TO_EXPLORE_ROUTE_STEP":
            if self.current_task is None or not self.current_task.get('path_to_explore'):
                self._task_failed_or_issue("invalid_explore_task_data");
                return
            path = self.current_task['path_to_explore']
            target_waypoint = path[self.current_path_to_explore_index]
            if self.pos == target_waypoint:
                self.current_path_to_explore_index += 1
                if self.current_path_to_explore_index >= len(path):
                    if self.is_current_task_initial_hotspot:
                        self._task_completed()
                    else:
                        self.state = "EXPLORING_LOCALLY_AFTER_SUPERVISOR_ROUTE"
                        self.local_exploration_steps_done = 0
            else:
                self._move_towards(target_waypoint)

        elif self.state == "EXPLORING_LOCALLY_AFTER_SUPERVISOR_ROUTE":
            if self.local_exploration_steps_done < self.local_exploration_max_steps:
                possible_moves = list(
                    self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1))
                if possible_moves: self._move_towards(self.model.random.choice(possible_moves))
                self.local_exploration_steps_done += 1
            else:
                self._task_completed()

        elif self.state == "MOVING_TO_CORRIDOR_ENTRY":
            if self.current_task is None or self.current_task.get('type') != 'explore_corridor':
                self._task_failed_or_issue("invalid_corridor_task_data_entry");
                return
            target_entry_pos = self.current_task.get('entry_pos')
            if target_entry_pos is None:
                self._task_failed_or_issue("corridor_task_missing_entry_pos");
                return
            if self.pos == target_entry_pos:
                self.state = "FOLLOWING_CORRIDOR_PATH"
                self.current_corridor_path_index = 0
            else:
                self._move_towards(target_entry_pos)

        elif self.state == "FOLLOWING_CORRIDOR_PATH":
            if self.current_task is None or self.current_task.get('type') != 'explore_corridor':
                self._task_failed_or_issue("invalid_corridor_task_data_follow");
                return
            path = self.current_corridor_path
            if not path:
                self._task_failed_or_issue("corridor_task_empty_path");
                return
            if self.current_corridor_path_index >= len(path):
                self._task_completed();
                return
            target_waypoint = path[self.current_corridor_path_index]
            if self.pos == target_waypoint:
                self.current_corridor_path_index += 1
            else:
                self._move_towards(target_waypoint)

    def set_task(self, task_details):
        """
        Nimmt eine neue Aufgabe vom Supervisor entgegen und initialisiert den Worker-Zustand.
        Kann jetzt den neuen Typ 'execute_tour' verarbeiten.
        """
        self.current_task = task_details
        task_type = self.current_task.get('type')
        log_target_info = "N/A"

        # Reset aller aufgabenspezifischen Attribute
        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0
        self.is_current_task_initial_hotspot = False
        self.current_corridor_path = []
        self.current_corridor_path_index = 0
        self.current_tour_steps = []
        self.current_tour_step_index = 0

        # Task-spezifische Initialisierung
        if task_type == 'execute_tour':
            self.state = "EXECUTING_TOUR"
            self.current_tour_steps = self.current_task.get('tour_steps', [])
            self.current_tour_step_index = 0
            log_target_info = f"Tour mit {len(self.current_tour_steps)} Schritten."
            if not self.current_tour_steps:
                self._task_failed_or_issue("tour_task_empty_steps")

        elif task_type == 'collect_resource':
            self.state = "MOVING_TO_COLLECT_TARGET"
            log_target_info = f"Resource at {self.current_task.get('target_pos')}"

        elif task_type == 'explore_area':
            if self.current_task.get('path_to_explore'):
                self.state = "MOVING_TO_EXPLORE_ROUTE_STEP"
                self.is_current_task_initial_hotspot = self.current_task.get('is_initial_hotspot_task', False)
                log_target_info = f"Area at {self.current_task.get('path_to_explore')[0]}"
            else:
                self._task_failed_or_issue("explore_task_missing_path")

        elif task_type == 'explore_corridor':
            entry_pos = self.current_task.get('entry_pos')
            self.current_corridor_path = self.current_task.get('corridor_path', [])
            if entry_pos and self.current_corridor_path:
                self.state = "MOVING_TO_CORRIDOR_ENTRY"
                log_target_info = f"Corridor entry {entry_pos}"
            else:
                self._task_failed_or_issue("corridor_task_missing_info")

        else:
            self.state = "IDLE"
            log_target_info = f"Error: Unknown task type {task_type}"

    def _report_and_request_new_task(self):
        if self.pos != self.supervisor_home_pos:
            if self.state != "MOVING_TO_SUPERVISOR": self.state = "MOVING_TO_SUPERVISOR"
            return

        # HIER PASSIERT DIE DATENÜBERGABE
        report_data = {
            'status': 'IDLE_AT_SUPERVISOR', 'current_pos': self.pos,
            'inventory': self.inventory_slot,
            'map_segment_updates': self.newly_observed_map_segment.copy()  # Sendet alle gesammelten Daten
        }
        self.model.submit_report_to_supervisor(self.unique_id, 'status_and_map_update', report_data)

        # HIER WERDEN DIE DATEN NACH DER ÜBERGABE GELÖSCHT
        self.newly_observed_map_segment.clear()

        new_task = self.model.request_task_from_supervisor(self.unique_id)
        if new_task:
            self.set_task(new_task)
        else:
            self.state = "AWAITING_TASK"

    def _task_completed(self):
        """
        Beendet eine Aufgabe und leitet die Rückkehr zum Supervisor ein.
        Es wird KEIN Bericht mehr vom Feld aus gesendet.
        Die gesammelten Kartendaten bleiben für die spätere Ablieferung erhalten.
        """
        # Die gesamte Berichtslogik wird hier entfernt.
        # self.model.submit_report_to_supervisor(...) WIRD HIER NICHT MEHR AUFGERUFEN.

        # Reset der aufgabenspezifischen Attribute
        self.current_task = None
        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0
        self.is_current_task_initial_hotspot = False
        self.current_corridor_path = []
        self.current_corridor_path_index = 0
        self.current_tour_steps = []
        self.current_tour_step_index = 0

        # Der Worker weiß, dass er seine Aufgabe erfüllt hat und kehrt nun zurück.
        self.state = "MOVING_TO_SUPERVISOR"

    def _task_failed_or_issue(self, reason="unknown"):
        """
        Beendet eine fehlgeschlagene Aufgabe und leitet die Rückkehr ein.
        Es wird KEIN Bericht mehr vom Feld aus gesendet.
        """
        # Die gesamte Berichtslogik wird hier entfernt.
        # self.model.submit_report_to_supervisor(...) WIRD HIER NICHT MEHR AUFGERUFEN.

        # Reset der aufgabenspezifischen Attribute
        self.current_task = None
        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0
        self.is_current_task_initial_hotspot = False
        self.current_corridor_path = []
        self.current_corridor_path_index = 0
        self.current_tour_steps = []
        self.current_tour_step_index = 0

        # Entscheide, wohin als Nächstes gegangen wird.
        if self.inventory_slot:
            self.state = "MOVING_TO_BASE_FOR_TASK_DELIVERY"
        else:
            self.state = "MOVING_TO_SUPERVISOR"