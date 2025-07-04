# src/worker_agent_touring.py
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
        self.state = "IDLE"  # Startet immer im Zustand IDLE
        self.newly_observed_map_segment = {}
        self.inventory_slot = None
        self.supervisor_home_pos = supervisor_home_pos
        self.vision_radius = self.model.agent_vision_radius_val
        self.worker_internal_map = np.full((self.model.grid_width_val, self.model.grid_height_val), UNKNOWN, dtype=int)

        # Aufgabenspezifische Attribute
        self.current_corridor_path = []
        self.current_corridor_path_index = 0
        self.current_tour_steps = []
        self.current_tour_step_index = 0
        self.current_sub_task = None
        self.is_current_task_initial_hotspot = False


    def _manhattan_distance(self, pos1, pos2):
        if pos1 is None or pos2 is None: return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self):
        self._update_own_perception()
        self._execute_fsm()

    def _update_own_perception(self):
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
        if self.pos is None or target_pos is None or self.pos == target_pos: return
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1)
        if not possible_steps: return

        best_step = None
        min_dist = self._manhattan_distance(self.pos, target_pos)
        shuffled_steps = list(possible_steps)
        self.random.shuffle(shuffled_steps)

        if len(shuffled_steps) == 1 and self._manhattan_distance(shuffled_steps[0], target_pos) >= min_dist:
            best_step = shuffled_steps[0]
        else:
            for step_option in shuffled_steps:
                dist = self._manhattan_distance(step_option, target_pos)
                if dist < min_dist: min_dist = dist; best_step = step_option

        if best_step:
            self.model.grid.move_agent(self, best_step)

    # MODIFIZIERT: Die Haupt-Zustandsmaschine wird um die Korridor-Touren erweitert
    def _execute_fsm(self):
        """
        Stellt die ursprüngliche, detaillierte Zustandslogik wieder her,
        um eine robuste Erkundung und Aufgabenbearbeitung zu gewährleisten.
        """
        previous_state = self.state

        # --- Standard-Zustände ---
        if self.state == "IDLE":
            self.state = "MOVING_TO_SUPERVISOR"
        elif self.state == "MOVING_TO_SUPERVISOR":
            if self.pos == self.supervisor_home_pos:
                self._report_and_request_new_task()
                if not self.current_task: self.state = "AWAITING_TASK"
            else:
                self._move_towards(self.supervisor_home_pos)
        elif self.state == "AWAITING_TASK":
            if not self.current_task:
                if self.pos == self.supervisor_home_pos and self.model.steps % 5 == self.unique_id % 5:
                    self._report_and_request_new_task()
            return

        # --- Logik für initiale Ankerpunkte (Hotspots) ---
        elif self.state == "MOVING_TO_EXPLORE_HOTSPOT":
            if self.current_task is None: self._task_failed_or_issue("no_hotspot_task"); return
            target_waypoint = self.current_task['path_to_explore'][0]
            if self.pos == target_waypoint:
                self._task_completed()  # Hotspot erreicht, Aufgabe erfüllt
            else:
                self._move_towards(target_waypoint)

        # --- Detaillierte Logik für Korridor-Touren ---
        elif self.state == "EXECUTING_CORRIDOR_TOUR":
            # Prüfe, ob die gesamte Tour (alle Korridore) fertig ist
            if self.current_tour_step_index >= len(self.current_tour_steps):
                self._task_completed()
                return

            # Nimm den nächsten Korridor aus der Tour und bereite ihn vor
            self.current_sub_task = self.current_tour_steps[self.current_tour_step_index]
            self.current_corridor_path = self.current_sub_task.get('corridor_path', [])
            self.current_corridor_path_index = 0
            self.state = "MOVING_TO_CORRIDOR_ENTRY"

        elif self.state == "MOVING_TO_CORRIDOR_ENTRY":
            if self.current_sub_task is None: self._task_failed_or_issue("no_active_corridor_sub_task"); return
            target_entry_pos = self.current_sub_task.get('entry_pos')
            if self.pos == target_entry_pos:
                self.state = "FOLLOWING_CORRIDOR_PATH"
            else:
                self._move_towards(target_entry_pos)

        elif self.state == "FOLLOWING_CORRIDOR_PATH":
            if self.current_sub_task is None: self._task_failed_or_issue("no_active_corridor_sub_task"); return
            path = self.current_corridor_path
            # Prüfe, ob der einzelne Korridor-Pfad fertig ist
            if self.current_corridor_path_index >= len(path):
                self.current_tour_step_index += 1  # Gehe zum nächsten Korridor in der Tour
                self.current_sub_task = None
                self.state = "EXECUTING_CORRIDOR_TOUR"  # Gehe zurück zum Tour-Manager
                return

            target_waypoint = path[self.current_corridor_path_index]
            if self.pos == target_waypoint:
                self.current_corridor_path_index += 1
            else:
                self._move_towards(target_waypoint)

        # --- Logik für Ressourcensammlung (unverändert) ---
        elif self.state == "MOVING_TO_COLLECT_TARGET":
            target = self.current_task.get('target_pos')
            if self.pos == target:
                self.state = "COLLECTING_AT_TARGET"
            else:
                self._move_towards(target)
        elif self.state == "COLLECTING_AT_TARGET":
            target_pos = self.current_task['target_pos']
            if self.inventory_slot is None and target_pos in self.model.resources_on_grid:
                resource_data = self.model.resources_on_grid.pop(target_pos)
                self.inventory_slot = {'type': resource_data['type']}
                self.state = "MOVING_TO_BASE_FOR_TASK_DELIVERY"
            else:
                self._task_failed_or_issue("resource_not_found_or_inv_full")
        elif self.state == "MOVING_TO_BASE_FOR_TASK_DELIVERY":
            base_target = self.model.base_deposit_point
            if self.pos == base_target:
                if self.inventory_slot:
                    self.model.base_resources_collected[self.inventory_slot['type']] += 1
                    self.inventory_slot = None
                self._task_completed()
            else:
                self._move_towards(base_target)

    def set_task(self, task_details):
        """
        Setzt die Zustände korrekt für die detaillierte FSM.
        """
        self.current_task = task_details

        # Setze alle aufgabenspezifischen Variablen zurück
        self.current_tour_steps = []
        self.current_tour_step_index = 0
        self.current_corridor_path = []
        self.current_corridor_path_index = 0
        self.current_sub_task = None
        self.inventory_slot = None

        if not task_details:
            self.state = "AWAITING_TASK"
            return

        task_type = self.current_task.get('type')

        if task_type == 'explore_area':
            # Dies ist der Zustand für die initialen Ankerpunkte (Hotspots)
            self.state = "MOVING_TO_EXPLORE_HOTSPOT"
        elif task_type == 'execute_corridor_tour':
            # Dies startet die komplexe Korridor-Logik
            self.state = "EXECUTING_CORRIDOR_TOUR"
            self.current_tour_steps = self.current_task.get('tour_steps', [])
        elif task_type == 'collect_resource':
            self.state = "MOVING_TO_COLLECT_TARGET"
        else:
            self._task_failed_or_issue(f"unknown_task_type_{task_type}")

    # Die restlichen Methoden bleiben unverändert
    def _report_and_request_new_task(self):
        if self.pos != self.supervisor_home_pos:
            if self.state != "MOVING_TO_SUPERVISOR": self.state = "MOVING_TO_SUPERVISOR"
            return

        report_data = {
            'status': 'IDLE_AT_SUPERVISOR', 'current_pos': self.pos,
            'inventory': self.inventory_slot,
            'map_segment_updates': self.newly_observed_map_segment.copy()
        }
        self.model.submit_report_to_supervisor(self.unique_id, 'status_and_map_update', report_data)
        self.newly_observed_map_segment.clear()

        new_task = self.model.request_task_from_supervisor(self.unique_id)
        if new_task:
            self.set_task(new_task)
        else:
            self.state = "AWAITING_TASK"

    def _task_completed(self):
        self.current_task = None
        self.current_sub_task = None
        # ... weitere Resets ...
        self.state = "MOVING_TO_SUPERVISOR"

    def _task_failed_or_issue(self, reason="unknown"):
        self.current_task = None
        self.current_sub_task = None
        # ... weitere Resets ...
        if self.inventory_slot:
            self.state = "MOVING_TO_BASE_FOR_TASK_DELIVERY"
        else:
            self.state = "MOVING_TO_SUPERVISOR"