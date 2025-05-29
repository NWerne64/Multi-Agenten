# src/worker_agent.py
import mesa
import numpy as np
from src.config import UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME


class WorkerAgent(mesa.Agent):
    def __init__(self, model, supervisor_home_pos, display_id_num):
        super().__init__(model=model)
        self.display_id = display_id_num

        self.current_task = None
        self.state = "IDLE"
        self.newly_observed_map_segment = {}
        self.inventory_slot = None
        self.supervisor_home_pos = supervisor_home_pos
        self.vision_radius = self.model.agent_vision_radius_val
        self.worker_internal_map = np.full((model.grid_width_val, model.grid_height_val), UNKNOWN, dtype=int)
        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0
        self.local_exploration_max_steps = self.model.random.randint(3, 6)  # Jeder Worker etwas anders

    def step(self):
        task_id_info = self.current_task.get('task_id') if self.current_task else 'None'
        # print(f"Worker {self.display_id} (MesaID: {self.unique_id}, Task: {task_id_info}, State: {self.state}) at {self.pos} on step {self.model.steps}")
        self._update_own_perception()
        self._execute_fsm()

    def _update_own_perception(self):
        if self.pos is None: return
        visible_cells_coords = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=self.vision_radius
        )
        for cell_pos in visible_cells_coords:
            cx, cy = cell_pos
            current_grid_content_type = EMPTY_EXPLORED
            if cell_pos in self.model.resources_on_grid:
                resource_type = self.model.resources_on_grid[cell_pos]['type']
                current_grid_content_type = WOOD_SEEN if resource_type == 'wood' else STONE_SEEN
            elif cell_pos in self.model.base_coords_list:
                current_grid_content_type = BASE_KNOWN

            if self.worker_internal_map[cx, cy] != current_grid_content_type or \
                    cell_pos not in self.newly_observed_map_segment:
                self.worker_internal_map[cx, cy] = current_grid_content_type
                self.newly_observed_map_segment[cell_pos] = current_grid_content_type

    def _execute_fsm(self):
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
                # Warte nicht passiv, wenn am Supervisor-Punkt, frage ggf. erneut (nicht zu oft)
                if self.pos == self.supervisor_home_pos and self.model.steps % 5 == 0:
                    # print(f"Worker {self.display_id} (MesaID: {self.unique_id}): Erneute Aufgabenanfrage beim Supervisor.")
                    self._report_and_request_new_task()
            # Der Zustand wird in set_task geändert, wenn eine Aufgabe erhalten wird

        elif self.state == "MOVING_TO_COLLECT_TARGET":
            if self.current_task is None: self.state = "IDLE"; return
            if self.pos == self.current_task['target_pos']:
                self.state = "COLLECTING_AT_TARGET"
            else:
                self._move_towards(self.current_task['target_pos'])

        elif self.state == "COLLECTING_AT_TARGET":
            if self.current_task is None: self.state = "IDLE"; return
            if self.inventory_slot is None and self.current_task['target_pos'] in self.model.resources_on_grid:
                expected_type = self.current_task.get('resource_type')
                actual_type_data = self.model.resources_on_grid.get(self.current_task['target_pos'])

                if actual_type_data and (not expected_type or expected_type == actual_type_data['type']):
                    resource_data = self.model.resources_on_grid.pop(self.current_task['target_pos'])
                    self.inventory_slot = {'type': resource_data['type']}
                    self.newly_observed_map_segment[self.current_task['target_pos']] = RESOURCE_COLLECTED_BY_ME
                    self.worker_internal_map[self.current_task['target_pos'][0], self.current_task['target_pos'][
                        1]] = RESOURCE_COLLECTED_BY_ME
                    print(
                        f"Worker {self.display_id} (MesaID: {self.unique_id}): Ressource {self.inventory_slot['type']} bei {self.pos} gesammelt.")
                    self.state = "MOVING_TO_BASE_FOR_TASK_DELIVERY"
                else:
                    reason = "wrong_resource_type" if actual_type_data else "resource_vanished_before_collection"
                    # print(f"Worker {self.display_id} (MesaID: {self.unique_id}): Problem bei {self.current_task['target_pos']}. Erwartet: {expected_type}, Status: {reason}.")
                    if self.current_task['target_pos'] not in self.model.resources_on_grid:
                        self.newly_observed_map_segment[self.current_task['target_pos']] = EMPTY_EXPLORED
                        self.worker_internal_map[
                            self.current_task['target_pos'][0], self.current_task['target_pos'][1]] = EMPTY_EXPLORED
                    self._task_failed_or_issue(reason)
            else:
                reason = "resource_not_found_at_target" if self.current_task[
                                                               'target_pos'] not in self.model.resources_on_grid else "inventory_full_or_other_collect_issue"
                # print(f"Worker {self.display_id} (MesaID: {self.unique_id}): Konnte Ressource bei {self.current_task['target_pos']} nicht sammeln. Grund: {reason}")
                if self.current_task['target_pos'] not in self.model.resources_on_grid:
                    self.newly_observed_map_segment[self.current_task['target_pos']] = EMPTY_EXPLORED
                    self.worker_internal_map[
                        self.current_task['target_pos'][0], self.current_task['target_pos'][1]] = EMPTY_EXPLORED
                self._task_failed_or_issue(reason)

        elif self.state == "MOVING_TO_BASE_FOR_TASK_DELIVERY":
            if self.current_task is None: self.state = "IDLE"; return
            if self.pos == self.model.base_deposit_point:
                if self.inventory_slot:
                    resource_type = self.inventory_slot['type']
                    self.model.base_resources_collected[resource_type] += 1
                    print(
                        f"Worker {self.display_id} (MesaID: {self.unique_id}): Ressource {resource_type} an Basis {self.model.base_deposit_point} abgeliefert.")
                    self.inventory_slot = None
                self._task_completed()
            else:
                self._move_towards(self.model.base_deposit_point)

        elif self.state == "MOVING_TO_EXPLORE_ROUTE_STEP":
            if self.current_task is None or not self.current_task.get('path_to_explore'):
                self._task_failed_or_issue("invalid_explore_task_data_in_fsm");
                return

            path = self.current_task['path_to_explore']
            if self.current_path_to_explore_index >= len(path):
                # print(f"Worker {self.display_id}: Explorationsroute bei {self.pos} beendet. Starte lokale Erkundung.")
                self.state = "EXPLORING_LOCALLY_AFTER_ROUTE"
                self.local_exploration_steps_done = 0
            else:
                next_waypoint = path[self.current_path_to_explore_index]
                if self.pos == next_waypoint:
                    self.current_path_to_explore_index += 1
                    if self.current_path_to_explore_index >= len(path):
                        self.state = "EXPLORING_LOCALLY_AFTER_ROUTE"
                        self.local_exploration_steps_done = 0
                else:
                    self._move_towards(next_waypoint)

        elif self.state == "EXPLORING_LOCALLY_AFTER_ROUTE":
            if self.current_task is None: self.state = "IDLE"; return

            if self.local_exploration_steps_done < self.local_exploration_max_steps:
                possible_moves = list(
                    self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1))
                if possible_moves:
                    # Einfache Logik: versuche, nicht direkt umzudrehen, falls möglich
                    # Diese Logik ist noch nicht perfekt, nur ein Anhaltspunkt.
                    best_move = self.model.random.choice(possible_moves)
                    # Noch einbauen Klügere lokale Exploration (z.B. spiralförmig, nicht nur random)
                    self._move_towards(best_move)
                self.local_exploration_steps_done += 1
            else:
                # print(f"Worker {self.display_id}: Lokale Erkundung nach Route beendet.")
                self._task_completed()

    def _report_and_request_new_task(self):
        if self.pos != self.supervisor_home_pos:
            if self.state != "MOVING_TO_SUPERVISOR": self.state = "MOVING_TO_SUPERVISOR"
            return

        # print(f"Worker {self.display_id} (MesaID: {self.unique_id}): Meldet sich beim Supervisor bei {self.pos}.")
        report_data = {
            'status': 'IDLE_AT_SUPERVISOR',
            'current_pos': self.pos,
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
            # print(f"Worker {self.display_id} (MesaID: {self.unique_id}): Keine neue Aufgabe vom Supervisor erhalten, warte bei {self.pos}.")

    def _task_completed(self):
        if not self.current_task:
            self.state = "MOVING_TO_SUPERVISOR";
            return

        # print(f"Worker {self.display_id} (MesaID: {self.unique_id}): Aufgabe {self.current_task['task_id']} ({self.current_task.get('type')}) erledigt.")
        report_data = {
            'status': 'TASK_COMPLETED',
            'task_id': self.current_task['task_id'],
            'current_pos': self.pos,
            'inventory': self.inventory_slot,
            'map_segment_updates': self.newly_observed_map_segment.copy()
        }
        self.model.submit_report_to_supervisor(self.unique_id, 'task_feedback', report_data)
        self.newly_observed_map_segment.clear()
        self.current_task = None
        self.current_path_to_explore_index = 0
        self.state = "MOVING_TO_SUPERVISOR"

    def _task_failed_or_issue(self, reason="unknown"):
        task_id_report = "N/A_NO_CURRENT_TASK"
        if self.current_task and 'task_id' in self.current_task:
            task_id_report = self.current_task['task_id']
        elif self.current_task:
            task_id_report = "N/A_TASK_ID_MISSING"

        # print(f"Worker {self.display_id} (MesaID: {self.unique_id}): Problem/Fehlschlag mit Aufgabe {task_id_report}. Grund: {reason}")
        report_data = {
            'status': 'TASK_FAILED',
            'task_id': task_id_report,
            'reason': reason,
            'current_pos': self.pos,
            'map_segment_updates': self.newly_observed_map_segment.copy()
        }
        self.model.submit_report_to_supervisor(self.unique_id, 'task_feedback', report_data)
        self.newly_observed_map_segment.clear()
        self.current_task = None
        self.current_path_to_explore_index = 0
        self.state = "MOVING_TO_SUPERVISOR"

    def set_task(self, task_details):
        self.current_task = task_details
        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0  # Wichtig für neue Explorationstasks

        print(
            f"Worker {self.display_id} (MesaID: {self.unique_id}): Aufgabe {self.current_task.get('task_id')} Typ {self.current_task.get('type')} erhalten.")
        if self.current_task['type'] == 'collect_resource':
            self.state = "MOVING_TO_COLLECT_TARGET"
        elif self.current_task['type'] == 'explore_area':
            if self.current_task.get('path_to_explore'):
                self.state = "MOVING_TO_EXPLORE_ROUTE_STEP"
            else:
                print(f"Worker {self.display_id} (MesaID: {self.unique_id}): Explorationsaufgabe ohne Pfad erhalten!")
                self._task_failed_or_issue("explore_task_missing_path")
        else:
            print(
                f"Worker {self.display_id} (MesaID: {self.unique_id}): Unbekannter Task-Typ in set_task: {self.current_task['type']}")
            self.state = "IDLE"

    def _move_towards(self, target_pos):
        if self.pos is None or target_pos is None or self.pos == target_pos:
            return
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False, radius=1
        )
        if not possible_steps:
            # print(f"Worker {self.display_id} (MesaID: {self.unique_id}): Keine möglichen Schritte von {self.pos}")
            return  # Bleibe stehen, wenn keine Schritte möglich sind

        best_step = None
        min_dist = self._manhattan_distance(self.pos, target_pos)
        shuffled_steps = list(possible_steps)
        self.random.shuffle(shuffled_steps)

        # Wenn nur ein möglicher Schritt da ist, der nicht näher kommt, aber der einzige ist
        if len(shuffled_steps) == 1 and self._manhattan_distance(shuffled_steps[0], target_pos) >= min_dist:
            best_step = shuffled_steps[0]
        else:
            for step_option in shuffled_steps:
                dist = self._manhattan_distance(step_option, target_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_step = step_option

        if best_step:
            self.model.grid.move_agent(self, best_step)
        # else:
        # print(f"Worker {self.display_id} (MesaID: {self.unique_id}): Konnte keinen besseren Schritt von {self.pos} nach {target_pos} finden.")

    def _manhattan_distance(self, pos1, pos2):
        if pos1 is None or pos2 is None: return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])