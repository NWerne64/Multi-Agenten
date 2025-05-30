# src/worker_agent.py
import mesa
import numpy as np
from src.config import (
    UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME,
    CHAINED_EXPLORATION_STEP_BUDGET,  # Importiere neue Konstanten
    MAX_CHAINED_FRONTIERS_VISITED,
    CHAINED_LOCAL_EXPLORE_STEPS
)


class WorkerAgent(mesa.Agent):
    NEIGHBOR_OFFSETS = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]

    def __init__(self, model, supervisor_home_pos, display_id_num):
        super().__init__(model=model)
        self.display_id = display_id_num

        self.current_task = None
        self.state = "IDLE"  # Startzustand
        self.newly_observed_map_segment = {}
        self.inventory_slot = None
        self.supervisor_home_pos = supervisor_home_pos
        self.vision_radius = self.model.agent_vision_radius_val
        self.worker_internal_map = np.full((model.grid_width_val, model.grid_height_val), UNKNOWN, dtype=int)

        # Für Supervisor-gegebene Routen
        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0  # Für lokale Erkundung nach Supervisor-Route / Kaskaden-Frontier
        self.local_exploration_max_steps = self.model.random.randint(3, 6)

        # NEU: Für Kaskaden-Exploration (Chained Exploration)
        self.chained_sortie_steps_taken = 0
        self.chained_frontiers_visited_count = 0
        self.current_chained_target_pos = None
        self.is_on_chained_exploration_sortie = False

        # Lade aus config.py, besser über self.model.
        self.CHAINED_EXPLORATION_STEP_BUDGET_val = getattr(self.model, 'CHAINED_EXPLORATION_STEP_BUDGET_val',
                                                           CHAINED_EXPLORATION_STEP_BUDGET)
        self.MAX_CHAINED_FRONTIERS_VISITED_val = getattr(self.model, 'MAX_CHAINED_FRONTIERS_VISITED_val',
                                                         MAX_CHAINED_FRONTIERS_VISITED)
        self.CHAINED_LOCAL_EXPLORE_STEPS_val = getattr(self.model, 'CHAINED_LOCAL_EXPLORE_STEPS_val',
                                                       CHAINED_LOCAL_EXPLORE_STEPS)

    def _manhattan_distance(self, pos1, pos2):
        if pos1 is None or pos2 is None: return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self):
        # task_id_info = self.current_task.get('task_id') if self.current_task else 'None'
        # print(f"Worker {self.display_id} (Task: {task_id_info}, State: {self.state}, ChainedSortie: {self.is_on_chained_exploration_sortie}) at {self.pos} on step {self.model.steps}")

        self._update_own_perception()
        self._execute_fsm()

        if self.is_on_chained_exploration_sortie:
            self.chained_sortie_steps_taken += 1

    def _update_own_perception(self):
        if self.pos is None: return
        visible_cells_coords = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=self.vision_radius
        )
        for cell_pos in visible_cells_coords:
            cx, cy = cell_pos
            # Nur aktualisieren, wenn wir die Zelle nicht als "von mir gesammelt" markiert haben
            # oder wenn die neue Beobachtung wichtiger ist.
            current_cell_state_on_worker_map = self.worker_internal_map[cx, cy]

            # Was der Worker tatsächlich sieht:
            actual_grid_content_type = EMPTY_EXPLORED
            if cell_pos in self.model.resources_on_grid:
                resource_type = self.model.resources_on_grid[cell_pos]['type']
                actual_grid_content_type = WOOD_SEEN if resource_type == 'wood' else STONE_SEEN
            elif cell_pos in self.model.base_coords_list:
                actual_grid_content_type = BASE_KNOWN

            # Update worker_internal_map und newly_observed_map_segment
            # Wenn sich der Zustand ändert ODER die Zelle noch nicht im aktuellen Segment ist
            if current_cell_state_on_worker_map != actual_grid_content_type or \
                    cell_pos not in self.newly_observed_map_segment or \
                    self.newly_observed_map_segment[
                        cell_pos] != actual_grid_content_type:  # Auch wenn sich der Wert im Segment ändert

                if current_cell_state_on_worker_map == RESOURCE_COLLECTED_BY_ME and \
                        actual_grid_content_type in [WOOD_SEEN, STONE_SEEN]:
                    # Seltsamer Fall: Ich dachte, ich hätte es gesammelt, aber da ist wieder was.
                    # Fürs Erste: Überschreibe mit der neuen Beobachtung.
                    pass  # Erlaube Überschreiben

                self.worker_internal_map[cx, cy] = actual_grid_content_type
                self.newly_observed_map_segment[cell_pos] = actual_grid_content_type

    def _find_own_local_frontier_target(self):
        """Einfache Frontier-Suche auf der eigenen Karte des Workers."""
        candidate_frontiers = []
        known_passable_rows, known_passable_cols = np.where(
            (self.worker_internal_map == EMPTY_EXPLORED) |
            (self.worker_internal_map == BASE_KNOWN) |
            (self.worker_internal_map == RESOURCE_COLLECTED_BY_ME)  # Auch von mir gesammelte sind passierbar
        )

        # Priorisiere Frontiers, die näher am aktuellen Standort sind
        if not known_passable_rows.size: return None

        parent_cells = []
        for r, c in zip(known_passable_rows, known_passable_cols):
            parent_cells.append(((int(r), int(c)), self._manhattan_distance(self.pos, (int(r), int(c)))))

        parent_cells.sort(key=lambda x: x[1])  # Sortiere nach Distanz

        for parent_data in parent_cells:
            parent_pos = parent_data[0]
            for dx, dy in self.NEIGHBOR_OFFSETS:
                frontier_x, frontier_y = parent_pos[0] + dx, parent_pos[1] + dy
                frontier_pos = (frontier_x, frontier_y)

                if 0 <= frontier_x < self.model.grid_width_val and \
                        0 <= frontier_y < self.model.grid_height_val and \
                        self.worker_internal_map[frontier_x, frontier_y] == UNKNOWN:
                    # Einfachste Logik: erste gefundene Frontier nehmen
                    # print(f"Worker {self.display_id}: Eigene lokale Frontier gefunden: {frontier_pos} von Parent {parent_pos}")
                    return frontier_pos
        return None

    def _start_chained_exploration_if_possible(self):
        """Versucht, die Kaskaden-Exploration zu starten oder fortzusetzen."""
        if not self.is_on_chained_exploration_sortie:  # Start einer neuen Kaskaden-Exploration-Phase
            self.is_on_chained_exploration_sortie = True
            self.chained_sortie_steps_taken = 0
            self.chained_frontiers_visited_count = 0
            # print(f"Worker {self.display_id}: Startet Kaskaden-Exploration.")

        # Prüfe Budgets
        if self.chained_sortie_steps_taken >= self.CHAINED_EXPLORATION_STEP_BUDGET_val or \
                self.chained_frontiers_visited_count >= self.MAX_CHAINED_FRONTIERS_VISITED_val:
            # print(f"Worker {self.display_id}: Kaskaden-Exploration Budget erschöpft (Schritte: {self.chained_sortie_steps_taken}, Frontiers: {self.chained_frontiers_visited_count}). Beende Aufgabe.")
            self._task_completed()
            return

        new_target = self._find_own_local_frontier_target()
        if new_target:
            self.current_chained_target_pos = new_target
            self.local_exploration_steps_done = 0  # Reset für lokale Schritte an neuer Frontier
            self.state = "CHAINING_TO_NEW_FRONTIER"
            # print(f"Worker {self.display_id}: Kaskaden-Explo: Nächstes Ziel {new_target}.")
        else:
            # print(f"Worker {self.display_id}: Keine weiteren lokalen Frontiers für Kaskaden-Explo gefunden. Beende Aufgabe.")
            self._task_completed()  # Keine lokalen Frontiers mehr, Aufgabe beenden

    def _execute_fsm(self):
        # --- Standard Zustände ---
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
                if self.pos == self.supervisor_home_pos and self.model.steps % 5 == 0:
                    self._report_and_request_new_task()
            # Zustand wird in set_task geändert

        elif self.state == "MOVING_TO_COLLECT_TARGET":
            # ... (Logik bleibt gleich wie in deiner Version) ...
            if self.current_task is None: self.state = "IDLE"; return
            if self.pos == self.current_task['target_pos']:
                self.state = "COLLECTING_AT_TARGET"
            else:
                self._move_towards(self.current_task['target_pos'])


        elif self.state == "COLLECTING_AT_TARGET":
            # ... (Logik bleibt gleich wie in deiner Version) ...
            if self.current_task is None: self.state = "IDLE"; return
            if self.inventory_slot is None and self.current_task['target_pos'] in self.model.resources_on_grid:
                expected_type = self.current_task.get('resource_type')
                actual_type_data = self.model.resources_on_grid.get(self.current_task['target_pos'])

                if actual_type_data and (not expected_type or expected_type == actual_type_data['type']):
                    resource_data = self.model.resources_on_grid.pop(self.current_task['target_pos'])
                    self.inventory_slot = {'type': resource_data['type']}
                    # Wichtig: Genaue Position als von mir gesammelt markieren
                    self.worker_internal_map[self.current_task['target_pos'][0], self.current_task['target_pos'][
                        1]] = RESOURCE_COLLECTED_BY_ME
                    self.newly_observed_map_segment[
                        self.current_task['target_pos']] = RESOURCE_COLLECTED_BY_ME  # Auch für Report
                    # print(f"Worker {self.display_id}: Ressource {self.inventory_slot['type']} bei {self.pos} gesammelt.")
                    self.state = "MOVING_TO_BASE_FOR_TASK_DELIVERY"
                else:  # Ressource nicht wie erwartet oder nicht mehr da
                    reason = "wrong_resource_type" if actual_type_data else "resource_vanished_before_collection"
                    if self.current_task['target_pos'] not in self.model.resources_on_grid:
                        # Markiere als leer, wenn nicht mehr da
                        self.worker_internal_map[
                            self.current_task['target_pos'][0], self.current_task['target_pos'][1]] = EMPTY_EXPLORED
                        self.newly_observed_map_segment[self.current_task['target_pos']] = EMPTY_EXPLORED
                    self._task_failed_or_issue(reason)
            else:  # Inventar voll oder Ressource nicht (mehr) da
                reason = "resource_not_found_at_target" if self.current_task[
                                                               'target_pos'] not in self.model.resources_on_grid else "inventory_full_or_other_collect_issue"
                if self.current_task['target_pos'] not in self.model.resources_on_grid:
                    self.worker_internal_map[
                        self.current_task['target_pos'][0], self.current_task['target_pos'][1]] = EMPTY_EXPLORED
                    self.newly_observed_map_segment[self.current_task['target_pos']] = EMPTY_EXPLORED
                self._task_failed_or_issue(reason)


        elif self.state == "MOVING_TO_BASE_FOR_TASK_DELIVERY":
            # ... (Logik bleibt gleich wie in deiner Version) ...
            if self.current_task is None: self.state = "IDLE"; return  # Sollte nicht passieren, wenn Inventar voll
            if self.pos == self.model.base_deposit_point:
                if self.inventory_slot:
                    resource_type = self.inventory_slot['type']
                    self.model.base_resources_collected[resource_type] += 1
                    # print(f"Worker {self.display_id}: Ressource {resource_type} an Basis {self.model.base_deposit_point} abgeliefert.")
                    self.inventory_slot = None
                self._task_completed()  # Aufgabe (Ressource abliefern) ist beendet
            else:
                self._move_towards(self.model.base_deposit_point)


        # --- Angepasste und neue Explorations-Zustände ---
        elif self.state == "MOVING_TO_EXPLORE_ROUTE_STEP":  # Supervisor-gegebene Route
            if self.current_task is None or not self.current_task.get('path_to_explore'):
                self._task_failed_or_issue("invalid_explore_task_data_in_fsm_supervisor_route");
                return

            path = self.current_task['path_to_explore']
            if self.current_path_to_explore_index >= len(path):
                # Supervisor-Route beendet, nun lokale Schritte an diesem Punkt
                self.state = "EXPLORING_LOCALLY_AFTER_SUPERVISOR_ROUTE"
                self.local_exploration_steps_done = 0
            else:
                next_waypoint = path[self.current_path_to_explore_index]
                if self.pos == next_waypoint:
                    self.current_path_to_explore_index += 1
                    # Wenn Pfad beendet, lokale Schritte starten
                    if self.current_path_to_explore_index >= len(path):
                        self.state = "EXPLORING_LOCALLY_AFTER_SUPERVISOR_ROUTE"
                        self.local_exploration_steps_done = 0
                else:
                    self._move_towards(next_waypoint)

        elif self.state == "EXPLORING_LOCALLY_AFTER_SUPERVISOR_ROUTE":  # Nach Supervisor-Route
            if self.local_exploration_steps_done < self.local_exploration_max_steps:
                possible_moves = list(
                    self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1))
                if possible_moves:
                    self._move_towards(self.model.random.choice(possible_moves))  # Einfach zufällig bewegen
                self.local_exploration_steps_done += 1
            else:
                # Lokale Schritte nach Supervisor-Route beendet, starte Kaskaden-Exploration
                # print(f"Worker {self.display_id}: Lokale Erkundung nach Supervisor-Route beendet. Starte Kaskaden-Explo.")
                self._start_chained_exploration_if_possible()

        elif self.state == "CHAINING_TO_NEW_FRONTIER":  # Auf dem Weg zu einer selbstgefundenen Frontier
            if self.current_chained_target_pos is None or not self.is_on_chained_exploration_sortie:
                self._task_failed_or_issue("invalid_state_for_chaining_to_frontier");
                return

            if self.pos == self.current_chained_target_pos:
                self.chained_frontiers_visited_count += 1
                self.state = "PERFORMING_CHAINED_LOCAL_EXPLORATION"
                self.local_exploration_steps_done = 0  # Reset für lokale Schritte
            else:
                self._move_towards(self.current_chained_target_pos)

            # Budget-Check auch während der Bewegung
            if self.chained_sortie_steps_taken >= self.CHAINED_EXPLORATION_STEP_BUDGET_val:
                # print(f"Worker {self.display_id}: Kaskaden-Exploration Budget (Schritte) beim Bewegen erschöpft. Beende Aufgabe.")
                self._task_completed()

        elif self.state == "PERFORMING_CHAINED_LOCAL_EXPLORATION":  # Lokale Schritte an einer Kaskaden-Frontier
            if not self.is_on_chained_exploration_sortie:
                self._task_failed_or_issue("invalid_state_for_performing_chained_local_exploration");
                return

            if self.local_exploration_steps_done < self.CHAINED_LOCAL_EXPLORE_STEPS_val:
                possible_moves = list(
                    self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1))
                if possible_moves:
                    self._move_towards(self.model.random.choice(possible_moves))
                self.local_exploration_steps_done += 1
            else:
                # Lokale Schritte an Kaskaden-Frontier beendet, versuche nächste Kaskaden-Frontier
                self._start_chained_exploration_if_possible()

    def _report_and_request_new_task(self):
        if self.pos != self.supervisor_home_pos:
            if self.state != "MOVING_TO_SUPERVISOR": self.state = "MOVING_TO_SUPERVISOR"
            return

        # print(f"Worker {self.display_id}: Meldet sich beim Supervisor bei {self.pos}.")
        report_data = {
            'status': 'IDLE_AT_SUPERVISOR',  # Wichtig für Supervisor, um zu wissen, dass Worker bereit ist
            'current_pos': self.pos,
            'inventory': self.inventory_slot,  # Inventar melden, falls noch was drin ist (sollte nicht sein)
            'map_segment_updates': self.newly_observed_map_segment.copy()
        }
        self.model.submit_report_to_supervisor(self.unique_id, 'status_and_map_update', report_data)
        self.newly_observed_map_segment.clear()  # Beobachtungen wurden gemeldet

        new_task = self.model.request_task_from_supervisor(self.unique_id)
        if new_task:
            self.set_task(new_task)
        else:
            self.state = "AWAITING_TASK"  # Bleibe im Wartezustand, wenn keine Aufgabe da ist

    def _task_completed(self):
        # Diese Methode wird aufgerufen, wenn eine Aufgabe (Sammeln, Liefern ODER die gesamte Explorationstour) beendet ist.
        # print(f"Worker {self.display_id}: Aufgabe {self.current_task.get('task_id') if self.current_task else 'Unknown'} Typ {self.current_task.get('type') if self.current_task else 'Unknown'} abgeschlossen.")
        report_data = {
            'status': 'TASK_COMPLETED',
            'task_id': self.current_task.get('task_id',
                                             'N/A_NO_CURRENT_TASK') if self.current_task else 'N/A_NO_TASK_AT_ALL',
            'current_pos': self.pos,
            'inventory': self.inventory_slot,  # Relevant falls Aufgabe war "Liefern" und Inventar jetzt leer
            'map_segment_updates': self.newly_observed_map_segment.copy()  # Alle gesammelten Beobachtungen
        }
        self.model.submit_report_to_supervisor(self.unique_id, 'task_feedback', report_data)
        self.newly_observed_map_segment.clear()

        # Reset für nächste Aufgabe
        self.current_task = None
        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0
        self.is_on_chained_exploration_sortie = False  # Wichtig: Kaskaden-Modus beenden
        self.chained_sortie_steps_taken = 0
        self.chained_frontiers_visited_count = 0
        self.current_chained_target_pos = None

        self.state = "MOVING_TO_SUPERVISOR"  # Nach jeder Aufgabe zurück zum Supervisor

    def _task_failed_or_issue(self, reason="unknown"):
        # print(f"Worker {self.display_id}: Problem/Fehlschlag mit Aufgabe. Grund: {reason}")
        task_id_report = "N/A_NO_CURRENT_TASK"
        if self.current_task and 'task_id' in self.current_task:
            task_id_report = self.current_task['task_id']
        elif self.current_task:  # Task existiert, aber ohne task_id Feld
            task_id_report = "N/A_TASK_ID_MISSING"

        report_data = {
            'status': 'TASK_FAILED',
            'task_id': task_id_report,
            'reason': reason,
            'current_pos': self.pos,
            'map_segment_updates': self.newly_observed_map_segment.copy()  # Auch bei Fehlschlag Karte melden
        }
        self.model.submit_report_to_supervisor(self.unique_id, 'task_feedback', report_data)
        self.newly_observed_map_segment.clear()

        # Reset
        self.current_task = None
        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0
        self.is_on_chained_exploration_sortie = False  # Wichtig
        self.chained_sortie_steps_taken = 0
        self.chained_frontiers_visited_count = 0
        self.current_chained_target_pos = None
        if self.inventory_slot:  # Wenn Ressource im Inventar war und Task fehlschlug (z.B. Basis nicht erreichbar)
            # print(f"Worker {self.display_id}: Task fehlgeschlagen, hat aber noch {self.inventory_slot['type']} im Inventar. Versucht zur Basis zu bringen.")
            self.state = "MOVING_TO_BASE_FOR_TASK_DELIVERY"  # Versuche trotzdem, es zur Basis zu bringen
            # Die "current_task" Information ist weg, aber der Worker hat eine klare Direktive.
            # Besser wäre, eine interne "Liefer-Notfallaufgabe" zu erstellen. Fürs Erste so.
        else:
            self.state = "MOVING_TO_SUPERVISOR"

    def set_task(self, task_details):
        self.current_task = task_details
        # Resets für route-basierte Tasks (Supervisor oder Kaskade intern)
        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0

        # Resets für Kaskaden-Exploration (wird beim Start der Kaskade neu initialisiert)
        self.is_on_chained_exploration_sortie = False
        self.chained_sortie_steps_taken = 0
        self.chained_frontiers_visited_count = 0
        self.current_chained_target_pos = None

        # print(f"Worker {self.display_id}: Aufgabe {self.current_task.get('task_id')} Typ {self.current_task.get('type')} erhalten.")
        if self.current_task['type'] == 'collect_resource':
            self.state = "MOVING_TO_COLLECT_TARGET"
        elif self.current_task['type'] == 'explore_area':
            if self.current_task.get('path_to_explore'):  # Sollte jetzt immer ein Pfad mit einem Ziel sein
                self.state = "MOVING_TO_EXPLORE_ROUTE_STEP"  # Start der Supervisor-Exploration
            else:
                # print(f"Worker {self.display_id}: Explorationsaufgabe ohne Pfad erhalten!")
                self._task_failed_or_issue("explore_task_missing_path")
        else:
            # print(f"Worker {self.display_id}: Unbekannter Task-Typ: {self.current_task['type']}")
            self.state = "IDLE"  # Fallback

    def _move_towards(self, target_pos):
        if self.pos is None or target_pos is None or self.pos == target_pos:
            return
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False, radius=1
        )
        if not possible_steps: return

        best_step = None
        min_dist = self._manhattan_distance(self.pos, target_pos)

        shuffled_steps = list(possible_steps)  # Konvertiere zu Liste für random.shuffle
        self.random.shuffle(shuffled_steps)

        # Wenn nur ein möglicher Schritt da ist, der nicht näher kommt, aber der einzige ist
        if len(shuffled_steps) == 1 and self._manhattan_distance(shuffled_steps[0], target_pos) >= min_dist:
            best_step = shuffled_steps[0]  # Nimm diesen Schritt, wenn es der einzige ist
        else:
            for step_option in shuffled_steps:
                dist = self._manhattan_distance(step_option, target_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_step = step_option

        if best_step:
            self.model.grid.move_agent(self, best_step)