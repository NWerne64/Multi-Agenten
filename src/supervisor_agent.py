# src/supervisor_agent.py
import mesa
import numpy as np
import itertools
from src.config import UNKNOWN, EMPTY_EXPLORED, BASE_KNOWN, WOOD_SEEN, STONE_SEEN, RESOURCE_COLLECTED_BY_ME


class SupervisorAgent(mesa.Agent):
    def __init__(self, model, home_pos, role_id_for_display="supervisor_0"):
        super().__init__(model=model)
        self.role_id = role_id_for_display

        self.home_pos = home_pos
        self.supervisor_known_map = np.full((model.grid_width_val, model.grid_height_val), UNKNOWN, dtype=int)

        if self.home_pos not in self.model.base_coords_list:
            if 0 <= self.home_pos[0] < self.model.grid_width_val and \
                    0 <= self.home_pos[1] < self.model.grid_height_val:
                self.supervisor_known_map[self.home_pos[0], self.home_pos[1]] = EMPTY_EXPLORED

        for base_coord in self.model.base_coords_list:
            if 0 <= base_coord[0] < self.model.grid_width_val and \
                    0 <= base_coord[1] < self.model.grid_height_val:
                self.supervisor_known_map[base_coord[0], base_coord[1]] = BASE_KNOWN

        self.worker_status = {}
        self.task_queue = []  # Aufgaben werden hier mit Priorität eingefügt (Sammeln > Exploration)
        self.assigned_tasks = {}
        self.resource_goals = self.model.resource_goals.copy()
        self._pending_worker_reports = []
        self._tasks_to_assign_to_worker = {}

        # Limits, wie viele *neue* Tasks pro Step maximal *geplant* werden sollen (nicht unbedingt zugewiesen)
        self.max_new_collect_tasks_per_planning = self.model.num_agents_val  # Plane potenziell für jeden Worker
        self.max_new_explore_tasks_per_planning = self.model.num_agents_val

        self.pending_exploration_targets = set()
        self.task_id_counter = itertools.count(1)

    def step(self):
        self._process_pending_reports()
        self._update_task_statuses_and_cleanup()
        self._plan_new_tasks()  # Füllt die task_queue
        self._prepare_tasks_for_assignment()  # Versucht, Aufgaben aus der Queue an wartende Worker zu geben

    def _process_pending_reports(self):
        # ... (Inhalt der Methode bleibt gleich wie in deiner letzten funktionierenden Version) ...
        for report in self._pending_worker_reports:
            worker_id = report['worker_id']
            report_type = report['report_type']
            data = report['data']
            self.worker_status.setdefault(worker_id, {})
            if 'current_pos' in data:
                self.worker_status[worker_id]['last_pos'] = data['current_pos']
            new_worker_state = data.get('status')
            if new_worker_state:
                self.worker_status[worker_id]['state'] = new_worker_state
            task_id_in_report = data.get('task_id')
            if new_worker_state in ['TASK_COMPLETED', 'TASK_FAILED']:
                self.worker_status[worker_id]['current_task_id'] = None
                if task_id_in_report and task_id_in_report in self.assigned_tasks:
                    task_details = self.assigned_tasks.get(task_id_in_report)
                    if task_details and task_details.get('worker_id') == worker_id:
                        if task_details.get('type') == 'explore_area' and 'path_to_explore' in task_details:
                            for pos_on_route in task_details['path_to_explore']:
                                self.pending_exploration_targets.discard(pos_on_route)
                        del self.assigned_tasks[task_id_in_report]
            elif new_worker_state == 'IDLE_AT_SUPERVISOR':
                self.worker_status[worker_id]['current_task_id'] = None
            if 'map_segment_updates' in data:
                for pos, reported_state in data['map_segment_updates'].items():
                    px, py = pos
                    if 0 <= px < self.supervisor_known_map.shape[0] and \
                            0 <= py < self.supervisor_known_map.shape[1]:
                        current_supervisor_state = self.supervisor_known_map[px, py]
                        new_state_to_set = current_supervisor_state
                        if reported_state == RESOURCE_COLLECTED_BY_ME:
                            new_state_to_set = EMPTY_EXPLORED
                        elif reported_state in [WOOD_SEEN, STONE_SEEN]:
                            if current_supervisor_state in [UNKNOWN, EMPTY_EXPLORED, BASE_KNOWN]:
                                new_state_to_set = reported_state
                            elif current_supervisor_state in [WOOD_SEEN,
                                                              STONE_SEEN] and current_supervisor_state != reported_state:
                                new_state_to_set = reported_state
                        elif reported_state == BASE_KNOWN:
                            if current_supervisor_state == UNKNOWN or current_supervisor_state == EMPTY_EXPLORED:
                                new_state_to_set = reported_state
                        elif reported_state == EMPTY_EXPLORED:
                            if current_supervisor_state != RESOURCE_COLLECTED_BY_ME:
                                if current_supervisor_state == UNKNOWN or current_supervisor_state in [WOOD_SEEN,
                                                                                                       STONE_SEEN]:  # Leer überschreibt auch gesehene Ressourcen
                                    new_state_to_set = reported_state
                        if current_supervisor_state != new_state_to_set:
                            print(
                                f"Supervisor DEBUG Map Update: Pos {pos} von {current_supervisor_state} zu {new_state_to_set} (Worker meldete: {reported_state})")
                            self.supervisor_known_map[px, py] = new_state_to_set
        self._pending_worker_reports.clear()

    def _update_task_statuses_and_cleanup(self):
        # ... (Inhalt der Methode bleibt gleich) ...
        pass

    def _is_target_already_assigned_or_queued(self, target_pos_to_check, task_type, resource_type=None):
        # ... (Inhalt der Methode bleibt gleich) ...
        for task_data in list(self.assigned_tasks.values()) + self.task_queue:
            current_task_main_target = None
            if task_data.get('type') == 'explore_area' and task_data.get('path_to_explore'):
                if not task_data['path_to_explore']: continue
                current_task_main_target = task_data['path_to_explore'][0]
            elif task_data.get('type') == 'collect_resource':
                current_task_main_target = task_data.get('target_pos')

            if task_data.get('type') == task_type and current_task_main_target == target_pos_to_check:
                if task_type == 'collect_resource':
                    if task_data.get('resource_type') == resource_type:
                        return True
                elif task_type == 'explore_area':
                    return True
        if task_type == 'explore_area':
            path_to_check = None
            if isinstance(target_pos_to_check, list):
                path_to_check = target_pos_to_check
            elif isinstance(target_pos_to_check, tuple):
                path_to_check = [target_pos_to_check]
            if path_to_check:
                for point_in_route in path_to_check:
                    if point_in_route in self.pending_exploration_targets:
                        return True
        return False

    # --- ÜBERARBEITET: _plan_new_tasks mit klarer Priorisierung ---
    def _plan_new_tasks(self):
        wood_on_map = np.count_nonzero(self.supervisor_known_map == WOOD_SEEN)
        stone_on_map = np.count_nonzero(self.supervisor_known_map == STONE_SEEN)
        print(
            f"Supervisor Planung (Step {self.model.steps}): Map: H={wood_on_map}, S={stone_on_map}. Tasks Queue: {len(self.task_queue)}, Assigned: {len(self.assigned_tasks)}")

        collect_tasks_added_now = 0
        explore_tasks_added_now = 0

        # --- Phase 1: Sammelaufgaben haben hohe Priorität ---
        # Sortiere Ziele nach Dringlichkeit (wie viel fehlt noch?)
        resource_priority = sorted(
            self.resource_goals.keys(),
            key=lambda r: (self.resource_goals[r] - self.model.base_resources_collected.get(r, 0)),
            reverse=True
        )

        for res_type in resource_priority:
            if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning:
                break  # Tageslimit für neue Sammelaufgaben erreicht

            needed_amount = self.resource_goals.get(res_type, 0) - self.model.base_resources_collected.get(res_type, 0)
            if needed_amount <= 0:
                continue  # Ziel für diesen Ressourcentyp bereits erfüllt

            resource_seen_constant = WOOD_SEEN if res_type == 'wood' else STONE_SEEN

            candidate_patches_coords = []
            rows, cols = np.where(self.supervisor_known_map == resource_seen_constant)
            for r_idx, c_idx in zip(rows, cols):
                candidate_patches_coords.append((r_idx, c_idx))

            self.model.random.shuffle(candidate_patches_coords)

            for patch_pos in candidate_patches_coords:
                if self._is_target_already_assigned_or_queued(patch_pos, 'collect_resource', res_type):
                    continue

                new_collect_task = {
                    'task_id': f"task_collect_{next(self.task_id_counter)}",
                    'type': 'collect_resource',
                    'target_pos': patch_pos,
                    'resource_type': res_type,
                    'status': 'pending_assignment'
                }
                # Füge Sammelaufgaben an den ANFANG der Queue (höhere Priorität)
                self.task_queue.insert(0, new_collect_task)
                print(
                    f"Supervisor: NEUE SAMMELAUFGABE {new_collect_task['task_id']} (Sammle {res_type} bei {patch_pos}) zur Queue hinzugefügt.")
                collect_tasks_added_now += 1
                if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning:
                    break  # Breche auch die patch_pos Schleife, wenn Limit erreicht
            # Das äußere `break` (für res_type) wird durch die Bedingung am Anfang der äußeren Schleife gehandhabt.

        # --- Phase 2: Explorationsaufgaben, wenn Kapazität und Bedarf ---
        # Bedarf an Exploration definieren: z.B. wenn noch Ziele offen sind und Ressourcen dafür nicht bekannt,
        # oder wenn ein Großteil der Karte unbekannt ist.

        # Zähle, wie viele Ressourcen noch fehlen, für die wir keine Standorte kennen
        unlocated_needed_resources = 0
        for res_type, target_amount in self.resource_goals.items():
            if self.model.base_resources_collected.get(res_type, 0) < target_amount:
                res_const = WOOD_SEEN if res_type == 'wood' else STONE_SEEN
                if np.count_nonzero(self.supervisor_known_map == res_const) == 0:
                    unlocated_needed_resources += 1

        # Grober Schätzer für "Karte ist noch sehr unbekannt"
        unknown_ratio = np.count_nonzero(self.supervisor_known_map == UNKNOWN) / (
                    self.model.grid_width_val * self.model.grid_height_val)

        # Exploriere, wenn (noch Ressourcen unbekannt sind UND wir nicht schon max Sammelaufgaben geplant haben)
        # ODER (die Karte noch sehr unbekannt ist UND wir nicht schon max Sammelaufgaben geplant haben)
        # UND das Limit für neue Explorationsaufgaben noch nicht erreicht ist.
        # Die Logik: Wenn wir schon viele Sammelaufgaben planen konnten, ist Exploration vllt. weniger dringend.

        should_explore_actively = (unlocated_needed_resources > 0 or unknown_ratio > 0.75)

        if should_explore_actively and explore_tasks_added_now < self.max_new_explore_tasks_per_planning:
            exploration_route = self._generate_exploration_route()
            if not exploration_route:
                exploration_route = self._generate_initial_random_exploration_target_route()

            if exploration_route:
                # Prüfe den Startpunkt der Route
                if not self._is_target_already_assigned_or_queued(exploration_route, 'explore_area'):
                    new_explore_task = {
                        'task_id': f"task_explore_{next(self.task_id_counter)}",
                        'type': 'explore_area',
                        'path_to_explore': exploration_route,
                        'status': 'pending_assignment',
                        'target_pos': exploration_route[0]
                    }
                    # Füge Explorationsaufgaben ans ENDE der Queue (niedrigere Priorität als Sammeln)
                    self.task_queue.append(new_explore_task)
                    print(
                        f"Supervisor: Neue Explorationsaufgabe {new_explore_task['task_id']} (Route: {exploration_route[:2]}...) zur Queue hinzugefügt.")
                    for pos_on_route in exploration_route:
                        self.pending_exploration_targets.add(pos_on_route)
                    explore_tasks_added_now += 1

    # --- Ende _plan_new_tasks ---

    def _generate_exploration_route(self):
        # ... (Inhalt bleibt wie in der vorherigen Antwort) ...
        NEIGHBOR_OFFSETS = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
        ROUTE_LENGTH = self.model.random.randint(3, 6)
        parent_candidates_info = []
        known_passable_rows, known_passable_cols = np.where(
            (self.supervisor_known_map == EMPTY_EXPLORED) |
            (self.supervisor_known_map == BASE_KNOWN)
        )
        if not known_passable_rows.size: return None
        all_parent_candidates = list(zip(known_passable_rows, known_passable_cols))
        self.model.random.shuffle(all_parent_candidates)
        for parent_pos in all_parent_candidates[:50]:
            self.model.random.shuffle(NEIGHBOR_OFFSETS)
            for dx_start, dy_start in NEIGHBOR_OFFSETS:
                first_unknown_step = (parent_pos[0] + dx_start, parent_pos[1] + dy_start)
                if 0 <= first_unknown_step[0] < self.model.grid_width_val and \
                        0 <= first_unknown_step[1] < self.model.grid_height_val and \
                        self.supervisor_known_map[first_unknown_step[0], first_unknown_step[1]] == UNKNOWN and \
                        first_unknown_step not in self.pending_exploration_targets:
                    route = [first_unknown_step]
                    current_pos_on_route = first_unknown_step
                    for _ in range(ROUTE_LENGTH - 1):
                        next_pos_on_route = (current_pos_on_route[0] + dx_start, current_pos_on_route[1] + dy_start)
                        if 0 <= next_pos_on_route[0] < self.model.grid_width_val and \
                                0 <= next_pos_on_route[1] < self.model.grid_height_val and \
                                (self.supervisor_known_map[next_pos_on_route[0], next_pos_on_route[1]] == UNKNOWN or \
                                 next_pos_on_route == route[0]) and \
                                next_pos_on_route not in self.pending_exploration_targets:
                            route.append(next_pos_on_route)
                            current_pos_on_route = next_pos_on_route
                        else:
                            break
                    if route: return route
        return None

    def _generate_initial_random_exploration_target_route(self):
        # ... (Inhalt bleibt wie in der vorherigen Antwort) ...
        unknown_cells_coords = []
        for r_idx in range(self.model.grid_width_val):
            for c_idx in range(self.model.grid_height_val):
                if self.supervisor_known_map[r_idx, c_idx] == UNKNOWN and (
                r_idx, c_idx) not in self.pending_exploration_targets:
                    unknown_cells_coords.append((r_idx, c_idx))
        if unknown_cells_coords:
            target_cell = self.model.random.choice(unknown_cells_coords)
            return [target_cell]
        return None

    def _prepare_tasks_for_assignment(self):
        # ... (Inhalt bleibt wie in der vorherigen Antwort) ...
        if not self.task_queue: return
        for worker_id, status_data in list(self.worker_status.items()):
            if not self.task_queue: break
            if status_data.get('state') == 'IDLE_AT_SUPERVISOR' and \
                    worker_id not in self._tasks_to_assign_to_worker:
                is_already_working_on_assigned_task = False
                active_task_id_for_worker = status_data.get('current_task_id')
                if active_task_id_for_worker and active_task_id_for_worker in self.assigned_tasks:
                    if self.assigned_tasks[active_task_id_for_worker].get('worker_id') == worker_id:
                        is_already_working_on_assigned_task = True
                if is_already_working_on_assigned_task:
                    continue
                task_to_assign = self.task_queue.pop(0)  # Nimmt Aufgabe mit höchster Priorität (Anfang der Liste)
                task_to_assign['status'] = 'assigned_pending_pickup'
                self._tasks_to_assign_to_worker[worker_id] = task_to_assign

    def receive_report_from_worker(self, worker_id, report_type, data):
        # ... (Inhalt bleibt wie in der vorherigen Antwort) ...
        self._pending_worker_reports.append({'worker_id': worker_id, 'report_type': report_type, 'data': data})
        new_worker_state = data.get('status')
        if new_worker_state:
            self.worker_status.setdefault(worker_id, {})['state'] = new_worker_state
            if new_worker_state in ['TASK_COMPLETED', 'TASK_FAILED', 'IDLE_AT_SUPERVISOR']:
                self.worker_status[worker_id]['current_task_id'] = None

    def request_task_from_worker(self, worker_id):
        # ... (Inhalt bleibt wie in der vorherigen Antwort) ...
        self.worker_status.setdefault(worker_id, {})['state'] = 'IDLE_AT_SUPERVISOR'
        self.worker_status[worker_id]['current_task_id'] = None
        self._prepare_tasks_for_assignment()
        if worker_id in self._tasks_to_assign_to_worker:
            task = self._tasks_to_assign_to_worker.pop(worker_id)
            task['status'] = 'assigned'
            task['worker_id'] = worker_id
            self.assigned_tasks[task['task_id']] = task
            self.worker_status[worker_id]['current_task_id'] = task['task_id']
            print(f"Supervisor: Aufgabe {task['task_id']} ({task.get('type')}) an Worker {worker_id} vergeben.")
            return task
        return None