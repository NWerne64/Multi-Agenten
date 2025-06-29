# src/supervisor_agent.py
import mesa
import numpy as np
import itertools
from collections import deque

from src.config import (
    UNKNOWN, EMPTY_EXPLORED, BASE_KNOWN,
    WOOD_SEEN, STONE_SEEN, RESOURCE_COLLECTED_BY_ME,
    SUPERVISOR_CLAIMED_RESOURCE,
    MIN_EXPLORE_TARGET_SEPARATION,
    SUPERVISOR_INITIAL_EXPLORATION_HOTSPOTS,
    MIN_UNKNOWN_RATIO_FOR_CONTINUED_EXPLORATION,
    SUPERVISOR_LOGISTICS_KNOWN_PASSABLE,
    SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED
)


class SupervisorAgent(mesa.Agent):
    NEIGHBOR_OFFSETS = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]  #
    DEFAULT_FRONTIER_SEARCH_DEPTH = 4
    DEFAULT_DEEP_DIVE_CANDIDATE_COUNT = 50
    BREAKOUT_PHASE_UNKNOWN_THRESHOLD = 0.80
    BREAKOUT_DISTANCE_WEIGHT = 5.0
    DEFAULT_CORRIDOR_LENGTH = 30
    CORRIDOR_ATTEMPT_THRESHOLD = 2

    def __init__(self, model, home_pos, role_id_for_display="supervisor_0"):
        super().__init__(model=model)
        self.role_id = role_id_for_display
        self.home_pos = home_pos

        self.supervisor_known_map = np.full((model.grid_width_val, model.grid_height_val), UNKNOWN, dtype=int)
        self.supervisor_exploration_logistics_map = np.full((model.grid_width_val, model.grid_height_val), UNKNOWN,
                                                            dtype=int)

        init_cells = [self.home_pos] + model.base_coords_list
        for pos in init_cells:
            if 0 <= pos[0] < self.model.grid_width_val and 0 <= pos[1] < self.model.grid_height_val:
                is_base = pos in model.base_coords_list
                self.supervisor_known_map[pos[0], pos[1]] = BASE_KNOWN if is_base else EMPTY_EXPLORED
                self.supervisor_exploration_logistics_map[pos[0], pos[1]] = SUPERVISOR_LOGISTICS_KNOWN_PASSABLE

        self.worker_status = {}
        self.task_queue = []
        self.assigned_tasks = {}
        self.resource_goals = self.model.resource_goals.copy()
        self._pending_worker_reports = []
        self._tasks_to_assign_to_worker = {}

        self.max_new_collect_tasks_per_planning = self.model.num_agents_val
        self.max_new_explore_tasks_per_planning = self.model.num_agents_val

        self.pending_exploration_targets = set()
        self.task_id_counter = itertools.count(1)

        self.MIN_EXPLORE_TARGET_SEPARATION_val = getattr(self.model, 'MIN_EXPLORE_TARGET_SEPARATION_val',
                                                         MIN_EXPLORE_TARGET_SEPARATION)
        self.min_unknown_ratio_for_continued_exploration_val = getattr(self.model,
                                                                       'MIN_UNKNOWN_RATIO_FOR_CONTINUED_EXPLORATION_val',
                                                                       MIN_UNKNOWN_RATIO_FOR_CONTINUED_EXPLORATION)
        self.initial_hotspots_abs = []
        hotspots_config = getattr(self.model, 'SUPERVISOR_INITIAL_EXPLORATION_HOTSPOTS_val',
                                  SUPERVISOR_INITIAL_EXPLORATION_HOTSPOTS)
        if hotspots_config:
            for rel_x, rel_y in hotspots_config:
                abs_x = int(self.model.grid_width_val * rel_x);
                abs_y = int(self.model.grid_height_val * rel_y)
                abs_x = max(0, min(abs_x, self.model.grid_width_val - 1));
                abs_y = max(0, min(abs_y, self.model.grid_height_val - 1))
                self.initial_hotspots_abs.append((abs_x, abs_y))
        self.model.random.shuffle(self.initial_hotspots_abs)
        self.attempted_hotspots = set()

        self.total_grid_cells = self.model.grid_width_val * self.model.grid_height_val
        self.initial_hotspot_planning_complete = False
        self.no_normal_target_found_count = 0

        self.needs_new_planning = True
        self.max_task_queue_buffer = 4

        self.active_corridors_viz = {}

        print(
            f"[S_AGENT {self.role_id}] (Init): Supervisor created. MIN_EXPLORE_TARGET_SEPARATION: {self.MIN_EXPLORE_TARGET_SEPARATION_val}, DEFAULT_CORRIDOR_LENGTH: {self.DEFAULT_CORRIDOR_LENGTH}")

    def get_initial_task(self):
        """
        Erstellt eine 'explore_area'-Aufgabe für einen initialen Hotspot,
        die mit der set_task-Methode des Workers kompatibel ist.
        """
        if self.initial_hotspots_abs:
            next_hotspot = self.initial_hotspots_abs.pop(0)
            task_id = next(self.task_id_counter)

            # Erstelle eine Aufgabe, die die 'set_task'-Methode des Workers versteht.
            # Wir formatieren sie als 'explore_area'-Task.
            task = {
                'task_id': task_id,
                'type': 'explore_area',
                # 'path_to_explore' muss eine Liste sein, auch wenn sie nur ein Element hat.
                'path_to_explore': [next_hotspot],
                'is_initial_hotspot_task': True,  # Wichtiger Flag für den Worker
                'status': 'assigned'
            }
            return task

        return None

    def _get_projected_explored_cells(self, target_pos, vision_radius):
        cells_to_mark = set();  #
        for dx in range(-vision_radius, vision_radius + 1):  #
            for dy in range(-vision_radius, vision_radius + 1):  #
                px, py = target_pos[0] + dx, target_pos[1] + dy  #
                if 0 <= px < self.model.grid_width_val and 0 <= py < self.model.grid_height_val:  #
                    cells_to_mark.add((px, py))  #
        return list(cells_to_mark)  #

    def _manhattan_distance(self, pos1, pos2):
        if pos1 is None or pos2 is None: return float('inf')  #
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])  #

    def _distance(self, pos1, pos2):
        """Calculates the Euclidean distance between two points."""
        if pos1 is None or pos2 is None: return float('inf')
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def step(self):
        self._process_pending_reports()  #
        # self._update_task_statuses_and_cleanup() # Auskommentiert, da nicht implementiert
        if self.needs_new_planning:  #
            self._plan_new_tasks()
            self.needs_new_planning = False  #
        self._prepare_tasks_for_assignment()  #

    def _process_pending_reports(self):
        if not self._pending_worker_reports: return  #
        reports_to_process = list(self._pending_worker_reports);  #
        self._pending_worker_reports.clear()  #

        for report in reports_to_process:  #
            worker_id = report['worker_id'];  #
            data = report['data']  #
            task_id_in_report = data.get('task_id');  #
            task_details_for_report = self.assigned_tasks.get(task_id_in_report) if task_id_in_report else None  #
            new_worker_state = data.get('status');  #
            worker_current_pos = data.get('current_pos')  #

            map_updates = data.get('map_segment_updates', {})  #
            report_caused_significant_map_change = False  #

            self.worker_status.setdefault(worker_id, {});  #
            if worker_current_pos: self.worker_status[worker_id]['last_pos'] = worker_current_pos  #

            if worker_current_pos == self.home_pos and map_updates:  #
                for pos, reported_state in map_updates.items():  #
                    px, py = pos;  #
                    if not (0 <= px < self.model.grid_width_val and 0 <= py < self.model.grid_height_val): continue  #
                    original_known_map_value = self.supervisor_known_map[px, py]  #
                    current_public_state = original_known_map_value  #
                    new_public_state = current_public_state  #
                    if current_public_state == SUPERVISOR_CLAIMED_RESOURCE:  #
                        if reported_state == RESOURCE_COLLECTED_BY_ME or reported_state == EMPTY_EXPLORED: new_public_state = EMPTY_EXPLORED  #
                    else:  #
                        if reported_state == RESOURCE_COLLECTED_BY_ME:  #
                            new_public_state = EMPTY_EXPLORED  #
                        elif reported_state in [WOOD_SEEN, STONE_SEEN]:  #
                            if current_public_state in [UNKNOWN, EMPTY_EXPLORED, BASE_KNOWN] or \
                                    (current_public_state in [WOOD_SEEN,
                                                              STONE_SEEN] and current_public_state != reported_state): new_public_state = reported_state  #
                        elif reported_state == BASE_KNOWN:  #
                            if current_public_state in [UNKNOWN, EMPTY_EXPLORED]: new_public_state = reported_state  #
                        elif reported_state == EMPTY_EXPLORED:  #
                            if current_public_state != SUPERVISOR_CLAIMED_RESOURCE:  #
                                if current_public_state in [UNKNOWN, WOOD_SEEN, STONE_SEEN,
                                                            BASE_KNOWN]: new_public_state = reported_state  #
                    if original_known_map_value != new_public_state:  #
                        self.supervisor_known_map[px, py] = new_public_state  #
                        report_caused_significant_map_change = True  #
                    original_logistics_map_value = self.supervisor_exploration_logistics_map[px, py]  #
                    new_logistics_map_value = original_logistics_map_value  #
                    if new_public_state == EMPTY_EXPLORED or new_public_state == BASE_KNOWN:  #
                        if original_logistics_map_value != SUPERVISOR_LOGISTICS_KNOWN_PASSABLE: new_logistics_map_value = SUPERVISOR_LOGISTICS_KNOWN_PASSABLE  #
                    elif new_public_state == UNKNOWN:  #
                        if original_logistics_map_value != UNKNOWN: new_logistics_map_value = UNKNOWN  #
                    elif new_public_state in [WOOD_SEEN, STONE_SEEN, SUPERVISOR_CLAIMED_RESOURCE]:  #
                        if original_logistics_map_value in [UNKNOWN,
                                                            SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED]: new_logistics_map_value = SUPERVISOR_LOGISTICS_KNOWN_PASSABLE  #
                    if original_logistics_map_value != new_logistics_map_value:  #
                        self.supervisor_exploration_logistics_map[px, py] = new_logistics_map_value  #
                        report_caused_significant_map_change = True  #

            if new_worker_state in ['TASK_COMPLETED', 'TASK_FAILED'] and task_details_for_report:  #
                self.needs_new_planning = True  #
                if task_id_in_report in self.active_corridors_viz:  #
                    del self.active_corridors_viz[task_id_in_report]  #

            if report_caused_significant_map_change:  #
                self.needs_new_planning = True  #

            if new_worker_state: self.worker_status[worker_id]['state'] = new_worker_state  #
            if new_worker_state in ['TASK_COMPLETED', 'TASK_FAILED']:  #
                self.worker_status[worker_id]['current_task_id'] = None  #
                if task_details_for_report and task_details_for_report.get('worker_id') == worker_id:  #
                    task_type = task_details_for_report.get('type');  #
                    target_pos = task_details_for_report.get('target_pos')  #
                    if task_type == 'explore_area':  #
                        explore_target = task_details_for_report.get('target_pos')  #
                        if explore_target: self.pending_exploration_targets.discard(explore_target)  #
                    elif task_type == 'explore_corridor':  #
                        corridor_path_reported = task_details_for_report.get('corridor_path', [])  #
                        if corridor_path_reported:  #
                            self.pending_exploration_targets.discard(corridor_path_reported[0])  #
                            if len(corridor_path_reported) > 1:  #
                                self.pending_exploration_targets.discard(corridor_path_reported[-1])  #
                    elif task_type == 'collect_resource' and target_pos and new_worker_state == 'TASK_FAILED':  #
                        current_map_val_at_target = self.supervisor_known_map[target_pos[0], target_pos[1]]  #
                        reported_state_at_target = map_updates.get(target_pos)  #
                        if current_map_val_at_target == SUPERVISOR_CLAIMED_RESOURCE:  #
                            original_known_map_value_at_target = self.supervisor_known_map[
                                target_pos[0], target_pos[1]]  #
                            new_val_for_target = original_known_map_value_at_target  #
                            if reported_state_at_target in [WOOD_SEEN, STONE_SEEN]:  #
                                new_val_for_target = reported_state_at_target  #
                            elif reported_state_at_target == EMPTY_EXPLORED or reported_state_at_target == RESOURCE_COLLECTED_BY_ME:  #
                                new_val_for_target = EMPTY_EXPLORED  #
                            if original_known_map_value_at_target != new_val_for_target:  #
                                self.supervisor_known_map[target_pos[0], target_pos[1]] = new_val_for_target  #
                                self.needs_new_planning = True  #
                    if task_id_in_report in self.assigned_tasks: del self.assigned_tasks[task_id_in_report]  #
            elif new_worker_state == 'IDLE_AT_SUPERVISOR':  #
                self.worker_status[worker_id]['current_task_id'] = None  #

    def _is_target_already_assigned_or_queued(self, target_pos_to_check, task_type, resource_type=None):  #
        for task_data in list(self.assigned_tasks.values()) + self.task_queue:  #
            current_task_main_target = None  #
            task_data_type = task_data.get('type')  #

            if task_data_type == 'explore_area':  #
                path = task_data.get('path_to_explore')  #
                if path and path[0]: current_task_main_target = path[0]  #
            elif task_data_type == 'explore_corridor':  #
                current_task_main_target = task_data.get('entry_pos')  #
            elif task_data_type == 'collect_resource':  #
                current_task_main_target = task_data.get('target_pos')  #

            if task_data_type == task_type and current_task_main_target == target_pos_to_check:  #
                if task_type == 'collect_resource' and task_data.get('resource_type') == resource_type: return True  #
                if task_type == 'explore_area' or task_type == 'explore_corridor': return True  #

        if task_type == 'explore_area' or task_type == 'explore_corridor':  #
            if self.supervisor_exploration_logistics_map[
                target_pos_to_check[0], target_pos_to_check[1]] == SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:  #
                return True  #
        return False  #

    def _plan_new_tasks(self):
        """
        Hauptplanungsfunktion, die jetzt auch Touren plant.
        Diese Funktion orchestriert die Erstellung von Sammel- und Erkundungsaufgaben.
        Die Erkundungsaufgaben werden nun als Touren geplant.
        """
        collect_tasks_added_now = 0
        explore_tasks_added_now = 0
        hotspots_created_this_step = 0

        # --- Initiales Hotspot-Planning (unverändert) ---
        if not self.initial_hotspot_planning_complete:
            unattempted_hotspots_for_planning = 0
            for hotspot_pos in self.initial_hotspots_abs:
                if explore_tasks_added_now >= self.max_new_explore_tasks_per_planning: break
                if hotspot_pos not in self.attempted_hotspots:
                    unattempted_hotspots_for_planning += 1
                    if not self._is_target_already_assigned_or_queued(hotspot_pos, 'explore_area'):
                        new_hotspot_task = {'task_id': f"task_hotspot_{next(self.task_id_counter)}",
                                            'type': 'explore_area', 'path_to_explore': [hotspot_pos],
                                            'status': 'pending_assignment', 'target_pos': hotspot_pos,
                                            'is_initial_hotspot_task': True}
                        self.task_queue.insert(0, new_hotspot_task)
                        self.attempted_hotspots.add(hotspot_pos)
                        explore_tasks_added_now += 1
                        hotspots_created_this_step += 1
            if unattempted_hotspots_for_planning == 0 and len(self.initial_hotspots_abs) > 0:
                self.initial_hotspot_planning_complete = True

        # --- Ressourcen-Sammel-Planung (unverändert) ---
        resource_priority = sorted(self.resource_goals.keys(), key=lambda r: (
                self.resource_goals[r] - self.model.base_resources_collected.get(r, 0)), reverse=True)
        for res_type in resource_priority:
            if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning: break
            needed = self.resource_goals.get(res_type, 0) - self.model.base_resources_collected.get(res_type, 0)
            if needed <= 0: continue
            resource_seen_constant = WOOD_SEEN if res_type == 'wood' else STONE_SEEN
            candidate_patches_coords = []
            rows, cols = np.where(self.supervisor_known_map == resource_seen_constant)
            for r_idx, c_idx in zip(rows, cols): candidate_patches_coords.append((int(r_idx), int(c_idx)))
            self.model.random.shuffle(candidate_patches_coords)
            for patch_pos in candidate_patches_coords:
                if self._is_target_already_assigned_or_queued(patch_pos, 'collect_resource', res_type): continue
                self.supervisor_known_map[patch_pos[0], patch_pos[1]] = SUPERVISOR_CLAIMED_RESOURCE
                new_collect_task = {'task_id': f"task_collect_{next(self.task_id_counter)}", 'type': 'collect_resource',
                                    'target_pos': patch_pos, 'resource_type': res_type,
                                    'status': 'pending_assignment'}
                self.task_queue.insert(0, new_collect_task)
                collect_tasks_added_now += 1
                if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning: break
            if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning: break

        # --- NEU: Logik, um zu entscheiden, ob Erkundung nötig ist (leicht angepasst) ---
        unlocated_needed_resources = 0
        for res_type_iter, target_amount_iter in self.resource_goals.items():
            if self.model.base_resources_collected.get(res_type_iter, 0) < target_amount_iter:
                res_const = WOOD_SEEN if res_type_iter == 'wood' else STONE_SEEN
                if np.count_nonzero(self.supervisor_known_map == res_const) == 0: unlocated_needed_resources += 1
        current_unknown_logistics_ratio = np.count_nonzero(
            self.supervisor_exploration_logistics_map == UNKNOWN) / self.total_grid_cells
        goals_fully_met = all(
            self.model.base_resources_collected.get(res_type, 0) >= target_amount for res_type, target_amount in
            self.resource_goals.items())

        should_explore_actively = False
        if not goals_fully_met:
            if unlocated_needed_resources > 0 or current_unknown_logistics_ratio > self.min_unknown_ratio_for_continued_exploration_val:
                should_explore_actively = True

        # --- NEU: Aufruf der Tourenplanung ---
        if should_explore_actively and (self.initial_hotspot_planning_complete or hotspots_created_this_step == 0):
            # Wir rufen jetzt eine dedizierte Funktion auf, um Touren zu planen.
            explore_tasks_added_now += self._plan_exploration_tours(
                self.max_new_explore_tasks_per_planning - explore_tasks_added_now)

        # Fallback auf Korridor-Exploration, wenn keine normalen Ziele gefunden werden (unverändert)
        if explore_tasks_added_now == 0 and should_explore_actively:
            self.no_normal_target_found_count += 1
            if self.no_normal_target_found_count >= self.CORRIDOR_ATTEMPT_THRESHOLD:
                if self._find_and_plan_corridor_task(set()):
                    self.no_normal_target_found_count = 0
        else:
            self.no_normal_target_found_count = 0

    # FÜGE DIESE GÄNZLICH NEUE FUNKTION IN DIE SupervisorAgent-KLASSE EIN
    def _plan_exploration_tours(self, max_tours_to_plan):
        """
        Plant Erkundungs-Touren, indem es Ziele zu Paketen schnürt.
        Iteration 1: Erstellt Touren mit genau zwei Zielen.
        """
        tours_planned = 0
        temp_pending_targets_this_step = set()

        # Hole alle aktuell zugewiesenen Ziele, um sie zu vermeiden
        assigned_targets = self._get_all_assigned_targets()

        while tours_planned < max_tours_to_plan:
            # Finde das erste Ziel für die Tour
            ziel_a = self._find_best_frontier_for_exploration(temp_pending_targets_this_step | assigned_targets)
            if not ziel_a:
                break  # Keine passenden Ziele mehr gefunden

            temp_pending_targets_this_step.add(ziel_a)

            # Finde das zweite, zu ziel_a nächstgelegene Ziel
            # Dafür brauchen wir eine Liste potenzieller Ziele
            candidate_frontiers = self._get_all_frontiers(temp_pending_targets_this_step | assigned_targets)

            if not candidate_frontiers:
                # Nur ein einziges Ziel gefunden, erstelle eine Tour mit einem Schritt
                tour_paket = [{'type': 'explore_area', 'target_pos': ziel_a}]
            else:
                # Sortiere Kandidaten nach der Distanz zu ziel_a
                candidate_frontiers.sort(key=lambda p: self._distance(p, ziel_a))
                ziel_b = candidate_frontiers[0]
                temp_pending_targets_this_step.add(ziel_b)

                tour_paket = [
                    {'type': 'explore_area', 'target_pos': ziel_a},
                    {'type': 'explore_area', 'target_pos': ziel_b}
                ]

            # Erstelle die Tour-Aufgabe und füge sie zur Warteschlange hinzu
            new_tour_task = {
                'task_id': f"task_tour_{next(self.task_id_counter)}",
                'type': 'execute_tour',  # Neuer Aufgabentyp
                'tour_steps': tour_paket,
                'status': 'pending_assignment'
            }
            self.task_queue.append(new_tour_task)
            tours_planned += 1

        return tours_planned

    # FÜGE DIESE ZWEI NEUEN HELFERFUNKTIONEN HINZU
    def _get_all_assigned_targets(self):
        """Sammelt alle Ziele, die aktuell zugewiesen oder in der Warteschlange sind."""
        targets = set()
        all_tasks = list(self.assigned_tasks.values()) + self.task_queue
        for task_data in all_tasks:
            if task_data.get('type') == 'execute_tour':
                for step in task_data.get('tour_steps', []):
                    targets.add(step['target_pos'])
            elif task_data.get('target_pos'):
                targets.add(task_data.get('target_pos'))
        return targets

    def _get_all_frontiers(self, excluded_targets):
        """Findet alle Frontier-Zellen, die nicht in excluded_targets sind."""
        frontiers = []
        logistics_map = self.supervisor_exploration_logistics_map
        known_passable_rows, known_passable_cols = np.where(logistics_map == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE)
        parent_cell_coords = list(zip(map(int, known_passable_rows), map(int, known_passable_cols)))
        self.model.random.shuffle(parent_cell_coords)

        evaluated = set()
        for parent_pos in parent_cell_coords:
            for dx, dy in self.NEIGHBOR_OFFSETS:
                frontier_pos = (parent_pos[0] + dx, parent_pos[1] + dy)
                if not (0 <= frontier_pos[0] < self.model.grid_width_val and 0 <= frontier_pos[
                    1] < self.model.grid_height_val):
                    continue

                if logistics_map[frontier_pos[0], frontier_pos[
                    1]] == UNKNOWN and frontier_pos not in excluded_targets and frontier_pos not in evaluated:
                    frontiers.append(frontier_pos)
                    evaluated.add(frontier_pos)
        return frontiers

    def _find_corridor_entry_candidate(self, temp_excluded_targets=None):  #
        if temp_excluded_targets is None: temp_excluded_targets = set()  #
        logistics_map = self.supervisor_exploration_logistics_map  #

        candidate_entries = []  #

        passable_rows, passable_cols = np.where(logistics_map == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE)  #
        parent_cells = list(zip(map(int, passable_rows), map(int, passable_cols)))  #
        self.model.random.shuffle(parent_cells)  #

        for p_pos in parent_cells:  #
            shuffled_neighbor_offsets = list(self.NEIGHBOR_OFFSETS)  #
            self.model.random.shuffle(shuffled_neighbor_offsets)  #

            for dx, dy in shuffled_neighbor_offsets:  #
                entry_U_cand = (p_pos[0] + dx, p_pos[1] + dy)  #

                if not (0 <= entry_U_cand[0] < self.model.grid_width_val and \
                        0 <= entry_U_cand[1] < self.model.grid_height_val):  #
                    continue

                if logistics_map[entry_U_cand[0], entry_U_cand[1]] == UNKNOWN and \
                        entry_U_cand not in self.pending_exploration_targets and \
                        entry_U_cand not in temp_excluded_targets and \
                        logistics_map[entry_U_cand[0], entry_U_cand[1]] != SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:  #

                    score = self._calculate_frontier_potential_score(entry_U_cand, logistics_map)  #
                    if score > 0:  #
                        candidate_entries.append({'parent_K': p_pos, 'entry_U': entry_U_cand, 'score': score})  #
                        if len(candidate_entries) > 50: break  #
            if len(candidate_entries) > 50: break  #

        if not candidate_entries:  #
            return None, None  #

        candidate_entries.sort(key=lambda x: x['score'], reverse=True)  #
        best_entry = candidate_entries[0]  #
        return best_entry['parent_K'], best_entry['entry_U']  #

    def _trace_corridor_from_entry(self, entry_U, parent_K, max_length):  #
        corridor_path = [entry_U]  #
        current_trace_pos = entry_U  #

        pref_dx = entry_U[0] - parent_K[0]  #
        pref_dy = entry_U[1] - parent_K[1]  #
        if pref_dx != 0: pref_dx = pref_dx // abs(pref_dx)  #
        if pref_dy != 0: pref_dy = pref_dy // abs(pref_dy)  #

        visited_on_this_trace = {parent_K, entry_U}  #

        for _i in range(max_length - 1):  #
            possible_next_steps = []  #

            straight_ahead = (current_trace_pos[0] + pref_dx, current_trace_pos[1] + pref_dy)  #
            if (0 <= straight_ahead[0] < self.model.grid_width_val and \
                0 <= straight_ahead[1] < self.model.grid_height_val) and \
                    self.supervisor_exploration_logistics_map[straight_ahead[0], straight_ahead[1]] == UNKNOWN and \
                    straight_ahead not in visited_on_this_trace:  #
                possible_next_steps.append({'pos': straight_ahead, 'priority': 0, 'dx': pref_dx, 'dy': pref_dy})  #

            shuffled_offsets = list(self.NEIGHBOR_OFFSETS)  #
            self.model.random.shuffle(shuffled_offsets)  #

            for dx_offset, dy_offset in shuffled_offsets:  #
                if dx_offset == pref_dx and dy_offset == pref_dy and possible_next_steps and possible_next_steps[0][
                    'pos'] == straight_ahead:  #
                    continue  #

                neighbor = (current_trace_pos[0] + dx_offset, current_trace_pos[1] + dy_offset)  #
                if not (0 <= neighbor[0] < self.model.grid_width_val and \
                        0 <= neighbor[1] < self.model.grid_height_val):  #
                    continue

                if self.supervisor_exploration_logistics_map[neighbor[0], neighbor[1]] == UNKNOWN and \
                        neighbor not in visited_on_this_trace:  #
                    turn_severity = abs(dx_offset - pref_dx) + abs(dy_offset - pref_dy)  #
                    priority = 1  #
                    if turn_severity > 1: priority = 2  #
                    possible_next_steps.append(
                        {'pos': neighbor, 'priority': priority, 'dx': dx_offset, 'dy': dy_offset})  #

            if not possible_next_steps: break  #

            possible_next_steps.sort(key=lambda x: x['priority'])  #
            chosen_step_info = possible_next_steps[0]  #
            chosen_step = chosen_step_info['pos']  #

            corridor_path.append(chosen_step)  #
            visited_on_this_trace.add(chosen_step)  #

            pref_dx = chosen_step_info['dx']  #
            pref_dy = chosen_step_info['dy']  #
            current_trace_pos = chosen_step  #

        return corridor_path  #

    def _find_and_plan_corridor_task(self, temp_excluded_targets):  #
        parent_K, entry_U = self._find_corridor_entry_candidate(temp_excluded_targets)  #

        if not parent_K or not entry_U:  #
            return False  #

        corridor_path = self._trace_corridor_from_entry(entry_U, parent_K, self.DEFAULT_CORRIDOR_LENGTH)  #

        if not corridor_path or len(corridor_path) < 2:  #
            return False  #

        new_corridor_task = {  #
            'task_id': f"task_corridor_{next(self.task_id_counter)}",  #
            'type': 'explore_corridor',  #
            'entry_pos': parent_K,  #
            'corridor_path': corridor_path,  #
            'target_pos': parent_K,  #
            'max_length': len(corridor_path),  #
            'status': 'pending_assignment',  #
            'is_initial_hotspot_task': False  #
        }
        self.task_queue.append(new_corridor_task)  #

        for cell_in_path in corridor_path:  #
            if (0 <= cell_in_path[0] < self.model.grid_width_val and \
                    0 <= cell_in_path[1] < self.model.grid_height_val):  #
                self.supervisor_exploration_logistics_map[
                    cell_in_path[0], cell_in_path[1]] = SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED  #
            self.pending_exploration_targets.add(cell_in_path)  #
            if temp_excluded_targets is not None: temp_excluded_targets.add(cell_in_path)  #
        return True  #

    def _calculate_frontier_potential_score(self, pos_to_evaluate, logistics_map_to_use):  #
        q = deque([(pos_to_evaluate, 0)]);  #
        visited_for_potential = {pos_to_evaluate};  #
        potential_unknown_count = 0  #
        if logistics_map_to_use[pos_to_evaluate[0], pos_to_evaluate[1]] == UNKNOWN:  #
            potential_unknown_count = 1  #
        else:
            return 0  #
        head_count = 0  #
        while q and head_count < 250:  #
            head_count += 1;  #
            curr_pos, depth = q.popleft()  #
            if depth >= self.DEFAULT_FRONTIER_SEARCH_DEPTH: continue  #
            for dx, dy in self.NEIGHBOR_OFFSETS:  #
                nx, ny = curr_pos[0] + dx, curr_pos[1] + dy;  #
                next_pos = (nx, ny)  #
                if 0 <= nx < self.model.grid_width_val and 0 <= ny < self.model.grid_height_val and \
                        next_pos not in visited_for_potential and logistics_map_to_use[nx, ny] == UNKNOWN:  #
                    visited_for_potential.add(next_pos);  #
                    potential_unknown_count += 1;  #
                    q.append((next_pos, depth + 1))  #
        return potential_unknown_count  #

    def _find_best_frontier_for_exploration(self, temp_excluded_targets=None):  #
        if temp_excluded_targets is None: temp_excluded_targets = set()  #
        candidate_targets = []  #
        logistics_map = self.supervisor_exploration_logistics_map  #
        current_unknown_logistics_r = np.count_nonzero(logistics_map == UNKNOWN) / self.total_grid_cells  #
        is_breakout_phase = current_unknown_logistics_r >= self.BREAKOUT_PHASE_UNKNOWN_THRESHOLD  #
        ref_point_for_dist = self.home_pos  #
        if self.model.base_deposit_point: ref_point_for_dist = self.model.base_deposit_point  #

        if is_breakout_phase:  #
            all_unknown_on_logistics = [];  #
            u_rows, u_cols = np.where(logistics_map == UNKNOWN)  #
            for r_idx, c_idx in zip(u_rows, u_cols):  #
                pos_tuple = (int(r_idx), int(c_idx))  #
                if pos_tuple not in self.pending_exploration_targets and pos_tuple not in temp_excluded_targets and \
                        pos_tuple not in self.attempted_hotspots and logistics_map[
                    pos_tuple[0], pos_tuple[1]] != SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:  #
                    all_unknown_on_logistics.append(pos_tuple)  #
            self.model.random.shuffle(all_unknown_on_logistics)  #
            for uc_pos in all_unknown_on_logistics[:self.DEFAULT_DEEP_DIVE_CANDIDATE_COUNT]:  #
                if logistics_map[uc_pos[0], uc_pos[1]] != UNKNOWN: continue  #
                potential_score = self._calculate_frontier_potential_score(uc_pos, logistics_map)  #
                if potential_score < 1: continue  #
                dist_to_ref = self._manhattan_distance(uc_pos, ref_point_for_dist)  #
                score = (dist_to_ref * self.BREAKOUT_DISTANCE_WEIGHT) + potential_score  #
                candidate_targets.append(
                    {'pos': uc_pos, 'score': score, 'dist': dist_to_ref, 'pot': potential_score})  #
        else:  #
            evaluated_frontiers = set()  #
            known_passable_rows, known_passable_cols = np.where(logistics_map == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE)  #
            parent_cell_coords = list(zip(map(int, known_passable_rows), map(int, known_passable_cols)));  #
            self.model.random.shuffle(parent_cell_coords)  #
            for parent_pos in parent_cell_coords[:200]:  #
                for dx, dy in self.NEIGHBOR_OFFSETS:  #
                    frontier_pos = (parent_pos[0] + dx, parent_pos[1] + dy)  #
                    if not (0 <= frontier_pos[0] < self.model.grid_width_val and 0 <= frontier_pos[
                        1] < self.model.grid_height_val): continue  #
                    if logistics_map[frontier_pos[0], frontier_pos[1]] == UNKNOWN and \
                            frontier_pos not in self.pending_exploration_targets and \
                            frontier_pos not in temp_excluded_targets and \
                            frontier_pos not in evaluated_frontiers and \
                            logistics_map[
                                frontier_pos[0], frontier_pos[1]] != SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:  #
                        evaluated_frontiers.add(frontier_pos)  #
                        score = float(self._calculate_frontier_potential_score(frontier_pos, logistics_map))  #
                        if score < 1: continue  #
                        candidate_targets.append({'pos': frontier_pos, 'score': score,
                                                  'dist': self._manhattan_distance(frontier_pos, ref_point_for_dist),
                                                  'pot': score})  #
                        if len(candidate_targets) > self.DEFAULT_DEEP_DIVE_CANDIDATE_COUNT * 2: break  #
                if len(candidate_targets) > self.DEFAULT_DEEP_DIVE_CANDIDATE_COUNT * 2: break  #
        if not candidate_targets: return None  #
        candidate_targets.sort(key=lambda f: f['score'], reverse=True)  #
        targets_to_avoid_proximity_to = set(
            self.pending_exploration_targets) | temp_excluded_targets | self.attempted_hotspots  #
        for task_data in list(self.assigned_tasks.values()) + self.task_queue:  #
            if task_data.get('type') == 'explore_area' and task_data.get('target_pos'):
                targets_to_avoid_proximity_to.add(task_data.get('target_pos'))  #
            elif task_data.get('type') == 'explore_corridor' and task_data.get('corridor_path') and task_data.get(
                    'corridor_path'):  #
                targets_to_avoid_proximity_to.add(task_data.get('corridor_path')[0])  #
                targets_to_avoid_proximity_to.add(task_data.get('corridor_path')[-1])  #
        targeted_rows, targeted_cols = np.where(logistics_map == SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED)  #
        for r, c in zip(targeted_rows, targeted_cols): targets_to_avoid_proximity_to.add((r, c))  #
        best_choice_satisfying_separation = None  #
        for cand in candidate_targets:  #
            candidate_pos = cand['pos'];
            is_far_enough = True  #
            if self.MIN_EXPLORE_TARGET_SEPARATION_val > 0:  #
                for existing_target in targets_to_avoid_proximity_to:  #
                    if self._manhattan_distance(candidate_pos,
                                                existing_target) < self.MIN_EXPLORE_TARGET_SEPARATION_val:  #
                        is_far_enough = False;
                        break  #
            if is_far_enough: best_choice_satisfying_separation = candidate_pos; break  #
        return best_choice_satisfying_separation  #

    def _prepare_tasks_for_assignment(self):  #
        if not self.task_queue: return  #
        idle_workers = []  #
        for worker_id, status_data in self.worker_status.items():  #
            is_busy = worker_id in self._tasks_to_assign_to_worker or \
                      (status_data.get('current_task_id') and status_data.get(
                          'current_task_id') in self.assigned_tasks)  #
            if status_data.get('state') == 'IDLE_AT_SUPERVISOR' and not is_busy: idle_workers.append(worker_id)  #
        if not idle_workers: return  #
        self.model.random.shuffle(idle_workers)  #
        for worker_id in idle_workers:  #
            if not self.task_queue: break  #
            task_to_assign = self.task_queue.pop(0)  #
            task_to_assign['status'] = 'assigned_pending_pickup'  #
            self._tasks_to_assign_to_worker[worker_id] = task_to_assign  #

    def receive_report_from_worker(self, worker_id, report_type, data):  #
        self._pending_worker_reports.append({'worker_id': worker_id, 'report_type': report_type, 'data': data})  #

    def request_task_from_worker(self, worker_id):
        """
        Vergibt eine Aufgabe an einen Worker. Kann jetzt auch 'execute_tour'-Aufgaben vergeben.
        """
        self.worker_status.setdefault(worker_id, {})['state'] = 'IDLE_AT_SUPERVISOR'
        self.worker_status[worker_id]['current_task_id'] = None

        if worker_id in self._tasks_to_assign_to_worker:
            task = self._tasks_to_assign_to_worker.pop(worker_id)
            task['status'] = 'assigned'
            task['worker_id'] = worker_id
            self.assigned_tasks[task['task_id']] = task
            self.worker_status[worker_id]['current_task_id'] = task['task_id']

            task_type_assigned = task.get('type')
            log_target_info = "N/A"

            # Markiere die Zellen der Tour als "geplant"
            if task_type_assigned == 'execute_tour':
                tour_steps = task.get('tour_steps', [])
                log_target_info = f"Tour mit {len(tour_steps)} Schritten. Ziele: {[step['target_pos'] for step in tour_steps]}"
                for step in tour_steps:
                    target_for_projection = step.get('target_pos')
                    if target_for_projection:
                        projected_cells = self._get_projected_explored_cells(target_for_projection,
                                                                             self.model.agent_vision_radius_val)
                        for cell_px, cell_py in projected_cells:
                            if 0 <= cell_px < self.model.grid_width_val and 0 <= cell_py < self.model.grid_height_val:
                                if self.supervisor_exploration_logistics_map[cell_px, cell_py] == UNKNOWN:
                                    self.supervisor_exploration_logistics_map[
                                        cell_px, cell_py] = SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED

            # Bestehende Logik für andere Aufgabentypen (unverändert)
            elif task_type_assigned == 'explore_area':
                target_for_projection = task.get('target_pos')
                log_target_info = f"Area at {target_for_projection}"
                if target_for_projection:
                    projected_cells = self._get_projected_explored_cells(target_for_projection,
                                                                         self.model.agent_vision_radius_val)
                    for cell_px, cell_py in projected_cells:
                        if 0 <= cell_px < self.model.grid_width_val and 0 <= cell_py < self.model.grid_height_val:
                            if self.supervisor_exploration_logistics_map[cell_px, cell_py] == UNKNOWN:
                                self.supervisor_exploration_logistics_map[
                                    cell_px, cell_py] = SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED
            elif task_type_assigned == 'explore_corridor':
                corridor_path = task.get('corridor_path', [])
                entry_pos = task.get('entry_pos')
                log_target_info = f"Corridor from {entry_pos} via path (len {len(corridor_path)})"
                if corridor_path:
                    task_viz_info = {'entry_U': corridor_path[0], 'end_U': corridor_path[-1]}
                    self.active_corridors_viz[task['task_id']] = task_viz_info
            elif task_type_assigned == 'collect_resource':
                log_target_info = f"Resource at {task.get('target_pos')}"

            return task

        return None