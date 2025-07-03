# src/supervisor_agent_corridor_only.py
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
    NEIGHBOR_OFFSETS = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
    DEFAULT_FRONTIER_SEARCH_DEPTH = 4
    DEFAULT_DEEP_DIVE_CANDIDATE_COUNT = 50
    BREAKOUT_PHASE_UNKNOWN_THRESHOLD = 0.80
    BREAKOUT_DISTANCE_WEIGHT = 5.0
    DEFAULT_CORRIDOR_LENGTH = 30
    CORRIDOR_ATTEMPT_THRESHOLD = 2
    DEFAULT_TOUR_SIZE = 4
    MIN_TOUR_SIZE = 2
    MIN_CORRIDOR_TOUR_SIZE = 2
    MAX_CORRIDOR_TOUR_SIZE = 4
    # Maximaler Abstand (Manhattan-Distanz) zwischen den Startpunkten der Korridore in einer Tour
    MAX_CORRIDOR_TOUR_DISTANCE = 40
    MIN_BUNDLE_SIZE = 2
    MAX_BUNDLE_SIZE = 4
    # Suchradius für weitere Korridore um den ersten Anker-Korridor herum
    BUNDLE_SEARCH_RADIUS = 40

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
        self.claimed_resources_by_supervisor = {'wood': 0, 'stone': 0}
        self.max_new_collect_tasks_per_planning = self.model.num_agents_val
        self.max_new_explore_tasks_per_planning = self.model.num_agents_val
        self.pending_exploration_targets = set()

        # KORRIGIERTE ZEILE: Verwendet itertools statt der alten Mesa-Methode
        self.task_id_counter = itertools.count(1)

        # Hotspot-Logik für die initiale Verteilung
        self.initial_hotspots_abs = []
        hotspots_config = getattr(self.model, 'SUPERVISOR_INITIAL_EXPLORATION_HOTSPOTS_val', [])
        if hotspots_config:
            for rel_x, rel_y in hotspots_config:
                abs_x = int(self.model.grid_width_val * rel_x)
                abs_y = int(self.model.grid_height_val * rel_y)
                self.initial_hotspots_abs.append((abs_x, abs_y))
        self.model.random.shuffle(self.initial_hotspots_abs)

        self.needs_new_planning = True
        self.active_corridors_viz = {}
        self.no_normal_target_found_count = 0

    def _check_if_exploration_is_needed(self):
        """Prüft, ob Erkundung basierend auf Ressourcenzielen und Kartenabdeckung notwendig ist."""
        unlocated_needed_resources = 0
        for res_type_iter, target_amount_iter in self.resource_goals.items():
            if self.model.base_resources_collected.get(res_type_iter, 0) < target_amount_iter:
                res_const = WOOD_SEEN if res_type_iter == 'wood' else STONE_SEEN
                if np.count_nonzero(self.supervisor_known_map == res_const) == 0:
                    unlocated_needed_resources += 1

        current_unknown_logistics_ratio = np.count_nonzero(
            self.supervisor_exploration_logistics_map == UNKNOWN) / (
                                                      self.model.grid_width_val * self.model.grid_height_val)

        goals_fully_met = all(
            self.model.base_resources_collected.get(res_type, 0) >= target_amount for res_type, target_amount in
            self.resource_goals.items())

        if not goals_fully_met:
            min_ratio = getattr(self.model, 'min_unknown_ratio_for_continued_exploration_cfg', 0.01)
            if unlocated_needed_resources > 0 or current_unknown_logistics_ratio > min_ratio:
                return True
        return False

    def _plan_corridor_bundles(self, max_bundles_to_plan):
        """Plant Pakete ("Bundles") aus 2-4 intelligent ausgewählten, nahegelegenen Korridoren."""
        bundles_planned = 0
        temp_excluded_targets = self._get_all_assigned_targets()

        while bundles_planned < max_bundles_to_plan:
            anchor_candidate = self._find_best_corridor_candidate(temp_excluded_targets)
            if not anchor_candidate:
                break  # Keine Kandidaten mehr gefunden

            anchor_task = self._create_single_corridor_task_dict(anchor_candidate['entry_U'],
                                                                 anchor_candidate['parent_K'])
            if not anchor_task:
                temp_excluded_targets.add(anchor_candidate['entry_U'])
                continue

            corridor_bundle = [anchor_task]
            temp_excluded_targets.add(anchor_candidate['entry_U'])
            for cell in anchor_task['corridor_path']: temp_excluded_targets.add(cell)

            # Finde nahegelegene Kandidaten für das Bündel
            nearby_candidates = self._get_all_corridor_candidates(temp_excluded_targets)
            nearby_candidates.sort(key=lambda c: self._manhattan_distance(c['entry_U'], anchor_candidate['entry_U']))

            for candidate in nearby_candidates:
                if len(corridor_bundle) >= self.MAX_BUNDLE_SIZE: break
                if self._manhattan_distance(candidate['entry_U'],
                                            anchor_candidate['entry_U']) > self.BUNDLE_SEARCH_RADIUS: continue

                task = self._create_single_corridor_task_dict(candidate['entry_U'], candidate['parent_K'])
                if task:
                    corridor_bundle.append(task)
                    temp_excluded_targets.add(candidate['entry_U'])
                    for cell in task['corridor_path']: temp_excluded_targets.add(cell)

            if len(corridor_bundle) >= self.MIN_BUNDLE_SIZE:
                new_bundle_task = {
                    'task_id': f"task_bundle_{next(self.task_id_counter)}",
                    'type': 'execute_corridor_tour',
                    'tour_steps': corridor_bundle,
                    'status': 'pending_assignment'
                }
                self.task_queue.append(new_bundle_task)
                bundles_planned += 1

        return bundles_planned

    def _find_best_corridor_candidate(self, excluded_targets):
        """Findet den einzelnen, besten Startpunkt für einen neuen Korridor."""
        candidates = self._get_all_corridor_candidates(excluded_targets)
        return candidates[0] if candidates else None

    def _get_all_corridor_candidates(self, excluded_targets):
        """Sammelt und bewertet ALLE möglichen Startpunkte (Frontiers) für Korridore."""
        logistics_map = self.supervisor_exploration_logistics_map
        candidate_entries = []
        passable_rows, passable_cols = np.where(logistics_map == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE)
        parent_cells = list(zip(map(int, passable_rows), map(int, passable_cols)))
        self.model.random.shuffle(parent_cells)
        evaluated_frontiers = set()

        for p_pos in parent_cells:
            for dx, dy in self.NEIGHBOR_OFFSETS:
                entry_U_cand = (p_pos[0] + dx, p_pos[1] + dy)
                if not (0 <= entry_U_cand[0] < self.model.grid_width_val and 0 <= entry_U_cand[
                    1] < self.model.grid_height_val): continue
                if logistics_map[entry_U_cand[0], entry_U_cand[
                    1]] != UNKNOWN or entry_U_cand in excluded_targets or entry_U_cand in evaluated_frontiers: continue

                evaluated_frontiers.add(entry_U_cand)
                score = self._calculate_frontier_potential_score(entry_U_cand, logistics_map)
                if score > 0:
                    candidate_entries.append({'parent_K': p_pos, 'entry_U': entry_U_cand, 'score': score})

        candidate_entries.sort(key=lambda x: x['score'], reverse=True)
        return candidate_entries

    def _create_single_corridor_task_dict(self, entry_U, parent_K):
        """Erstellt das Dictionary für eine einzelne Korridor-Aufgabe."""
        corridor_path = self._trace_corridor_from_entry(entry_U, parent_K, self.DEFAULT_CORRIDOR_LENGTH)
        if not corridor_path or len(corridor_path) < 2: return None

        for cell_in_path in corridor_path:
            if (0 <= cell_in_path[0] < self.model.grid_width_val and 0 <= cell_in_path[1] < self.model.grid_height_val):
                self.supervisor_exploration_logistics_map[
                    cell_in_path[0], cell_in_path[1]] = SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED
            self.pending_exploration_targets.add(cell_in_path)

        return {
            'task_id': f"task_single_corridor_{next(self.task_id_counter)}", 'type': 'explore_corridor',
            'entry_pos': parent_K, 'corridor_path': corridor_path,
            'target_pos': parent_K, 'status': 'part_of_bundle'
        }

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
        cells_to_mark = set()
        for dx in range(-vision_radius, vision_radius + 1):
            for dy in range(-vision_radius, vision_radius + 1):
                px, py = target_pos[0] + dx, target_pos[1] + dy
                if 0 <= px < self.model.grid_width_val and 0 <= py < self.model.grid_height_val:
                    cells_to_mark.add((px, py))
        return list(cells_to_mark)

    def _manhattan_distance(self, pos1, pos2):
        if pos1 is None or pos2 is None: return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _distance(self, pos1, pos2):
        """Calculates the Euclidean distance between two points."""
        if pos1 is None or pos2 is None: return float('inf')
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def step(self):
        self._process_pending_reports()
        if self.needs_new_planning:
            self._plan_new_tasks()
            self.needs_new_planning = False
        self._prepare_tasks_for_assignment()

    def _process_pending_reports(self):
        if not self._pending_worker_reports: return
        reports_to_process = list(self._pending_worker_reports)
        self._pending_worker_reports.clear()

        significant_map_change_occurred = False

        for report in reports_to_process:
            worker_id = report['worker_id']
            data = report['data']
            task_id_in_report = data.get('task_id')
            task_details_for_report = self.assigned_tasks.get(task_id_in_report) if task_id_in_report else None
            new_worker_state = data.get('status')

            # Update der Karte, wenn der Worker bei der Basis ist
            map_updates = data.get('map_segment_updates', {})
            if map_updates:
                for pos, reported_state in map_updates.items():
                    px, py = pos
                    if not (0 <= px < self.model.grid_width_val and 0 <= py < self.model.grid_height_val): continue

                    original_known_map_value = self.supervisor_known_map[px, py]
                    new_public_state = original_known_map_value

                    # Logik zur Aktualisierung des Kartenzustands (vereinfacht)
                    if reported_state in [WOOD_SEEN, STONE_SEEN, EMPTY_EXPLORED, BASE_KNOWN]:
                        if original_known_map_value != reported_state:
                            new_public_state = reported_state
                    elif reported_state == RESOURCE_COLLECTED_BY_ME:
                        new_public_state = EMPTY_EXPLORED

                    if self.supervisor_known_map[px, py] != new_public_state:
                        self.supervisor_known_map[px, py] = new_public_state
                        significant_map_change_occurred = True

                    # Logik zur Aktualisierung der Logistikkarte
                    if new_public_state in [EMPTY_EXPLORED, BASE_KNOWN]:
                        if self.supervisor_exploration_logistics_map[px, py] != SUPERVISOR_LOGISTICS_KNOWN_PASSABLE:
                            self.supervisor_exploration_logistics_map[px, py] = SUPERVISOR_LOGISTICS_KNOWN_PASSABLE
                            significant_map_change_occurred = True

            # WICHTIG: Bearbeitung des Aufgabenstatus
            if new_worker_state in ['TASK_COMPLETED', 'TASK_FAILED'] and task_details_for_report:
                self.needs_new_planning = True  # Eine neue Planung ist nötig

                # ENTFERNE ABGESCHLOSSENE AUFGABE AUS DER BUCHFÜHRUNG
                if task_id_in_report in self.assigned_tasks:
                    print(f"LOG: Task {task_id_in_report} completed/failed. Removing from assigned_tasks.")
                    del self.assigned_tasks[task_id_in_report]

                # Bereinige Visualisierungsdaten, falls vorhanden
                if task_id_in_report in self.active_corridors_viz:
                    del self.active_corridors_viz[task_id_in_report]

                # Gib beanspruchte Ressourcen bei Fehlschlag wieder frei
                if task_details_for_report.get('type') == 'collect_resource':
                    res_type = task_details_for_report.get('resource_type')
                    if res_type in self.claimed_resources_by_supervisor:
                        self.claimed_resources_by_supervisor[res_type] -= 1

            if significant_map_change_occurred:
                self.needs_new_planning = True

            # Update des Worker-Status
            self.worker_status.setdefault(worker_id, {})
            self.worker_status[worker_id]['state'] = new_worker_state
            if new_worker_state in ['TASK_COMPLETED', 'TASK_FAILED', 'IDLE_AT_SUPERVISOR']:
                self.worker_status[worker_id]['current_task_id'] = None

    def _is_target_already_assigned_or_queued(self, target_pos_to_check, task_type, resource_type=None):
        for task_data in list(self.assigned_tasks.values()) + self.task_queue:
            current_task_main_target = None
            task_data_type = task_data.get('type')

            if task_data_type == 'explore_area':
                path = task_data.get('path_to_explore')
                if path and path[0]: current_task_main_target = path[0]
            elif task_data_type == 'explore_corridor':
                current_task_main_target = task_data.get('entry_pos')
            elif task_data_type == 'collect_resource':
                current_task_main_target = task_data.get('target_pos')
            # Prüfen Sie auch Touren, um Duplikate zu vermeiden
            elif task_data_type == 'execute_tour':
                tour_steps = task_data.get('tour_steps', [])
                for step in tour_steps:
                    if step.get('target_pos') == target_pos_to_check:
                        return True

            if task_data_type == task_type and current_task_main_target == target_pos_to_check:
                if task_type == 'collect_resource' and task_data.get('resource_type') == resource_type: return True
                if task_type == 'explore_area' or task_type == 'explore_corridor': return True

        if task_type == 'explore_area' or task_type == 'explore_corridor':
            if self.supervisor_exploration_logistics_map[
                target_pos_to_check[0], target_pos_to_check[1]] == SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:
                return True
        return False

    def _plan_new_tasks(self):
        """
        Hauptplanungsfunktion, die proaktiv einen Vorrat an Erkundungsaufgaben erstellt,
        um sicherzustellen, dass alle Worker beschäftigt bleiben.

        KORRIGIERT: Stellt sicher, dass zuerst alle initialen Hotspots als Aufgaben
        erstellt werden, bevor die reguläre Korridorplanung beginnt.
        """
        print(
            f"LOG [{self.role_id} @ Step {self.model.steps}]: Running _plan_new_tasks. Current task queue: {len(self.task_queue)}")

        # --- Ressourcen-Planung (unverändert) ---
        collect_tasks_added_now = 0
        self.claimed_resources_by_supervisor = {'wood': 0, 'stone': 0}
        for task_data in self.assigned_tasks.values():
            if task_data.get('type') == 'collect_resource':
                res_type = task_data.get('resource_type')
                if res_type in self.claimed_resources_by_supervisor:
                    self.claimed_resources_by_supervisor[res_type] += 1

        # Bereinigen der Task-Warteschlange von nicht mehr benötigten Sammelaufgaben
        new_task_queue = []
        for task_data in self.task_queue:
            if task_data.get('type') == 'collect_resource':
                res_type = task_data.get('resource_type')
                needed_goal = self.resource_goals.get(res_type, 0)
                current_claimed = self.claimed_resources_by_supervisor.get(res_type, 0)
                if current_claimed < needed_goal:
                    new_task_queue.append(task_data)
            else:
                new_task_queue.append(task_data)
        self.task_queue = new_task_queue

        # Planung neuer Sammelaufgaben
        resource_priority = sorted(self.resource_goals.keys(), key=lambda r: (
                self.resource_goals[r] - self.model.base_resources_collected.get(r, 0)), reverse=True)
        for res_type in resource_priority:
            if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning: break
            needed_goal = self.resource_goals.get(res_type, 0)
            current_claimed = self.claimed_resources_by_supervisor.get(res_type, 0)
            if current_claimed >= needed_goal: continue
            resource_seen_constant = WOOD_SEEN if res_type == 'wood' else STONE_SEEN
            candidate_patches_coords = []
            rows, cols = np.where(self.supervisor_known_map == resource_seen_constant)
            for r_idx, c_idx in zip(rows, cols): candidate_patches_coords.append((int(r_idx), int(c_idx)))
            self.model.random.shuffle(candidate_patches_coords)
            for patch_pos in candidate_patches_coords:
                if self._is_target_already_assigned_or_queued(patch_pos, 'collect_resource', res_type): continue
                if self.claimed_resources_by_supervisor.get(res_type, 0) >= needed_goal: break
                self.supervisor_known_map[patch_pos[0], patch_pos[1]] = SUPERVISOR_CLAIMED_RESOURCE
                new_collect_task = {'task_id': f"task_collect_{next(self.task_id_counter)}", 'type': 'collect_resource',
                                    'target_pos': patch_pos, 'resource_type': res_type,
                                    'status': 'pending_assignment'}
                self.task_queue.insert(0, new_collect_task)
                collect_tasks_added_now += 1
                if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning: break
            if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning: break

        # --- KORRIGIERTE ZWEI-PHASEN-ERKUNDUNGSPLANUNG ---

        # Phase 1: Hotspot-Verteilung (stellt sicher, dass alle Hotspots zuerst zugewiesen werden)
        if self.initial_hotspots_abs:
            print(f"LOG: Phase 1: Planning Hotspots. Remaining: {len(self.initial_hotspots_abs)}")
            for hotspot_pos in list(self.initial_hotspots_abs):
                if not self._is_target_already_assigned_or_queued(hotspot_pos, 'explore_area'):
                    new_hotspot_task = {
                        'task_id': f"task_hotspot_{next(self.task_id_counter)}",
                        'type': 'explore_area', 'path_to_explore': [hotspot_pos],
                        'is_initial_hotspot_task': True, 'status': 'pending_assignment'
                    }
                    self.task_queue.insert(0, new_hotspot_task)
                    self.initial_hotspots_abs.remove(
                        hotspot_pos)  # Entferne den Hotspot, nachdem eine Aufgabe erstellt wurde
            print(f"LOG: Finished planning all initial Hotspots. Queue size is now {len(self.task_queue)}.")
            # Beende die Planung für diesen Schritt hier, um die Hotspot-Zuweisung zu priorisieren
            return

        # Phase 2: Korridor-Bündel - wird nur ausgeführt, wenn keine Hotspots mehr übrig sind
        print(f"LOG: Phase 2: Considering Corridor Bundles.")
        should_explore_actively = self._check_if_exploration_is_needed()
        if should_explore_actively:
            # Berechne, wie viele Aufgaben fehlen, um einen Vorrat für alle Agenten zu haben
            num_active_and_queued_tasks = len(self.assigned_tasks) + len(self.task_queue)
            num_tasks_to_plan = self.model.num_agents_val - num_active_and_queued_tasks

            print(f"LOG: Active+Queued Tasks: {num_active_and_queued_tasks}. Num Agents: {self.model.num_agents_val}.")
            print(f"LOG: Planning to create {num_tasks_to_plan} new bundle(s) to fill the stockpile.")

            if num_tasks_to_plan > 0:
                tasks_planned = self._plan_corridor_bundles(max_bundles_to_plan=num_tasks_to_plan)
                print(f"LOG: Actually planned {tasks_planned} new corridor bundles.")
                if tasks_planned == 0:
                    self.no_normal_target_found_count += 1
                    print(
                        f"LOG: No new bundles could be planned. No-target-found count: {self.no_normal_target_found_count}")
                else:
                    self.no_normal_target_found_count = 0
        else:
            print("LOG: Exploration is not currently needed.")

    def _get_all_assigned_targets(self):
        """Sammelt alle Ziele, die aktuell zugewiesen oder in der Warteschlange sind."""
        targets = set()
        all_tasks = list(self.assigned_tasks.values()) + self.task_queue + list(self._tasks_to_assign_to_worker.values())

        for task_data in all_tasks:
            task_type = task_data.get('type')
            if task_type == 'execute_corridor_tour':
                for step in task_data.get('tour_steps', []):
                    entry_pos = step.get('entry_pos')
                    if entry_pos: targets.add(entry_pos)
            elif task_data.get('target_pos'):
                targets.add(task_data.get('target_pos'))
        return targets
    # NEUE FUNKTION
    def _plan_corridor_tours(self, max_tours_to_plan):
        """
        Plant "Touren", die aus 2-4 nahegelegenen Korridor-Aufgaben bestehen.
        """
        tours_planned = 0
        # Sammelt alle Ziele, die in diesem Planungsschritt vergeben werden, um Duplikate zu vermeiden
        temp_pending_targets_this_step = self._get_all_assigned_targets()

        while tours_planned < max_tours_to_plan:
            # 1. Finde alle möglichen Startpunkte für Korridore
            all_candidates = self._get_all_corridor_candidates(temp_pending_targets_this_step)
            if not all_candidates:
                break  # Keine Kandidaten mehr, Abbruch

            # 2. Wähle den besten Kandidaten als Anker für die neue Tour
            start_candidate = all_candidates.pop(0)
            start_entry_pos = start_candidate['entry_U']
            temp_pending_targets_this_step.add(start_entry_pos)

            # 3. Erstelle die erste Korridor-Aufgabe für die Tour
            first_corridor_task = self._create_single_corridor_task_dict(start_candidate['entry_U'],
                                                                         start_candidate['parent_K'])
            if not first_corridor_task:
                continue  # Konnte aus irgendeinem Grund keinen Korridor erstellen

            corridor_tour_steps = [first_corridor_task]

            # 4. Suche nach weiteren nahegelegenen Korridoren für die Tour
            # Sortiere die verbleibenden Kandidaten nach ihrer Nähe zum Ankerpunkt
            all_candidates.sort(key=lambda c: self._manhattan_distance(c['entry_U'], start_entry_pos))

            for nearby_candidate in all_candidates:
                if len(corridor_tour_steps) >= self.MAX_CORRIDOR_TOUR_SIZE:
                    break  # Die Tour ist voll

                candidate_pos = nearby_candidate['entry_U']
                if candidate_pos in temp_pending_targets_this_step:
                    continue  # Dieser Punkt wurde bereits in einer anderen Tour verplant

                # Prüfe, ob der Kandidat nahe genug am Ankerpunkt ist
                if self._manhattan_distance(candidate_pos, start_entry_pos) <= self.MAX_CORRIDOR_TOUR_DISTANCE:
                    nearby_corridor_task = self._create_single_corridor_task_dict(nearby_candidate['entry_U'],
                                                                                  nearby_candidate['parent_K'])
                    if nearby_corridor_task:
                        corridor_tour_steps.append(nearby_corridor_task)
                        temp_pending_targets_this_step.add(candidate_pos)

            # 5. Erstelle die finale Tour-Aufgabe, wenn sie groß genug ist
            if len(corridor_tour_steps) >= self.MIN_CORRIDOR_TOUR_SIZE:
                new_tour_task = {
                    'task_id': f"task_corridor_tour_{next(self.task_id_counter)}",
                    'type': 'execute_corridor_tour',  # NEUER AUFGABENTYP
                    'tour_steps': corridor_tour_steps,
                    'status': 'pending_assignment'
                }
                self.task_queue.append(new_tour_task)
                tours_planned += 1
            else:
                # Tour war zu klein, gib die Ziele wieder frei (optional, aber sauber)
                for task_dict in corridor_tour_steps:
                    path = task_dict.get('corridor_path', [])
                    for cell in path:
                        temp_pending_targets_this_step.discard(cell)

        return tours_planned

    # NEUE FUNKTION
    def _get_all_corridor_candidates(self, excluded_targets):
        """
        Sammelt und bewertet ALLE möglichen Startpunkte (Frontiers) für Korridore.
        Gibt eine nach Score sortierte Liste zurück.
        """
        logistics_map = self.supervisor_exploration_logistics_map
        candidate_entries = []

        # Finde alle bekannten, passierbaren Zellen
        passable_rows, passable_cols = np.where(logistics_map == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE)
        parent_cells = list(zip(map(int, passable_rows), map(int, passable_cols)))
        self.model.random.shuffle(parent_cells)

        evaluated_frontiers = set()

        for p_pos in parent_cells:
            for dx, dy in self.NEIGHBOR_OFFSETS:
                entry_U_cand = (p_pos[0] + dx, p_pos[1] + dy)

                # Überspringe ungültige, bereits vergebene oder schon evaluierte Zellen
                if not (0 <= entry_U_cand[0] < self.model.grid_width_val and 0 <= entry_U_cand[
                    1] < self.model.grid_height_val):
                    continue
                if logistics_map[entry_U_cand[0], entry_U_cand[
                    1]] != UNKNOWN or entry_U_cand in excluded_targets or entry_U_cand in evaluated_frontiers:
                    continue

                evaluated_frontiers.add(entry_U_cand)
                score = self._calculate_frontier_potential_score(entry_U_cand, logistics_map)
                if score > 0:
                    candidate_entries.append({'parent_K': p_pos, 'entry_U': entry_U_cand, 'score': score})

        # Sortiere alle gefundenen Kandidaten nach ihrem Potenzial
        candidate_entries.sort(key=lambda x: x['score'], reverse=True)
        return candidate_entries

    # NEUE FUNKTION
    def _create_single_corridor_task_dict(self, entry_U, parent_K):
        """
        Erstellt das Dictionary für eine einzelne Korridor-Aufgabe, ohne es zur Task-Queue hinzuzufügen.
        Gibt das Dictionary oder None bei einem Fehler zurück.
        """
        corridor_path = self._trace_corridor_from_entry(entry_U, parent_K, self.DEFAULT_CORRIDOR_LENGTH)

        if not corridor_path or len(corridor_path) < 2:
            return None

        # Markiere die Zellen als "angezielt", um Überschneidungen zu vermeiden
        for cell_in_path in corridor_path:
            if (0 <= cell_in_path[0] < self.model.grid_width_val and 0 <= cell_in_path[1] < self.model.grid_height_val):
                self.supervisor_exploration_logistics_map[
                    cell_in_path[0], cell_in_path[1]] = SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED
            self.pending_exploration_targets.add(cell_in_path)

        # Erstelle das Task-Dictionary
        corridor_task = {
            'task_id': f"task_single_corridor_{next(self.task_id_counter)}",
            'type': 'explore_corridor',
            'entry_pos': parent_K,
            'corridor_path': corridor_path,
            'target_pos': parent_K,
            'status': 'part_of_tour'  # Interner Status
        }
        return corridor_task

    def _find_corridor_entry_candidate(self, temp_excluded_targets=None):
        if temp_excluded_targets is None: temp_excluded_targets = set()
        logistics_map = self.supervisor_exploration_logistics_map

        candidate_entries = []

        passable_rows, passable_cols = np.where(logistics_map == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE)
        parent_cells = list(zip(map(int, passable_rows), map(int, passable_cols)))
        self.model.random.shuffle(parent_cells)

        for p_pos in parent_cells:
            shuffled_neighbor_offsets = list(self.NEIGHBOR_OFFSETS)
            self.model.random.shuffle(shuffled_neighbor_offsets)

            for dx, dy in shuffled_neighbor_offsets:
                entry_U_cand = (p_pos[0] + dx, p_pos[1] + dy)

                if not (0 <= entry_U_cand[0] < self.model.grid_width_val and \
                        0 <= entry_U_cand[1] < self.model.grid_height_val):
                    continue

                if logistics_map[entry_U_cand[0], entry_U_cand[1]] == UNKNOWN and \
                        entry_U_cand not in self.pending_exploration_targets and \
                        entry_U_cand not in temp_excluded_targets and \
                        logistics_map[entry_U_cand[0], entry_U_cand[1]] != SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:

                    score = self._calculate_frontier_potential_score(entry_U_cand, logistics_map)
                    if score > 0:
                        candidate_entries.append({'parent_K': p_pos, 'entry_U': entry_U_cand, 'score': score})
                        if len(candidate_entries) > 50: break
            if len(candidate_entries) > 50: break

        if not candidate_entries:
            return None, None

        candidate_entries.sort(key=lambda x: x['score'], reverse=True)
        best_entry = candidate_entries[0]
        return best_entry['parent_K'], best_entry['entry_U']

    def _trace_corridor_from_entry(self, entry_U, parent_K, max_length):
        corridor_path = [entry_U]
        current_trace_pos = entry_U

        pref_dx = entry_U[0] - parent_K[0]
        pref_dy = entry_U[1] - parent_K[1]
        if pref_dx != 0: pref_dx = pref_dx // abs(pref_dx)
        if pref_dy != 0: pref_dy = pref_dy // abs(pref_dy)

        visited_on_this_trace = {parent_K, entry_U}

        for _i in range(max_length - 1):
            possible_next_steps = []

            straight_ahead = (current_trace_pos[0] + pref_dx, current_trace_pos[1] + pref_dy)
            if (0 <= straight_ahead[0] < self.model.grid_width_val and \
                0 <= straight_ahead[1] < self.model.grid_height_val) and \
                    self.supervisor_exploration_logistics_map[straight_ahead[0], straight_ahead[1]] == UNKNOWN and \
                    straight_ahead not in visited_on_this_trace:
                possible_next_steps.append({'pos': straight_ahead, 'priority': 0, 'dx': pref_dx, 'dy': pref_dy})

            shuffled_offsets = list(self.NEIGHBOR_OFFSETS)
            self.model.random.shuffle(shuffled_offsets)

            for dx_offset, dy_offset in shuffled_offsets:
                if dx_offset == pref_dx and dy_offset == pref_dy and possible_next_steps and possible_next_steps[0][
                    'pos'] == straight_ahead:
                    continue

                neighbor = (current_trace_pos[0] + dx_offset, current_trace_pos[1] + dy_offset)
                if not (0 <= neighbor[0] < self.model.grid_width_val and \
                        0 <= neighbor[1] < self.model.grid_height_val):
                    continue

                if self.supervisor_exploration_logistics_map[neighbor[0], neighbor[1]] == UNKNOWN and \
                        neighbor not in visited_on_this_trace:
                    turn_severity = abs(dx_offset - pref_dx) + abs(dy_offset - pref_dy)
                    priority = 1
                    if turn_severity > 1: priority = 2
                    possible_next_steps.append(
                        {'pos': neighbor, 'priority': priority, 'dx': dx_offset, 'dy': dy_offset})

            if not possible_next_steps: break

            possible_next_steps.sort(key=lambda x: x['priority'])
            chosen_step_info = possible_next_steps[0]
            chosen_step = chosen_step_info['pos']

            corridor_path.append(chosen_step)
            visited_on_this_trace.add(chosen_step)

            pref_dx = chosen_step_info['dx']
            pref_dy = chosen_step_info['dy']
            current_trace_pos = chosen_step

        return corridor_path

    def _find_and_plan_corridor_task(self, temp_excluded_targets):
        parent_K, entry_U = self._find_corridor_entry_candidate(temp_excluded_targets)

        if not parent_K or not entry_U:
            return False

        corridor_path = self._trace_corridor_from_entry(entry_U, parent_K, self.DEFAULT_CORRIDOR_LENGTH)

        if not corridor_path or len(corridor_path) < 2:
            return False

        new_corridor_task = {
            'task_id': f"task_corridor_{next(self.task_id_counter)}",
            'type': 'explore_corridor',
            'entry_pos': parent_K,
            'corridor_path': corridor_path,
            'target_pos': parent_K,
            'max_length': len(corridor_path),
            'status': 'pending_assignment',
            'is_initial_hotspot_task': False
        }
        self.task_queue.append(new_corridor_task)

        for cell_in_path in corridor_path:
            if (0 <= cell_in_path[0] < self.model.grid_width_val and \
                    0 <= cell_in_path[1] < self.model.grid_height_val):
                self.supervisor_exploration_logistics_map[
                    cell_in_path[0], cell_in_path[1]] = SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED
            self.pending_exploration_targets.add(cell_in_path)
            if temp_excluded_targets is not None: temp_excluded_targets.add(cell_in_path)
        return True

    def _calculate_frontier_potential_score(self, pos_to_evaluate, logistics_map_to_use):
        q = deque([(pos_to_evaluate, 0)])
        visited_for_potential = {pos_to_evaluate}
        potential_unknown_count = 0
        if logistics_map_to_use[pos_to_evaluate[0], pos_to_evaluate[1]] == UNKNOWN:
            potential_unknown_count = 1
        else:
            return 0
        head_count = 0
        while q and head_count < 250:
            head_count += 1
            curr_pos, depth = q.popleft()
            if depth >= self.DEFAULT_FRONTIER_SEARCH_DEPTH: continue
            for dx, dy in self.NEIGHBOR_OFFSETS:
                nx, ny = curr_pos[0] + dx, curr_pos[1] + dy
                next_pos = (nx, ny)
                if 0 <= nx < self.model.grid_width_val and 0 <= ny < self.model.grid_height_val and \
                        next_pos not in visited_for_potential and logistics_map_to_use[nx, ny] == UNKNOWN:
                    visited_for_potential.add(next_pos)
                    potential_unknown_count += 1
                    q.append((next_pos, depth + 1))
        return potential_unknown_count

    def _prepare_tasks_for_assignment(self):
        if not self.task_queue: return
        idle_workers = []
        for worker_id, status_data in self.worker_status.items():
            is_busy = worker_id in self._tasks_to_assign_to_worker or \
                      (status_data.get('current_task_id') and status_data.get(
                          'current_task_id') in self.assigned_tasks)
            if status_data.get('state') == 'IDLE_AT_SUPERVISOR' and not is_busy: idle_workers.append(worker_id)
        if not idle_workers: return
        self.model.random.shuffle(idle_workers)
        for worker_id in idle_workers:
            if not self.task_queue: break
            task_to_assign = self.task_queue.pop(0)
            task_to_assign['status'] = 'assigned_pending_pickup'
            self._tasks_to_assign_to_worker[worker_id] = task_to_assign

    def receive_report_from_worker(self, worker_id, report_type, data):
        self._pending_worker_reports.append({'worker_id': worker_id, 'report_type': report_type, 'data': data})

    def request_task_from_worker(self, worker_id):
        """
        Vergibt eine Aufgabe an einen Worker und inkrementiert den Zähler für beanspruchte Ressourcen,
        wenn es sich um eine Sammelaufgabe handelt.
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

            # Inkrementiere claimed_resources_by_supervisor, wenn eine collect_resource-Aufgabe zugewiesen wird
            if task_type_assigned == 'collect_resource':
                res_type = task.get('resource_type')
                if res_type in self.claimed_resources_by_supervisor:
                    self.claimed_resources_by_supervisor[res_type] += 1
                log_target_info = f"Resource at {task.get('target_pos')} (Type: {res_type})"
                # NEU: Log, wenn eine Ressource als 'beansprucht' markiert wird
                #print(
                    #f"[S_AGENT {self.role_id}] - Assigned task {task['task_id']} ({res_type}) to Worker {worker_id}. Incrementing claimed. Current claimed: {self.claimed_resources_by_supervisor}")


            elif task_type_assigned == 'execute_tour':
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

            return task

        return None