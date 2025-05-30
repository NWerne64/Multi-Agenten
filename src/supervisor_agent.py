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
    NEIGHBOR_OFFSETS = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
    # Diagonale und direkte Nachbarn für Stripe Following (Priorität)
    STRIPE_NEIGHBORS_PRIO = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]

    DEFAULT_FRONTIER_SEARCH_DEPTH = 4
    DEFAULT_DEEP_DIVE_CANDIDATE_COUNT = 50
    BREAKOUT_PHASE_UNKNOWN_THRESHOLD = 0.80
    BREAKOUT_DISTANCE_WEIGHT = 5.0
    DEFAULT_STRIPE_LENGTH = 20  # NEU

    def __init__(self, model, home_pos, role_id_for_display="supervisor_0"):
        # ... (Init bleibt größtenteils gleich wie in deiner letzten guten Version)
        super().__init__(model=model)
        self.role_id = role_id_for_display
        self.home_pos = home_pos
        self.supervisor_known_map = np.full((model.grid_width_val, model.grid_height_val), UNKNOWN, dtype=int)
        self.supervisor_exploration_logistics_map = np.full((model.grid_width_val, model.grid_height_val), UNKNOWN,
                                                            dtype=int)
        if self.home_pos not in self.model.base_coords_list:
            if 0 <= self.home_pos[0] < self.model.grid_width_val and \
                    0 <= self.home_pos[1] < self.model.grid_height_val:
                self.supervisor_known_map[self.home_pos[0], self.home_pos[1]] = EMPTY_EXPLORED
                self.supervisor_exploration_logistics_map[
                    self.home_pos[0], self.home_pos[1]] = SUPERVISOR_LOGISTICS_KNOWN_PASSABLE
        for base_coord in self.model.base_coords_list:
            if 0 <= base_coord[0] < self.model.grid_width_val and \
                    0 <= base_coord[1] < self.model.grid_height_val:
                self.supervisor_known_map[base_coord[0], base_coord[1]] = BASE_KNOWN
                self.supervisor_exploration_logistics_map[
                    base_coord[0], base_coord[1]] = SUPERVISOR_LOGISTICS_KNOWN_PASSABLE
        if self.supervisor_exploration_logistics_map[self.home_pos[0], self.home_pos[1]] == UNKNOWN:
            self.supervisor_exploration_logistics_map[
                self.home_pos[0], self.home_pos[1]] = SUPERVISOR_LOGISTICS_KNOWN_PASSABLE
        if self.supervisor_known_map[self.home_pos[0], self.home_pos[1]] == UNKNOWN:
            self.supervisor_known_map[self.home_pos[0], self.home_pos[1]] = EMPTY_EXPLORED

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

        print(
            f"[S_AGENT {self.role_id}] (Init): Supervisor created. Vision for projection: {self.model.agent_vision_radius_val}, FrontierSearchDepth: {self.DEFAULT_FRONTIER_SEARCH_DEPTH}")

    def _get_projected_explored_cells(self, target_pos, vision_radius):
        # ... (bleibt gleich)
        cells_to_mark = set()
        for dx in range(-vision_radius, vision_radius + 1):
            for dy in range(-vision_radius, vision_radius + 1):
                px, py = target_pos[0] + dx, target_pos[1] + dy
                if 0 <= px < self.model.grid_width_val and 0 <= py < self.model.grid_height_val:
                    cells_to_mark.add((px, py))
        return list(cells_to_mark)

    def _manhattan_distance(self, pos1, pos2):
        # ... (bleibt gleich)
        if pos1 is None or pos2 is None: return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self):
        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): --- Supervisor Step Start ---")
        self._process_pending_reports()
        self._update_task_statuses_and_cleanup()
        self._plan_new_tasks()
        self._prepare_tasks_for_assignment()
        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): --- Supervisor Step End ---")

    def _process_pending_reports(self):
        # ... (Logik wie in deiner letzten Version, mit Logging-Anpassungen)
        if not self._pending_worker_reports: return
        reports_to_process = list(self._pending_worker_reports)
        self._pending_worker_reports.clear()

        for report_idx, report in enumerate(reports_to_process):
            worker_id = report['worker_id'];
            data = report['data']
            task_id_in_report = data.get('task_id')
            task_details_for_report = self.assigned_tasks.get(task_id_in_report) if task_id_in_report else None
            new_worker_state = data.get('status');
            worker_current_pos = data.get('current_pos')

            if new_worker_state in ['TASK_COMPLETED', 'TASK_FAILED']:
                print(
                    f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Report from W{worker_id}: {new_worker_state}, Task: {task_id_in_report}, Pos: {worker_current_pos}")

            self.worker_status.setdefault(worker_id, {});
            if worker_current_pos: self.worker_status[worker_id]['last_pos'] = worker_current_pos
            map_updates_from_worker = data.get('map_segment_updates', {})

            if new_worker_state in ['TASK_COMPLETED', 'TASK_FAILED'] and task_details_for_report:
                task_target_pos = task_details_for_report.get('target_pos')
                task_type = task_details_for_report.get('type')
                is_hotspot_task = task_details_for_report.get('is_initial_hotspot_task', False)

                if is_hotspot_task and task_target_pos: self.attempted_hotspots.add(task_target_pos)

                if task_type == 'explore_area' or task_type == 'explore_stripe':  # Stripe-Task auch hier behandeln
                    if task_target_pos:  # task_target_pos ist path_to_explore[0] oder start_pos für stripe
                        projected_cells = self._get_projected_explored_cells(task_target_pos,
                                                                             self.model.agent_vision_radius_val)
                        # Wenn Stripe, dann ist der *gesamte Stripe-Pfad* potentiell erkundet, nicht nur der Start.
                        # Das ist komplexer, für jetzt behandeln wir nur den Startpunkt/Zielpunkt der Projektion.
                        for cell in projected_cells:
                            if 0 <= cell[0] < self.model.grid_width_val and 0 <= cell[1] < self.model.grid_height_val:
                                if self.supervisor_exploration_logistics_map[
                                    cell[0], cell[1]] == SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:
                                    reported_cell_state = map_updates_from_worker.get(cell)
                                    if reported_cell_state in [EMPTY_EXPLORED, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME]:
                                        self.supervisor_exploration_logistics_map[
                                            cell[0], cell[1]] = SUPERVISOR_LOGISTICS_KNOWN_PASSABLE
                                    else:
                                        self.supervisor_exploration_logistics_map[cell[0], cell[1]] = UNKNOWN

            if worker_current_pos == self.home_pos and map_updates_from_worker:
                # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Worker {worker_id} @ Home. Updating PUBLIC map.")
                for pos, reported_state in map_updates_from_worker.items():
                    px, py = pos;  # Python mag keine Semikolons am Zeilenende für Variablendeklaration
                    if not (0 <= px < self.model.grid_width_val and 0 <= py < self.model.grid_height_val): continue
                    current_public_state = self.supervisor_known_map[px, py];
                    new_public_state = current_public_state
                    if current_public_state == SUPERVISOR_CLAIMED_RESOURCE:
                        if reported_state == RESOURCE_COLLECTED_BY_ME or reported_state == EMPTY_EXPLORED: new_public_state = EMPTY_EXPLORED
                    else:
                        if reported_state == RESOURCE_COLLECTED_BY_ME:
                            new_public_state = EMPTY_EXPLORED
                        elif reported_state in [WOOD_SEEN, STONE_SEEN]:
                            if current_public_state in [UNKNOWN, EMPTY_EXPLORED, BASE_KNOWN] or \
                                    (current_public_state in [WOOD_SEEN,
                                                              STONE_SEEN] and current_public_state != reported_state): new_public_state = reported_state
                        elif reported_state == BASE_KNOWN:
                            if current_public_state in [UNKNOWN, EMPTY_EXPLORED]: new_public_state = reported_state
                        elif reported_state == EMPTY_EXPLORED:
                            if current_public_state != SUPERVISOR_CLAIMED_RESOURCE:
                                if current_public_state in [UNKNOWN, WOOD_SEEN, STONE_SEEN,
                                                            BASE_KNOWN]: new_public_state = reported_state
                    if self.supervisor_known_map[px, py] != new_public_state: self.supervisor_known_map[
                        px, py] = new_public_state
                    if new_public_state == EMPTY_EXPLORED or new_public_state == BASE_KNOWN:
                        if self.supervisor_exploration_logistics_map[px, py] != SUPERVISOR_LOGISTICS_KNOWN_PASSABLE:
                            self.supervisor_exploration_logistics_map[px, py] = SUPERVISOR_LOGISTICS_KNOWN_PASSABLE
                    elif new_public_state == UNKNOWN and self.supervisor_exploration_logistics_map[px, py] != UNKNOWN:
                        self.supervisor_exploration_logistics_map[px, py] = UNKNOWN
                    elif new_public_state in [WOOD_SEEN, STONE_SEEN, SUPERVISOR_CLAIMED_RESOURCE]:
                        if self.supervisor_exploration_logistics_map[px, py] in [UNKNOWN,
                                                                                 SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED]:
                            self.supervisor_exploration_logistics_map[px, py] = SUPERVISOR_LOGISTICS_KNOWN_PASSABLE

            if new_worker_state: self.worker_status[worker_id]['state'] = new_worker_state
            if new_worker_state in ['TASK_COMPLETED', 'TASK_FAILED']:
                self.worker_status[worker_id]['current_task_id'] = None
                if task_details_for_report and task_details_for_report.get('worker_id') == worker_id:
                    task_type = task_details_for_report.get('type');
                    target_pos = task_details_for_report.get('target_pos')
                    if task_type == 'explore_area' or task_type == 'explore_stripe':
                        explore_target_to_check = target_pos
                        path = task_details_for_report.get(
                            'path_to_explore')  # Bei explore_stripe nicht relevant für pending_targets
                        if task_type == 'explore_area' and path and path[0]: explore_target_to_check = path[0]
                        if explore_target_to_check: self.pending_exploration_targets.discard(explore_target_to_check)
                    elif task_type == 'collect_resource' and target_pos and new_worker_state == 'TASK_FAILED':
                        current_map_val_at_target = self.supervisor_known_map[target_pos[0], target_pos[1]]
                        reported_state_at_target = map_updates_from_worker.get(target_pos)
                        if current_map_val_at_target == SUPERVISOR_CLAIMED_RESOURCE:
                            if reported_state_at_target in [WOOD_SEEN, STONE_SEEN]:
                                self.supervisor_known_map[target_pos[0], target_pos[1]] = reported_state_at_target
                            elif reported_state_at_target == EMPTY_EXPLORED or reported_state_at_target == RESOURCE_COLLECTED_BY_ME:
                                self.supervisor_known_map[target_pos[0], target_pos[1]] = EMPTY_EXPLORED
                    if task_id_in_report in self.assigned_tasks: del self.assigned_tasks[task_id_in_report]
            elif new_worker_state == 'IDLE_AT_SUPERVISOR':
                self.worker_status[worker_id]['current_task_id'] = None

    def _update_task_statuses_and_cleanup(self):
        pass

    def _is_target_already_assigned_or_queued(self, target_pos_to_check, task_type, resource_type=None):
        # ... (bleibt gleich)
        for task_data in list(self.assigned_tasks.values()) + self.task_queue:
            current_task_main_target = None
            path = task_data.get('path_to_explore')  # Für explore_area
            start_pos_stripe = task_data.get('start_pos')  # Für explore_stripe

            if task_data.get('type') == 'explore_area' and path and path[0]:
                current_task_main_target = path[0]
            elif task_data.get('type') == 'explore_stripe' and start_pos_stripe:
                current_task_main_target = start_pos_stripe
            elif task_data.get('type') == 'collect_resource':
                current_task_main_target = task_data.get('target_pos')

            if task_data.get('type') == task_type and current_task_main_target == target_pos_to_check:
                if task_type == 'collect_resource' and task_data.get('resource_type') == resource_type: return True
                if task_type == 'explore_area' or task_type == 'explore_stripe': return True

        if task_type == 'explore_area' or task_type == 'explore_stripe':
            if self.supervisor_exploration_logistics_map[
                target_pos_to_check[0], target_pos_to_check[1]] == SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:
                return True
        return False

    def _plan_new_tasks(self):
        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Planning new tasks. TaskQ: {len(self.task_queue)}, Assigned: {len(self.assigned_tasks)}")
        collect_tasks_added_now = 0;
        explore_tasks_added_now = 0;
        hotspots_created_this_step = 0

        if not self.initial_hotspot_planning_complete:
            # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Planning Hotspots. Attempted: {len(self.attempted_hotspots)}/{len(self.initial_hotspots_abs)}")
            for hotspot_pos in self.initial_hotspots_abs:
                if explore_tasks_added_now >= self.max_new_explore_tasks_per_planning: break
                if hotspot_pos not in self.attempted_hotspots:
                    if not self._is_target_already_assigned_or_queued(hotspot_pos, 'explore_area'):
                        new_hotspot_task = {'task_id': f"task_hotspot_{next(self.task_id_counter)}",
                                            'type': 'explore_area', 'path_to_explore': [hotspot_pos],
                                            'status': 'pending_assignment', 'target_pos': hotspot_pos,
                                            'is_initial_hotspot_task': True}
                        self.task_queue.insert(0, new_hotspot_task)
                        self.attempted_hotspots.add(hotspot_pos)  # Sofort als "in Planung/versucht" markieren
                        # Markierung der Logistikkarte erfolgt in request_task_from_worker
                        print(
                            f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Hotspot task for {hotspot_pos} added to queue (now in attempted_hotspots).")
                        explore_tasks_added_now += 1;
                        hotspots_created_this_step += 1
            if len(self.attempted_hotspots) >= len(self.initial_hotspots_abs):
                self.initial_hotspot_planning_complete = True
                print(
                    f"[S_AGENT {self.role_id}] (Step {self.model.steps}): All initial hotspots are now in 'attempted' set. Initial hotspot planning complete.")

        # (Sammelaufgaben Logik bleibt gleich)
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
                # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Creating collect task for {res_type} at {patch_pos}.")
                self.supervisor_known_map[patch_pos[0], patch_pos[1]] = SUPERVISOR_CLAIMED_RESOURCE
                new_collect_task = {'task_id': f"task_collect_{next(self.task_id_counter)}", 'type': 'collect_resource',
                                    'target_pos': patch_pos, 'resource_type': res_type, 'status': 'pending_assignment'}
                self.task_queue.insert(0, new_collect_task)
                collect_tasks_added_now += 1
                if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning: break
            if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning: break

        unlocated_needed_resources = 0
        for res_type_iter, target_amount_iter in self.resource_goals.items():
            if self.model.base_resources_collected.get(res_type_iter, 0) < target_amount_iter:
                res_const = WOOD_SEEN if res_type_iter == 'wood' else STONE_SEEN
                if np.count_nonzero(self.supervisor_known_map == res_const) == 0: unlocated_needed_resources += 1

        current_unknown_logistics_ratio = np.count_nonzero(
            self.supervisor_exploration_logistics_map == UNKNOWN) / self.total_grid_cells
        goals_fully_met = all(self.model.base_resources_collected.get(res_type, 0) >= target_amount
                              for res_type, target_amount in self.resource_goals.items())
        should_explore_actively = False
        if not goals_fully_met:
            if unlocated_needed_resources > 0 or current_unknown_logistics_ratio > self.min_unknown_ratio_for_continued_exploration_val:
                should_explore_actively = True

        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Plan Regular/Stripe Explo: GoalsMet={goals_fully_met}, UnlocatedRes={unlocated_needed_resources}, UnknownLogiRatio={current_unknown_logistics_ratio:.2f}, ShouldExplore={should_explore_actively}")

        if self.initial_hotspot_planning_complete or hotspots_created_this_step == 0:
            temp_max_explore_tasks = self.max_new_explore_tasks_per_planning
            temp_pending_targets_this_step = set()

            no_frontier_target_found_count = 0  # Zähler für vergebliche Frontier-Suchen

            while should_explore_actively and explore_tasks_added_now < temp_max_explore_tasks:
                exploration_target_cell = self._find_best_frontier_for_exploration(temp_pending_targets_this_step)

                if exploration_target_cell:
                    no_frontier_target_found_count = 0  # Zurücksetzen, da Ziel gefunden
                    if not self._is_target_already_assigned_or_queued(exploration_target_cell, 'explore_area'):
                        new_explore_task = {
                            'task_id': f"task_explore_{next(self.task_id_counter)}", 'type': 'explore_area',
                            'path_to_explore': [exploration_target_cell], 'status': 'pending_assignment',
                            'target_pos': exploration_target_cell, 'is_initial_hotspot_task': False}
                        self.task_queue.append(new_explore_task)
                        self.pending_exploration_targets.add(exploration_target_cell)
                        temp_pending_targets_this_step.add(exploration_target_cell)
                        explore_tasks_added_now += 1
                        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Regular explore task for {exploration_target_cell} added.")
                    else:
                        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Best regular target {exploration_target_cell} already assigned/targeted. Adding to temp_excluded.")
                        temp_excluded_targets.add(exploration_target_cell)
                        continue
                else:  # Keine "normalen" Frontiers gefunden
                    no_frontier_target_found_count += 1
                    print(
                        f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No regular/breakout targets from _find_best_frontier. Attempting stripe. (Count: {no_frontier_target_found_count})")
                    # Wenn wiederholt keine Frontiers gefunden werden, versuche Stripe Exploration
                    if no_frontier_target_found_count >= 2:  # Schwelle, bevor Stripe-Suche beginnt
                        print(
                            f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Trying to plan stripe exploration tasks.")
                        stripe_start_point = self._find_stripe_start_point(temp_pending_targets_this_step)
                        if stripe_start_point:
                            if not self._is_target_already_assigned_or_queued(stripe_start_point,
                                                                              'explore_stripe'):  # Neuer Typ
                                new_stripe_task = {
                                    'task_id': f"task_stripe_{next(self.task_id_counter)}",
                                    'type': 'explore_stripe',
                                    'start_pos': stripe_start_point,  # Worker startet hier den Stripe
                                    'target_pos': stripe_start_point,  # Für Kompatibilität mit is_target_already...
                                    'max_length': self.DEFAULT_STRIPE_LENGTH,
                                    'status': 'pending_assignment',
                                    'is_initial_hotspot_task': False  # Kein Hotspot
                                }
                                self.task_queue.append(new_stripe_task)  # Nicht ganz vorne einreihen
                                self.pending_exploration_targets.add(stripe_start_point)  # Start als Zielpunkt merken
                                temp_pending_targets_this_step.add(stripe_start_point)
                                explore_tasks_added_now += 1
                                no_frontier_target_found_count = 0  # Zurücksetzen nach erfolgreichem Stripe-Task
                                print(
                                    f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Stripe explore task for start {stripe_start_point} added.")
                                continue  # Nächste Iteration der while-Schleife für weitere Tasks
                            else:
                                temp_excluded_targets.add(
                                    stripe_start_point)  # Nicht nochmal diesen Stripe-Start versuchen
                        else:  # Auch kein Stripe-Startpunkt gefunden
                            print(
                                f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No stripe start points found. Breaking explore planning.")
                            break
                    if no_frontier_target_found_count < 2:  # Wenn erster Versuch scheitert, aber noch keine Stripe-Suche
                        print(
                            f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No regular targets, but not trying stripe yet ({no_frontier_target_found_count}).")
                        break  # Für diesen Step keine weiteren Explo-Tasks, wenn keine Frontiers da sind

                    if no_frontier_target_found_count >= 2 and not stripe_start_point:  # Wenn 2x Frontier fehlgeschlagen UND kein Stripe gefunden
                        break  # Keine Ziele mehr

    def _find_stripe_start_point(self, temp_excluded_targets=None):
        """Sucht einen geeigneten Startpunkt für eine Stripe-Exploration."""
        if temp_excluded_targets is None: temp_excluded_targets = set()
        logistics_map = self.supervisor_exploration_logistics_map
        candidate_stripe_starts = []

        # Suche UNKNOWN Zellen, die an KNOWN_PASSABLE grenzen (Eingänge zu Streifen)
        passable_rows, passable_cols = np.where(logistics_map == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE)
        parent_cells = list(zip(map(int, passable_rows), map(int, passable_cols)))
        self.model.random.shuffle(parent_cells)

        for p_pos in parent_cells:
            for dx, dy in self.NEIGHBOR_OFFSETS:  # Direkte Nachbarn zuerst
                s_x, s_y = p_pos[0] + dx, p_pos[1] + dy
                stripe_start_cand = (s_x, s_y)
                if 0 <= s_x < self.model.grid_width_val and 0 <= s_y < self.model.grid_height_val and \
                        logistics_map[s_x, s_y] == UNKNOWN and \
                        stripe_start_cand not in self.pending_exploration_targets and \
                        stripe_start_cand not in temp_excluded_targets and \
                        logistics_map[s_x, s_y] != SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:
                    # Bevorzuge Zellen, die wenige bekannte Nachbarn haben (typisch für Korridore)
                    known_neighbors = 0
                    unknown_neighbors = 0
                    for ndx, ndy in self.NEIGHBOR_OFFSETS:
                        nnx, nny = s_x + ndx, s_y + ndy
                        if 0 <= nnx < self.model.grid_width_val and 0 <= nny < self.model.grid_height_val:
                            if logistics_map[nnx, nny] == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE:
                                known_neighbors += 1
                            elif logistics_map[nnx, nny] == UNKNOWN:
                                unknown_neighbors += 1

                    # Einfache Heuristik: Wenige bekannte Nachbarn, einige unbekannte Nachbarn
                    if 1 <= unknown_neighbors <= 4 and known_neighbors <= 3:  # Werte sind experimentell
                        candidate_stripe_starts.append(
                            {'pos': stripe_start_cand, 'score': unknown_neighbors - known_neighbors})

        if not candidate_stripe_starts:
            print(
                f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No suitable stripe start points found directly adjacent to known passable.")
            # Fallback: Nimm eine beliebige UNKNOWN Zelle, die nicht TARGETED ist, als Notfall-Stripe-Start
            u_rows, u_cols = np.where(logistics_map == UNKNOWN)
            fallback_starts = []
            for r, c in zip(u_rows, u_cols):
                pos = (int(r), int(c))
                if pos not in self.pending_exploration_targets and \
                        pos not in temp_excluded_targets and \
                        logistics_map[pos[0], pos[1]] != SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:
                    fallback_starts.append({'pos': pos, 'score': self._calculate_frontier_potential_score(pos,
                                                                                                          logistics_map)})  # Nutze Potential
            if fallback_starts:
                fallback_starts.sort(key=lambda x: x['score'], reverse=True)
                print(
                    f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Found {len(fallback_starts)} fallback stripe starts. Best: {fallback_starts[0]['pos']}")
                return fallback_starts[0]['pos']
            return None

        candidate_stripe_starts.sort(key=lambda x: x['score'], reverse=True)  # Höherer Score (mehr U, weniger K) besser
        print(
            f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Found {len(candidate_stripe_starts)} potential stripe starts. Best: {candidate_stripe_starts[0]['pos']}")
        return candidate_stripe_starts[0]['pos']

    def _calculate_frontier_potential_score(self, pos_to_evaluate, logistics_map_to_use):
        # ... (bleibt gleich)
        q = deque([(pos_to_evaluate, 0)]);
        visited_for_potential = {pos_to_evaluate};
        potential_unknown_count = 0
        if logistics_map_to_use[pos_to_evaluate[0], pos_to_evaluate[1]] == UNKNOWN:
            potential_unknown_count = 1
        else:
            return 0
        head_count = 0
        while q and head_count < 250:
            head_count += 1;
            curr_pos, depth = q.popleft()
            if depth >= self.DEFAULT_FRONTIER_SEARCH_DEPTH: continue
            for dx, dy in self.NEIGHBOR_OFFSETS:
                nx, ny = curr_pos[0] + dx, curr_pos[1] + dy;
                next_pos = (nx, ny)
                if 0 <= nx < self.model.grid_width_val and 0 <= ny < self.model.grid_height_val and \
                        next_pos not in visited_for_potential and logistics_map_to_use[nx, ny] == UNKNOWN:
                    visited_for_potential.add(next_pos);
                    potential_unknown_count += 1;
                    q.append((next_pos, depth + 1))
        return potential_unknown_count

    def _find_best_frontier_for_exploration(self, temp_excluded_targets=None):
        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Finding best exploration target (non-stripe)...")
        # ... (Rest der Methode bleibt strukturell gleich, nutzt Logistikkarte, Breakout-Logik etc.)
        # Wichtig: Die Logs für Kandidaten sollten jetzt weniger häufig sein oder nur Top-Kandidaten zeigen.
        if temp_excluded_targets is None: temp_excluded_targets = set()
        candidate_targets = []
        logistics_map = self.supervisor_exploration_logistics_map
        current_unknown_logistics_r = np.count_nonzero(logistics_map == UNKNOWN) / self.total_grid_cells
        is_breakout_phase = current_unknown_logistics_r >= self.BREAKOUT_PHASE_UNKNOWN_THRESHOLD
        ref_point_for_dist = self.home_pos
        if self.model.base_deposit_point: ref_point_for_dist = self.model.base_deposit_point

        if is_breakout_phase:
            # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Breakout phase. Searching deep dive targets.")
            all_unknown_on_logistics = []
            u_rows, u_cols = np.where(logistics_map == UNKNOWN)
            for r_idx, c_idx in zip(u_rows, u_cols):
                pos_tuple = (int(r_idx), int(c_idx))
                if pos_tuple not in self.pending_exploration_targets and \
                        pos_tuple not in temp_excluded_targets and \
                        pos_tuple not in self.attempted_hotspots and \
                        logistics_map[pos_tuple[0], pos_tuple[1]] != SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:
                    all_unknown_on_logistics.append(pos_tuple)
            self.model.random.shuffle(all_unknown_on_logistics)
            for uc_pos in all_unknown_on_logistics[:self.DEFAULT_DEEP_DIVE_CANDIDATE_COUNT]:
                if logistics_map[uc_pos[0], uc_pos[1]] != UNKNOWN: continue
                potential_score = self._calculate_frontier_potential_score(uc_pos, logistics_map)
                if potential_score < 1: continue
                dist_to_ref = self._manhattan_distance(uc_pos, ref_point_for_dist)
                score = (dist_to_ref * self.BREAKOUT_DISTANCE_WEIGHT) + potential_score
                candidate_targets.append({'pos': uc_pos, 'score': score, 'dist': dist_to_ref, 'pot': potential_score})
        else:
            # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Regular phase. Searching edge frontiers.")
            evaluated_frontiers = set()
            known_passable_rows, known_passable_cols = np.where(logistics_map == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE)
            parent_cell_coords = list(zip(map(int, known_passable_rows), map(int, known_passable_cols)))
            self.model.random.shuffle(parent_cell_coords)
            for parent_pos in parent_cell_coords:
                for dx, dy in self.NEIGHBOR_OFFSETS:
                    frontier_pos = (parent_pos[0] + dx, parent_pos[1] + dy)
                    if not (0 <= frontier_pos[0] < self.model.grid_width_val and 0 <= frontier_pos[
                        1] < self.model.grid_height_val): continue
                    if logistics_map[frontier_pos[0], frontier_pos[1]] == UNKNOWN and \
                            frontier_pos not in self.pending_exploration_targets and \
                            frontier_pos not in temp_excluded_targets and \
                            frontier_pos not in evaluated_frontiers and \
                            logistics_map[
                                frontier_pos[0], frontier_pos[1]] != SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:
                        evaluated_frontiers.add(frontier_pos)
                        score = float(self._calculate_frontier_potential_score(frontier_pos, logistics_map))
                        if score < 1: continue
                        candidate_targets.append({'pos': frontier_pos, 'score': score,
                                                  'dist': self._manhattan_distance(frontier_pos, ref_point_for_dist),
                                                  'pot': score})

        if not candidate_targets:
            # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No candidates from primary logic in _find_best_frontier.")
            return None

        candidate_targets.sort(key=lambda f: f['score'], reverse=True)
        if candidate_targets:  # Log nur wenn Kandidaten da sind
            print(
                f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Top candidate (before sep): {candidate_targets[0]['pos']} Score: {candidate_targets[0]['score']:.2f} (Pot: {candidate_targets[0]['pot']})")

        targets_to_avoid_proximity_to = set(
            self.pending_exploration_targets) | temp_excluded_targets | self.attempted_hotspots
        for task_data in list(self.assigned_tasks.values()) + self.task_queue:
            if task_data.get('type') == 'explore_area' and task_data.get('target_pos'):
                targets_to_avoid_proximity_to.add(task_data.get('target_pos'))
            elif task_data.get('type') == 'explore_stripe' and task_data.get('start_pos'):
                targets_to_avoid_proximity_to.add(task_data.get('start_pos'))

        targeted_rows, targeted_cols = np.where(logistics_map == SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED)
        for r, c in zip(targeted_rows, targeted_cols): targets_to_avoid_proximity_to.add((r, c))

        best_choice_satisfying_separation = None
        for cand in candidate_targets:
            candidate_pos = cand['pos'];
            is_far_enough = True
            if self.MIN_EXPLORE_TARGET_SEPARATION_val > 0:
                for existing_target in targets_to_avoid_proximity_to:
                    if self._manhattan_distance(candidate_pos,
                                                existing_target) < self.MIN_EXPLORE_TARGET_SEPARATION_val:
                        is_far_enough = False;
                        break
            if is_far_enough:
                best_choice_satisfying_separation = candidate_pos;
                break

        if best_choice_satisfying_separation:
            print(
                f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Chosen explore target (met separation): {best_choice_satisfying_separation}")
            return best_choice_satisfying_separation
        elif candidate_targets:
            chosen_target = candidate_targets[0]['pos']
            print(
                f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No target met separation criteria. Choosing highest score: {chosen_target}")
            return chosen_target

        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No suitable target from _find_best_frontier after all filters.")
        return None

    def _prepare_tasks_for_assignment(self):
        # (Logik bleibt gleich, Logging reduziert)
        if not self.task_queue: return
        idle_workers = []
        for worker_id, status_data in self.worker_status.items():
            is_busy = worker_id in self._tasks_to_assign_to_worker or \
                      (status_data.get('current_task_id') and status_data.get('current_task_id') in self.assigned_tasks)
            if status_data.get('state') == 'IDLE_AT_SUPERVISOR' and not is_busy:
                idle_workers.append(worker_id)
        if not idle_workers: return
        self.model.random.shuffle(idle_workers)
        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Preparing tasks for {len(idle_workers)} idle workers. TaskQ size: {len(self.task_queue)}")
        for worker_id in idle_workers:
            if not self.task_queue: break
            task_to_assign = self.task_queue.pop(0)
            task_to_assign['status'] = 'assigned_pending_pickup'
            self._tasks_to_assign_to_worker[worker_id] = task_to_assign
            # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Task {task_to_assign.get('task_id', 'N/A')} PREPARED for W{worker_id}.")

    def receive_report_from_worker(self, worker_id, report_type, data):
        self._pending_worker_reports.append({'worker_id': worker_id, 'report_type': report_type, 'data': data})

    def request_task_from_worker(self, worker_id):
        self.worker_status.setdefault(worker_id, {})['state'] = 'IDLE_AT_SUPERVISOR'
        self.worker_status[worker_id]['current_task_id'] = None
        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Worker {worker_id} requesting task.")

        if worker_id in self._tasks_to_assign_to_worker:
            task = self._tasks_to_assign_to_worker.pop(worker_id)
            task['status'] = 'assigned';
            task['worker_id'] = worker_id
            self.assigned_tasks[task['task_id']] = task
            self.worker_status[worker_id]['current_task_id'] = task['task_id']

            task_type_assigned = task.get('type')
            target_for_projection = None
            if task_type_assigned == 'explore_area':
                target_for_projection = task.get('target_pos')
            elif task_type_assigned == 'explore_stripe':
                target_for_projection = task.get('start_pos')

            if target_for_projection:
                projected_cells = self._get_projected_explored_cells(target_for_projection,
                                                                     self.model.agent_vision_radius_val)
                cells_marked_targeted = 0
                for cell_px, cell_py in projected_cells:
                    if self.supervisor_exploration_logistics_map[cell_px, cell_py] == UNKNOWN:
                        self.supervisor_exploration_logistics_map[
                            cell_px, cell_py] = SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED
                        cells_marked_targeted += 1
                if cells_marked_targeted > 0:
                    print(
                        f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Marked {cells_marked_targeted} cells around {target_for_projection} as TARGETED on logistics for W{worker_id}.")

            print(
                f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Assigning task {task['task_id']} (Type: {task_type_assigned}, Target: {target_for_projection if target_for_projection else task.get('target_pos')}) to W{worker_id}.")
            return task

        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No task for W{worker_id} (tasks_to_assign empty).")
        return None