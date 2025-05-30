# src/supervisor_agent.py
import mesa
import numpy as np
import itertools
from collections import deque

from src.config import (
    UNKNOWN, EMPTY_EXPLORED, BASE_KNOWN,
    WOOD_SEEN, STONE_SEEN, RESOURCE_COLLECTED_BY_ME,
    SUPERVISOR_CLAIMED_RESOURCE,
    MIN_EXPLORE_TARGET_SEPARATION,  # Sollte in config.py auf 20 gesetzt werden
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
    DEFAULT_STRIPE_LENGTH = 30
    STRIPE_ATTEMPT_THRESHOLD = 2  # Nach 2 fehlgeschlagenen normalen Suchen -> Stripe

    def __init__(self, model, home_pos, role_id_for_display="supervisor_0"):
        super().__init__(model=model)
        self.role_id = role_id_for_display
        self.home_pos = home_pos

        self.supervisor_known_map = np.full((model.grid_width_val, model.grid_height_val), UNKNOWN, dtype=int)
        self.supervisor_exploration_logistics_map = np.full((model.grid_width_val, model.grid_height_val), UNKNOWN,
                                                            dtype=int)

        # Initiale bekannte Zellen (Basis, Supervisor-Position)
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
        self.attempted_hotspots = set()  # Hotspots, für die schon ein Task erstellt wurde

        self.total_grid_cells = self.model.grid_width_val * self.model.grid_height_val
        self.initial_hotspot_planning_complete = False

        self.no_normal_target_found_count = 0  # WICHTIG: Als Instanzattribut

        print(
            f"[S_AGENT {self.role_id}] (Init): Supervisor created. MIN_EXPLORE_TARGET_SEPARATION: {self.MIN_EXPLORE_TARGET_SEPARATION_val}, DEFAULT_STRIPE_LENGTH: {self.DEFAULT_STRIPE_LENGTH}")

    def _get_projected_explored_cells(self, target_pos, vision_radius):
        # ... (bleibt gleich)
        cells_to_mark = set();
        for dx in range(-vision_radius, vision_radius + 1):
            for dy in range(-vision_radius, vision_radius + 1):
                px, py = target_pos[0] + dx, target_pos[1] + dy
                if 0 <= px < self.model.grid_width_val and 0 <= py < self.model.grid_height_val:
                    cells_to_mark.add((px, py))
        return list(cells_to_mark)

    def _manhattan_distance(self, pos1, pos2):
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
        # ... (Logik wie in der letzten Version, Logs reduziert)
        if not self._pending_worker_reports: return
        reports_to_process = list(self._pending_worker_reports);
        self._pending_worker_reports.clear()
        for report in reports_to_process:
            worker_id = report['worker_id'];
            data = report['data']
            task_id_in_report = data.get('task_id');
            task_details_for_report = self.assigned_tasks.get(task_id_in_report) if task_id_in_report else None
            new_worker_state = data.get('status');
            worker_current_pos = data.get('current_pos')

            if new_worker_state in ['TASK_COMPLETED', 'TASK_FAILED'] and task_details_for_report:
                print(
                    f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Report from W{worker_id}: {new_worker_state}, Task: {task_id_in_report}, Target: {task_details_for_report.get('target_pos') or task_details_for_report.get('start_pos')}")

            self.worker_status.setdefault(worker_id, {});
            if worker_current_pos: self.worker_status[worker_id]['last_pos'] = worker_current_pos
            map_updates = data.get('map_segment_updates', {})

            if new_worker_state in ['TASK_COMPLETED', 'TASK_FAILED'] and task_details_for_report:
                task_target_pos = task_details_for_report.get('target_pos') or task_details_for_report.get('start_pos')
                task_type = task_details_for_report.get('type')
                is_hotspot_task = task_details_for_report.get('is_initial_hotspot_task', False)

                # Hotspot wird als "versucht" markiert, wenn der Task dafür zurückkommt.
                # Die Neuplanung wird durch `not in self.attempted_hotspots` in `_plan_new_tasks` gesteuert.
                if is_hotspot_task and task_target_pos:
                    # self.attempted_hotspots.add(task_target_pos) # Wird jetzt in _plan_new_tasks gesetzt
                    pass

                if task_type == 'explore_area' or task_type == 'explore_stripe':
                    if task_target_pos:
                        projected_cells = self._get_projected_explored_cells(task_target_pos,
                                                                             self.model.agent_vision_radius_val)
                        for cell in projected_cells:
                            if 0 <= cell[0] < self.model.grid_width_val and 0 <= cell[1] < self.model.grid_height_val:
                                if self.supervisor_exploration_logistics_map[
                                    cell[0], cell[1]] == SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:
                                    reported_state = map_updates.get(cell)
                                    if reported_state in [EMPTY_EXPLORED, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME]:
                                        self.supervisor_exploration_logistics_map[
                                            cell[0], cell[1]] = SUPERVISOR_LOGISTICS_KNOWN_PASSABLE
                                    else:
                                        self.supervisor_exploration_logistics_map[cell[0], cell[1]] = UNKNOWN

            if worker_current_pos == self.home_pos and map_updates:
                # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Worker {worker_id} @ Home. Updating PUBLIC map.")
                for pos, reported_state in map_updates.items():
                    px, py = pos;
                    if not (0 <= px < self.model.grid_width_val and 0 <= py < self.model.grid_height_val): continue
                    current_public_state = self.supervisor_known_map[px, py];
                    new_public_state = current_public_state
                    # ... (Update-Logik für Public Map bleibt gleich) ...
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
                    if new_public_state == EMPTY_EXPLORED or new_public_state == BASE_KNOWN:  # Sync to logistics
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
                        explore_target = task_details_for_report.get('target_pos') or task_details_for_report.get(
                            'start_pos')
                        if explore_target: self.pending_exploration_targets.discard(explore_target)
                    elif task_type == 'collect_resource' and target_pos and new_worker_state == 'TASK_FAILED':  # ... (Collect Fail Logic)
                        current_map_val_at_target = self.supervisor_known_map[target_pos[0], target_pos[1]]
                        reported_state_at_target = map_updates.get(target_pos)
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
            current_task_main_target = None;
            path = task_data.get('path_to_explore');
            start_pos_stripe = task_data.get('start_pos')
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
            if self.supervisor_exploration_logistics_map[target_pos_to_check[0], target_pos_to_check[
                1]] == SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED: return True
        return False

    def _plan_new_tasks(self):
        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Planning. TaskQ: {len(self.task_queue)}, Assigned: {len(self.assigned_tasks)}")
        collect_tasks_added_now = 0;
        explore_tasks_added_now = 0;
        hotspots_created_this_step = 0

        # --- Phase 0: Initiale Hotspot-Explorationsaufgaben ---
        if not self.initial_hotspot_planning_complete:
            # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Hotspot Planning. Attempted: {len(self.attempted_hotspots)}/{len(self.initial_hotspots_abs)}")
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
                        self.task_queue.insert(0, new_hotspot_task);
                        self.attempted_hotspots.add(hotspot_pos)  # Sofort als "versucht/geplant" markieren
                        explore_tasks_added_now += 1;
                        hotspots_created_this_step += 1
                        print(
                            f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Hotspot task for {hotspot_pos} added to queue (now in attempted_hotspots).")

            if unattempted_hotspots_for_planning == 0 and len(self.initial_hotspots_abs) > 0:
                self.initial_hotspot_planning_complete = True
                print(
                    f"[S_AGENT {self.role_id}] (Step {self.model.steps}): All initial hotspots have been added to attempted_hotspots. Initial hotspot planning phase complete.")

        # --- Phase 1: Sammelaufgaben (Logik bleibt gleich) ---
        # ... (Code für Sammelaufgaben hier einfügen)
        resource_priority = sorted(self.resource_goals.keys(), key=lambda r: (
                    self.resource_goals[r] - self.model.base_resources_collected.get(r, 0)), reverse=True)
        for res_type in resource_priority:  # ... (Rest der Sammellogik)
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

        # --- Phase 2: Reguläre Explorationsaufgaben & Stripe Exploration ---
        unlocated_needed_resources = 0  # ... (wie gehabt)
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

        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Plan Explo: GoalsMet={goals_fully_met}, UnlocRes={unlocated_needed_resources}, UnkLogRatio={current_unknown_logistics_ratio:.2f}, ShouldExplo={should_explore_actively}")

        if self.initial_hotspot_planning_complete or hotspots_created_this_step == 0:
            temp_max_explore_tasks = self.max_new_explore_tasks_per_planning
            temp_pending_targets_this_step = set()

            while should_explore_actively and explore_tasks_added_now < temp_max_explore_tasks:
                exploration_target_cell = self._find_best_frontier_for_exploration(temp_pending_targets_this_step)

                if exploration_target_cell:
                    self.no_normal_target_found_count = 0
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
                        temp_excluded_targets.add(exploration_target_cell)
                        continue
                else:
                    self.no_normal_target_found_count += 1
                    print(
                        f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No regular/breakout targets found (Attempt count: {self.no_normal_target_found_count}).")
                    if self.no_normal_target_found_count >= self.STRIPE_ATTEMPT_THRESHOLD:
                        print(
                            f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Threshold reached, trying to plan stripe exploration tasks.")
                        stripe_start_point = self._find_stripe_start_point(temp_pending_targets_this_step)
                        if stripe_start_point:
                            if not self._is_target_already_assigned_or_queued(stripe_start_point, 'explore_stripe'):
                                new_stripe_task = {
                                    'task_id': f"task_stripe_{next(self.task_id_counter)}", 'type': 'explore_stripe',
                                    'start_pos': stripe_start_point, 'target_pos': stripe_start_point,
                                    'max_length': self.DEFAULT_STRIPE_LENGTH, 'status': 'pending_assignment',
                                    'is_initial_hotspot_task': False}
                                self.task_queue.append(new_stripe_task)
                                self.pending_exploration_targets.add(stripe_start_point)
                                temp_pending_targets_this_step.add(stripe_start_point)
                                explore_tasks_added_now += 1
                                print(
                                    f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Stripe explore task for start {stripe_start_point} added.")
                                self.no_normal_target_found_count = 0
                                continue
                            else:
                                temp_excluded_targets.add(stripe_start_point)
                                print(
                                    f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Stripe start {stripe_start_point} already targeted. Skipping this stripe attempt.")
                        else:
                            print(
                                f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No stripe start points found. Breaking exploration planning for this step.")
                            break
                    else:
                        print(
                            f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Not trying stripe yet (Count: {self.no_normal_target_found_count}). Breaking explore planning for this step.")
                        break
                        # else:
            # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Skipping regular/stripe exploration as initial hotspots are still being planned/assigned.")

    def _find_stripe_start_point(self, temp_excluded_targets=None):
        # ... (Logik wie in der letzten Antwort, ggf. Logging anpassen)
        if temp_excluded_targets is None: temp_excluded_targets = set()
        logistics_map = self.supervisor_exploration_logistics_map;
        candidate_stripe_starts = []
        passable_rows, passable_cols = np.where(logistics_map == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE)
        parent_cells = list(zip(map(int, passable_rows), map(int, passable_cols)));
        self.model.random.shuffle(parent_cells)

        # Sortiere parent_cells nach Nähe zum Supervisor (oder Kartenmitte)
        # um "naheliegende" Stripes zuerst zu finden.
        # ref_point = self.home_pos
        # parent_cells.sort(key=lambda p: self._manhattan_distance(p, ref_point))

        for p_pos in parent_cells:
            # Sortiere Nachbarn, um eine konsistentere Auswahl zu ermöglichen (optional)
            # sorted_offsets = sorted(self.NEIGHBOR_OFFSETS, key=lambda x: (abs(x[0])+abs(x[1])))
            for dx, dy in self.NEIGHBOR_OFFSETS:  # Oder sorted_offsets
                s_x, s_y = p_pos[0] + dx, p_pos[1] + dy;
                stripe_start_cand = (s_x, s_y)
                if 0 <= s_x < self.model.grid_width_val and 0 <= s_y < self.model.grid_height_val and \
                        logistics_map[s_x, s_y] == UNKNOWN and \
                        stripe_start_cand not in self.pending_exploration_targets and \
                        stripe_start_cand not in temp_excluded_targets and \
                        logistics_map[s_x, s_y] != SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:
                    known_neighbors = 0;
                    unknown_neighbors = 0
                    for ndx, ndy in self.NEIGHBOR_OFFSETS:
                        nn_x, nn_y = s_x + ndx, s_y + ndy
                        if 0 <= nn_x < self.model.grid_width_val and 0 <= nn_y < self.model.grid_height_val:
                            if logistics_map[nn_x, nn_y] == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE:
                                known_neighbors += 1
                            elif logistics_map[nn_x, nn_y] == UNKNOWN:
                                unknown_neighbors += 1
                    # Heuristik: Ein guter Stripe-Start hat einige unbekannte Nachbarn (um reinzugehen)
                    # und einige bekannte (der Eingang). Zu viele bekannte -> kein Korridor.
                    if 1 <= unknown_neighbors <= 6 and known_neighbors >= 1 and known_neighbors <= 4:
                        candidate_stripe_starts.append({'pos': stripe_start_cand,
                                                        'score': (
                                                                             unknown_neighbors * 2) - known_neighbors + self._calculate_frontier_potential_score(
                                                            stripe_start_cand, logistics_map)})

        if not candidate_stripe_starts:
            # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No ideal stripe start points found. Trying fallback (any UNKNOWN adjacent to KNOWN_PASSABLE).")
            # Fallback: Jede UNKNOWN Zelle neben einer KNOWN_PASSABLE, wenn die striktere Heuristik fehlschlägt
            for p_pos in parent_cells:  # Nutze die bereits sortierten (oder unsortierten) parent_cells
                for dx, dy in self.NEIGHBOR_OFFSETS:
                    s_x, s_y = p_pos[0] + dx, p_pos[1] + dy;
                    stripe_start_cand = (s_x, s_y)
                    if 0 <= s_x < self.model.grid_width_val and 0 <= s_y < self.model.grid_height_val and \
                            logistics_map[s_x, s_y] == UNKNOWN and \
                            stripe_start_cand not in self.pending_exploration_targets and \
                            stripe_start_cand not in temp_excluded_targets and \
                            logistics_map[s_x, s_y] != SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:
                        candidate_stripe_starts.append({'pos': stripe_start_cand,
                                                        'score': self._calculate_frontier_potential_score(
                                                            stripe_start_cand, logistics_map)})
                        if len(candidate_stripe_starts) > 20: break  # Begrenze Anzahl der Fallback-Kandidaten
                if len(candidate_stripe_starts) > 20: break

            if not candidate_stripe_starts:
                # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Still no stripe start points after fallback.")
                return None

        candidate_stripe_starts.sort(key=lambda x: x['score'], reverse=True)
        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Found {len(candidate_stripe_starts)} potential stripe starts. Best chosen: {candidate_stripe_starts[0]['pos']}")
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
        while q and head_count < 250:  # Sicherheitslimit
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
        if temp_excluded_targets is None: temp_excluded_targets = set()
        candidate_targets = []
        logistics_map = self.supervisor_exploration_logistics_map
        current_unknown_logistics_r = np.count_nonzero(logistics_map == UNKNOWN) / self.total_grid_cells
        is_breakout_phase = current_unknown_logistics_r >= self.BREAKOUT_PHASE_UNKNOWN_THRESHOLD
        ref_point_for_dist = self.home_pos
        if self.model.base_deposit_point: ref_point_for_dist = self.model.base_deposit_point

        if is_breakout_phase:
            # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Breakout phase. Searching deep dive targets.")
            all_unknown_on_logistics = [];
            u_rows, u_cols = np.where(logistics_map == UNKNOWN)
            for r_idx, c_idx in zip(u_rows, u_cols):
                pos_tuple = (int(r_idx), int(c_idx))
                if pos_tuple not in self.pending_exploration_targets and pos_tuple not in temp_excluded_targets and \
                        pos_tuple not in self.attempted_hotspots and logistics_map[
                    pos_tuple[0], pos_tuple[1]] != SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:
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
            parent_cell_coords = list(zip(map(int, known_passable_rows), map(int, known_passable_cols)));
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

        if not candidate_targets: return None
        candidate_targets.sort(key=lambda f: f['score'], reverse=True)
        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Top candidate (before sep): {candidate_targets[0]['pos']} Score: {candidate_targets[0]['score']:.2f} (Pot: {candidate_targets[0]['pot']})")

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
            # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Chosen explore target (met separation): {best_choice_satisfying_separation}")
            return best_choice_satisfying_separation
        # Strict Separation: If no candidate satisfies separation, return None.
        # This will trigger the stripe logic more often if areas are "crowded" with potential (but too close) targets.
        print(
            f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No target met MIN_EXPLORE_TARGET_SEPARATION ({self.MIN_EXPLORE_TARGET_SEPARATION_val}). Returning None from _find_best_frontier.")
        return None

    def _prepare_tasks_for_assignment(self):
        # ... (Logik gleich, Logs reduziert)
        if not self.task_queue: return
        idle_workers = []
        for worker_id, status_data in self.worker_status.items():
            is_busy = worker_id in self._tasks_to_assign_to_worker or \
                      (status_data.get('current_task_id') and status_data.get('current_task_id') in self.assigned_tasks)
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
        # ... (Logik gleich, aber mit Logging für zugewiesenen Task)
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
                cells_marked_targeted_count = 0
                for cell_px, cell_py in projected_cells:
                    if 0 <= cell_px < self.model.grid_width_val and 0 <= cell_py < self.model.grid_height_val:
                        if self.supervisor_exploration_logistics_map[cell_px, cell_py] == UNKNOWN:
                            self.supervisor_exploration_logistics_map[
                                cell_px, cell_py] = SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED
                            cells_marked_targeted_count += 1
                # if cells_marked_targeted_count > 0:
                # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Marked {cells_marked_targeted_count} cells around {target_for_projection} as TARGETED for W{worker_id}.")
            print(
                f"[S_AGENT {self.role_id}] (Step {self.model.steps}): Assigning task {task['task_id']} (Type: {task_type_assigned}, Target: {target_for_projection if target_for_projection else task.get('target_pos')}) to W{worker_id}.")
            return task

        # print(f"[S_AGENT {self.role_id}] (Step {self.model.steps}): No task for W{worker_id} (tasks_to_assign_worker empty).")
        return None