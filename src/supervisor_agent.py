# src/supervisor_agent.py
import mesa
import numpy as np
import itertools
from src.config import (
    UNKNOWN, EMPTY_EXPLORED, BASE_KNOWN,
    WOOD_SEEN, STONE_SEEN, RESOURCE_COLLECTED_BY_ME,
    SUPERVISOR_CLAIMED_RESOURCE,
    MIN_EXPLORE_TARGET_SEPARATION
)


class SupervisorAgent(mesa.Agent):
    NEIGHBOR_OFFSETS = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]

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
        self.task_queue = []
        self.assigned_tasks = {}
        self.resource_goals = self.model.resource_goals.copy()
        self._pending_worker_reports = []
        self._tasks_to_assign_to_worker = {}

        self.max_new_collect_tasks_per_planning = self.model.num_agents_val
        self.max_new_explore_tasks_per_planning = self.model.num_agents_val

        self.pending_exploration_targets = set()
        self.task_id_counter = itertools.count(1)

        self.MIN_EXPLORE_TARGET_SEPARATION_val = MIN_EXPLORE_TARGET_SEPARATION

    def _manhattan_distance(self, pos1, pos2):
        if pos1 is None or pos2 is None: return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self):
        self._process_pending_reports()
        self._update_task_statuses_and_cleanup()
        self._plan_new_tasks()  # Hier war der Fehler
        self._prepare_tasks_for_assignment()

    def _process_pending_reports(self):
        reports_to_process = list(
            self._pending_worker_reports)  # Bearbeite eine Kopie, um Modifikationen während der Iteration zu erlauben
        self._pending_worker_reports.clear()  # Leere die Originalliste sofort

        for report in reports_to_process:
            worker_id = report['worker_id']
            data = report['data']

            self.worker_status.setdefault(worker_id, {})
            if 'current_pos' in data: self.worker_status[worker_id]['last_pos'] = data['current_pos']

            new_worker_state = data.get('status')
            task_id_in_report = data.get('task_id')
            task_details_for_report = self.assigned_tasks.get(task_id_in_report) if task_id_in_report else None

            if 'map_segment_updates' in data:
                for pos, reported_state in data['map_segment_updates'].items():
                    px, py = pos
                    if not (0 <= px < self.supervisor_known_map.shape[0] and 0 <= py < self.supervisor_known_map.shape[
                        1]):
                        continue

                    current_supervisor_state = self.supervisor_known_map[px, py]
                    new_state_to_set = current_supervisor_state

                    if current_supervisor_state == SUPERVISOR_CLAIMED_RESOURCE:
                        if reported_state == RESOURCE_COLLECTED_BY_ME or reported_state == EMPTY_EXPLORED:
                            new_state_to_set = EMPTY_EXPLORED
                    else:
                        if reported_state == RESOURCE_COLLECTED_BY_ME:
                            new_state_to_set = EMPTY_EXPLORED
                        elif reported_state in [WOOD_SEEN, STONE_SEEN]:
                            if current_supervisor_state in [UNKNOWN, EMPTY_EXPLORED, BASE_KNOWN]:
                                new_state_to_set = reported_state
                            elif current_supervisor_state in [WOOD_SEEN,
                                                              STONE_SEEN] and current_supervisor_state != reported_state:
                                new_state_to_set = reported_state
                        elif reported_state == BASE_KNOWN:
                            if current_supervisor_state in [UNKNOWN, EMPTY_EXPLORED]:
                                new_state_to_set = reported_state
                        elif reported_state == EMPTY_EXPLORED:
                            if current_supervisor_state in [UNKNOWN, WOOD_SEEN, STONE_SEEN, BASE_KNOWN]:
                                new_state_to_set = reported_state

                    if current_supervisor_state != new_state_to_set:
                        # print(f"Supervisor DEBUG Map Update (Step {self.model.steps}): Pos {pos} von {current_supervisor_state} zu {new_state_to_set} (Worker {worker_id} meldete: {reported_state}, Report-TaskID: {task_id_in_report})")
                        self.supervisor_known_map[px, py] = new_state_to_set

            if new_worker_state:
                self.worker_status[worker_id]['state'] = new_worker_state

            if new_worker_state in ['TASK_COMPLETED', 'TASK_FAILED']:
                self.worker_status[worker_id]['current_task_id'] = None
                if task_details_for_report and task_details_for_report.get('worker_id') == worker_id:
                    task_type = task_details_for_report.get('type')
                    target_pos = task_details_for_report.get('target_pos')
                    path_to_explore = task_details_for_report.get('path_to_explore')

                    if task_type == 'explore_area' and path_to_explore:
                        for pos_on_route in path_to_explore:
                            self.pending_exploration_targets.discard(pos_on_route)

                    elif task_type == 'collect_resource' and target_pos:
                        current_map_val_at_target = self.supervisor_known_map[target_pos[0], target_pos[1]]
                        if new_worker_state == 'TASK_FAILED':
                            reported_state_at_target = data.get('map_segment_updates', {}).get(target_pos)
                            if current_map_val_at_target == SUPERVISOR_CLAIMED_RESOURCE:
                                if reported_state_at_target in [WOOD_SEEN, STONE_SEEN]:
                                    self.supervisor_known_map[target_pos[0], target_pos[1]] = reported_state_at_target
                                    # print(f"Supervisor (Step {self.model.steps}): CLAIMED Ressource bei {target_pos} (Task {task_id_in_report} FAILED) als {reported_state_at_target} wieder freigegeben.")
                                elif reported_state_at_target == EMPTY_EXPLORED or reported_state_at_target == RESOURCE_COLLECTED_BY_ME:
                                    self.supervisor_known_map[target_pos[0], target_pos[1]] = EMPTY_EXPLORED
                                    # print(f"Supervisor (Step {self.model.steps}): CLAIMED Ressource bei {target_pos} (Task {task_id_in_report} FAILED) als EMPTY markiert.")

                    if task_id_in_report in self.assigned_tasks:
                        del self.assigned_tasks[task_id_in_report]

            elif new_worker_state == 'IDLE_AT_SUPERVISOR':
                self.worker_status[worker_id]['current_task_id'] = None

    def _update_task_statuses_and_cleanup(self):
        pass

    def _is_target_already_assigned_or_queued(self, target_pos_to_check, task_type, resource_type=None):
        for task_data in list(self.assigned_tasks.values()) + self.task_queue:
            current_task_main_target = None
            if task_data.get('type') == 'explore_area' and task_data.get('path_to_explore'):
                if not task_data['path_to_explore']: continue
                current_task_main_target = task_data['path_to_explore'][0]
            elif task_data.get('type') == 'collect_resource':
                current_task_main_target = task_data.get('target_pos')

            if task_data.get('type') == task_type and current_task_main_target == target_pos_to_check:
                if task_type == 'collect_resource':
                    if task_data.get('resource_type') == resource_type: return True
                elif task_type == 'explore_area':
                    return True

        if task_type == 'explore_area':
            if isinstance(target_pos_to_check, tuple) and target_pos_to_check in self.pending_exploration_targets:
                return True
        return False

    def _plan_new_tasks(self):
        # KORREKTUR: Initialisierung außerhalb des if-Blocks
        collect_tasks_added_now = 0
        explore_tasks_added_now = 0

        if self.model.steps % 10 == 0:
            wood_on_map = np.count_nonzero(self.supervisor_known_map == WOOD_SEEN)
            stone_on_map = np.count_nonzero(self.supervisor_known_map == STONE_SEEN)
            claimed_resources = np.count_nonzero(self.supervisor_known_map == SUPERVISOR_CLAIMED_RESOURCE)
            pending_expl_count = len(self.pending_exploration_targets)
            # print(f"Supervisor Planung (Step {self.model.steps}): Map: H={wood_on_map}, S={stone_on_map}, Claimed={claimed_resources}. Tasks Q: {len(self.task_queue)}, Assigned: {len(self.assigned_tasks)}, PendingExplo: {pending_expl_count}")

        # --- Phase 1: Sammelaufgaben ---
        resource_priority = sorted(
            self.resource_goals.keys(),
            key=lambda r: (self.resource_goals[r] - self.model.base_resources_collected.get(r, 0)),
            reverse=True
        )
        for res_type in resource_priority:
            if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning: break
            needed_amount = self.resource_goals.get(res_type, 0) - self.model.base_resources_collected.get(res_type, 0)
            if needed_amount <= 0: continue
            resource_seen_constant = WOOD_SEEN if res_type == 'wood' else STONE_SEEN

            candidate_patches_coords = []
            rows, cols = np.where(self.supervisor_known_map == resource_seen_constant)
            for r_idx, c_idx in zip(rows, cols):
                candidate_patches_coords.append((int(r_idx), int(c_idx)))
            self.model.random.shuffle(candidate_patches_coords)

            for patch_pos in candidate_patches_coords:
                if self._is_target_already_assigned_or_queued(patch_pos, 'collect_resource', res_type):
                    continue
                px, py = patch_pos
                self.supervisor_known_map[px, py] = SUPERVISOR_CLAIMED_RESOURCE
                new_collect_task = {
                    'task_id': f"task_collect_{next(self.task_id_counter)}",
                    'type': 'collect_resource', 'target_pos': patch_pos,
                    'resource_type': res_type, 'status': 'pending_assignment'
                }
                self.task_queue.insert(0, new_collect_task)
                # print(f"Supervisor (Step {self.model.steps}): NEUE SAMMELAUFGABE {new_collect_task['task_id']} für {res_type} bei {patch_pos}. Map -> CLAIMED.")
                collect_tasks_added_now += 1
                if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning: break
            if collect_tasks_added_now >= self.max_new_collect_tasks_per_planning: break

        # --- Phase 2: Explorationsaufgaben ---
        unlocated_needed_resources = 0
        for res_type_iter, target_amount in self.resource_goals.items():
            if self.model.base_resources_collected.get(res_type_iter, 0) < target_amount:
                res_const = WOOD_SEEN if res_type_iter == 'wood' else STONE_SEEN
                if np.count_nonzero(self.supervisor_known_map == res_const) == 0:
                    unlocated_needed_resources += 1
        unknown_ratio = np.count_nonzero(self.supervisor_known_map == UNKNOWN) / (
                self.model.grid_width_val * self.model.grid_height_val)
        should_explore_actively = (unlocated_needed_resources > 0 or unknown_ratio > 0.65)

        temp_pending_targets_this_step = set()

        while should_explore_actively and explore_tasks_added_now < self.max_new_explore_tasks_per_planning:
            exploration_target_cell = self._find_best_frontier_for_exploration(temp_pending_targets_this_step)

            if exploration_target_cell:
                if not self._is_target_already_assigned_or_queued(exploration_target_cell, 'explore_area'):
                    new_explore_task = {
                        'task_id': f"task_explore_{next(self.task_id_counter)}",
                        'type': 'explore_area', 'path_to_explore': [exploration_target_cell],
                        'status': 'pending_assignment', 'target_pos': exploration_target_cell
                    }
                    self.task_queue.append(new_explore_task)
                    self.pending_exploration_targets.add(exploration_target_cell)
                    temp_pending_targets_this_step.add(exploration_target_cell)
                    # print(f"Supervisor (Step {self.model.steps}): Neue Frontier-Explo {new_explore_task['task_id']} für {exploration_target_cell} zur Queue.")
                    explore_tasks_added_now += 1
                else:
                    break
            else:
                break

    def _find_best_frontier_for_exploration(self, temp_excluded_targets=None):
        if temp_excluded_targets is None:
            temp_excluded_targets = set()

        candidate_frontiers = []
        evaluated_frontiers = set()

        known_passable_rows, known_passable_cols = np.where(
            (self.supervisor_known_map == EMPTY_EXPLORED) |
            (self.supervisor_known_map == BASE_KNOWN)
        )
        parent_cell_coords = list(zip(map(int, known_passable_rows), map(int, known_passable_cols)))
        self.model.random.shuffle(parent_cell_coords)

        for parent_pos in parent_cell_coords:
            for dx, dy in self.NEIGHBOR_OFFSETS:
                frontier_pos = (parent_pos[0] + dx, parent_pos[1] + dy)
                if not (0 <= frontier_pos[0] < self.model.grid_width_val and 0 <= frontier_pos[
                    1] < self.model.grid_height_val):
                    continue

                if self.supervisor_known_map[frontier_pos[0], frontier_pos[1]] == UNKNOWN and \
                        frontier_pos not in self.pending_exploration_targets and \
                        frontier_pos not in temp_excluded_targets and \
                        frontier_pos not in evaluated_frontiers:
                    evaluated_frontiers.add(frontier_pos)
                    unknown_neighbors_of_frontier = 0
                    for ndx, ndy in self.NEIGHBOR_OFFSETS:
                        nnx, nny = frontier_pos[0] + ndx, frontier_pos[1] + ndy
                        if 0 <= nnx < self.model.grid_width_val and \
                                0 <= nny < self.model.grid_height_val and \
                                self.supervisor_known_map[nnx, nny] == UNKNOWN:
                            unknown_neighbors_of_frontier += 1
                    score = unknown_neighbors_of_frontier
                    candidate_frontiers.append({'pos': frontier_pos, 'score': score})

        if not candidate_frontiers:
            all_unknown_coords = []
            u_rows, u_cols = np.where(self.supervisor_known_map == UNKNOWN)
            for r_idx, c_idx in zip(u_rows, u_cols):
                pos_tuple = (int(r_idx), int(c_idx))
                if pos_tuple not in self.pending_exploration_targets and pos_tuple not in temp_excluded_targets:
                    all_unknown_coords.append(pos_tuple)
            if all_unknown_coords: return self.model.random.choice(all_unknown_coords)
            return None

        candidate_frontiers.sort(key=lambda f: f['score'], reverse=True)

        targets_to_avoid_proximity_to = set(self.pending_exploration_targets)
        targets_to_avoid_proximity_to.update(temp_excluded_targets)
        for task_data in list(self.assigned_tasks.values()) + self.task_queue:
            if task_data.get('type') == 'explore_area' and task_data.get('target_pos'):
                targets_to_avoid_proximity_to.add(task_data.get('target_pos'))

        top_score = candidate_frontiers[0]['score']
        equally_good_candidates = [cand for cand in candidate_frontiers if cand['score'] == top_score]
        self.model.random.shuffle(equally_good_candidates)

        best_choice = None
        for frontier_candidate in equally_good_candidates:
            candidate_pos = frontier_candidate['pos']
            is_far_enough = True
            for existing_target in targets_to_avoid_proximity_to:
                if self._manhattan_distance(candidate_pos, existing_target) < self.MIN_EXPLORE_TARGET_SEPARATION_val:
                    is_far_enough = False;
                    break
            if is_far_enough:
                best_choice = candidate_pos;
                break

        if not best_choice:
            self.model.random.shuffle(candidate_frontiers)
            for frontier_candidate in candidate_frontiers:
                candidate_pos = frontier_candidate['pos']
                is_far_enough = True
                for existing_target in targets_to_avoid_proximity_to:
                    if self._manhattan_distance(candidate_pos,
                                                existing_target) < self.MIN_EXPLORE_TARGET_SEPARATION_val:
                        is_far_enough = False;
                        break
                if is_far_enough:
                    best_choice = candidate_pos;
                    break

        if best_choice: return best_choice
        if candidate_frontiers: return candidate_frontiers[0]['pos']
        return None

    def _prepare_tasks_for_assignment(self):
        if not self.task_queue: return
        idle_workers = []
        for worker_id, status_data in self.worker_status.items():
            is_busy_or_getting_task = worker_id in self._tasks_to_assign_to_worker or \
                                      (status_data.get('current_task_id') and \
                                       status_data.get('current_task_id') in self.assigned_tasks)
            if status_data.get('state') == 'IDLE_AT_SUPERVISOR' and not is_busy_or_getting_task:
                idle_workers.append(worker_id)
        self.model.random.shuffle(idle_workers)
        for worker_id in idle_workers:
            if not self.task_queue: break
            task_to_assign = self.task_queue.pop(0)
            task_to_assign['status'] = 'assigned_pending_pickup'
            self._tasks_to_assign_to_worker[worker_id] = task_to_assign

    def receive_report_from_worker(self, worker_id, report_type, data):
        self._pending_worker_reports.append({'worker_id': worker_id, 'report_type': report_type, 'data': data})

    def request_task_from_worker(self, worker_id):
        self.worker_status.setdefault(worker_id, {})['state'] = 'IDLE_AT_SUPERVISOR'
        self.worker_status[worker_id]['current_task_id'] = None
        self._prepare_tasks_for_assignment()
        if worker_id in self._tasks_to_assign_to_worker:
            task = self._tasks_to_assign_to_worker.pop(worker_id)
            task['status'] = 'assigned'
            task['worker_id'] = worker_id
            self.assigned_tasks[task['task_id']] = task
            self.worker_status[worker_id]['current_task_id'] = task['task_id']
            return task
        return None