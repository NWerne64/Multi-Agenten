# src/worker_agent.py
import mesa
import numpy as np
from src.config import (
    UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME,
    # Die folgenden Konstanten sind durch die Änderungen weniger relevant oder ungenutzt für Worker:
    # CHAINED_EXPLORATION_STEP_BUDGET,
    # MAX_CHAINED_FRONTIERS_VISITED,
    # CHAINED_LOCAL_EXPLORE_STEPS
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
        self.vision_radius = self.model.agent_vision_radius_val
        self.worker_internal_map = np.full((model.grid_width_val, model.grid_height_val), UNKNOWN, dtype=int)

        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0
        self.local_exploration_max_steps = self.model.random.randint(3, 6)

        self.is_current_task_initial_hotspot = False

        # Attribute für Kaskaden-Exploration (derzeit nicht aktiv genutzt durch FSM-Änderungen)
        self.chained_sortie_steps_taken = 0
        self.chained_frontiers_visited_count = 0
        self.current_chained_target_pos = None
        self.is_on_chained_exploration_sortie = False

        # Attribute für Stripe Exploration
        self.stripe_steps_taken = 0
        self.current_stripe_direction = None
        # DEFAULT_STRIPE_LENGTH wird nicht mehr aus config geladen, Supervisor gibt max_length vor
        self.DEFAULT_STRIPE_LENGTH_val = getattr(self.model, 'DEFAULT_STRIPE_LENGTH_val', 20)

        # print(f"[W_AGENT {self.display_id}] (Init): Worker created. Vision: {self.vision_radius}, Max local steps: {self.local_exploration_max_steps}")

    def _manhattan_distance(self, pos1, pos2):
        if pos1 is None or pos2 is None: return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self):
        # task_id_info = self.current_task.get('task_id') if self.current_task else 'None'
        # print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Pos: {self.pos}, State: {self.state}, Task: {task_id_info}")
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

    def _move_towards(self, target_pos):  # WIEDERHERGESTELLTE METHODE
        if self.pos is None or target_pos is None or self.pos == target_pos: return
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1)
        if not possible_steps:
            print(
                f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Cannot move from {self.pos}, no possible steps to reach {target_pos}.")
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
            # print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Moving from {self.pos} to {best_step} towards {target_pos}.")
            self.model.grid.move_agent(self, best_step)
        # else:
        # print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): No better step from {self.pos} towards {target_pos}.")

    def _execute_fsm(self):
        previous_state = self.state

        if self.state == "IDLE":
            self.state = "MOVING_TO_SUPERVISOR"
        elif self.state == "MOVING_TO_SUPERVISOR":
            if self.pos == self.supervisor_home_pos:
                self.state = "AWAITING_TASK"
                # print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Reached supervisor. Requesting task.")
                self._report_and_request_new_task()
            else:
                self._move_towards(self.supervisor_home_pos)
        elif self.state == "AWAITING_TASK":
            if not self.current_task:
                if self.pos == self.supervisor_home_pos and self.model.steps % 10 == self.unique_id % 10:
                    # print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Still AWAITING_TASK. Re-requesting.")
                    self._report_and_request_new_task()
        elif self.state == "MOVING_TO_COLLECT_TARGET":
            if self.current_task is None: self.state = "IDLE"; return
            target = self.current_task['target_pos']
            if self.pos == target:
                # print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Reached collect target {target}.")
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
                    # print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Collected {self.inventory_slot['type']} at {target_pos}.")
                    self.state = "MOVING_TO_BASE_FOR_TASK_DELIVERY"
                else:
                    reason = "wrong_res_type" if actual_type_data else "res_vanished"
                    # print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Collect failed at {target_pos}. Reason: {reason}")
                    if target_pos not in self.model.resources_on_grid:
                        self.worker_internal_map[target_pos[0], target_pos[1]] = EMPTY_EXPLORED
                        self.newly_observed_map_segment[target_pos] = EMPTY_EXPLORED
                    self._task_failed_or_issue(reason)
            else:
                reason = "inv_full_or_not_found_at_target"  # Zusammengefasst
                # print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Collect failed at {target_pos}. Reason: {reason}")
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
                    # print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Deposited {res_type} at base {self.pos}.")
                    self.inventory_slot = None
                self._task_completed()
            else:
                self._move_towards(self.model.base_deposit_point)
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
                        print(
                            f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Initial Hotspot {target_waypoint} reached. Task completed.")
                        self._task_completed()
                    else:
                        print(
                            f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Regular explore route to {target_waypoint} done. Starting local scan.")
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
                print(
                    f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Local scan finished after {self.local_exploration_max_steps} steps. Task completed.")
                self._task_completed()
        elif self.state == "MOVING_TO_STRIPE_START":
            if self.current_task is None or self.current_task.get('start_pos') is None:
                self._task_failed_or_issue("invalid_stripe_task_data");
                return
            stripe_start_pos = self.current_task['start_pos']
            if self.pos == stripe_start_pos:
                print(
                    f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Reached stripe start {stripe_start_pos}. Beginning stripe follow.")
                self.state = "FOLLOWING_STRIPE";
                self.stripe_steps_taken = 0;
                self.current_stripe_direction = None
            else:
                self._move_towards(stripe_start_pos)
        elif self.state == "FOLLOWING_STRIPE":
            if self.current_task is None: self._task_failed_or_issue("no_task_in_stripe_follow"); return
            max_len = self.current_task.get('max_length', self.DEFAULT_STRIPE_LENGTH_val)
            if self.stripe_steps_taken >= max_len:
                print(
                    f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Stripe max length {max_len} reached at {self.pos}. Task completed.")
                self._task_completed();
                return

            neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1)
            best_stripe_step = None;
            min_known_neighbors_of_next = 9

            if self.current_stripe_direction:  # Versuche Richtung beizubehalten
                pn_x = self.pos[0] + self.current_stripe_direction[0]
                pn_y = self.pos[1] + self.current_stripe_direction[1]
                potential_next = (pn_x, pn_y)
                if 0 <= pn_x < self.model.grid_width_val and 0 <= pn_y < self.model.grid_height_val and \
                        potential_next in neighbors and self.worker_internal_map[pn_x, pn_y] == UNKNOWN:
                    temp_known_count = 0
                    for dx_n, dy_n in self.NEIGHBOR_OFFSETS:
                        nn_x, nn_y = pn_x + dx_n, pn_y + dy_n
                        if 0 <= nn_x < self.model.grid_width_val and 0 <= nn_y < self.model.grid_height_val:
                            if self.worker_internal_map[nn_x, nn_y] != UNKNOWN: temp_known_count += 1
                    if temp_known_count < min_known_neighbors_of_next:
                        min_known_neighbors_of_next = temp_known_count;
                        best_stripe_step = potential_next

            if not best_stripe_step:  # Wenn nicht, suche besten Nachbarn
                shuffled_neighbors = list(neighbors);
                self.model.random.shuffle(shuffled_neighbors)
                for next_pos_cand in shuffled_neighbors:
                    if 0 <= next_pos_cand[0] < self.model.grid_width_val and 0 <= next_pos_cand[
                        1] < self.model.grid_height_val and \
                            self.worker_internal_map[next_pos_cand[0], next_pos_cand[1]] == UNKNOWN:
                        known_count = 0
                        for dx_n, dy_n in self.NEIGHBOR_OFFSETS:
                            nn_x, nn_y = next_pos_cand[0] + dx_n, next_pos_cand[1] + dy_n
                            if 0 <= nn_x < self.model.grid_width_val and 0 <= nn_y < self.model.grid_height_val:
                                if self.worker_internal_map[nn_x, nn_y] != UNKNOWN: known_count += 1
                        if known_count < min_known_neighbors_of_next:
                            min_known_neighbors_of_next = known_count;
                            best_stripe_step = next_pos_cand
                        elif known_count == min_known_neighbors_of_next and best_stripe_step is None:
                            best_stripe_step = next_pos_cand

            if best_stripe_step:
                # print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Stripe step {self.stripe_steps_taken + 1}/{max_len}. Moving from {self.pos} to {best_stripe_step}.")
                if self.pos != best_stripe_step:
                    self.current_stripe_direction = (
                    best_stripe_step[0] - self.pos[0], best_stripe_step[1] - self.pos[1])
                self._move_towards(best_stripe_step)
                self.stripe_steps_taken += 1
            else:
                print(
                    f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Stripe ended at {self.pos} after {self.stripe_steps_taken} steps (no UNKNOWN continuation). Task completed.")
                self._task_completed()

        if previous_state != self.state and self.state not in ["IDLE", "AWAITING_TASK"]:
            print(
                f"[W_AGENT {self.display_id}] (Step {self.model.steps}): State change {previous_state} -> {self.state}")

    def set_task(self, task_details):  # KORREKTE POSITIONIERUNG AUSSERHALB VON _execute_fsm
        self.current_task = task_details
        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0
        self.is_current_task_initial_hotspot = self.current_task.get('is_initial_hotspot_task', False)
        self.stripe_steps_taken = 0
        self.current_stripe_direction = None

        task_type = self.current_task.get('type')
        target_log = self.current_task.get('target_pos')
        if task_type == 'explore_stripe':
            target_log = self.current_task.get('start_pos')
        elif task_type == 'explore_area' and self.current_task.get('path_to_explore'):
            if self.current_task.get('path_to_explore'):  # Sicherstellen, dass path_to_explore existiert
                target_log = self.current_task.get('path_to_explore')[0]
            else:  # Sollte nicht passieren, wenn Supervisor korrekt plant
                target_log = "Error_No_Path_Explore_Area"

        print(
            f"[W_AGENT {self.display_id}] (Step {self.model.steps}): SET_TASK - ID: {self.current_task.get('task_id')}, Type: {task_type}, Target: {target_log}, IsHotspot: {self.is_current_task_initial_hotspot}")

        if task_type == 'collect_resource':
            self.state = "MOVING_TO_COLLECT_TARGET"
        elif task_type == 'explore_area':
            if self.current_task.get('path_to_explore'):
                self.state = "MOVING_TO_EXPLORE_ROUTE_STEP"
            else:
                self._task_failed_or_issue("explore_task_missing_path")
        elif task_type == 'explore_stripe':
            if self.current_task.get('start_pos'):
                self.state = "MOVING_TO_STRIPE_START"
            else:
                self._task_failed_or_issue("stripe_task_missing_start_pos")
        else:
            print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): ERROR - Unknown task type: {task_type}")
            self.state = "IDLE"

    def _report_and_request_new_task(self):
        # ... (Logik bleibt gleich)
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
        # ... (Logik bleibt gleich)
        task_id_log = self.current_task.get('task_id', 'N/A') if self.current_task else 'N/A'
        print(f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Task {task_id_log} COMPLETED. Reporting.")
        report_data = {
            'status': 'TASK_COMPLETED', 'task_id': task_id_log, 'current_pos': self.pos,
            'inventory': self.inventory_slot, 'map_segment_updates': self.newly_observed_map_segment.copy()
        }
        self.model.submit_report_to_supervisor(self.unique_id, 'task_feedback', report_data)
        self.newly_observed_map_segment.clear();
        self.current_task = None;
        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0;
        self.is_current_task_initial_hotspot = False
        self.stripe_steps_taken = 0;
        self.current_stripe_direction = None
        self.state = "MOVING_TO_SUPERVISOR"

    def _task_failed_or_issue(self, reason="unknown"):
        # ... (Logik bleibt gleich)
        task_id_log = self.current_task.get('task_id', 'N/A') if self.current_task else 'N/A'
        print(
            f"[W_AGENT {self.display_id}] (Step {self.model.steps}): Task {task_id_log} FAILED/Issue. Reason: {reason}. Reporting.")
        report_data = {
            'status': 'TASK_FAILED', 'task_id': task_id_log, 'reason': reason, 'current_pos': self.pos,
            'map_segment_updates': self.newly_observed_map_segment.copy()}
        self.model.submit_report_to_supervisor(self.unique_id, 'task_feedback', report_data)
        self.newly_observed_map_segment.clear();
        self.current_task = None;
        self.current_path_to_explore_index = 0
        self.local_exploration_steps_done = 0;
        self.is_current_task_initial_hotspot = False
        self.stripe_steps_taken = 0;
        self.current_stripe_direction = None
        if self.inventory_slot:
            self.state = "MOVING_TO_BASE_FOR_TASK_DELIVERY"
        else:
            self.state = "MOVING_TO_SUPERVISOR"