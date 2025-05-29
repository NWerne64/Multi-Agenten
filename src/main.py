# src/main.py
import pygame
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, GRID_LINE_COLOR, WOOD_COLOR, STONE_COLOR,
    BASE_COLOR, TEXT_COLOR, CELL_WIDTH, CELL_HEIGHT, FRAMES_PER_SECOND, SIMULATION_STEPS_PER_SECOND,
    UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME,
    UNKNOWN_CELL_COLOR, BLACKBOARD_OBJECT_COLOR,
    KEY_VIEW_BLACKBOARD, KEY_VIEW_AGENT_1, KEY_VIEW_AGENT_2, KEY_VIEW_AGENT_3, KEY_VIEW_AGENT_4,
    KEY_VIEW_AGENT_5, KEY_VIEW_AGENT_6, KEY_VIEW_AGENT_7, KEY_VIEW_AGENT_8, KEY_VIEW_AGENT_9,
    SUPERVISOR_COLOR, WORKER_COLOR, AGENT_COLOR
)
from src.model import AoELiteModel
from src.supervisor_agent import SupervisorAgent
from src.worker_agent import WorkerAgent
from src.agent import ResourceCollectorAgent


def draw_grid(screen):
    # ... (unverändert)
    for x in range(0, SCREEN_WIDTH, CELL_WIDTH): pygame.draw.line(screen, GRID_LINE_COLOR, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_HEIGHT): pygame.draw.line(screen, GRID_LINE_COLOR, (0, y), (SCREEN_WIDTH, y))


def draw_physical_blackboard(screen, model):
    # ... (unverändert)
    if model.strategy == "decentralized" and model.blackboard_coords_list:
        for (bbx, bby) in model.blackboard_coords_list:
            pygame.draw.rect(screen, BLACKBOARD_OBJECT_COLOR,
                             (bbx * CELL_WIDTH, SCREEN_HEIGHT - (bby + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))


def draw_world_view(screen, model, view_mode, selected_agent_idx):
    map_to_display = None
    agent_to_view_for_text = None

    if view_mode == "BLACKBOARD":
        if model.strategy == "supervisor" and model.supervisor_agent_instance:
            map_to_display = model.supervisor_agent_instance.supervisor_known_map
            agent_to_view_for_text = model.supervisor_agent_instance
        elif model.strategy == "decentralized":
            map_to_display = model.blackboard_map
        else:
            view_mode = "NONE"
    elif view_mode == "AGENT":
        # ANPASSUNG: model.agents statt model.schedule.agents
        # Und sicherstellen, dass die Agentenliste nicht leer ist.
        # selected_agent_idx ist der Index für model.agents (0-basiert)
        # Die Agenten in model.agents haben möglicherweise keine feste Reihenfolge mehr wie im alten Scheduler.
        # Für eine konsistente Auswahl müsste man die Agentenliste ggf. sortieren (z.B. nach unique_id)
        # oder eine feste Zuordnung von Tasten zu spezifischen Agenten-IDs implementieren.
        # Hier verwenden wir erstmal die direkte Indexierung in der (potenziell unsortierten) AgentSet-Iteration.

        # Um eine feste Reihenfolge für die Tasten zu bekommen, könnten wir die Agentenliste einmalig sortieren:
        # agents_list_for_view = sorted(list(model.agents), key=lambda a: a.unique_id) # Beispiel Sortierung
        agents_list_for_view = list(model.agents)  # Direkte Konvertierung zu Liste für Indexierung

        if agents_list_for_view and 0 <= selected_agent_idx < len(agents_list_for_view):
            agent = agents_list_for_view[selected_agent_idx]
            agent_to_view_for_text = agent

            if isinstance(agent, SupervisorAgent):
                map_to_display = agent.supervisor_known_map
            elif isinstance(agent, WorkerAgent):
                map_to_display = agent.worker_internal_map
            elif isinstance(agent, ResourceCollectorAgent):
                map_to_display = agent.known_map
            else:
                view_mode = "NONE"
        else:
            view_mode = "NONE"

    if map_to_display is not None:
        # ... (Rest der Zeichenlogik für die Karte, unverändert)
        for gx in range(model.grid_width_val):
            for gy in range(model.grid_height_val):
                pygame_rect = pygame.Rect(gx * CELL_WIDTH, SCREEN_HEIGHT - (gy + 1) * CELL_HEIGHT, CELL_WIDTH,
                                          CELL_HEIGHT)
                if gx < map_to_display.shape[0] and gy < map_to_display.shape[1]:
                    known_status = map_to_display[gx, gy]
                    cell_color_to_draw = UNKNOWN_CELL_COLOR
                    if known_status != UNKNOWN:
                        if known_status == WOOD_SEEN:
                            cell_color_to_draw = WOOD_COLOR
                        elif known_status == STONE_SEEN:
                            cell_color_to_draw = STONE_COLOR
                        elif known_status == BASE_KNOWN or \
                                ((gx, gy) in model.base_coords_list and \
                                 (view_mode == "BLACKBOARD" or \
                                  (agent_to_view_for_text and isinstance(agent_to_view_for_text, SupervisorAgent)))):
                            cell_color_to_draw = BASE_COLOR
                        elif known_status == EMPTY_EXPLORED or known_status == RESOURCE_COLLECTED_BY_ME:
                            cell_color_to_draw = WHITE
                        else:
                            cell_color_to_draw = WHITE
                    pygame.draw.rect(screen, cell_color_to_draw, pygame_rect)
                else:
                    pygame.draw.rect(screen, BLACK, pygame_rect)
    else:
        for gx_ in range(model.grid_width_val):
            for gy_ in range(model.grid_height_val):
                pygame.draw.rect(screen, UNKNOWN_CELL_COLOR, (
                    gx_ * CELL_WIDTH, SCREEN_HEIGHT - (gy_ + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))


def draw_agents(screen, model):
    # ANPASSUNG: model.agents statt model.schedule.agents
    for agent in model.agents:  # model.agents ist ein AgentSet, iterierbar
        if agent.pos is not None:
            x, y = agent.pos
            pygame_rect = pygame.Rect(x * CELL_WIDTH, SCREEN_HEIGHT - (y + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            agent_color_to_draw = AGENT_COLOR
            if isinstance(agent, SupervisorAgent):
                agent_color_to_draw = SUPERVISOR_COLOR
            elif isinstance(agent, WorkerAgent):
                agent_color_to_draw = WORKER_COLOR
            pygame.draw.rect(screen, agent_color_to_draw, pygame_rect)


def draw_base_resource_text(screen, model, font):
    # ... (unverändert)
    if not model.base_deposit_point: return
    wood_text = f"Wood: {model.base_resources_collected['wood']}/{model.resource_goals.get('wood', 0)}"
    stone_text = f"Stone: {model.base_resources_collected['stone']}/{model.resource_goals.get('stone', 0)}"
    wood_surface = font.render(wood_text, True, TEXT_COLOR);
    stone_surface = font.render(stone_text, True, TEXT_COLOR)
    text_x = (model.base_coords_list[0][0]) * CELL_WIDTH
    text_y_base = SCREEN_HEIGHT - (max(c[1] for c in model.base_coords_list) + 1) * CELL_HEIGHT - 5
    screen.blit(wood_surface, (text_x, text_y_base - wood_surface.get_height()));
    screen.blit(stone_surface, (text_x, text_y_base))


def draw_view_mode_text(screen, view_mode, selected_agent_idx, model, font):
    text_lines = []
    text_lines.append(f"Strategie: {model.strategy}")

    # ANPASSUNG: model.agents und len(model.agents)
    # agents_list_for_view = sorted(list(model.agents), key=lambda a: a.unique_id) # Für konsistente Reihenfolge
    agents_list_for_view = list(model.agents)

    if view_mode == "BLACKBOARD":
        if model.strategy == "supervisor":
            text_lines.append(f"View: Supervisor Karte (0)")
        else:
            text_lines.append("View: Blackboard (0)")
    elif view_mode == "AGENT":
        if agents_list_for_view and 0 <= selected_agent_idx < len(agents_list_for_view):
            agent = agents_list_for_view[selected_agent_idx]
            agent_type_str = "Agent"
            agent_id_str = agent.unique_id

            if isinstance(agent, SupervisorAgent):
                agent_type_str = "Supervisor"
            elif isinstance(agent, WorkerAgent):
                agent_type_str = "Worker"
            elif isinstance(agent, ResourceCollectorAgent) and hasattr(agent, 'agent_display_id'):
                agent_id_str = agent.agent_display_id
            text_lines.append(f"View: {agent_type_str} {agent_id_str} (Taste {selected_agent_idx + 1})")
        else:
            text_lines.append(f"View: Agent (Index {selected_agent_idx + 1} ungültig)")
    else:
        text_lines.append("View: Global (Unbekannt)")

    # ANPASSUNG: model.steps statt model.current_step
    text_lines.append(f"Step: {model.steps}")  # Mesa 3.0+
    if not model.simulation_running and model.completion_step > -1:
        text_lines.append(f"Ziele erreicht in {model.completion_step} Schritten!")

    y_offset = 5
    for line in text_lines:
        status_surface = font.render(line, True, TEXT_COLOR)
        screen.blit(status_surface, (5, y_offset))
        y_offset += status_surface.get_height() + 2


def run_simulation():
    pygame.init()
    pygame.font.init()
    try:
        game_font = pygame.font.SysFont("arial", 20)
    except pygame.error:
        game_font = pygame.font.Font(None, 28)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Age of Empires Lite - Kollaborationsstrategien")

    CHOSEN_STRATEGY = "supervisor"
    num_agents_for_run = 4
    vision_for_run = 2

    model = AoELiteModel(
        strategy=CHOSEN_STRATEGY,
        num_agents_val=num_agents_for_run,
        agent_vision_radius=vision_for_run
    )

    running = True
    clock = pygame.time.Clock()
    time_since_last_model_step = 0
    current_view_mode = "AGENT"
    selected_agent_idx_for_view = 0

    agent_view_keys = {
        KEY_VIEW_AGENT_1: 0, KEY_VIEW_AGENT_2: 1, KEY_VIEW_AGENT_3: 2,
        KEY_VIEW_AGENT_4: 3, KEY_VIEW_AGENT_5: 4, KEY_VIEW_AGENT_6: 5,
        KEY_VIEW_AGENT_7: 6, KEY_VIEW_AGENT_8: 7, KEY_VIEW_AGENT_9: 8,
    }

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == KEY_VIEW_BLACKBOARD:
                    current_view_mode = "BLACKBOARD"
                elif event.key in agent_view_keys:
                    idx = agent_view_keys[event.key]
                    # ANPASSUNG: len(model.agents)
                    # agents_list_for_view_keys = sorted(list(model.agents), key=lambda a: a.unique_id)
                    agents_list_for_view_keys = list(model.agents)

                    if idx < len(agents_list_for_view_keys):
                        current_view_mode = "AGENT"
                        selected_agent_idx_for_view = idx
                    else:
                        print(f"Warnung: Agenten-Index {idx} für Ansicht ungültig.")

        if model.simulation_running:
            time_since_last_model_step += clock.get_rawtime()
            if time_since_last_model_step >= 1000.0 / SIMULATION_STEPS_PER_SECOND:
                model.step()  # Ruft intern self.agents.shuffle_do("step") etc. auf
                time_since_last_model_step = 0

        if current_view_mode == "AGENT":
            # agents_list_for_view_check = sorted(list(model.agents), key=lambda a: a.unique_id)
            agents_list_for_view_check = list(model.agents)
            if not (agents_list_for_view_check and 0 <= selected_agent_idx_for_view < len(agents_list_for_view_check)):
                selected_agent_idx_for_view = 0
                if not agents_list_for_view_check:
                    current_view_mode = "NONE"

        screen.fill(WHITE)
        draw_world_view(screen, model, current_view_mode, selected_agent_idx_for_view)
        draw_grid(screen)
        if model.strategy == "decentralized":
            draw_physical_blackboard(screen, model)
        draw_agents(screen, model)
        draw_base_resource_text(screen, model, game_font)
        draw_view_mode_text(screen, current_view_mode, selected_agent_idx_for_view, model, game_font)

        pygame.display.flip()
        clock.tick(FRAMES_PER_SECOND)

    pygame.font.quit()
    pygame.quit()


if __name__ == "__main__":
    run_simulation()