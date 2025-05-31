# src/main.py
import pygame
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GRID_LINE_COLOR, WOOD_COLOR, STONE_COLOR,
    BASE_COLOR, TEXT_COLOR, CELL_WIDTH, CELL_HEIGHT, FRAMES_PER_SECOND, SIMULATION_STEPS_PER_SECOND,
    UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME,
    UNKNOWN_CELL_COLOR, BLACKBOARD_OBJECT_COLOR,
    SUPERVISOR_COLOR, WORKER_COLOR, AGENT_COLOR, SUPERVISOR_CLAIMED_RESOURCE,
    SUPERVISOR_LOGISTICS_KNOWN_PASSABLE, LOGISTICS_KNOWN_PASSABLE_COLOR,
    SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED, LOGISTICS_EXPLORATION_TARGETED_COLOR,
    # NEUE Farbkonstanten für Korridore
    CORRIDOR_ENTRY_POINT_COLOR, CORRIDOR_END_POINT_COLOR,
    KEY_VIEW_SUPERVISOR_PUBLIC, KEY_VIEW_SUPERVISOR_LOGISTICS,
    KEY_VIEW_AGENT_1, KEY_VIEW_AGENT_2, KEY_VIEW_AGENT_3, KEY_VIEW_AGENT_4,
    KEY_VIEW_AGENT_5, KEY_VIEW_AGENT_6, KEY_VIEW_AGENT_7, KEY_VIEW_AGENT_8, KEY_VIEW_AGENT_9
)
from src.model import AoELiteModel #
from src.supervisor_agent import SupervisorAgent # Sicherstellen, dass SupervisorAgent importiert wird #
from src.worker_agent import WorkerAgent #
from src.agent import ResourceCollectorAgent #


def draw_grid(screen):
    # Unverändert
    for x in range(0, SCREEN_WIDTH, CELL_WIDTH): pygame.draw.line(screen, GRID_LINE_COLOR, (x, 0), (x, SCREEN_HEIGHT)) #
    for y in range(0, SCREEN_HEIGHT, CELL_HEIGHT): pygame.draw.line(screen, GRID_LINE_COLOR, (0, y), (SCREEN_WIDTH, y)) #


def draw_physical_blackboard(screen, model):
    # Unverändert
    if model.strategy == "decentralized" and model.blackboard_coords_list: #
        for (bbx, bby) in model.blackboard_coords_list: #
            pygame.draw.rect(screen, BLACKBOARD_OBJECT_COLOR, #
                             (bbx * CELL_WIDTH, SCREEN_HEIGHT - (bby + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))


def draw_world_view(screen, model, view_mode, selected_agent_idx, show_supervisor_logistics_map):
    map_to_display = None
    agent_to_view_for_text = None
    displaying_logistics_map = False

    if view_mode == "SUPERVISOR_VIEW":
        if model.strategy == "supervisor" and model.supervisor_agent_instance: #
            agent_to_view_for_text = model.supervisor_agent_instance #
            if show_supervisor_logistics_map: #
                map_to_display = model.supervisor_agent_instance.supervisor_exploration_logistics_map #
                displaying_logistics_map = True #
            else:
                map_to_display = model.supervisor_agent_instance.supervisor_known_map #
        elif model.strategy == "decentralized":
            map_to_display = model.blackboard_map #
        else:
            view_mode = "NONE"


    elif view_mode == "AGENT_VIEW":
        agents_list_for_view = list(model.agents) #
        if agents_list_for_view and 0 <= selected_agent_idx < len(agents_list_for_view): #
            agent = agents_list_for_view[selected_agent_idx] #
            agent_to_view_for_text = agent #

            if isinstance(agent, SupervisorAgent):
                if show_supervisor_logistics_map and hasattr(agent, 'supervisor_exploration_logistics_map'): #
                    map_to_display = agent.supervisor_exploration_logistics_map #
                    displaying_logistics_map = True #
                elif hasattr(agent, 'supervisor_known_map'): #
                    map_to_display = agent.supervisor_known_map #
            elif isinstance(agent, WorkerAgent) and hasattr(agent, 'worker_internal_map'): #
                map_to_display = agent.worker_internal_map #
            elif isinstance(agent, ResourceCollectorAgent) and hasattr(agent, 'known_map'): #
                map_to_display = agent.known_map #
            else:
                view_mode = "NONE" #
        else:
            view_mode = "NONE" #

    # Karten-Rendering
    if map_to_display is not None:
        for gx in range(model.grid_width_val):  #
            for gy in range(model.grid_height_val):  #
                pygame_rect = pygame.Rect(gx * CELL_WIDTH, SCREEN_HEIGHT - (gy + 1) * CELL_HEIGHT, CELL_WIDTH,
                                          CELL_HEIGHT)  #
                if gx < map_to_display.shape[0] and gy < map_to_display.shape[1]:  #
                    known_status = map_to_display[gx, gy]  #
                    cell_color_to_draw = UNKNOWN_CELL_COLOR  #

                    if displaying_logistics_map:
                        if known_status == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE:  #
                            cell_color_to_draw = LOGISTICS_KNOWN_PASSABLE_COLOR  #
                        elif known_status == SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:  #
                            cell_color_to_draw = LOGISTICS_EXPLORATION_TARGETED_COLOR  #
                        elif known_status == UNKNOWN:  #
                            cell_color_to_draw = UNKNOWN_CELL_COLOR  #
                        else:
                            # Dieser Print sollte erscheinen, wenn die hellbraune Farbe hier entsteht
                            print(
                                f"DEBUG: Unexpected logistics status at ({gx},{gy}): {known_status}. Fallback to Magenta.")
                            cell_color_to_draw = (255, 0, 255)  # Magenta als Fehler/Unbekannter Logistikstatus #

                        # DEBUG: Gib Status und Farbe für jede Zelle der Logistikkarte aus
                        print(f"DEBUG LOGI: ({gx},{gy}) Status: {known_status}, Color: {cell_color_to_draw}")

                    else:  # Farben für öffentliche Supervisor-Karte oder Agenten-Karten
                        # ... (bisherige Logik für nicht-Logistikkarten)
                        if known_status != UNKNOWN:  #
                            if known_status == WOOD_SEEN:
                                cell_color_to_draw = WOOD_COLOR  #
                            elif known_status == STONE_SEEN:
                                cell_color_to_draw = STONE_COLOR  #
                            elif known_status == SUPERVISOR_CLAIMED_RESOURCE:
                                cell_color_to_draw = (160, 160, 160)  #
                            elif known_status == BASE_KNOWN or \
                                    ((gx,
                                      gy) in model.base_coords_list and view_mode == "SUPERVISOR_VIEW" and not displaying_logistics_map):  #
                                cell_color_to_draw = BASE_COLOR  #
                            elif known_status == EMPTY_EXPLORED or known_status == RESOURCE_COLLECTED_BY_ME:  #
                                cell_color_to_draw = WHITE  #
                            else:
                                cell_color_to_draw = WHITE  #
                    pygame.draw.rect(screen, cell_color_to_draw, pygame_rect)
                else:
                    pygame.draw.rect(screen, BLACK, pygame_rect) #
    else:
        for gx_ in range(model.grid_width_val): #
            for gy_ in range(model.grid_height_val): #
                pygame.draw.rect(screen, UNKNOWN_CELL_COLOR, (gx_ * CELL_WIDTH, SCREEN_HEIGHT - (gy_ + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)) #

    # NEU: Korridor Start-/Endpunkte zeichnen, wenn Logistikkarte des Supervisors aktiv ist
    if displaying_logistics_map and model.strategy == "supervisor" and model.supervisor_agent_instance:
        # Iteriere über die in active_corridors_viz gespeicherten Informationen
        # (task_id -> {'entry_U':(x,y), 'end_U':(x,y)})
        for task_id, corridor_info in model.supervisor_agent_instance.active_corridors_viz.items():
            entry_u_pos = corridor_info.get('entry_U')
            end_u_pos = corridor_info.get('end_U')

            if entry_u_pos:
                pygame.draw.circle(screen, CORRIDOR_ENTRY_POINT_COLOR,
                                   (int(entry_u_pos[0] * CELL_WIDTH + CELL_WIDTH / 2),
                                    int(SCREEN_HEIGHT - (entry_u_pos[1] + 1) * CELL_HEIGHT + CELL_HEIGHT / 2)),
                                   int(CELL_WIDTH / 2)) # Zeichne einen Kreis für den Start
            if end_u_pos:
                # Zeichne das Ende etwas anders, z.B. ein Quadrat oder kleineren Kreis
                end_rect = pygame.Rect(end_u_pos[0] * CELL_WIDTH + CELL_WIDTH // 4,
                                       SCREEN_HEIGHT - (end_u_pos[1] + 1) * CELL_HEIGHT + CELL_HEIGHT // 4,
                                       CELL_WIDTH // 2, CELL_HEIGHT // 2)
                pygame.draw.rect(screen, CORRIDOR_END_POINT_COLOR, end_rect)


def draw_agents(screen, model):
    # Unverändert
    for agent in model.agents: #
        if agent.pos is not None: #
            x, y = agent.pos #
            pygame_rect = pygame.Rect(x * CELL_WIDTH, SCREEN_HEIGHT - (y + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT) #
            agent_color_to_draw = AGENT_COLOR #
            if isinstance(agent, SupervisorAgent): agent_color_to_draw = SUPERVISOR_COLOR #
            elif isinstance(agent, WorkerAgent): agent_color_to_draw = WORKER_COLOR #
            pygame.draw.rect(screen, agent_color_to_draw, pygame_rect) #


def draw_base_resource_text(screen, model, font):
    # Unverändert
    if not model.base_deposit_point: return #
    wood_text = f"Wood: {model.base_resources_collected['wood']}/{model.resource_goals.get('wood', 0)}" #
    stone_text = f"Stone: {model.base_resources_collected['stone']}/{model.resource_goals.get('stone', 0)}" #
    wood_surface = font.render(wood_text, True, TEXT_COLOR); #
    stone_surface = font.render(stone_text, True, TEXT_COLOR) #
    text_x = (model.base_coords_list[0][0]) * CELL_WIDTH #
    text_y_base = SCREEN_HEIGHT - (max(c[1] for c in model.base_coords_list) + 1) * CELL_HEIGHT - 5 #
    screen.blit(wood_surface, (text_x, text_y_base - wood_surface.get_height())); #
    screen.blit(stone_surface, (text_x, text_y_base)) #


def draw_view_mode_text(screen, view_mode, selected_agent_idx, model, font, show_supervisor_logistics_map):
    # Unverändert
    text_lines = [] #
    text_lines.append(f"Strategie: {model.strategy}") #
    agents_list_for_view = list(model.agents) #

    if view_mode == "SUPERVISOR_VIEW": #
        if model.strategy == "supervisor": #
            map_type = "Logistik (P)" if show_supervisor_logistics_map else "Öffentlich (0)" #
            text_lines.append(f"View: Supervisor Karte [{map_type}]") #
        else:
            text_lines.append("View: Blackboard (Taste 0)") #
    elif view_mode == "AGENT_VIEW": #
        if agents_list_for_view and 0 <= selected_agent_idx < len(agents_list_for_view): #
            agent = agents_list_for_view[selected_agent_idx] #
            agent_id_str = getattr(agent, 'display_id', agent.unique_id) #
            agent_type_str = agent.__class__.__name__.replace("Agent", "") #
            text_lines.append(f"View: {agent_type_str} {agent_id_str} (Taste {selected_agent_idx + 1})") #
            if isinstance(agent, SupervisorAgent) and show_supervisor_logistics_map: #
                text_lines.append("  -> zeigt Logistik-Karte") #
        else:
            text_lines.append(f"View: Agent (Index {selected_agent_idx + 1} ungültig)") #
    else:
        text_lines.append("View: Global (Keine spezifische Karte)") #

    text_lines.append(f"Step: {model.steps}") #
    if not model.simulation_running and model.completion_step > -1: #
        text_lines.append(f"Ziele erreicht in {model.completion_step} Schritten!") #

    y_offset = 5 #
    for line in text_lines: #
        status_surface = font.render(line, True, TEXT_COLOR) #
        screen.blit(status_surface, (5, y_offset)) #
        y_offset += status_surface.get_height() + 2 #


def run_simulation():
    # Kleinere Anpassung im Tasten-Handling für Logistikkarte
    pygame.init() #
    pygame.font.init() #
    try:
        game_font = pygame.font.SysFont("arial", 18) #
    except pygame.error:
        game_font = pygame.font.Font(None, 24) #

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) #
    pygame.display.set_caption("Age of Empires Lite - Kollaborationsstrategien") #

    CHOSEN_STRATEGY = "supervisor"
    num_agents_for_run = 4 #
    vision_for_run = 2 #

    model = AoELiteModel(strategy=CHOSEN_STRATEGY, num_agents_val=num_agents_for_run, agent_vision_radius=vision_for_run) #

    running = True; clock = pygame.time.Clock(); time_since_last_model_step = 0 #
    current_view_mode = "SUPERVISOR_VIEW" if CHOSEN_STRATEGY == "supervisor" else "AGENT_VIEW" #
    selected_agent_idx_for_view = 0 #
    show_supervisor_logistics_map = False #

    agent_view_keys = { #
        KEY_VIEW_AGENT_1: 0, KEY_VIEW_AGENT_2: 1, KEY_VIEW_AGENT_3: 2, #
        KEY_VIEW_AGENT_4: 3, KEY_VIEW_AGENT_5: 4, KEY_VIEW_AGENT_6: 5, #
        KEY_VIEW_AGENT_7: 6, KEY_VIEW_AGENT_8: 7, KEY_VIEW_AGENT_9: 8, #
    }

    while running: #
        for event in pygame.event.get(): #
            if event.type == pygame.QUIT: running = False #
            if event.type == pygame.KEYDOWN: #
                if event.key == pygame.K_ESCAPE: running = False #
                if event.key == KEY_VIEW_SUPERVISOR_PUBLIC: # Taste 0 #
                    current_view_mode = "SUPERVISOR_VIEW" #
                    # show_supervisor_logistics_map bleibt unverändert, wird mit KEY_VIEW_SUPERVISOR_LOGISTICS umgeschaltet
                elif event.key == KEY_VIEW_SUPERVISOR_LOGISTICS: # Taste P (oder was auch immer in config.py steht) #
                    if model.strategy == "supervisor": #
                        current_view_mode = "SUPERVISOR_VIEW" # Sicherstellen, dass wir im Supervisor-Modus sind #
                        show_supervisor_logistics_map = not show_supervisor_logistics_map #
                elif event.key in agent_view_keys: #
                    idx = agent_view_keys[event.key] #
                    agents_list_for_view_keys = list(model.agents) #
                    if idx < len(agents_list_for_view_keys): #
                        current_view_mode = "AGENT_VIEW" #
                        selected_agent_idx_for_view = idx #

        if model.simulation_running: #
            time_since_last_model_step += clock.get_rawtime() #
            if time_since_last_model_step >= 1000.0 / SIMULATION_STEPS_PER_SECOND: #
                model.step() #
                time_since_last_model_step = 0 #

        if current_view_mode == "AGENT_VIEW": #
            agents_list_for_view_check = list(model.agents) #
            if not (agents_list_for_view_check and 0 <= selected_agent_idx_for_view < len(agents_list_for_view_check)): #
                selected_agent_idx_for_view = 0
                if not agents_list_for_view_check: current_view_mode = "NONE" #

        screen.fill(WHITE) #
        draw_world_view(screen, model, current_view_mode, selected_agent_idx_for_view, show_supervisor_logistics_map) #
        draw_grid(screen) #
        if model.strategy == "decentralized": draw_physical_blackboard(screen, model) #
        draw_agents(screen, model) #
        draw_base_resource_text(screen, model, game_font) #
        draw_view_mode_text(screen, current_view_mode, selected_agent_idx_for_view, model, game_font, show_supervisor_logistics_map) #

        pygame.display.flip() #
        clock.tick(FRAMES_PER_SECOND) #

    pygame.font.quit(); #
    pygame.quit() #


if __name__ == "__main__":
    run_simulation() #