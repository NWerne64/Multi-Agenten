# src/main.py
import pygame
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GRID_LINE_COLOR, WOOD_COLOR, STONE_COLOR,
    BASE_COLOR, TEXT_COLOR, CELL_WIDTH, CELL_HEIGHT, FRAMES_PER_SECOND, SIMULATION_STEPS_PER_SECOND,
    UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME,
    UNKNOWN_CELL_COLOR, BLACKBOARD_OBJECT_COLOR,
    SUPERVISOR_COLOR, WORKER_COLOR, AGENT_COLOR, SUPERVISOR_CLAIMED_RESOURCE,
    # NEUE IMPORTE für Logistik-Karten-Visualisierung und Keys
    SUPERVISOR_LOGISTICS_KNOWN_PASSABLE, LOGISTICS_KNOWN_PASSABLE_COLOR,
    SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED, LOGISTICS_EXPLORATION_TARGETED_COLOR,
    KEY_VIEW_SUPERVISOR_PUBLIC, KEY_VIEW_SUPERVISOR_LOGISTICS,  # Angepasste Supervisor-View Keys
    KEY_VIEW_AGENT_1, KEY_VIEW_AGENT_2, KEY_VIEW_AGENT_3, KEY_VIEW_AGENT_4,
    KEY_VIEW_AGENT_5, KEY_VIEW_AGENT_6, KEY_VIEW_AGENT_7, KEY_VIEW_AGENT_8, KEY_VIEW_AGENT_9  # Fehlende Keys ergänzt
)
from src.model import AoELiteModel
from src.supervisor_agent import SupervisorAgent  # Sicherstellen, dass SupervisorAgent importiert wird
from src.worker_agent import WorkerAgent
from src.agent import ResourceCollectorAgent


def draw_grid(screen):
    for x in range(0, SCREEN_WIDTH, CELL_WIDTH): pygame.draw.line(screen, GRID_LINE_COLOR, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_HEIGHT): pygame.draw.line(screen, GRID_LINE_COLOR, (0, y), (SCREEN_WIDTH, y))


def draw_physical_blackboard(screen, model):
    if model.strategy == "decentralized" and model.blackboard_coords_list:
        for (bbx, bby) in model.blackboard_coords_list:
            pygame.draw.rect(screen, BLACKBOARD_OBJECT_COLOR,
                             (bbx * CELL_WIDTH, SCREEN_HEIGHT - (bby + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))


def draw_world_view(screen, model, view_mode, selected_agent_idx, show_supervisor_logistics_map):  # Neuer Parameter
    map_to_display = None
    agent_to_view_for_text = None  # Für Textanzeige, wer gerade betrachtet wird
    displaying_logistics_map = False

    if view_mode == "SUPERVISOR_VIEW":  # Umbenannt von BLACKBOARD für Klarheit
        if model.strategy == "supervisor" and model.supervisor_agent_instance:
            agent_to_view_for_text = model.supervisor_agent_instance
            if show_supervisor_logistics_map:
                map_to_display = model.supervisor_agent_instance.supervisor_exploration_logistics_map
                displaying_logistics_map = True
                print(f"[MainDraw] Displaying Supervisor Logistics Map")  # Debug
            else:
                map_to_display = model.supervisor_agent_instance.supervisor_known_map
                print(f"[MainDraw] Displaying Supervisor Public Map")  # Debug
        elif model.strategy == "decentralized":  # Fallback für dezentral, falls K_0 noch so genutzt wird
            map_to_display = model.blackboard_map  # Blackboard bei dezentraler Strategie
            # print(f"[MainDraw] Displaying Blackboard (Decentralized)") # Debug
        else:  # Sollte nicht passieren
            view_mode = "NONE"  # Fallback
            # print(f"[MainDraw] Error: Supervisor instance not found for SUPERVISOR_VIEW") # Debug


    elif view_mode == "AGENT_VIEW":  # Umbenannt von AGENT
        agents_list_for_view = list(model.agents)
        if agents_list_for_view and 0 <= selected_agent_idx < len(agents_list_for_view):
            agent = agents_list_for_view[selected_agent_idx]
            agent_to_view_for_text = agent
            # print(f"[MainDraw] Displaying map for Agent {getattr(agent, 'display_id', agent.unique_id)}") # Debug

            if isinstance(agent, SupervisorAgent):  # Supervisor kann auch als "Agent" ausgewählt werden
                if show_supervisor_logistics_map and hasattr(agent, 'supervisor_exploration_logistics_map'):
                    map_to_display = agent.supervisor_exploration_logistics_map
                    displaying_logistics_map = True
                elif hasattr(agent, 'supervisor_known_map'):
                    map_to_display = agent.supervisor_known_map
            elif isinstance(agent, WorkerAgent) and hasattr(agent, 'worker_internal_map'):
                map_to_display = agent.worker_internal_map
            elif isinstance(agent, ResourceCollectorAgent) and hasattr(agent, 'known_map'):
                map_to_display = agent.known_map
            else:
                view_mode = "NONE"
        else:
            view_mode = "NONE"

    # Karten-Rendering
    if map_to_display is not None:
        for gx in range(model.grid_width_val):
            for gy in range(model.grid_height_val):
                pygame_rect = pygame.Rect(gx * CELL_WIDTH, SCREEN_HEIGHT - (gy + 1) * CELL_HEIGHT, CELL_WIDTH,
                                          CELL_HEIGHT)
                if gx < map_to_display.shape[0] and gy < map_to_display.shape[1]:
                    known_status = map_to_display[gx, gy]
                    cell_color_to_draw = UNKNOWN_CELL_COLOR

                    if displaying_logistics_map:  # Spezifische Farben für Logistik-Karte
                        if known_status == SUPERVISOR_LOGISTICS_KNOWN_PASSABLE:
                            cell_color_to_draw = LOGISTICS_KNOWN_PASSABLE_COLOR
                        elif known_status == SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED:
                            cell_color_to_draw = LOGISTICS_EXPLORATION_TARGETED_COLOR
                        elif known_status == UNKNOWN:  # UNKNOWN auf Logistikkarte
                            cell_color_to_draw = UNKNOWN_CELL_COLOR
                        else:  # Fallback für andere Werte auf Logistikkarte (sollte nicht oft vorkommen)
                            cell_color_to_draw = (255, 0, 255)  # Magenta als Fehler/Unbekannter Logistikstatus

                    else:  # Farben für öffentliche Supervisor-Karte oder Agenten-Karten
                        if known_status != UNKNOWN:
                            if known_status == WOOD_SEEN:
                                cell_color_to_draw = WOOD_COLOR
                            elif known_status == STONE_SEEN:
                                cell_color_to_draw = STONE_COLOR
                            elif known_status == SUPERVISOR_CLAIMED_RESOURCE:
                                cell_color_to_draw = (160, 160, 160)
                            elif known_status == BASE_KNOWN or \
                                    ((gx,
                                      gy) in model.base_coords_list and view_mode == "SUPERVISOR_VIEW" and not displaying_logistics_map):  # Basis auf öffentlicher Sup.Karte
                                cell_color_to_draw = BASE_COLOR
                            elif known_status == EMPTY_EXPLORED or known_status == RESOURCE_COLLECTED_BY_ME:
                                cell_color_to_draw = WHITE
                            else:
                                cell_color_to_draw = WHITE

                    pygame.draw.rect(screen, cell_color_to_draw, pygame_rect)
                else:
                    pygame.draw.rect(screen, BLACK, pygame_rect)
    else:  # Fallback, wenn keine Karte zum Anzeigen da ist
        # print(f"[MainDraw] map_to_display is None. View_mode: {view_mode}") # Debug
        for gx_ in range(model.grid_width_val):
            for gy_ in range(model.grid_height_val):
                pygame.draw.rect(screen, UNKNOWN_CELL_COLOR, (
                    gx_ * CELL_WIDTH, SCREEN_HEIGHT - (gy_ + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))


def draw_agents(screen, model):
    for agent in model.agents:
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
    if not model.base_deposit_point: return
    wood_text = f"Wood: {model.base_resources_collected['wood']}/{model.resource_goals.get('wood', 0)}"
    stone_text = f"Stone: {model.base_resources_collected['stone']}/{model.resource_goals.get('stone', 0)}"
    wood_surface = font.render(wood_text, True, TEXT_COLOR);
    stone_surface = font.render(stone_text, True, TEXT_COLOR)
    text_x = (model.base_coords_list[0][0]) * CELL_WIDTH
    text_y_base = SCREEN_HEIGHT - (max(c[1] for c in model.base_coords_list) + 1) * CELL_HEIGHT - 5
    screen.blit(wood_surface, (text_x, text_y_base - wood_surface.get_height()));
    screen.blit(stone_surface, (text_x, text_y_base))


def draw_view_mode_text(screen, view_mode, selected_agent_idx, model, font,
                        show_supervisor_logistics_map):  # Neuer Parameter
    text_lines = []
    text_lines.append(f"Strategie: {model.strategy}")
    agents_list_for_view = list(model.agents)

    if view_mode == "SUPERVISOR_VIEW":
        if model.strategy == "supervisor":
            map_type = "Logistik (Intern)" if show_supervisor_logistics_map else "Öffentlich (Bestätigt)"
            text_lines.append(f"View: Supervisor Karte [{map_type}] (Taste 0, Umschalten mit ' )")
        else:  # Dezentral
            text_lines.append("View: Blackboard (Taste 0)")  # Dezentral zeigt Blackboard mit K_0
    elif view_mode == "AGENT_VIEW":
        if agents_list_for_view and 0 <= selected_agent_idx < len(agents_list_for_view):
            agent = agents_list_for_view[selected_agent_idx]
            agent_id_str = getattr(agent, 'display_id', agent.unique_id)  # Nutze display_id wenn vorhanden
            agent_type_str = agent.__class__.__name__.replace("Agent", "")  # Kürzerer Typname

            text_lines.append(f"View: {agent_type_str} {agent_id_str} (Taste {selected_agent_idx + 1})")
            if isinstance(agent, SupervisorAgent) and show_supervisor_logistics_map:
                text_lines.append("  -> zeigt Logistik-Karte (Supervisor-Agent)")
        else:
            text_lines.append(f"View: Agent (Index {selected_agent_idx + 1} ungültig)")
    else:  # "NONE"
        text_lines.append("View: Global (Keine spezifische Karte)")

    text_lines.append(f"Step: {model.steps}")
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
        game_font = pygame.font.SysFont("arial", 18)  # Kleinere Schrift für mehr Text
    except pygame.error:
        game_font = pygame.font.Font(None, 24)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Age of Empires Lite - Kollaborationsstrategien")

    CHOSEN_STRATEGY = "supervisor"  # oder "decentralized"
    num_agents_for_run = 4
    vision_for_run = 2

    model = AoELiteModel(strategy=CHOSEN_STRATEGY, num_agents_val=num_agents_for_run,
                         agent_vision_radius=vision_for_run)

    running = True;
    clock = pygame.time.Clock()
    time_since_last_model_step = 0

    # View Modes: "SUPERVISOR_VIEW", "AGENT_VIEW", "NONE"
    current_view_mode = "SUPERVISOR_VIEW" if CHOSEN_STRATEGY == "supervisor" else "AGENT_VIEW"  # Startansicht
    selected_agent_idx_for_view = 0  # Index für die Agentenliste
    show_supervisor_logistics_map = False  # Umschaltbarer Zustand

    # Agenten-Tasten von 1-9 (0 ist für Supervisor/Blackboard)
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

                if event.key == KEY_VIEW_SUPERVISOR_PUBLIC:  # Taste 0
                    if model.strategy == "supervisor":
                        current_view_mode = "SUPERVISOR_VIEW"
                        # show_supervisor_logistics_map bleibt unverändert, kann mit ' umgeschaltet werden
                        print("[MainEvent] View switched to Supervisor Map (Public/Logistics based on toggle)")
                    elif model.strategy == "decentralized":
                        current_view_mode = "SUPERVISOR_VIEW"  # Zeigt Blackboard in draw_world_view
                        print("[MainEvent] View switched to Blackboard (Decentralized)")


                elif event.key == KEY_VIEW_SUPERVISOR_LOGISTICS:  # Taste '
                    if model.strategy == "supervisor":
                        current_view_mode = "SUPERVISOR_VIEW"  # Sicherstellen, dass wir im Supervisor-Modus sind
                        show_supervisor_logistics_map = not show_supervisor_logistics_map
                        map_type = "Logistics" if show_supervisor_logistics_map else "Public"
                        print(f"[MainEvent] Toggled Supervisor Map to: {map_type}")

                elif event.key in agent_view_keys:
                    idx = agent_view_keys[event.key]
                    agents_list_for_view_keys = list(model.agents)
                    if idx < len(agents_list_for_view_keys):
                        current_view_mode = "AGENT_VIEW"
                        selected_agent_idx_for_view = idx
                        # print(f"[MainEvent] View switched to Agent {idx}")
                    else:
                        print(
                            f"Warnung: Agenten-Index {idx} für Ansicht ungültig (nur {len(agents_list_for_view_keys)} Agenten).")

        if model.simulation_running:
            time_since_last_model_step += clock.get_rawtime()
            if time_since_last_model_step >= 1000.0 / SIMULATION_STEPS_PER_SECOND:
                model.step()
                time_since_last_model_step = 0

        # Sicherstellen, dass selected_agent_idx_for_view gültig ist, falls Agenten entfernt werden etc. (unwahrscheinlich hier)
        if current_view_mode == "AGENT_VIEW":
            agents_list_for_view_check = list(model.agents)
            if not (agents_list_for_view_check and 0 <= selected_agent_idx_for_view < len(agents_list_for_view_check)):
                selected_agent_idx_for_view = 0  # Fallback auf ersten Agenten
                if not agents_list_for_view_check: current_view_mode = "NONE"  # Kein Agent zum Anzeigen

        screen.fill(WHITE)
        draw_world_view(screen, model, current_view_mode, selected_agent_idx_for_view, show_supervisor_logistics_map)
        draw_grid(screen)
        if model.strategy == "decentralized": draw_physical_blackboard(screen, model)
        draw_agents(screen, model)
        draw_base_resource_text(screen, model, game_font)
        draw_view_mode_text(screen, current_view_mode, selected_agent_idx_for_view, model, game_font,
                            show_supervisor_logistics_map)

        pygame.display.flip()
        clock.tick(FRAMES_PER_SECOND)

    pygame.font.quit();
    pygame.quit()


if __name__ == "__main__":
    run_simulation()