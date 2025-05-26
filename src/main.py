# src/main.py

import pygame
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, WHITE,
    AGENT_COLOR, GRID_LINE_COLOR, WOOD_COLOR, STONE_COLOR, BASE_COLOR, TEXT_COLOR,
    CELL_WIDTH, CELL_HEIGHT,
    FRAMES_PER_SECOND, SIMULATION_STEPS_PER_SECOND,
    UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME,
    UNKNOWN_CELL_COLOR, BLACKBOARD_OBJECT_COLOR,  # Neue Farben
    KEY_VIEW_BLACKBOARD, KEY_VIEW_AGENT_1, KEY_VIEW_AGENT_2,  # Tasten
    KEY_VIEW_AGENT_3, KEY_VIEW_AGENT_4, NUM_AGENTS
)
from src.model import AoELiteModel


def draw_grid(screen):
    """Zeichnet das Grid."""
    for x in range(0, SCREEN_WIDTH, CELL_WIDTH):
        pygame.draw.line(screen, GRID_LINE_COLOR, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_HEIGHT):
        pygame.draw.line(screen, GRID_LINE_COLOR, (0, y), (SCREEN_WIDTH, y))


def draw_physical_blackboard(screen, model):
    """Zeichnet das physische Blackboard-Objekt auf der Karte."""
    if model.blackboard_coords_list:
        for (bbx, bby) in model.blackboard_coords_list:
            pygame_rect = pygame.Rect(
                bbx * CELL_WIDTH,
                SCREEN_HEIGHT - (bby + 1) * CELL_HEIGHT,  # y-Koordinate umrechnen
                CELL_WIDTH,
                CELL_HEIGHT
            )
            pygame.draw.rect(screen, BLACKBOARD_OBJECT_COLOR, pygame_rect)


def draw_world_view(screen, model, view_mode, selected_agent_idx):
    """
    Zeichnet die Welt basierend auf dem ausgewählten Ansichtsmodus:
    - "BLACKBOARD": Zeigt den Inhalt des model.blackboard_map.
    - "AGENT": Zeigt die known_map des model.agents[selected_agent_idx].
    """
    map_to_display = None
    if view_mode == "BLACKBOARD":
        map_to_display = model.blackboard_map
    elif view_mode == "AGENT":
        if model.agents and 0 <= selected_agent_idx < len(model.agents):
            map_to_display = model.agents[selected_agent_idx].known_map
        else:  # Fallback, falls Agent nicht existiert oder Index ungültig
            view_mode = "NONE"  # Umschalten auf eine leere Ansicht

    if map_to_display is None and view_mode != "NONE":  # Sollte nicht passieren bei korrekter Logik
        # Zeichne alles als unbekannt, wenn keine gültige Karte zum Anzeigen da ist
        for gx_ in range(model.grid.width):
            for gy_ in range(model.grid.height):
                pygame.draw.rect(screen, UNKNOWN_CELL_COLOR,
                                 (gx_ * CELL_WIDTH, SCREEN_HEIGHT - (gy_ + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
        return
    elif view_mode == "NONE":  # Explizit keine Agentenansicht (z.B. am Anfang, wenn Agenten noch nicht da)
        for gx_ in range(model.grid.width):
            for gy_ in range(model.grid.height):
                pygame.draw.rect(screen, UNKNOWN_CELL_COLOR,
                                 (gx_ * CELL_WIDTH, SCREEN_HEIGHT - (gy_ + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
        return

    for gx in range(model.grid.width):
        for gy in range(model.grid.height):
            pygame_rect = pygame.Rect(
                gx * CELL_WIDTH,
                SCREEN_HEIGHT - (gy + 1) * CELL_HEIGHT,  # y-Koordinate umrechnen
                CELL_WIDTH,
                CELL_HEIGHT
            )
            known_status = map_to_display[gx, gy]
            cell_color_to_draw = UNKNOWN_CELL_COLOR  # Standard für UNKNOWN

            if known_status == UNKNOWN:
                pass  # Bereits auf UNKNOWN_CELL_COLOR gesetzt
            else:  # Zelle ist bekannt
                if known_status == WOOD_SEEN:
                    cell_color_to_draw = WOOD_COLOR
                elif known_status == STONE_SEEN:
                    cell_color_to_draw = STONE_COLOR
                elif known_status == BASE_KNOWN or \
                        (gx,
                         gy) in model.base_coords_list and view_mode == "BLACKBOARD":  # Basis immer als Basis auf BB zeigen
                    cell_color_to_draw = BASE_COLOR
                elif known_status == EMPTY_EXPLORED or known_status == RESOURCE_COLLECTED_BY_ME:
                    cell_color_to_draw = WHITE  # Explizit als leer oder von mir abgebaut
                else:  # Falls ein anderer bekannter Status (unwahrscheinlich mit aktueller Def.)
                    cell_color_to_draw = WHITE

            pygame.draw.rect(screen, cell_color_to_draw, pygame_rect)


def draw_agents(screen, model):
    if hasattr(model, 'agents'):
        for agent in model.agents:
            if agent.pos is not None:
                x, y = agent.pos
                pygame_rect = pygame.Rect(
                    x * CELL_WIDTH,
                    SCREEN_HEIGHT - (y + 1) * CELL_HEIGHT,
                    CELL_WIDTH,
                    CELL_HEIGHT
                )
                pygame.draw.rect(screen, AGENT_COLOR, pygame_rect)
                # Hier könnte man die Agenten-Display-ID zeichnen
                # agent_id_surface = game_font.render(str(agent.agent_display_id), True, BLACK)
                # screen.blit(agent_id_surface, (pygame_rect.centerx - agent_id_surface.get_width() // 2,
                #                               pygame_rect.centery - agent_id_surface.get_height() // 2))


def draw_base_resource_text(screen, model, font):
    if not model.base_deposit_point: return
    wood_text = f"Wood: {model.base_resources_collected['wood']}"
    stone_text = f"Stone: {model.base_resources_collected['stone']}"
    wood_surface = font.render(wood_text, True, TEXT_COLOR)
    stone_surface = font.render(stone_text, True, TEXT_COLOR)
    text_x = (model.base_coords_list[0][0]) * CELL_WIDTH
    text_y_base = SCREEN_HEIGHT - (max(c[1] for c in model.base_coords_list) + 1) * CELL_HEIGHT - 5
    screen.blit(wood_surface, (text_x, text_y_base - wood_surface.get_height()))
    screen.blit(stone_surface, (text_x, text_y_base))


def draw_view_mode_text(screen, view_mode, selected_agent_idx, num_total_agents, font):
    """Zeigt den aktuellen Ansichtsmodus an."""
    text = ""
    if view_mode == "BLACKBOARD":
        text = "View: Blackboard (Key 0)"
    elif view_mode == "AGENT":
        # Agenten-Display-ID ist selected_agent_idx + 1
        text = f"View: Agent {selected_agent_idx + 1} (Key {selected_agent_idx + 1})"

    if text:
        status_surface = font.render(text, True, TEXT_COLOR)
        screen.blit(status_surface, (5, 5))  # Oben links auf dem Bildschirm


def run_simulation():
    pygame.init()
    pygame.font.init()
    try:
        game_font = pygame.font.SysFont("arial", 20)  # Etwas kleinere Schrift für Status
        agent_id_font = pygame.font.SysFont("arial", 12)  # Kleinere Schrift für Agenten-IDs
    except pygame.error:
        game_font = pygame.font.Font(None, 24)
        agent_id_font = pygame.font.Font(None, 18)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Age of Empires Lite - Iteration 3: Blackboard")

    model = AoELiteModel()

    running = True
    clock = pygame.time.Clock()
    time_since_last_model_step = 0

    current_view_mode = "AGENT"  # Startansicht
    # Agenten-Indizes sind 0-basiert, Display-IDs 1-basiert
    selected_agent_idx_for_view = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == KEY_VIEW_BLACKBOARD:
                    current_view_mode = "BLACKBOARD"
                elif event.key == KEY_VIEW_AGENT_1 and NUM_AGENTS >= 1:
                    current_view_mode = "AGENT"
                    selected_agent_idx_for_view = 0
                elif event.key == KEY_VIEW_AGENT_2 and NUM_AGENTS >= 2:
                    current_view_mode = "AGENT"
                    selected_agent_idx_for_view = 1
                elif event.key == KEY_VIEW_AGENT_3 and NUM_AGENTS >= 3:
                    current_view_mode = "AGENT"
                    selected_agent_idx_for_view = 2
                elif event.key == KEY_VIEW_AGENT_4 and NUM_AGENTS >= 4:
                    current_view_mode = "AGENT"
                    selected_agent_idx_for_view = 3

        time_since_last_model_step += clock.get_rawtime()
        if time_since_last_model_step >= 1000.0 / SIMULATION_STEPS_PER_SECOND:
            model.step()
            time_since_last_model_step = 0

        # Wähle den Agenten für die FoW-Ansicht (falls im Agentenmodus)
        # Stelle sicher, dass der Index gültig ist, falls sich NUM_AGENTS ändert und der Index noch alt ist
        if current_view_mode == "AGENT":
            if not (model.agents and 0 <= selected_agent_idx_for_view < len(model.agents)):
                selected_agent_idx_for_view = 0  # Fallback auf ersten Agenten, falls Index ungültig
                if not model.agents:  # Falls gar keine Agenten, zeige nichts Spezifisches
                    current_view_mode = "NONE"

        screen.fill(WHITE)
        draw_world_view(screen, model, current_view_mode, selected_agent_idx_for_view)
        draw_grid(screen)
        draw_physical_blackboard(screen, model)  # Zeichne das physische Blackboard-Objekt
        draw_agents(screen, model)  # Agenten darüber zeichnen
        draw_base_resource_text(screen, model, game_font)
        draw_view_mode_text(screen, current_view_mode, selected_agent_idx_for_view, NUM_AGENTS, game_font)

        pygame.display.flip()
        clock.tick(FRAMES_PER_SECOND)

    pygame.font.quit()
    pygame.quit()


if __name__ == "__main__":
    run_simulation()