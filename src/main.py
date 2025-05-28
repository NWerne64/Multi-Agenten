# src/main.py
# ... (Imports bleiben gleich) ...
import pygame
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, AGENT_COLOR, GRID_LINE_COLOR, WOOD_COLOR, STONE_COLOR,
    BASE_COLOR, TEXT_COLOR, CELL_WIDTH, CELL_HEIGHT, FRAMES_PER_SECOND, SIMULATION_STEPS_PER_SECOND,
    UNKNOWN, EMPTY_EXPLORED, WOOD_SEEN, STONE_SEEN, BASE_KNOWN, RESOURCE_COLLECTED_BY_ME,
    UNKNOWN_CELL_COLOR, BLACKBOARD_OBJECT_COLOR, KEY_VIEW_BLACKBOARD, KEY_VIEW_AGENT_1,
    KEY_VIEW_AGENT_2, KEY_VIEW_AGENT_3, KEY_VIEW_AGENT_4, NUM_AGENTS
)
from src.model import AoELiteModel


# --- Zeichenfunktionen draw_grid, draw_physical_blackboard, draw_world_view,
# --- draw_agents, draw_base_resource_text, draw_view_mode_text
# --- bleiben exakt wie in der letzten Version. ---
def draw_grid(screen):
    for x in range(0, SCREEN_WIDTH, CELL_WIDTH): pygame.draw.line(screen, GRID_LINE_COLOR, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_HEIGHT): pygame.draw.line(screen, GRID_LINE_COLOR, (0, y), (SCREEN_WIDTH, y))


def draw_physical_blackboard(screen, model):
    if model.blackboard_coords_list:
        for (bbx, bby) in model.blackboard_coords_list:
            pygame.draw.rect(screen, BLACKBOARD_OBJECT_COLOR,
                             (bbx * CELL_WIDTH, SCREEN_HEIGHT - (bby + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))


def draw_world_view(screen, model, view_mode, selected_agent_idx):
    map_to_display = None
    if view_mode == "BLACKBOARD":
        map_to_display = model.blackboard_map
    elif view_mode == "AGENT":
        if model.agents and 0 <= selected_agent_idx < len(model.agents):
            map_to_display = model.agents[selected_agent_idx].known_map
        else:
            view_mode = "NONE"
    if map_to_display is None and view_mode != "NONE":
        for gx_ in range(model.grid.width):
            for gy_ in range(model.grid.height): pygame.draw.rect(screen, UNKNOWN_CELL_COLOR, (
            gx_ * CELL_WIDTH, SCREEN_HEIGHT - (gy_ + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
        return
    elif view_mode == "NONE":
        for gx_ in range(model.grid.width):
            for gy_ in range(model.grid.height): pygame.draw.rect(screen, UNKNOWN_CELL_COLOR, (
            gx_ * CELL_WIDTH, SCREEN_HEIGHT - (gy_ + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
        return
    for gx in range(model.grid.width):
        for gy in range(model.grid.height):
            pygame_rect = pygame.Rect(gx * CELL_WIDTH, SCREEN_HEIGHT - (gy + 1) * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            known_status = map_to_display[gx, gy];
            cell_color_to_draw = UNKNOWN_CELL_COLOR
            if known_status != UNKNOWN:
                if known_status == WOOD_SEEN:
                    cell_color_to_draw = WOOD_COLOR
                elif known_status == STONE_SEEN:
                    cell_color_to_draw = STONE_COLOR
                elif known_status == BASE_KNOWN or ((gx, gy) in model.base_coords_list and view_mode == "BLACKBOARD"):
                    cell_color_to_draw = BASE_COLOR
                elif known_status == EMPTY_EXPLORED or known_status == RESOURCE_COLLECTED_BY_ME:
                    cell_color_to_draw = WHITE
                else:
                    cell_color_to_draw = WHITE
            pygame.draw.rect(screen, cell_color_to_draw, pygame_rect)


def draw_agents(screen, model):
    if hasattr(model, 'agents'):
        for agent in model.agents:  # enumerate für potenziellen Farbwechsel
            if agent.pos is not None:
                x, y = agent.pos;
                pygame_rect = pygame.Rect(x * CELL_WIDTH, SCREEN_HEIGHT - (y + 1) * CELL_HEIGHT, CELL_WIDTH,
                                          CELL_HEIGHT)
                pygame.draw.rect(screen, AGENT_COLOR, pygame_rect)


def draw_base_resource_text(screen, model, font):
    if not model.base_deposit_point: return
    wood_text = f"Wood: {model.base_resources_collected['wood']}/{model.resource_goals.get('wood', 0)}"  # Zeige Ziel an
    stone_text = f"Stone: {model.base_resources_collected['stone']}/{model.resource_goals.get('stone', 0)}"  # Zeige Ziel an
    wood_surface = font.render(wood_text, True, TEXT_COLOR);
    stone_surface = font.render(stone_text, True, TEXT_COLOR)
    text_x = (model.base_coords_list[0][0]) * CELL_WIDTH
    text_y_base = SCREEN_HEIGHT - (max(c[1] for c in model.base_coords_list) + 1) * CELL_HEIGHT - 5
    screen.blit(wood_surface, (text_x, text_y_base - wood_surface.get_height()));
    screen.blit(stone_surface, (text_x, text_y_base))


def draw_view_mode_text(screen, view_mode, selected_agent_idx, num_total_agents, font, model_steps):
    text_lines = []
    if view_mode == "BLACKBOARD":
        text_lines.append("View: Blackboard (Key 0)")
    elif view_mode == "AGENT":
        text_lines.append(f"View: Agent {selected_agent_idx + 1} (Key {selected_agent_idx + 1})")
    text_lines.append(f"Step: {model_steps}")  # Zeige Simulationsschritte an

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
        game_font = pygame.font.Font(None, 28)  # Etwas größer für bessere Lesbarkeit

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Age of Empires Lite - Iteration 4: Goals")

    model = AoELiteModel()

    running = True  # Wird jetzt durch model.simulation_running gesteuert
    clock = pygame.time.Clock()
    time_since_last_model_step = 0

    current_view_mode = "AGENT"
    selected_agent_idx_for_view = 0

    while model.simulation_running and running:  # Prüfe beide Flags
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Erlaube manuelles Beenden
            if event.type == pygame.KEYDOWN:
                if event.key == KEY_VIEW_BLACKBOARD:
                    current_view_mode = "BLACKBOARD"
                elif event.key == KEY_VIEW_AGENT_1 and NUM_AGENTS >= 1:
                    current_view_mode = "AGENT"; selected_agent_idx_for_view = 0
                elif event.key == KEY_VIEW_AGENT_2 and NUM_AGENTS >= 2:
                    current_view_mode = "AGENT"; selected_agent_idx_for_view = 1
                elif event.key == KEY_VIEW_AGENT_3 and NUM_AGENTS >= 3:
                    current_view_mode = "AGENT"; selected_agent_idx_for_view = 2
                elif event.key == KEY_VIEW_AGENT_4 and NUM_AGENTS >= 4:
                    current_view_mode = "AGENT"; selected_agent_idx_for_view = 3

        if model.simulation_running:  # Nur Modellschritt ausführen, wenn Simulation noch läuft
            time_since_last_model_step += clock.get_rawtime()
            if time_since_last_model_step >= 1000.0 / SIMULATION_STEPS_PER_SECOND:
                model.step()
                time_since_last_model_step = 0

        if current_view_mode == "AGENT":
            if not (model.agents and 0 <= selected_agent_idx_for_view < len(model.agents)):
                selected_agent_idx_for_view = 0
                if not model.agents: current_view_mode = "NONE"

        screen.fill(WHITE)
        draw_world_view(screen, model, current_view_mode, selected_agent_idx_for_view)
        draw_grid(screen)
        draw_physical_blackboard(screen, model)
        draw_agents(screen, model)
        draw_base_resource_text(screen, model, game_font)
        draw_view_mode_text(screen, current_view_mode, selected_agent_idx_for_view, NUM_AGENTS, game_font,
                            model.current_step)

        pygame.display.flip()
        clock.tick(FRAMES_PER_SECOND)

    # Schleife für die Endanzeige, falls Ziele erreicht wurden
    if not model.simulation_running and model.completion_step > -1:  # Ziele erreicht
        end_font = pygame.font.Font(None, 48)
        end_text = f"Ziele erreicht in {model.completion_step} Schritten!"
        end_surface = end_font.render(end_text, True, TEXT_COLOR, WHITE)
        end_rect = end_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(end_surface, end_rect)
        pygame.display.flip()

        # Warte ein paar Sekunden oder auf Tastendruck/Schließen-Event
        wait_for_exit = True
        while wait_for_exit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    wait_for_exit = False
            clock.tick(10)  # Reduziere CPU-Last im Wartezustand

    pygame.font.quit()
    pygame.quit()


if __name__ == "__main__":
    run_simulation()