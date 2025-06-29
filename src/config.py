# src/config.py
import pygame

# Grid-Dimensionen
GRID_WIDTH = 100
GRID_HEIGHT = 100

# PyGame Zellendarstellung
CELL_WIDTH = 6
CELL_HEIGHT = 6
SCREEN_WIDTH = GRID_WIDTH * CELL_WIDTH
SCREEN_HEIGHT = GRID_HEIGHT * CELL_HEIGHT

# Farben (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
AGENT_COLOR = (0, 128, 255) #
GRID_LINE_COLOR = (200, 200, 200) #
WOOD_COLOR = (139, 69, 19) #
STONE_COLOR = (128, 128, 128) #
BASE_COLOR = (205, 133, 63) #
TEXT_COLOR = (50, 50, 50) #
UNKNOWN_CELL_COLOR = (70, 70, 70) #
BLACKBOARD_OBJECT_COLOR = (30, 30, 30) #
SUPERVISOR_COLOR = (255, 165, 0) # Orange #
WORKER_COLOR = (100, 149, 237)   # Hellblau (CornflowerBlue) #

# Farben für Supervisor Logistik-Karte Visualisierung
LOGISTICS_KNOWN_PASSABLE_COLOR = (200, 200, 255) # Sehr helles Blau/Lila #
LOGISTICS_EXPLORATION_TARGETED_COLOR = (255, 200, 200) # Sehr helles Rot/Pink #

# NEU: Farben für Korridor-Visualisierung
CORRIDOR_ENTRY_POINT_COLOR = (0, 255, 0)  # Leuchtendes Grün für den Start des unbekannten Korridorteils
CORRIDOR_END_POINT_COLOR = (255, 0, 255)    # Magenta für das Ende des Korridors


# Ressourcen-Konfiguration
NUM_WOOD_PATCHES = 15 #
NUM_STONE_PATCHES = 15 #
MIN_CLUSTER_SIZE = 3 #
MAX_CLUSTER_SIZE = 10 #

# Basis-Konfiguration
BASE_SIZE = 2 #

# Agenten-Konfiguration
NUM_AGENTS = 4 #
VISION_RANGE_RADIUS = 2 # Wird vom Supervisor für Projektionen genutzt #
BLACKBOARD_SYNC_INTERVAL = 100 # OPTIMIERT am 30.06.2025 basierend auf Batch-Run


# Karten-Zellzustände (Integer-IDs)
UNKNOWN = 0 #
EMPTY_EXPLORED = 1 #
WOOD_SEEN = 2 #
STONE_SEEN = 3 #
BASE_KNOWN = 4 #
RESOURCE_COLLECTED_BY_ME = 5 #
SUPERVISOR_CLAIMED_RESOURCE = 6 #
SUPERVISOR_LOGISTICS_KNOWN_PASSABLE = 8 #
SUPERVISOR_LOGISTICS_EXPLORATION_TARGETED = 9 #

# Blackboard-Konfiguration
BLACKBOARD_SIZE_X = 2 #
BLACKBOARD_SIZE_Y = 1 #

# Tastenbelegung für Ansichtswechsel
KEY_VIEW_SUPERVISOR_PUBLIC = pygame.K_0 # Zeigt öffentliche Karte des Supervisors (oder Blackboard dezentral) #
KEY_VIEW_SUPERVISOR_LOGISTICS = pygame.K_p # Taste: 'p' (statt Apostroph für einfachere Erreichbarkeit) für Logistik-Karte #
KEY_VIEW_AGENT_1 = pygame.K_1 #
KEY_VIEW_AGENT_2 = pygame.K_2 #
KEY_VIEW_AGENT_3 = pygame.K_3 #
KEY_VIEW_AGENT_4 = pygame.K_4 #
KEY_VIEW_AGENT_5 = pygame.K_5 #
KEY_VIEW_AGENT_6 = pygame.K_6 #
KEY_VIEW_AGENT_7 = pygame.K_7 #
KEY_VIEW_AGENT_8 = pygame.K_8 #
KEY_VIEW_AGENT_9 = pygame.K_9 #


# Globale Ressourcenziele
RESOURCE_GOALS = {'wood': 15, 'stone': 10} #

# Simulationsgeschwindigkeit
SIMULATION_STEPS_PER_SECOND = 100 #
FRAMES_PER_SECOND = 30 #

# Für dezentrale Strategie:
INITIAL_EXPLORATION_ANCHORS = [ #
    (0.15, 0.15), (0.85, 0.15), #
    (0.15, 0.85), (0.85, 0.85) ] #
ANCHOR_REACHED_THRESHOLD_DISTANCE = 5 #

# Für Supervisor-gesteuerte Exploration (vereinfacht)
MIN_EXPLORE_TARGET_SEPARATION = 20 #
SUPERVISOR_INITIAL_EXPLORATION_HOTSPOTS = [ # Beibehalten für initiale Streuung #
    (0.15, 0.15), (0.85, 0.15), #
    (0.15, 0.85), (0.85, 0.85) ] #
MIN_UNKNOWN_RATIO_FOR_CONTINUED_EXPLORATION = 0.02 # Weiter explorieren, wenn >2% unbekannt und Ziele nicht erreicht #