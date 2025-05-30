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
AGENT_COLOR = (0, 128, 255)  # Blau (für ResourceCollectorAgent)
GRID_LINE_COLOR = (200, 200, 200) # Helles Grau

WOOD_COLOR = (139, 69, 19)   # Braun (SaddleBrown)
STONE_COLOR = (128, 128, 128) # Grau
BASE_COLOR = (205, 133, 63)   # Peru (ein anderer Braunton für die Basis)
TEXT_COLOR = (50, 50, 50)     # Dunkelgrau für Text
UNKNOWN_CELL_COLOR = (70, 70, 70) # Farbe für unbekannte Zellen unter FoW
BLACKBOARD_OBJECT_COLOR = (30, 30, 30) # Farbe für das physische Blackboard-Objekt

# NEUE FARBEN für Supervisor/Worker Strategie
SUPERVISOR_COLOR = (255, 165, 0)  # Orange
WORKER_COLOR = (100, 149, 237)     # Hellblau (CornflowerBlue)


# Ressourcen-Konfiguration
NUM_WOOD_PATCHES = 15
NUM_STONE_PATCHES = 15
MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SIZE = 10

# Basis-Konfiguration
BASE_SIZE = 2

# Agenten-Konfiguration (diese Werte dienen als Default, wenn das Modell instanziiert wird)
NUM_AGENTS = 4 # Anzahl der ResourceCollectorAgents oder WorkerAgents
VISION_RANGE_RADIUS = 2
BLACKBOARD_SYNC_INTERVAL = 75 # Agent geht ca. alle X Schritte periodisch zum Blackboard (für dezentrale Strategie)


# Known_map und Blackboard Zellzustände (Integer-IDs)
UNKNOWN = 0
EMPTY_EXPLORED = 1
WOOD_SEEN = 2
STONE_SEEN = 3
BASE_KNOWN = 4
RESOURCE_COLLECTED_BY_ME = 5
SUPERVISOR_CLAIMED_RESOURCE = 6

# Blackboard-Konfiguration (für dezentrale Strategie)
BLACKBOARD_SIZE_X = 2
BLACKBOARD_SIZE_Y = 1

# Tastenbelegung für Ansichtswechsel
KEY_VIEW_BLACKBOARD = pygame.K_0 # Zeigt Blackboard (dezentral) oder Supervisor-Karte (zentral)
KEY_VIEW_AGENT_1 = pygame.K_1
KEY_VIEW_AGENT_2 = pygame.K_2
KEY_VIEW_AGENT_3 = pygame.K_3
KEY_VIEW_AGENT_4 = pygame.K_4
# Weitere Tasten, falls mehr als 4 Agenten plus Supervisor visualisiert werden sollen
KEY_VIEW_AGENT_5 = pygame.K_5
KEY_VIEW_AGENT_6 = pygame.K_6
KEY_VIEW_AGENT_7 = pygame.K_7
KEY_VIEW_AGENT_8 = pygame.K_8
KEY_VIEW_AGENT_9 = pygame.K_9


# Globale Ressourcenziele für die Basis
RESOURCE_GOALS = {'wood': 15, 'stone': 10}

# Simulationsgeschwindigkeit
SIMULATION_STEPS_PER_SECOND = 100
FRAMES_PER_SECOND = 30

# Für dezentrale Strategie:
INITIAL_EXPLORATION_ANCHORS = [
    (0.15, 0.15),
    (0.85, 0.15),
    (0.15, 0.85),
    (0.85, 0.85)
]
ANCHOR_REACHED_THRESHOLD_DISTANCE = 5

# NEU: Für Supervisor-gesteuerte Exploration
MIN_EXPLORE_TARGET_SEPARATION = 20  # Mindestabstand zwischen neuen Explorationszielen des Supervisors

# NEU: Für Worker-gesteuerte Kaskaden-Exploration (Chained Exploration)
CHAINED_EXPLORATION_STEP_BUDGET = 75  # Max. Schritte, die ein Worker für die gesamte Kaskaden-Explo ausgibt
MAX_CHAINED_FRONTIERS_VISITED = 5     # Max. Anzahl von Folge-Frontiers, die ein Worker pro Einsatz besucht
CHAINED_LOCAL_EXPLORE_STEPS = 2       # Anzahl lokaler Schritte nach Erreichen einer Kaskaden-Frontier