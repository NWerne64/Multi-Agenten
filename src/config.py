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
AGENT_COLOR = (0, 128, 255)  # Blau
GRID_LINE_COLOR = (200, 200, 200) # Helles Grau

WOOD_COLOR = (139, 69, 19)   # Braun (SaddleBrown)
STONE_COLOR = (128, 128, 128) # Grau
BASE_COLOR = (205, 133, 63)   # Peru (ein anderer Braunton für die Basis)
TEXT_COLOR = (50, 50, 50)     # Dunkelgrau für Text
UNKNOWN_CELL_COLOR = (70, 70, 70) # Farbe für unbekannte Zellen unter FoW
BLACKBOARD_OBJECT_COLOR = (30, 30, 30) # Farbe für das physische Blackboard-Objekt

# Ressourcen-Konfiguration
NUM_WOOD_PATCHES = 15
NUM_STONE_PATCHES = 15
MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SIZE = 10

# Basis-Konfiguration
BASE_SIZE = 2

# Agenten-Konfiguration
NUM_AGENTS = 4 # Erhöht auf 4 Agenten
VISION_RANGE_RADIUS = 2
# EXPLORATION_STEPS_PER_DIRECTION wird für Frontier-Exploration nicht mehr primär genutzt

# Known_map und Blackboard Zellzustände (Integer-IDs)
UNKNOWN = 0
EMPTY_EXPLORED = 1
WOOD_SEEN = 2
STONE_SEEN = 3
BASE_KNOWN = 4
RESOURCE_COLLECTED_BY_ME = 5 # Wenn der Agent selbst hier etwas abgebaut hat
# RESOURCE_COLLECTED_OTHER könnte ein Zustand sein, den das Blackboard setzt,
# wenn es von einem anderen Agenten die Info "abgebaut" bekommt.
# Fürs Erste reicht es, wenn Blackboard direkt auf EMPTY_EXPLORED oder einen generischen "COLLECTED" setzt.
# Die Regel ist: Abgebaut/Leer ist stärker als Gesehen.

# Blackboard-Konfiguration
BLACKBOARD_SIZE_X = 2
BLACKBOARD_SIZE_Y = 1
BLACKBOARD_SYNC_INTERVAL = 75 # Agent geht ca. alle X Schritte periodisch zum Blackboard

# Tastenbelegung für Ansichtswechsel
KEY_VIEW_BLACKBOARD = pygame.K_0
KEY_VIEW_AGENT_1 = pygame.K_1
KEY_VIEW_AGENT_2 = pygame.K_2
KEY_VIEW_AGENT_3 = pygame.K_3
KEY_VIEW_AGENT_4 = pygame.K_4 # Entspricht Agent mit Index 3

# Globale Ressourcenziele für die Basis
RESOURCE_GOALS = {'wood': 15, 'stone': 10}

# Simulationsgeschwindigkeit
SIMULATION_STEPS_PER_SECOND = 20 # Etwas schneller für mehr Interaktion
FRAMES_PER_SECOND = 30

INITIAL_EXPLORATION_ANCHORS = [
    (0.15, 0.15),  # Agent auf Basis[0] ((49,49) -> UL der Karte)
    (0.85, 0.15),  # Agent auf Basis[1] ((50,49) -> UR der Karte)
    (0.15, 0.85),  # Agent auf Basis[2] ((49,50) -> OL der Karte)
    (0.85, 0.85)   # Agent auf Basis[3] ((50,50) -> OR der Karte)
]
ANCHOR_REACHED_THRESHOLD_DISTANCE = 5 # Distanz, bei der ein Ankerpunkt als erreicht gilt