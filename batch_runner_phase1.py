# batch_runner_phase1.py
import pandas as pd
from mesa.batchrunner import batch_run
from src.model import ExplorationModel  # Sicherstellen, dass der Modellname korrekt ist
import time

# ====================================================================================
# PARAMETER-DEFINITION FÜR DIE BREITE SUCHE (PHASE 1)
# ====================================================================================

# 1. Definiere die Parameter, die wir in Kombination testen wollen.
variable_params = {
    "MIN_EXPLORE_TARGET_SEPARATION_val": [10, 15, 20, 25],
    "DEFAULT_TOUR_SIZE": [3, 4, 5, 6],
    "MIN_TOUR_SIZE": [2, 3],
    "DEFAULT_CORRIDOR_LENGTH": [20, 30, 40]
}

# 2. Definiere die festen Parameter, die bei jedem Lauf gleich bleiben.
fixed_params = {
    "num_agents_val": 4
}

# ====================================================================================
# MODELL-DATEN-SAMMLER
# Definiert, welche Daten wir am Ende jedes Laufs speichern wollen.
# ====================================================================================
model_reporters = {
    # In welchem Schritt wurde das Ziel erreicht? (-1 falls nicht)
    "GoalMetStep": lambda m: m.goal_met_step,
    # Wie viele Schritte hat die Simulation insgesamt gedauert?
    "FinalStep": lambda m: m.schedule.steps,
    # Wie viel Prozent der Karte wurde am Ende aufgedeckt?
    "ExplorationPercentage": lambda m: m.get_exploration_percentage()
}

# ====================================================================================
# BATCHRUNNER AUSFÜHREN
# ====================================================================================
if __name__ == '__main__':
    print("Starte Batchrun Phase 1 (Breite Suche)...")
    print(f"Feste Parameter: {fixed_params}")

    # Kombiniere feste und variable Parameter für den Batchrunner
    # Mesa's batch_run kann dies direkt verarbeiten
    all_params = {**fixed_params, **variable_params}

    # Führe den Batch-Run aus.
    results_list = batch_run(
        model_cls=ExplorationModel,
        parameters=all_params,
        iterations=10,  # 10 Durchläufe pro Parameter-Kombination
        max_steps=4000,  # Sicherheitslimit pro Simulation
        model_reporters=model_reporters,
        display_progress=True
    )
    print("\nBatch-Run abgeschlossen.")

    # ====================================================================================
    # ERGEBNISSE ANALYSIEREN
    # ====================================================================================

    # 1. Konvertiere die Ergebnisliste in einen Pandas DataFrame
    results_df = pd.DataFrame(results_list)

    # 2. Filtere ungültige Kombinationen heraus (wichtig!)
    # Entferne alle Läufe, bei denen die Mindest-Tour-Größe größer als die maximale war.
    results_df = results_df[results_df['MIN_TOUR_SIZE'] <= results_df['DEFAULT_TOUR_SIZE']].copy()

    # Speichere die vollständigen Rohdaten für eine eventuelle tiefere Analyse
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    raw_results_filename = f"batch_phase1_raw_results_{timestamp}.csv"
    try:
        results_df.to_csv(raw_results_filename, index=False)
        print(f"\Vollständige Rohdaten wurden in '{raw_results_filename}' gespeichert.")
    except Exception as e:
        print(f"\nFehler beim Speichern der Rohdaten-CSV: {e}")

    # 3. Analysiere die Performance der erfolgreichen Läufe
    print("\n--- Analyse der Performance ---")

    # Wir wollen nur die Läufe betrachten, die das Ziel auch erreicht haben
    successful_runs_df = results_df[results_df['GoalMetStep'] != -1].copy()

    if not successful_runs_df.empty:
        # Gruppiere nach den Parametern und berechne den Durchschnitt der Abschluss-Schritte
        parameter_cols = list(variable_params.keys())

        # Berechne Durchschnitt und Standardabweichung für jede Parameter-Kombination
        summary = successful_runs_df.groupby(parameter_cols)['GoalMetStep'].agg(['mean', 'std', 'count']).reset_index()
        summary = summary.rename(
            columns={'mean': 'MeanGoalMetStep', 'std': 'StdDevGoalMetStep', 'count': 'SuccessfulRuns'})

        # Sortiere die Ergebnisse, sodass die beste Kombination (schnellster Durchschnitt) oben steht
        summary = summary.sort_values(by='MeanGoalMetStep', ascending=True)

        print(f"\nTop-Performer (sortiert nach schnellster durchschnittlicher Zielerreichung):")
        print(summary.head(10).to_string(index=False))  # Zeige die Top 10 Kombinationen an

        # Speichere die Zusammenfassung
        summary_filename = f"batch_phase1_summary_{timestamp}.csv"
        try:
            summary.to_csv(summary_filename, index=False)
            print(f"\nPerformance-Zusammenfassung wurde in '{summary_filename}' gespeichert.")
        except Exception as e:
            print(f"\nFehler beim Speichern der Zusammenfassungs-CSV: {e}")
    else:
        print("\nKeine der Simulationen hat das Ziel innerhalb des Limits von 1500 Schritten erreicht.")