import pandas as pd
from mesa.batchrunner import batch_run
from src.model import AoELiteModel
from src.config import NUM_AGENTS

# ====================================================================================
# BATCH-RUN ZUR OPTIMIERUNG DES BLACKBOARD-SYNC-INTERVALLS
# (Angepasst an die bestehende batch_run.py Struktur)
# ====================================================================================

# 1. Definiere die Parameter für die Simulation.
#    'batch_run' erstellt automatisch alle Kombinationen der hier definierten Parameter.
parameters = {
    "strategy": "decentralized",
    "num_agents_val": NUM_AGENTS,
    # Dieser Parameter wird variiert:
    "blackboard_sync_interval": range(30, 151, 10) # Testet Intervalle von 30, 40, 50, ..., 150
}

# 2. Führe den Batch-Lauf aus.
if __name__ == '__main__':
    print("Starte Batch-Lauf zur Optimierung des Blackboard-Sync-Intervalls...")

    # 'batch_run' führt die Simulation für jeden Wert des blackboard_sync_interval durch.
    # Die Anzahl der 'iterations' gilt pro einzelnem Intervall-Wert.
    results_list = batch_run(
        model_cls=AoELiteModel,
        parameters=parameters,
        iterations=10,  # Führe jeden Intervall-Wert 10 Mal aus
        max_steps=4000,
        # 'data_collection_period=-1' stellt sicher, dass nur Enddaten gesammelt werden,
        # was bei diesem Aufbau effizienter ist. Wir brauchen nur das Endergebnis.
        data_collection_period=-1,
        display_progress=True
    )
    print("Batch-Lauf abgeschlossen.")

    # 3. Konvertiere die Ergebnisliste in einen Pandas DataFrame
    results_df = pd.DataFrame(results_list)

    # 4. Analysiere die Ergebnisse
    print("\n--- Analyse der Ergebnisse ---")

    # Ersetze nicht abgeschlossene Läufe (-1) durch die maximale Schrittzahl für eine faire Analyse
    results_df['CompletionSteps'] = results_df['CompletionSteps'].apply(lambda x: 4000 if x == -1 else x)

    # Gruppiere die Daten nach dem getesteten Intervall und berechne den Durchschnitt der Schritte.
    average_completion_steps = results_df.groupby("blackboard_sync_interval")["CompletionSteps"].mean()

    # Finde den Intervall-Wert, der im Durchschnitt die wenigsten Schritte benötigt hat.
    if not average_completion_steps.empty:
        best_interval = average_completion_steps.idxmin()
        min_steps = average_completion_steps.min()

        print("\nDurchschnittliche Schritte bis zur Fertigstellung pro getestetem Intervall:")
        print(average_completion_steps)

        print("\n--- Fazit ---")
        print(f"Der optimale Sync-Intervall ist: {best_interval} Schritte")
        print(f"(führte zu durchschnittlich {min_steps:.0f} Schritten bis zum Ziel).")

        try:
            # Speichere die aggregierten Ergebnisse
            average_completion_steps.to_csv("blackboard_optimization_summary.csv")
            print("\nZusammengefasste Ergebnisse wurden in 'blackboard_optimization_summary.csv' gespeichert.")
        except Exception as e:
            print(f"\nFehler beim Speichern der CSV-Datei: {e}")
    else:
        print("Es konnten keine Ergebnisse für die Analyse gefunden werden.")