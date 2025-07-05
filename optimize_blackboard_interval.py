# optimize_blackboard_interval_batch1.py
import pandas as pd
from mesa.batchrunner import batch_run
from src.model import AoELiteModel
from src.config import MIN_EXPLORE_TARGET_SEPARATION # Importieren Sie MIN_EXPLORE_TARGET_SEPARATION


# 1. Definiere die festen und variablen Parameter für die Simulation.
fixed_params = {
    "strategy": "decentralized",
    "blackboard_sync_interval": 700, # Optimaler Wert aus vorheriger Analyse
    "min_explore_target_separation_cfg": MIN_EXPLORE_TARGET_SEPARATION, # Aus config.py
    "agent_vision_radius": 4, # Fixiert für diesen Batch
}

# Variable Parameter für Batch 1: Agentenanzahl und Kartengröße
variable_params = {
    "num_agents_val": [4, 8, 16],
    "map_dimension": [100], # Nur quadratische Karten
}

# 2. Führe den Batch-Run aus.
if __name__ == '__main__':
    print(f"Starte Batch 1: Skalierbarkeit (Agentenanzahl & Kartengröße) mit {fixed_params['blackboard_sync_interval']} Sync-Intervall und {fixed_params['agent_vision_radius']} Sichtradius...")

    all_parameters = {}
    all_parameters.update(fixed_params)
    all_parameters.update(variable_params)

    results_list = batch_run(
        model_cls=AoELiteModel,
        parameters=all_parameters,
        iterations=100, # 100 Iterationen pro Kombination
        max_steps=5000,
        display_progress=True
    )
    print("Batch 1 abgeschlossen.")

    # 3. Konvertiere die Ergebnisliste in einen Pandas DataFrame
    results_df = pd.DataFrame(results_list)

    # 4. Analysiere die Ergebnisse
    print("\n--- Analyse Batch 1 Ergebnisse ---")

    grouping_cols = list(variable_params.keys()) # ['num_agents_val', 'map_dimension']

    final_steps_df = results_df.loc[results_df.groupby(grouping_cols + ["RunId"])["Step"].idxmax()]

    grouped_stats = final_steps_df.groupby(grouping_cols).apply(
        lambda x: pd.Series({
            'TotalRuns': len(x),
            'SuccessfulRuns': (x['CompletionSteps'] != -1).sum(),
            'SuccessRate': (x['CompletionSteps'] != -1).mean() * 100,
            'AvgCompletionSteps': x[x['CompletionSteps'] != -1]['CompletionSteps'].mean(),
            'StdCompletionSteps': x[x['CompletionSteps'] != -1]['CompletionSteps'].std(),
            'AvgCommunicationEvents': x['TotalCommunicationEvents'].mean(),
            'StdCommunicationEvents': x['TotalCommunicationEvents'].std(),
        })
    ).reset_index()

    grouped_stats = grouped_stats.sort_values(by=['num_agents_val', 'map_dimension'])

    print(grouped_stats.to_string())

    try:
        grouped_stats.to_csv("batch1_scalability_summary.csv", index=False)
        print("\nZusammengefasste Ergebnisse Batch 1 in 'batch1_scalability_summary.csv' gespeichert.")
        results_df.to_csv("batch1_scalability_raw_results.csv", index=False)
        print("Rohdaten Batch 1 in 'batch1_scalability_raw_results.csv' gespeichert.")
    except Exception as e:
        print(f"\nFehler beim Speichern der CSV-Dateien für Batch 1: {e}")