# batch_run.py
import pandas as pd
from mesa.batchrunner import batch_run
from src.model import AoELiteModel
from src.config import NUM_AGENTS, MIN_EXPLORE_TARGET_SEPARATION

# 1. Definiere die festen Parameter für die Simulation.
fixed_params = {
    "strategy": "decentralized",
    "num_agents_val": NUM_AGENTS,
    "min_explore_target_separation_cfg": MIN_EXPLORE_TARGET_SEPARATION
}

# 2. Führe den Batch-Run aus.
if __name__ == '__main__':
    print(f"Starte 10 Baseline-Runs für die Supervisor-Strategie mit {NUM_AGENTS} Agenten...")

    # ÄNDERUNG: Wir entfernen 'data_collection_period'.
    # Der Runner sammelt nun bei jedem Schritt, was robuster ist.
    results_list = batch_run(
        model_cls=AoELiteModel,
        parameters=fixed_params,
        iterations=100,
        max_steps=4000,
        display_progress=True
    )
    print("Batch-Run abgeschlossen.")

    # 3. Konvertiere die Ergebnisliste in einen Pandas DataFrame
    results_df = pd.DataFrame(results_list)

    # 4. Analysiere die Ergebnisse - ROBUSTE METHODE
    print("\n--- Analyse der Baseline ---")

    # Wir wollen nur die letzte Zeile von jedem einzelnen Lauf (identifiziert durch "RunId")
    final_steps_df = results_df.loc[results_df.groupby("RunId")["Step"].idxmax()]

    # Entferne Läufe, die das Ziel nicht erreicht haben (completion_step war -1)
    successful_runs_df = final_steps_df[final_steps_df['CompletionSteps'] != -1].copy()

    if not successful_runs_df.empty:
        # Berechne die Kernstatistiken für die Abschlussrunden
        mean_steps = successful_runs_df['CompletionSteps'].mean()
        std_dev_steps = successful_runs_df['CompletionSteps'].std()
        min_steps = successful_runs_df['CompletionSteps'].min()
        max_steps = successful_runs_df['CompletionSteps'].max()

        print(f"Ergebnisse aus {len(successful_runs_df)} erfolgreichen Läufen:")
        print(f"  -> Durchschnittliche Abschlussrunden: {mean_steps:.2f}")
        print(f"  -> Standardabweichung:              {std_dev_steps:.2f}")
        print(f"  -> Bester Lauf (wenigste Runden):     {int(min_steps)}")
        print(f"  -> Schlechtester Lauf (meiste Runden):  {int(max_steps)}")

        try:
            successful_runs_df.to_csv("supervisor_baseline_results.csv")
            print("\nZusammengefasste Ergebnisse wurden in 'supervisor_baseline_results.csv' gespeichert.")
        except Exception as e:
            print(f"\nFehler beim Speichern der CSV-Datei: {e}")
    else:
        print("\nKeine der Simulationen hat das Ziel innerhalb des max_steps-Limits von 4000 erreicht.")