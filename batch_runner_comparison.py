import pandas as pd
from mesa.batchrunner import batch_run
# WICHTIG: Stelle sicher, dass dein Modell den neuen Parameter verarbeiten kann.
from src.model import AoELiteModel
import time

# ====================================================================================
# PARAMETER-DEFINITION FÜR DEN VERGLEICHS-RUN
# ====================================================================================

# 1. Definiere die Parameter, die wir vergleichen wollen.
#    In diesem Fall ist es der Typ des Supervisors.
#    HINWEIS: Du musst deine AoELiteModel.__init__ so anpassen, dass sie einen
#    Parameter `supervisor_type` akzeptiert und die entsprechende Agenten-Klasse lädt.
variable_params = {
    "supervisor_type": ["mixed_strategy", "corridor_only"]
}

# 2. Definiere die festen Parameter, die bei jedem Lauf gleich bleiben.
#    Hier sollten die "besten" oder vernünftigsten Werte aus deiner
#    vorherigen Analyse verwendet werden, um einen fairen Vergleich zu gewährleisten.
fixed_params = {
    "num_agents_val": 4,
    "strategy": "supervisor",
    "min_explore_target_separation_cfg": 60,  # Fester Wert für fairen Vergleich
    "min_unknown_ratio_for_continued_exploration_cfg": 0.0001,  # Fester Wert für fairen Vergleich
}

# ====================================================================================
# MODELL-DATEN-SAMMLER
# (unverändert)
# ====================================================================================
model_reporters = {
    "CompletionSteps": lambda m: m.completion_step,
    "FinalStep": lambda m: m.schedule.steps,
    "ExplorationPercentage": lambda m: m.get_exploration_percentage()
}

# ====================================================================================
# BATCHRUNNER AUSFÜHREN
# ====================================================================================
if __name__ == '__main__':
    print("Starte Batchrun (Vergleich: Gemischte Strategie vs. Nur-Korridor)...")
    print(f"Feste Parameter: {fixed_params}")
    print(f"Variable Parameter ({len(variable_params)}):")
    for param, values in variable_params.items():
        print(f"  - {param}: {values}")

    # Kombiniere feste und variable Parameter
    all_parameters = {**fixed_params, **variable_params}

    # Führe den Batch-Run aus
    results_list = batch_run(
        model_cls=AoELiteModel,
        parameters=all_parameters,
        iterations=5,             # 100 Durchläufe pro Strategie
        max_steps=8000,
        display_progress=True
    )
    print("\nBatch-Run abgeschlossen.")

    # ====================================================================================
    # ERGEBNISSE ANALYSIEREN
    # ====================================================================================

    # 1. Konvertiere die Ergebnisliste in einen Pandas DataFrame
    results_df = pd.DataFrame(results_list)

    # 2. Speichere die vollständigen Rohdaten
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    raw_results_filename = f"batch_comparison_raw_results_{timestamp}.csv"
    try:
        results_df.to_csv(raw_results_filename, index=False)
        print(f"\nVollständige Rohdaten wurden in '{raw_results_filename}' gespeichert.")
    except Exception as e:
        print(f"\nFehler beim Speichern der Rohdaten-CSV: {e}")

    # 3. Analysiere die Performance der erfolgreichen Läufe
    print("\n--- Analyse der Performance ---")

    # Nur die letzte Zeile von jedem einzelnen Lauf verwenden
    final_steps_df = results_df.loc[results_df.groupby("RunId")["Step"].idxmax()]

    # Nur die Läufe betrachten, die das Ziel erreicht haben
    successful_runs_df = final_steps_df[final_steps_df['CompletionSteps'] != -1].copy()

    if not successful_runs_df.empty:
        print(f"Analyse basiert auf {len(successful_runs_df)} erfolgreichen von insgesamt {len(final_steps_df)} Läufen.")

        # Gruppiere nach dem Supervisor-Typ und berechne die Statistiken
        parameter_cols = list(variable_params.keys())
        summary = successful_runs_df.groupby(parameter_cols)['CompletionSteps'].agg(
            ['mean', 'std', 'min', 'max', 'count']).reset_index()
        summary = summary.rename(
            columns={
                'mean': 'MeanCompletionSteps',
                'std': 'StdDevCompletionSteps',
                'min': 'MinCompletionSteps',
                'max': 'MaxCompletionSteps',
                'count': 'SuccessfulRuns'
            }
        )

        # Sortiere die Ergebnisse
        summary = summary.sort_values(by='MeanCompletionSteps', ascending=True)

        print(f"\nVergleich der Strategien (sortiert nach schnellster durchschnittlicher Zielerreichung):")
        print(summary.to_string(index=False))

        # Speichere die Zusammenfassung
        summary_filename = f"batch_comparison_summary_{timestamp}.csv"
        try:
            summary.to_csv(summary_filename, index=False)
            print(f"\nPerformance-Zusammenfassung wurde in '{summary_filename}' gespeichert.")
        except Exception as e:
            print(f"\nFehler beim Speichern der Zusammenfassungs-CSV: {e}")
    else:
        print("\nKeine der Simulationen hat das Ziel innerhalb des max_steps-Limits von 8000 erreicht.")