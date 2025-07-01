import pandas as pd
from mesa.batchrunner import batch_run
from src.model import AoELiteModel  # Import des Modells wie von Ihnen gewünscht
import time

# ====================================================================================
# PARAMETER-DEFINITION FÜR DIE BREITE SUCHE (PHASE 1)
# ====================================================================================

# 1. Definiere die Parameter, die wir in Kombination testen wollen.
# WICHTIG: Diese Parameter müssen EXAKT den Argumentnamen der AoELiteModel.__init__-Methode entsprechen.
variable_params = {
    "min_explore_target_separation_cfg": [25, 30], # Entspricht min_explore_target_separation_cfg
    "min_unknown_ratio_for_continued_exploration_cfg": [0.02, 0.7], # Entspricht min_unknown_ratio_for_continued_exploration_cfg
    # agent_vision_radius wurde entfernt, da nicht gewünscht.
    # Weitere Parameter, die nicht direkt in AoELiteModel.__init__ sind, können hier nicht variiert werden.
}

# 2. Definiere die festen Parameter, die bei jedem Lauf gleich bleiben.
fixed_params = {
    "num_agents_val": 4, # KORRIGIERT: Parametername angepasst an AoELiteModel.__init__
    "strategy": "supervisor" # Hinzugefügt aus Ihrer Batch-Run-Vorlage
}

# ====================================================================================
# MODELL-DATEN-SAMMLER
# Definiert, welche Daten wir am Ende jedes Laufs speichern wollen.
# ====================================================================================
model_reporters = {
    # KORRIGIERT: Hier verwenden wir "CompletionSteps" und greifen auf m.completion_step zu,
    # da dies der Name ist, den das Modell intern verwendet und meldet.
    "CompletionSteps": lambda m: m.completion_step,
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
    print(f"Variable Parameter ({len(variable_params)}):")
    for param, values in variable_params.items():
        print(f"  - {param}: {values}")

    # KORREKTUR: Kombiniere feste und variable Parameter in einem einzigen Dictionary
    all_parameters = {**fixed_params, **variable_params}

    # Führe den Batch-Run aus.
    results_list = batch_run(
        model_cls=AoELiteModel,     # Modellklasse
        parameters=all_parameters,  # ALLE Parameter hier übergeben
        iterations=50,             # 100 Durchläufe pro Parameter-Kombination
        max_steps=8000,             # ERHÖHT: Sicherheitslimit pro Simulation
        display_progress=True
    )
    print("\nBatch-Run abgeschlossen.")

    # ====================================================================================
    # ERGEBNISSE ANALYSIEREN
    # ====================================================================================

    # 1. Konvertiere die Ergebnisliste in einen Pandas DataFrame
    results_df = pd.DataFrame(results_list)

    # 2. Filtere ungültige Kombinationen heraus (wichtig!)
    # Diese Zeile wurde ENTFERNT, da 'min_tour_size' und 'default_tour_size'
    # nicht mehr als Spalten in results_df existieren.
    # if 'min_tour_size' in results_df.columns and 'default_tour_size' in results_df.columns:
    #     results_df = results_df[results_df['min_tour_size'] <= results_df['default_tour_size']].copy()
    # else:
    #     print("Hinweis: 'min_tour_size' oder 'default_tour_size' nicht in den Ergebnissen, Filterung übersprungen.")


    # Speichere die vollständigen Rohdaten für eine eventuelle tiefere Analyse
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    raw_results_filename = f"batch_phase1_raw_results_{timestamp}.csv"
    try:
        print(f"Vollständige Rohdaten wurden in '{raw_results_filename}' gespeichert.")
        results_df.to_csv(raw_results_filename, index=False)
    except Exception as e:
        print(f"\nFehler beim Speichern der Rohdaten-CSV: {e}")

    # 3. Analysiere die Performance der erfolgreichen Läufe
    print("\n--- Analyse der Performance ---")

    # Wir wollen nur die letzte Zeile von jedem einzelnen Lauf (identifiziert durch "RunId")
    final_steps_df = results_df.loc[results_df.groupby("RunId")["Step"].idxmax()]

    # Wir wollen nur die Läufe betrachten, die das Ziel auch erreicht haben
    # KORRIGIERT: Filtern anhand von 'CompletionSteps' != -1
    successful_runs_df = final_steps_df[final_steps_df['CompletionSteps'] != -1].copy()

    if not successful_runs_df.empty:
        # Berechne die Kernstatistiken für die Abschlussrunden
        # KORRIGIERT: Nutzen von 'CompletionSteps'
        mean_steps = successful_runs_df['CompletionSteps'].mean()
        std_dev_steps = successful_runs_df['CompletionSteps'].std()
        min_steps = successful_runs_df['CompletionSteps'].min()
        max_steps = successful_runs_df['CompletionSteps'].max()

        print(f"Ergebnisse aus {len(successful_runs_df)} erfolgreichen Läufen:")
        print(f"  -> Durchschnittliche Abschlussrunden: {mean_steps:.2f}")
        print(f"  -> Standardabweichung:              {std_dev_steps:.2f}")
        print(f"  -> Bester Lauf (wenigste Runden):     {int(min_steps)}")
        print(f"  -> Schlechtester Lauf (meiste Runden):  {int(max_steps)}")

        # Gruppiere nach den Parametern und berechne den Durchschnitt der Abschluss-Schritte
        parameter_cols = list(variable_params.keys())

        # Berechne Durchschnitt und Standardabweichung für jede Parameter-Kombination
        # KORRIGIERT: Nutzen von 'CompletionSteps'
        summary = successful_runs_df.groupby(parameter_cols)['CompletionSteps'].agg(['mean', 'std', 'count']).reset_index()
        summary = summary.rename(
            columns={'mean': 'MeanCompletionSteps', 'std': 'StdDevCompletionSteps', 'count': 'SuccessfulRuns'})

        # Sortiere die Ergebnisse, sodass die beste Kombination (schnellster Durchschnitt) oben steht
        summary = summary.sort_values(by='MeanCompletionSteps', ascending=True)

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
        print("\nKeine der Simulationen hat das Ziel innerhalb des max_steps-Limits von 8000 erreicht.")