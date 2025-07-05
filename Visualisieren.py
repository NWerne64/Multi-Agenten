import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Daten erstellen, basierend auf deinem Screenshot.
data = {
    'num_agents_val': [4, 4, 4, 8, 8, 8, 16, 16, 16],
    'map_dimension': [50, 100, 200, 50, 100, 200, 50, 100, 200],
    'AvgCommunicationEvents': [25.35, 28.61, 37.53, 27.08, 31.35, 44.93, 28.45, 36.88, 53.80]
}
df = pd.DataFrame(data)

# 2. Einen professionellen Stil und die Farben für die Grafik festlegen
sns.set_theme(style="whitegrid")
custom_palette = {
    50: 'green',
    100: 'orange',
    200: 'purple'
}

# 3. Die Grafik erstellen
plt.figure(figsize=(10, 6))
plot = sns.lineplot(
    data=df,
    x='num_agents_val',
    y='AvgCommunicationEvents', # Die Spalte für die y-Achse wird geändert
    hue='map_dimension',
    style='map_dimension',
    palette=custom_palette,
    marker='o',
    linewidth=2.5
)

# 4. Titel und Beschriftungen hinzufügen
#    Wie gewünscht, wird der Titel hier direkt in die Grafik eingefügt.
plot.set_title('Kommunikationsaufwand in Abhängigkeit der Agentenanzahl', fontsize=16)
plot.set_xlabel('Anzahl der Agenten', fontsize=12)
plot.set_ylabel('Durchschnittliche Kommunikationsevents', fontsize=12) # Die Beschriftung wird angepasst
plot.set_xticks(df['num_agents_val'].unique())
plot.get_legend().set_title('Kartengröße')

# 5. Grafik speichern und anzeigen
plt.savefig('kommunikationsaufwand_dezentral_mit_titel.png', dpi=300, bbox_inches='tight')
plt.show()