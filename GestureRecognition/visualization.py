import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GestureRecognition.hmmclassifier import HMMClassifier

def load_dataset(data_path):
    """Lädt den Trainingsdatensatz oder gibt None zurück, falls nicht vorhanden/ungültig."""
    if not os.path.exists(data_path):
        print(f"Fehler: '{data_path}' nicht gefunden. Bitte erst Datensatz erstellen!")
        return None
    try:
        with open(data_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return None


def visualize_dataset(data_path="data/training_data.pkl", max_examples=8, save_path="data/dataset_visualization.png"):
    """
    TODO: Visualisierung des eigenen Datensatzes

    Ziel:
    -----
    Entwickle eine Möglichkeit, deinen aufgenommenen Datensatz visuell zu
    inspizieren und zu verstehen.

    Warum ist das wichtig?
    ----------------------
    - Du musst nachvollziehen können, was dein Modell eigentlich „sieht“
    - Fehler im Datensatz lassen sich visuell oft sofort erkennen
    - Qualität der Daten ist entscheidend für die Modellperformance

    Anforderungen / Ideen:
    ----------------------
    - Lade deinen Trainingsdatensatz
    - Visualisiere mehrere Sequenzen pro Klasse
    - Stelle sicher, dass:
        - unterschiedliche Gesten klar unterscheidbar sind
        - Sequenzen sinnvoll aussehen (keine Ausreißer, keine leeren Daten)

    .. tip::
       Ein einfacher Ansatz:
         - Plotte Trajektorien (z. B. x/y-Koordinaten)
         - Zeige mehrere Beispiele pro Klasse übereinander

    .. note::
       Du kannst selbst entscheiden:
         - Wie viele Sequenzen du anzeigst
         - Welche Features du visualisierst
         - Ob du interaktive Elemente einbaust

    .. tip::
       Interaktivität (z. B. Klick auf eine Sequenz) kann hilfreich sein,
       um einzelne Beispiele genauer zu untersuchen.

    Abgabe:
    -------
    - Du musst in der Lage sein, deinen Datensatz visuell zu präsentieren
    - Du solltest erklären können:
        - Wie unterscheiden sich die Klassen?
        - Gibt es problematische Beispiele?

    Erweiterung (optional):
    -----------------------
    - Mittelwerte oder typische Sequenzen pro Klasse darstellen
    - Ausreißer automatisch erkennen
    """

    data_dict = load_dataset(data_path)
    if not data_dict:
        return
 
    labels = sorted(label for label, seqs in data_dict.items() if seqs)
    if not labels:
        print("Keine gültigen Sequenzen im Datensatz gefunden.")
        return

    n_cols = int(np.ceil(np.sqrt(len(labels))))
    n_rows = int(np.ceil(len(labels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
 
    for ax, label in zip(axes.flat, labels):
        for seq in data_dict[label][:max_examples]:
            x, y = np.array(seq, dtype=float).T
            line, = ax.plot(x, y, alpha=0.8)
            ax.scatter(x[0], y[0], color=line.get_color(), marker="o")  
            ax.scatter(x[-1], y[-1], color=line.get_color(), marker="s")  
 
        ax.set_title(f"{label.upper()} (n={len(data_dict[label])})")
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
 
    for ax in axes.flat[len(labels):]:
        ax.axis("off")  
 
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Visualisierung gespeichert unter: {save_path}")
    plt.show()


def evaluate_classifier():
    """
    TODO: Evaluation deines Klassifikators

    Ziel:
    -----
    Implementiere eine sinnvolle Auswertung deines Modells auf Testdaten.

    Warum ist das wichtig?
    ----------------------
    - Du brauchst objektive Metriken für die Qualität deines Modells
    - Training allein reicht nicht, entscheidend ist die Generalisierung

    Anforderungen / Ideen:
    ----------------------
    - Lade ein trainiertes Modell
    - Lade Testdaten (getrennt vom Training!)
    - Berechne Vorhersagen
    - Vergleiche Vorhersagen mit Ground Truth

    Metriken:
    ---------
    - Klassifikationsgenauigkeit (Accuracy)
    - Confusion Matrix

    .. tip::
       Eine Confusion Matrix zeigt dir:
         - Welche Klassen gut erkannt werden
         - Wo dein Modell Fehler macht

    .. warning::
       Testdaten dürfen **nicht** aus dem Training stammen!

    Interpretation:
    ---------------
    Du solltest erklären können:
    - Welche Klassen gut funktionieren
    - Welche Klassen verwechselt werden
    - Warum das passieren könnte

    .. note::
       Schlechte Performance liegt oft an:
         - schlechten Trainingsdaten
         - zu wenigen Beispielen
         - ungeeigneten Features

    Erweiterung (optional):
    -----------------------
    - Weitere Metriken (Precision, Recall, F1)
    - Vergleich verschiedener Modelle
    """

    data_dict = load_dataset("data/training_data.pkl")
    if not data_dict:
        return

    X_all = []
    y_all = []
    for label, seqs in data_dict.items():
        for seq in seqs:
            X_all.append(seq)
            y_all.append(label.upper())

    if not X_all:
        print("Keine gültigen Sequenzen für die Evaluation gefunden.")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42
        )

    print("Daten erfolgreich geteilt:")
    print(f"-> {len(X_train)} Trainings-Sequenzen")
    print(f"-> {len(X_test)} Test-Sequenzen\n")

    train_dict = {label: [] for label in set(y_train)}
    for x, y in zip(X_train, y_train):
        train_dict[y].append(x)

    print("Trainiere temporäres Modell auf 80% der Daten...")
    clf = HMMClassifier()
    clf.fit(train_dict)

    print("\nEvaluiere Modell auf den unbekannten Testdaten...")
    y_pred = []
    y_true_valid = []

    for x_test, true_label in zip(X_test, y_test):
        best_label, scores = clf.predict(x_test)
        if best_label is not None:
            y_pred.append(best_label.upper())
            y_true_valid.append(true_label)


    acc = accuracy_score(y_true_valid, y_pred)
    print("\n" + "---------------------------------------------")
    print(f"ERGEBNIS: Genauigkeit (Accuracy) = {acc * 100:.2f} %")
    print("---------------------------------------------" + "\n")

    labels = sorted(list(set(y_true_valid)))
    cm = confusion_matrix(y_true_valid, y_pred, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - Gestenerkennung (Accuracy: {acc*100:.1f}%)', fontsize=14)
    plt.ylabel('Tatsächliche Geste (Ground Truth)', fontsize=12)
    plt.xlabel('Vom Modell erkannte Geste (Prediction)', fontsize=12)
    plt.tight_layout()
    
    print("Zeige Confusion Matrix an... (Schließe das Fenster, um das Programm zu beenden)")
    plt.show()


def replay_recordings(data_path="data/training_data.pkl"):
    """
    TODO: Exploration und Replay der aufgenommenen Rohdaten

    Ziel:
    -----
    Ermögliche es, aufgenommene Sequenzen erneut abzuspielen
    und qualitativ zu überprüfen.

    Warum ist das wichtig?
    ----------------------
    - Du kannst überprüfen, ob deine Aufnahmen korrekt sind
    - Fehler in der Datenerfassung werden früh sichtbar
    - Du entwickelst ein besseres Verständnis für deine Daten

    Anforderungen / Ideen:
    ----------------------
    - Lade gespeicherte Aufnahmen
    - Spiele diese erneut ab (z. B. über SignalHub / Replay-Modus)
    - Iteriere über verschiedene Labels und Beispiele

    .. tip::
       Besonders hilfreich:
         - Vergleiche mehrere Beispiele derselben Klasse
         - Suche nach inkonsistenten Bewegungen

    .. warning::
       Schlechte oder inkonsistente Aufnahmen führen fast immer zu
       schlechten Modellen. Überprüfe deine Daten frühzeitig!

    Abgabe:
    -------
    - Du solltest zeigen können, wie deine Daten aussehen (Replay)
    - Du solltest erklären können:
        - Welche Beispiele gut sind
        - Welche problematisch sind

    Erweiterung (optional):
    -----------------------
    - Automatisches Filtern schlechter Sequenzen
    - Kombination mit Visualisierung
    """
    
    data_dict = load_dataset(data_path)
    if not data_dict:
        return


    labels = sorted([key for key, seqs in data_dict.items() if seqs])
    if not labels:
        print("Keine gültigen Sequenzen gefunden.")
        return

    print(f"Verfügbare Klassen: {[label.upper() for label in labels]}")
    user_input = input("Welche Klasse abspielen? ").strip().strip("'\"").lower()

    key_lookup = {key.lower(): key for key in data_dict.keys()}
    label = key_lookup.get(user_input)

    if label is None or not data_dict[label]:
        print(f"Unbekannte oder leere Klasse: '{user_input}'")
        print(f"Vorhandene Keys (exakt): {list(data_dict.keys())}")
        return

    sequences = data_dict[label]
    plt.ion()

    fig, ax = plt.subplots(figsize=(5, 5))

    for i, seq in enumerate(sequences):
        seq = np.array(seq, dtype=float)
        ax.clear()
        ax.set_title(f"{label.upper()} - Sequenz {i + 1}/{len(sequences)}")
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")

        for frame in range(2, len(seq) + 1):
            ax.plot(seq[:frame, 0], seq[:frame, 1], color="steelblue")
            ax.scatter(seq[frame - 1, 0], seq[frame - 1, 1], color="red", zorder=3)
            plt.pause(0.02)

        if input("Enter = nächste Sequenz, q = beenden: ").strip().lower() == "q":
            break

    plt.ioff()
    plt.close(fig)

if __name__ == "__main__":
    while True:
        print("\n" + "---------------------------------------------")
        print("VISUALISIERUNG & EVALUATION")
        print("---------------------------------------------")
        print("1: Datensatz visualizieren (Trajektorien pro Klasse)")
        print("2: Klassifikator evaluieren (Confusion Matrix)")
        print("3: Aufnahmen abspielen (Replay)")
        print("4: Beenden")
        
        wahl = input("\nBitte wähle (1/2/3/4): ")
        
        if wahl == "1":
            visualize_dataset()
        elif wahl == "2":
            evaluate_classifier()
        elif wahl == "3":
            replay_recordings()
        elif wahl == "4":
            break