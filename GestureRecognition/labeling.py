import os
import pickle
import subprocess
import sys
import numpy as np
import msvcrt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def data_labeling(label: str, times: int):
    """
    TODO: data_labeling: Datenerfassung für Gesten (SignalHub)

    Ziel:
    -----
    Implementiere eine Funktion, mit der Trainingsdaten für eine bestimmte
    Geste aufgenommen und gespeichert werden können.

    Anforderungen / Ideen:
    ----------------------

    1. Aufnahme starten

       - Starte SignalHub über einen Subprocess mit ``--mode record``
       - Übergib einen Dateipfad für die Aufnahme ``--recorder <path_to_save_recording_at>.pkl``
       - Überlege, welche Module aufgenommen werden sollen
       - Nimm entsprechende Änderungen in der ``config.yaml`` vor

    2. Interaktive Steuerung (optional)

       - Implementiere eine einfache Benutzerinteraktion:
         - Aufnahme speichern
         - Aufnahme verwerfen
         - Programm beenden

    .. tip::

       Die Funktion ``getch()`` (Aus dem Modul Linux :mod:`getch` oder bei Windows :mod:`msvcrt`) ist sehr hilfreich, um einzelne Tastendrücke
       direkt auszulesen (ohne Enter). Damit kannst du dir ein schnelles
       Labeling-Interface bauen.

       Beispiel:

       .. code-block:: text

           ESC → speichern
           andere Taste → verwerfen

    3. Daten sichten und bereinigen

       - Lade die aufgenommenen Daten
       - Überlege:
         - Welche Teile sind relevant?
         - Welche Frames sind leer oder unbrauchbar?
         - Sollten gewisse Sequenzen evtl. gar nicht benutzt werden?
       - Entferne unnötige Anteile (z. B. keine erkannte Hand am Anfang/Ende)

    4. Speicherung

       - Speichere Daten strukturiert nach Labels (z. B. Ordnerstruktur)
       - Jede Aufnahme sollte einzeln gespeichert werden

    .. note::

       Die konkrete Umsetzung (Dateiformat, Struktur, Ablauf) ist bewusst offen.
       Entwickle ein System, das für dich sinnvoll ist und sich gut weiterverarbeiten lässt.

    .. warning::

       Ziel ist nicht nur, dass es „funktioniert“, sondern ein sauberer und
       effizienter Workflow für Datensammlung.

    Parameters
    ----------
    label : str
        Name der Geste / Klasse.
        Kann ebenfalls frei gestaltet werden (z. B. dynamische Labels, mehrere Klassen gleichzeitig).

    times : int
        Wie viele Aufnahmen gemacht werden sollen.
        Kann frei angepasst werden (z. B. Endlosschleife oder interaktive Steuerung).
    """

    dataset_dir = f"dataset/{label}"
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"\n--- Starte Labeling für Geste: '{label.upper()}' ---")
    print("Anleitung: Mache die Geste vor der Kamera. Fenster schließen mit ESC.")

    count = 1
    while count <= times:
        print(f"\nAufnahme {count} von {times} läuft...")
        
        temp_file = os.path.abspath("temp_record.pickle").replace("\\", "/")
        
        if os.path.exists(temp_file):
            os.remove(temp_file)

        subprocess.run([
            sys.executable, 
            "main.py", 
            "--mode", "record",
            "--recorder.file", temp_file
        ])

        print("\n-> Aktion wählen: [ESC] Speichern | [w] Wiederholen | [q] Abbrechen")
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8", errors="ignore").lower()

                if key == "\x1b":
                    target_file = os.path.join(dataset_dir, f"{label}_{count}.pkl")
                    if os.path.exists(temp_file):
                        os.replace(temp_file, target_file)
                        print(f"Erfolgreich gespeichert: {target_file}")
                    count += 1
                    break
                    
                elif key == "w":
                    print("Aufnahme verworfen. Wird wiederholt.")
                    break

                elif key == "q":
                    print("Labeling-Prozess abgebrochen.")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    return


def extract_trajectory(data, signal="preprocessor"):
    """
    Extrahiert die längste (= vollständigste) Trajektorie eines Signals
    aus einer aufgezeichneten Datei.
 
    Erwartete Struktur von ``data``:
        {signalname: [frame_dict, frame_dict, ...]}
 
    Jeder ``frame_dict`` ist entweder leer (``{}``, z.B. Start-/Stop-Frames
    des Frameworks) oder enthält den Wert des Signals unter dem Key
    ``signal`` -- entweder ``None`` (noch nicht genug Punkte gesammelt)
    oder eine wachsende Liste von [x, y]-Punkten.
 
    Da die Trajektorie beim Preprocessor über die Zeit anwächst, ist die
    längste gefundene Liste automatisch die vollständigste Aufnahme vor
    einem Reset (z.B. durch Handverlust).
 
    Parameters
    ----------
    data : dict
        Geladene Aufnahme (z.B. via ``pickle.load``).
    signal : str
        Name des Signals, dessen Trajektorie extrahiert werden soll.
 
    Returns
    -------
    np.ndarray oder None
        Die längste gefundene Trajektorie als ``(N, 2)`` Array, oder
        ``None``, falls keine gefunden wurde.
    """
    frames = data.get(signal, []) if isinstance(data, dict) else []
 
    best = None
    for frame in frames:
        val = frame.get(signal) if isinstance(frame, dict) else None
        if val is not None and (best is None or len(val) > len(best)):
            best = val
 
    return np.array(best, dtype=float) if best is not None else None
 
 
def dataset_building(output_path="data/training_data.pkl"):
    """
    TODO: dataset_building: Trainingsdatensatz aus aufgenommenen Gesten erstellen

    Ziel:
    -----
    Implementiere eine Funktion, die alle aufgenommenen Daten lädt,
    verarbeitet und in eine Form bringt, die von eurem
    Hidden-Markov-Modell (HMM) Classifier verwendet werden kann.

    Anforderungen / Ideen:
    ----------------------

    1. Daten laden

       - Durchsuche deinen Trainingsdaten-Ordner
       - Organisiere Daten nach Labels

    2. Feature-Extraktion / Preprocessing

       - Überlege:
         - Welche Features braucht dein Modell?
         - Wie transformierst du die Rohdaten sinnvoll?
       - Wende eine konsistente Verarbeitung auf alle Sequenzen an

    3. Umgang mit Sequenzen

       - Daten sind zeitliche Sequenzen
       - Achte auf:
         - Unterschiedliche Längen
         - Konsistente Struktur

    4. Validierung

       - Entferne unbrauchbare Daten
         (z. B. zu kurze oder fehlerhafte Sequenzen)

    5. Ausgabeformat

       - Baue den Datensatz so, dass dein HMM direkt damit arbeiten kann
       - Das Format sollst du selbst definieren

    .. note::

       Es gibt hier keine vorgegebene „richtige“ Lösung.
       Wichtig ist, dass dein Datensatz konsistent und nutzbar ist.

    .. tip::

       Denke wie ein System-Designer:
       Wie müssen Daten aussehen, damit Training und Inferenz sauber funktionieren?

    .. warning::

       Inkonsistente Datenstrukturen sind eine der häufigsten Fehlerquellen
       beim Training von Sequenzmodellen.

    Erweiterung (optional):
    -----------------------

    - Normalisierung der Daten
    - Datenaugmentation
    - Debug-Ausgaben oder Visualisierung

    Parameters
    ----------
    output_path : Path or str
        Zielpfad für den erzeugten Trainingsdatensatz.
    """

    print("\nErstelle Trainingsdatensatz...")
    X_dict = {}

    dataset_dir = ["dataset"]
    labels = set()

    for dir in dataset_dir:
        if os.path.exists(dir):
            labels.update(os.listdir(dir))

    for label in sorted(labels):
        X_dict[label] = []
        files = []

        for dir in dataset_dir:
            label_dir = os.path.join(dir, label)
            files = [
                os.path.join(label_dir, f)
                for f in os.listdir(label_dir)
                if f.endswith((".pkl", ".pickle"))
            ]


        for file in files:
            with open(file, "rb") as f:
                try:
                    data = pickle.load(f)
                except Exception:
                    continue

            traj = extract_trajectory(data, signal="preprocessor")

            if traj is not None and len(traj) >= 5:
                X_dict[label].append(traj)

        if X_dict[label]:
            print(f"Klasse '{label.upper()}': {len(X_dict[label])} gültige Sequenzen extrahiert.")
        else:
            del X_dict[label]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(X_dict, f)

    print(f"\nDatensatz erfolgreich gespeichert unter: {output_path}")

    return X_dict


if __name__ == "__main__":
    while True:
        print("\n" + "---------------------------------------------")
        print("LABELING MENÜ")
        print("---------------------------------------------")
        print("1: Neue Geste aufnehmen")
        print("2: Datensatz verarbeiten & KI trainieren")
        print("3: Beenden")

        wahl = input("\nBitte wähle (1/2/3): ")

        if wahl == "1":
            geste = input("Name der Geste: ").strip().lower()
            anzahl = int(input(f"Wie viele Aufnahmen möchtest du für '{geste}' machen?: "))
            data_labeling(geste, anzahl)

        elif wahl == "2":
            X_dict = dataset_building("data/training_data.pkl")

            if X_dict and any(len(v) > 0 for v in X_dict.values()):
                from GestureRecognition.hmmclassifier import HMMClassifier

                print("\nStarte KI Training...")
                model = HMMClassifier()
                model.fit(X_dict)
                model.save("data/hmm.pkl")

                print("\nTraining komplett! Live-Erkennung: python main.py")
            else:
                print("\nFehler: Keine gültigen Daten gefunden.")

        elif wahl == "3":
            break