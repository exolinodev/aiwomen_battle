#!/usr/bin/env python3
import subprocess
import os
import numpy as np
import argparse
import random
from glob import glob
import librosa
from scipy.signal import find_peaks
import shutil
import sys

class RhythmicVideoEditor:
    def __init__(self, clips_dir, audio_path, output_file=None, temp_dir=None):
        self.clips_dir = clips_dir
        self.audio_path = audio_path
        
        if output_file is None:
            output_file = os.path.join(os.path.dirname(clips_dir), "final_rhythmic_video.mp4")
        self.output_file = output_file
        
        if temp_dir is None:
            temp_dir = os.path.join(os.path.dirname(output_file), "temp")
        self.temp_dir = temp_dir
        
        # Stellen Sie sicher, dass das temporäre Verzeichnis existiert und leer ist
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Liste für alle gefundenen Clips
        self.video_clips = []
        self.beat_times = []
        self.segments = []

    def find_video_clips(self, extensions=None):
        """Alle Videoclips im angegebenen Ordner finden"""
        if extensions is None:
            extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv']
            
        self.video_clips = []
        for ext in extensions:
            self.video_clips.extend(glob(os.path.join(self.clips_dir, f'*{ext}')))
            
        if not self.video_clips:
            raise FileNotFoundError(f"Keine Videoclips im Ordner {self.clips_dir} gefunden")
            
        print(f"{len(self.video_clips)} Videoclips gefunden:")
        for clip in self.video_clips:
            print(f" - {os.path.basename(clip)}")
            
        return self.video_clips
    
    def get_clip_duration(self, clip_path):
        """Dauer eines Clips ermitteln"""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", clip_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            print(f"Warnung: Konnte Dauer für {clip_path} nicht ermitteln")
            return 5.0  # Standard-Dauer zurückgeben
            
    def detect_beats(self, sensitivity=1.2, min_beats=10):
        """Verbesserte Beat-Erkennung mit mehreren Methoden und Optimierungen"""
        print(f"Analysiere Audiodatei: {os.path.basename(self.audio_path)}...")
        
        # Lade Audio mit erhöhter Genauigkeit
        y, sr = librosa.load(self.audio_path, sr=44100)
        
        # Kombiniere mehrere Beat-Erkennungsmethoden für bessere Ergebnisse
        beats = []
        
        # 1. Onset-Erkennung (erkennt plötzliche Änderungen im Audiosignal)
        print("Methode 1: Onset-Detektion...")
        onset_env = librosa.onset.onset_strength(
            y=y, 
            sr=sr,
            hop_length=512,
            aggregate=np.median  # Robustere Aggregation
        )
        
        # Dynamischer Schwellenwert basierend auf Perzentilen statt Mittelwert
        threshold = np.percentile(onset_env, 75) * sensitivity
        peaks, _ = find_peaks(
            onset_env, 
            height=threshold,
            distance=sr/512/4  # Mindestabstand zwischen Peaks: 1/4 Takt bei 120 BPM
        )
        beat_times_1 = librosa.frames_to_time(peaks, sr=sr, hop_length=512)
        
        # 2. Tempogram-basierte Beat-Erkennung
        print("Methode 2: Tempogram-Analyse...")
        # Dynamisches Tempo-Tracking
        tempo, beats_frames = librosa.beat.beat_track(
            y=y, 
            sr=sr,
            hop_length=512,
            tightness=100  # Strengere Taktbindung
        )
        beat_times_2 = librosa.frames_to_time(beats_frames, sr=sr, hop_length=512)
        
        # 3. Spektralfluss-basierte Beat-Erkennung
        print("Methode 3: Spektralfluss-Analyse...")
        spec = np.abs(librosa.stft(y, hop_length=512))
        spec_flux = np.sum(np.diff(spec, axis=0), axis=0)
        spec_flux = np.maximum(spec_flux, 0.0)
        
        # Normalisieren und anwenden eines gleitenden Durchschnitts
        spec_flux = spec_flux / np.max(spec_flux)
        window_size = 5
        weights = np.hamming(window_size)
        spec_flux_smooth = np.convolve(spec_flux, weights/weights.sum(), mode='same')
        
        # Schwellenwert auf Basis des geglätteten Spektralflusses
        sf_threshold = np.percentile(spec_flux_smooth, 75) * sensitivity
        sf_peaks, _ = find_peaks(spec_flux_smooth, height=sf_threshold, distance=sr/512/4)
        beat_times_3 = librosa.frames_to_time(sf_peaks, sr=sr, hop_length=512)
        
        # 4. RMS-Energie-basierte Beat-Erkennung (gut für Bass-Beats)
        print("Methode 4: Energiebasierte Detektion...")
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        rms_threshold = np.percentile(rms, 75) * sensitivity
        rms_peaks, _ = find_peaks(rms, height=rms_threshold, distance=sr/512/4)
        beat_times_4 = librosa.frames_to_time(rms_peaks, sr=sr, hop_length=512)
        
        # Intelligente Kombination aller Methoden
        print("Kombiniere Ergebnisse aller Methoden...")
        
        # Sammle alle Beats in einer Liste
        all_beats = np.concatenate([beat_times_1, beat_times_2, beat_times_3, beat_times_4])
        
        # Gruppiere nah beieinander liegende Beats (Clustering)
        all_beats = np.sort(all_beats)
        grouped_beats = []
        last_beat = -1
        min_beat_distance = 0.1  # 100ms Mindestabstand
        
        for beat in all_beats:
            if last_beat == -1 or beat - last_beat >= min_beat_distance:
                grouped_beats.append(beat)
                last_beat = beat
        
        self.beat_times = np.array(grouped_beats)
        
        # Bestimme das dominante Tempo und fülle Lücken
        if len(self.beat_times) >= 2:
            # Berechne Abstände zwischen den Beats
            beat_diffs = np.diff(self.beat_times)
            
            # Schätze das dominante Tempo aus den Abständen (Mode/Median)
            from scipy import stats
            avg_beat_length = np.median(beat_diffs)
            
            # Füge Beats hinzu, wo große Lücken sind (über 2x durchschnittlicher Beat-Länge)
            enhanced_beats = [self.beat_times[0]]
            for i in range(1, len(self.beat_times)):
                current_diff = self.beat_times[i] - self.beat_times[i-1]
                if current_diff > 2.2 * avg_beat_length:
                    # Füge synthetische Beats in die Lücke ein
                    num_missing = int(current_diff / avg_beat_length) - 1
                    for j in range(1, num_missing + 1):
                        synthetic_beat = self.beat_times[i-1] + j * avg_beat_length
                        enhanced_beats.append(synthetic_beat)
                enhanced_beats.append(self.beat_times[i])
            
            self.beat_times = np.array(enhanced_beats)
        
        # Stellen Sie sicher, dass wir genügend Beats haben
        if len(self.beat_times) < min_beats:
            print(f"Zu wenige Beats erkannt ({len(self.beat_times)}), erzeuge künstliche Beats...")
            # Bestimme automatisch ein sinnvolles Tempo
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Versuche, ein musikalisch sinnvolles Tempo zu ermitteln
            if tempo < 40:  # Wenn Tempo-Schätzung zu niedrig ist
                tempo = 120  # Standard-Tempo
                
            beat_interval = 60.0 / tempo  # Umrechnung BPM in Sekundenabstand
            self.beat_times = np.arange(0, duration, beat_interval)
        
        # Fügen Sie den Anfang und das Ende hinzu
        if self.beat_times[0] > 0.5:
            self.beat_times = np.insert(self.beat_times, 0, 0)
        
        duration = librosa.get_duration(y=y, sr=sr)
        if self.beat_times[-1] < duration - 0.5:
            self.beat_times = np.append(self.beat_times, duration)
        
        # Sortieren und Duplikate entfernen
        self.beat_times = np.unique(self.beat_times)
        
        print(f"{len(self.beat_times)} Beats erkannt")
        return self.beat_times
        
    def create_beat_synchronized_video(self, min_clip_duration=0.5, max_clip_duration=5.0, transition_duration=0.5):
        """Video erstellen, das im Rhythmus der Musik geschnitten ist, mit weichen Übergängen"""
        if len(self.beat_times) < 2:
            raise ValueError("Nicht genügend Beats erkannt!")
            
        if not self.video_clips:
            raise ValueError("Keine Videoclips gefunden!")
            
        # Erstelle eine erweiterte Liste von Clips mit verschiedenen Variationen
        enhanced_clips = []
        
        # Füge jeden Clip und seine rückwärts-Version hinzu
        for clip in self.video_clips:
            enhanced_clips.append({"path": clip, "reverse": False})
            enhanced_clips.append({"path": clip, "reverse": True})
        
        # Shuffle mehrmals, um bessere Durchmischung zu garantieren
        for _ in range(3):
            random.shuffle(enhanced_clips)
        
        # Hilfsvariable, um zu verfolgen, welcher Clip zuletzt verwendet wurde
        last_clip_path = None
        last_was_reverse = False
        used_count = {clip: 0 for clip in self.video_clips}
        
        # Liste für alle erstellten Segmente
        raw_segments = []
        
        # Für jeden Beat-Abschnitt einen Clip erstellen
        for i in range(len(self.beat_times) - 1):
            # Beat-Zeitpunkte
            start_time = self.beat_times[i]
            end_time = self.beat_times[i+1]
            
            # Berücksichtige die Übergangsdauer (erweitere die Dauer etwas)
            duration = (end_time - start_time) + transition_duration
            
            # Zu kurze Segmente überspringen, zu lange kürzen
            if duration < min_clip_duration:
                continue
            if duration > max_clip_duration:
                duration = max_clip_duration
            
            # Intelligentere Clip-Auswahl
            # 1. Vermeide Wiederholung des gleichen Clips
            # 2. Bevorzuge weniger benutzte Clips
            
            # Beginne mit einem zufälligen Index
            start_idx = random.randint(0, len(enhanced_clips) - 1)
            
            # Gehe die erweiterte Clipliste durch, beginnend beim zufälligen Index
            chosen_clip = None
            for j in range(len(enhanced_clips)):
                idx = (start_idx + j) % len(enhanced_clips)
                clip_info = enhanced_clips[idx]
                clip_path = clip_info["path"]
                is_reverse = clip_info["reverse"]
                
                # Vermeide den gleichen Clip zweimal hintereinander
                if clip_path == last_clip_path and is_reverse == last_was_reverse:
                    continue
                
                # Vermeide den ersten Clip, wenn er bereits häufig verwendet wurde
                if clip_path == self.video_clips[0] and used_count[clip_path] > len(self.video_clips):
                    # Überspringen, wenn es der erste Clip ist und schon oft benutzt wurde
                    continue
                
                # Wir haben einen geeigneten Clip gefunden
                chosen_clip = clip_info
                break
            
            # Falls keiner der Clips passt, nehme einen zufälligen (sollte selten vorkommen)
            if chosen_clip is None:
                chosen_clip = random.choice(enhanced_clips)
            
            clip_path = chosen_clip["path"]
            is_reverse = chosen_clip["reverse"]
            
            # Aktualisiere Tracking-Variablen
            last_clip_path = clip_path
            last_was_reverse = is_reverse
            used_count[clip_path] = used_count.get(clip_path, 0) + 1
            
            # Zufälligen Startpunkt im Clip wählen
            clip_duration = self.get_clip_duration(clip_path)
            if clip_duration <= duration:
                clip_start = 0
            else:
                max_start = clip_duration - duration
                clip_start = random.uniform(0, max_start)
            
            # Ausgabedatei für das Segment
            segment_file = os.path.join(self.temp_dir, f"segment_{i:03d}.mp4")
            
            # FFmpeg-Befehl zum Extrahieren des Segments
            cmd = [
                "ffmpeg", "-i", clip_path,
                "-ss", f"{clip_start:.3f}",
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-c:a", "aac",
                "-an",  # Keine Audiospur
            ]
            
            # Wenn rückwärts, füge entsprechende Filter hinzu
            if is_reverse:
                cmd += ["-vf", "reverse"]
                
            cmd += [segment_file, "-y"]
            
            # Ausgabe mit Rückwärts-Information
            rev_info = " (rückwärts)" if is_reverse else ""
            print(f"Erstelle Segment {i+1}/{len(self.beat_times)-1}: {os.path.basename(clip_path)}{rev_info} "
                  f"von {clip_start:.2f}s für {duration:.2f}s")
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Segmentinfo speichern
                segment_info = {
                    "file": segment_file,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "index": i
                }
                raw_segments.append(segment_info)
            except subprocess.CalledProcessError as e:
                print(f"Fehler beim Erstellen von Segment {i}: {str(e)}")
                print(f"Fehlerausgabe: {e.stderr.decode() if e.stderr else 'Keine Fehlerausgabe'}")
                continue
        
        # Prüfen, ob Segmente erstellt wurden
        if not raw_segments:
            raise ValueError("Keine Segmente wurden erstellt!")
        
        print(f"\nErstelle Übergänge zwischen den Segmenten...")
        
        # Temporäre Datei für die endgültige Liste der Segmente mit Übergängen
        segments_file = os.path.join(self.temp_dir, "segments.txt")
        
        # Erstelle eine Xfade-Filterkette für alle Segmente
        segments_with_transitions = []
        
        with open(segments_file, 'w') as f:
            # Für jedes Segment einzeln eine Datei erstellen (ohne Übergänge)
            for i, segment in enumerate(raw_segments):
                # Schreibe das Segment in die Dateiliste
                f.write(f"file '{segment['file']}'\n")
                
                # Wenn es nicht das letzte Segment ist, setze Outpoint vor dem Ende
                # um Überlappung für Crossfade zu ermöglichen
                if i < len(raw_segments) - 1:
                    # Duration anpassen, um Überlappung für Übergang zu berücksichtigen
                    actual_duration = segment['duration'] - transition_duration
                    if actual_duration > 0:
                        f.write(f"outpoint {actual_duration:.3f}\n")
                
                # Wenn es nicht das erste Segment ist, verwende Crossfade
                if i > 0:
                    f.write(f"inpoint {transition_duration:.3f}\n")
        
        # Alle Segmente zusammenfügen (ohne Audio) mit Übergängen
        silent_output = os.path.join(self.temp_dir, "silent_output.mp4")
        cmd = [
            "ffmpeg", "-f", "concat", 
            "-safe", "0", 
            "-i", segments_file, 
            "-c:v", "libx264", 
            "-video_track_timescale", "30000",  # Genaue Timeline für Übergänge
            "-movflags", "+faststart",
            silent_output, "-y"
        ]
        print("\nFüge Videosegmente zusammen...")
        subprocess.run(cmd, check=True)
        
        # Finale Ausgabe mit originaler Audiospur
        cmd = [
            "ffmpeg", "-i", silent_output,
            "-i", self.audio_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac",
            "-shortest",  # Beende wenn kürzeste Spur (Video oder Audio) endet
            self.output_file, "-y"
        ]
        print("\nFüge Audiospur hinzu...")
        subprocess.run(cmd, check=True)
        
        print(f"\nFertig! Ausgabe gespeichert als: {self.output_file}")
        return self.output_file
        
        # Prüfen, ob Segmente erstellt wurden
        if not segments_list or not os.path.exists(segments_list[0]):
            raise ValueError("Keine Segmente wurden erstellt!")
            
        print(f"Insgesamt {len(segments_list)} Segmente erstellt")
        
        # Dateiliste überprüfen
        with open(segments_file, 'r') as f:
            content = f.read()
            print(f"Inhalt der Segment-Liste (ersten 200 Zeichen):\n{content[:200]}...")
            
        # Alle Segmente zusammenfügen (ohne Audio)
        silent_output = os.path.join(self.temp_dir, "silent_output.mp4")
        cmd = [
            "ffmpeg", "-f", "concat", 
            "-safe", "0", 
            "-i", segments_file, 
            "-c", "copy", 
            silent_output, "-y"
        ]
        print("\nFüge Videosegmente zusammen...")
        subprocess.run(cmd, check=True)
        
        # Finale Ausgabe mit originaler Audiospur
        cmd = [
            "ffmpeg", "-i", silent_output,
            "-i", self.audio_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac",
            "-shortest",  # Beende wenn kürzeste Spur (Video oder Audio) endet
            self.output_file, "-y"
        ]
        print("\nFüge Audiospur hinzu...")
        subprocess.run(cmd, check=True)
        
        print(f"\nFertig! Ausgabe gespeichert als: {self.output_file}")
        return self.output_file
        
    def cleanup(self):
        """Temporäre Dateien löschen"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Temporäre Dateien gelöscht: {self.temp_dir}")

def main():
    # Feste Pfadangaben
    input_dir = "/Users/dev/womanareoundtheworld/Music_sync/input"
    output_dir = "/Users/dev/womanareoundtheworld/Music_sync/output"
    
    # Prüfen, ob die Verzeichnisse existieren
    if not os.path.exists(input_dir):
        print(f"Fehler: Eingabeverzeichnis existiert nicht: {input_dir}")
        return 1
    
    # Ausgabeverzeichnis erstellen, falls es nicht existiert
    os.makedirs(output_dir, exist_ok=True)
    
    # Finde Audio-Datei im Eingabeverzeichnis
    audio_files = []
    for ext in ['.mp3', '.wav', '.m4a', '.aac', '.flac']:
        audio_files.extend(glob(os.path.join(input_dir, f'*{ext}')))
    
    if not audio_files:
        print(f"Fehler: Keine Audiodateien im Verzeichnis gefunden: {input_dir}")
        return 1
    
    # Erste gefundene Audiodatei verwenden
    audio_file = audio_files[0]
    print(f"Verwende Audiodatei: {os.path.basename(audio_file)}")
    
    # Standard-Ausgabedatei im Ausgabeverzeichnis
    output_file = os.path.join(output_dir, "final_music_sync_video.mp4")
    temp_dir = os.path.join(output_dir, "temp")
    
    # Command-Line-Parameter für zusätzliche Konfiguration
    parser = argparse.ArgumentParser(description="Erstellt ein Video aus Clips, synchronisiert zum Rhythmus der Musik")
    parser.add_argument("--sensitivity", "-s", type=float, default=1.2, 
                       help="Empfindlichkeit der Beat-Erkennung (höher = mehr Schnitte)")
    parser.add_argument("--min-segment", "-m", type=float, default=0.5,
                       help="Minimale Segmentlänge in Sekunden")
    parser.add_argument("--max-segment", "-M", type=float, default=4.0,
                       help="Maximale Segmentlänge in Sekunden")
    parser.add_argument("--transition", "-t", type=float, default=0.5,
                       help="Übergangsdauer in Sekunden (Crossfade)")
    parser.add_argument("--keep-temp", "-k", action="store_true",
                       help="Temporäre Dateien nicht löschen")
    
    args = parser.parse_args()
    
    # Editor initialisieren
    editor = RhythmicVideoEditor(
        clips_dir=input_dir,
        audio_path=audio_file,
        output_file=output_file,
        temp_dir=temp_dir
    )
    
    try:
        # Prozess ausführen
        editor.find_video_clips()
        editor.detect_beats(sensitivity=args.sensitivity)
        output_file = editor.create_beat_synchronized_video(
            min_clip_duration=args.min_segment,
            max_clip_duration=args.max_segment,
            transition_duration=args.transition
        )
        
        if not args.keep_temp:
            editor.cleanup()
            
        print(f"\nErfolgreich abgeschlossen!\nVideo gespeichert als: {output_file}")
        
    except Exception as e:
        import traceback
        print(f"Fehler: {str(e)}")
        print(traceback.format_exc())
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())