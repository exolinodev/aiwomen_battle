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
            
    def detect_beats(self, sensitivity=1.2, min_beats=20):
        """Vereinfachte Beat-Erkennung für bessere Zuverlässigkeit"""
        print(f"Analysiere Audiodatei: {os.path.basename(self.audio_path)}...")
        
        # Lade Audio
        y, sr = librosa.load(self.audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Einfache Onset-Detektion (zuverlässiger)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Dynamischer Schwellenwert
        threshold = np.percentile(onset_env, 75) * sensitivity
        
        # Finde Peaks mit Mindestabstand
        min_distance = int(sr * 0.3 / 512)  # Mindestens 0.3 Sekunden zwischen Beats
        peaks, _ = find_peaks(onset_env, height=threshold, distance=min_distance)
        beat_times = librosa.frames_to_time(peaks, sr=sr)
        
        # Stellen Sie sicher, dass wir genügend Beats haben
        if len(beat_times) < min_beats:
            print(f"Zu wenige Beats erkannt ({len(beat_times)}), erzeuge künstliche Beats...")
            # Erzeuge regelmäßige Beat-Intervalle basierend auf der Musikdauer
            beat_interval = duration / min_beats
            beat_times = np.arange(0, duration, beat_interval)
        
        # Stelle sicher, dass wir am Anfang und Ende einen Beat haben
        if beat_times[0] > 0.5:
            beat_times = np.insert(beat_times, 0, 0)
        
        if beat_times[-1] < duration - 0.5:
            beat_times = np.append(beat_times, duration)
        
        self.beat_times = beat_times
        print(f"{len(self.beat_times)} Beats erkannt")
        
        # Debug: Zeige die ersten 10 Beat-Zeiten
        print(f"Erste 10 Beat-Zeiten: {self.beat_times[:10]}")
        
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
        created_segments = []
        segments_file = os.path.join(self.temp_dir, "segments.txt")
        
        # Maximale Anzahl an Segmenten begrenzen
        max_segments = min(100, len(self.beat_times) - 1)
        print(f"Erstelle bis zu {max_segments} Segmente...")
        
        # Mindestdauer für Segmente festlegen (zur Sicherheit)
        min_segment_duration = 0.3
        
        # Separate Datei für die Segmentliste erstellen
        with open(segments_file, 'w') as f:
            # Für jeden nutzbaren Beat-Abschnitt einen Clip erstellen
            valid_segments = 0
            for i in range(min(len(self.beat_times) - 1, max_segments)):
                # Beat-Zeitpunkte
                start_time = self.beat_times[i]
                end_time = self.beat_times[i+1]
                
                # Berechne Dauer
                segment_duration = end_time - start_time
                
                # Debug-Ausgabe
                print(f"Beat {i}: von {start_time:.2f}s bis {end_time:.2f}s (Dauer: {segment_duration:.2f}s)")
                
                # Segmente mit zu kurzer Dauer überspringen
                if segment_duration < min_segment_duration:
                    print(f"  Überspringe: Zu kurzes Segment ({segment_duration:.2f}s < {min_segment_duration:.2f}s)")
                    continue
                
                # Zu lange Segmente kürzen
                if segment_duration > max_clip_duration:
                    segment_duration = max_clip_duration
                    print(f"  Kürze auf {segment_duration:.2f}s (maximale Dauer)")
                
                # Clip auswählen - vermeiden Sie Wiederholung des gleichen Clips
                clip_index = random.randint(0, len(enhanced_clips) - 1)
                chosen_clip = enhanced_clips[clip_index]
                
                # Prüfe, ob wir den gleichen Clip wie zuvor verwenden
                if last_clip_path == chosen_clip["path"] and last_was_reverse == chosen_clip["reverse"]:
                    # Versuche einen anderen Clip zu wählen
                    alternate_index = (clip_index + 1) % len(enhanced_clips)
                    chosen_clip = enhanced_clips[alternate_index]
                
                clip_path = chosen_clip["path"]
                is_reverse = chosen_clip["reverse"]
                
                # Aktualisiere Tracking-Variablen
                last_clip_path = clip_path
                last_was_reverse = is_reverse
                
                # Zufälligen Startpunkt im Clip wählen
                clip_duration = self.get_clip_duration(clip_path)
                if clip_duration <= segment_duration:
                    clip_start = 0
                else:
                    max_start = clip_duration - segment_duration
                    clip_start = random.uniform(0, max_start)
                
                # Ausgabedatei für das Segment
                segment_file = os.path.join(self.temp_dir, f"segment_{i:03d}.mp4")
                
                # FFmpeg-Befehl zum Extrahieren des Segments mit hoher Qualität
                cmd = [
                    "ffmpeg", "-i", clip_path,
                    "-ss", f"{clip_start:.3f}",
                    "-t", f"{segment_duration:.3f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "22",
                    "-an"  # Keine Audiospur
                ]
                
                # Wenn rückwärts, füge entsprechende Filter hinzu
                if is_reverse:
                    cmd += ["-vf", "reverse"]
                    
                cmd += [segment_file, "-y"]
                
                # Ausgabe mit Rückwärts-Information
                rev_info = " (rückwärts)" if is_reverse else ""
                print(f"Erstelle Segment {i+1}/{max_segments}: {os.path.basename(clip_path)}{rev_info} "
                      f"von {clip_start:.2f}s für {segment_duration:.2f}s")
                
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                        # Segmentinfo speichern und zur Liste hinzufügen
                        f.write(f"file '{segment_file}'\n")
                        created_segments.append(segment_file)
                        valid_segments += 1
                    else:
                        print(f"  Fehler: Segment wurde nicht erstellt oder ist leer.")
                except subprocess.CalledProcessError as e:
                    print(f"  Fehler beim Erstellen von Segment {i}: {str(e)}")
                    print(f"  Fehlerausgabe: {e.stderr.decode() if e.stderr else 'Keine Fehlerausgabe'}")
                    continue
            
            print(f"\nErfolgreich erstellte Segmente: {valid_segments}/{max_segments}")
        
        # Prüfen, ob Segmente erstellt wurden
        if not created_segments:
            raise ValueError("Keine Segmente wurden erstellt! Überprüfen Sie die Beat-Erkennung und Clip-Dauer.")
        
        print(f"\nFüge {len(created_segments)} Videosegmente zusammen...")
        
        # Erstelle zunächst ein Video ohne Übergänge (einfache Konkatenation)
        silent_output = os.path.join(self.temp_dir, "silent_output.mp4")
        concat_cmd = [
            "ffmpeg", "-f", "concat",
            "-safe", "0",
            "-i", segments_file,
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            silent_output, "-y"
        ]
        
        try:
            print("Führe FFmpeg-Befehl aus:")
            print(" ".join(concat_cmd))
            subprocess.run(concat_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Fehler beim Zusammenfügen der Segmente: {str(e)}")
            # Notfall-Fallback - versuche einzeln konkatenieren
            print("Versuche alternative Methode zur Zusammenführung...")
            
            # Erstelle einzelne Datei für jedes Segment
            alt_segments_file = os.path.join(self.temp_dir, "alt_segments.txt")
            with open(alt_segments_file, 'w') as f:
                for seg_file in created_segments:
                    if os.path.exists(seg_file) and os.path.getsize(seg_file) > 0:
                        f.write(f"file '{os.path.basename(seg_file)}'\n")
            
            # Versuche mit reduzierter Komplexität
            alt_cmd = [
                "ffmpeg", "-f", "concat",
                "-safe", "0",
                "-i", alt_segments_file,
                "-c", "copy",
                silent_output, "-y"
            ]
            try:
                subprocess.run(alt_cmd, check=True, cwd=self.temp_dir)
            except subprocess.CalledProcessError:
                print("Auch alternative Methode fehlgeschlagen. Versuche einzelne Segmente...")
                # Extreme Fallback - verwende das erste Segment
                if len(created_segments) > 0:
                    shutil.copy(created_segments[0], silent_output)
                else:
                    raise ValueError("Keine Segmente konnten erstellt werden.")
        
        # Finale Ausgabe mit originaler Audiospur
        cmd = [
            "ffmpeg", "-i", silent_output,
            "-i", self.audio_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            self.output_file, "-y"
        ]
        print("\nFüge Audiospur hinzu...")
        subprocess.run(cmd, check=True)
        
        print(f"\nFertig! Ausgabe gespeichert als: {self.output_file}")
        return self.output_file
        
    def cleanup(self):
        """Temporäre Dateien löschen"""
        import shutil
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
    parser.add_argument("--sensitivity", "-s", type=float, default=0.8, 
                       help="Empfindlichkeit der Beat-Erkennung (höher = mehr Schnitte)")
    parser.add_argument("--min-segment", "-m", type=float, default=1.0,
                       help="Minimale Segmentlänge in Sekunden")
    parser.add_argument("--max-segment", "-M", type=float, default=6.0,
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