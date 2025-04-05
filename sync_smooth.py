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
        all_beats = np.sort(all_beats)
        
        # Kombiniere Beats, die zu nah beieinander liegen
        min_segment_duration = 0.5  # Minimale Segmentdauer in Sekunden
        combined_beats = [all_beats[0]]  # Start mit dem ersten Beat
        
        for beat in all_beats[1:]:
            # Wenn der Abstand zum letzten Beat zu klein ist, überspringe diesen Beat
            if beat - combined_beats[-1] < min_segment_duration:
                continue
            combined_beats.append(beat)
        
        self.beat_times = np.array(combined_beats)
        
        # Stellen Sie sicher, dass wir genügend Beats haben
        if len(self.beat_times) < min_beats:
            print(f"Zu wenige Beats erkannt ({len(self.beat_times)}), erzeuge künstliche Beats...")
            # Bestimme automatisch ein sinnvolles Tempo
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Versuche, ein musikalisch sinnvolles Tempo zu ermitteln
            if tempo < 40:  # Wenn Tempo-Schätzung zu niedrig ist
                tempo = 120  # Standard-Tempo
                
            beat_interval = 60.0 / tempo  # Umrechnung BPM in Sekundenabstand
            
            # Stelle sicher, dass das Intervall mindestens min_segment_duration ist
            if beat_interval < min_segment_duration:
                beat_interval = min_segment_duration
                
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
            
        print(f"\nDebug: Beat times range: {self.beat_times[0]:.2f}s to {self.beat_times[-1]:.2f}s")
        print(f"Debug: Total number of beats: {len(self.beat_times)}")
        
        # Erstelle eine erweiterte Liste von Clips mit verschiedenen Variationen
        enhanced_clips = []
        
        # Füge jeden Clip mehrmals mit verschiedenen Variationen hinzu
        for clip in self.video_clips:
            # Normal vorwärts
            enhanced_clips.append({
                "path": clip,
                "reverse": False,
                "speed": 1.0,
                "start_bias": "random"  # random, start, middle, end
            })
            # Rückwärts
            enhanced_clips.append({
                "path": clip,
                "reverse": True,
                "speed": 1.0,
                "start_bias": "random"
            })
            # Schneller vorwärts
            enhanced_clips.append({
                "path": clip,
                "reverse": False,
                "speed": 1.5,
                "start_bias": "random"
            })
            # Langsamer vorwärts
            enhanced_clips.append({
                "path": clip,
                "reverse": False,
                "speed": 0.75,
                "start_bias": "start"
            })
            # Schneller rückwärts
            enhanced_clips.append({
                "path": clip,
                "reverse": True,
                "speed": 1.5,
                "start_bias": "end"
            })
        
        print(f"Debug: Enhanced clips count: {len(enhanced_clips)}")
        
        # Mehrfaches Shuffling für bessere Durchmischung
        for _ in range(5):
            random.shuffle(enhanced_clips)
        
        # Tracking-Variablen für Clip-Auswahl
        last_clips = []  # Liste der letzten N verwendeten Clips
        max_last_clips = 3  # Wie viele letzte Clips wir tracken
        used_count = {clip: 0 for clip in self.video_clips}
        
        # Liste für alle erstellten Segmente
        raw_segments = []
        
        # Maximale Anzahl an Segmenten begrenzen
        max_segments = min(100, len(self.beat_times) - 1)
        
        print(f"Debug: Will process up to {max_segments} segments")
        
        # Für jeden Beat-Abschnitt einen Clip erstellen
        for i in range(min(len(self.beat_times) - 1, max_segments)):
            # Beat-Zeitpunkte
            start_time = self.beat_times[i]
            end_time = self.beat_times[i+1]
            
            # Berechne die Dauer ohne Übergang
            base_duration = end_time - start_time
            
            print(f"\nDebug: Processing segment {i+1}")
            print(f"Debug: Beat times: {start_time:.2f}s to {end_time:.2f}s")
            print(f"Debug: Base duration: {base_duration:.2f}s")
            
            # Zu kurze Segmente überspringen
            if base_duration < min_clip_duration:
                print(f"Debug: Skipping segment - too short ({base_duration:.2f}s < {min_clip_duration}s)")
                continue
                
            # Die erweiterte Dauer für das Extrahieren des Clips berechnen
            extract_duration = base_duration
            if i < len(self.beat_times) - 2:
                extract_duration += transition_duration
                
            # Zu lange Segmente kürzen
            if extract_duration > max_clip_duration:
                extract_duration = max_clip_duration
                print(f"Debug: Truncating duration to {max_clip_duration}s")
            
            print(f"Debug: Final extract duration: {extract_duration:.2f}s")
            
            # Intelligentere Clip-Auswahl mit mehr Zufallsfaktoren
            available_clips = []
            for clip_info in enhanced_clips:
                clip_path = clip_info["path"]
                
                # Überspringe Clips, die zu oft verwendet wurden
                if used_count[clip_path] > len(self.video_clips) * 2:
                    continue
                    
                # Überspringe Clips, die kürzlich verwendet wurden
                if any(last["path"] == clip_path and last["reverse"] == clip_info["reverse"] for last in last_clips):
                    continue
                
                available_clips.append(clip_info)
            
            # Wenn keine Clips verfügbar sind, reset die Verwendungszähler
            if not available_clips:
                used_count = {clip: 0 for clip in self.video_clips}
                available_clips = enhanced_clips
            
            # Wähle einen zufälligen Clip aus den verfügbaren
            chosen_clip = random.choice(available_clips)
            
            # Update tracking variables
            if len(last_clips) >= max_last_clips:
                last_clips.pop(0)
            last_clips.append(chosen_clip)
            used_count[chosen_clip["path"]] += 1
            
            clip_path = chosen_clip["path"]
            is_reverse = chosen_clip["reverse"]
            speed = chosen_clip["speed"]
            start_bias = chosen_clip["start_bias"]
            
            print(f"Debug: Selected clip: {os.path.basename(clip_path)} {'(reverse)' if is_reverse else ''} speed={speed}")
            
            # Zufälligen Startpunkt im Clip wählen basierend auf start_bias
            clip_duration = self.get_clip_duration(clip_path)
            print(f"Debug: Clip duration: {clip_duration:.2f}s")
            
            if clip_duration <= extract_duration:
                clip_start = 0
            else:
                max_start = clip_duration - extract_duration
                if start_bias == "start":
                    clip_start = random.uniform(0, max_start * 0.3)
                elif start_bias == "middle":
                    clip_start = random.uniform(max_start * 0.3, max_start * 0.7)
                elif start_bias == "end":
                    clip_start = random.uniform(max_start * 0.7, max_start)
                else:  # "random"
                    clip_start = random.uniform(0, max_start)
            
            print(f"Debug: Clip start time: {clip_start:.2f}s")
            
            # Ausgabedatei für das Segment
            segment_file = os.path.join(self.temp_dir, f"segment_{i:03d}.mp4")
            
            # FFmpeg-Befehl mit zusätzlichen Effekten
            filter_complex = []
            if is_reverse:
                filter_complex.append("reverse")
            if speed != 1.0:
                filter_complex.append(f"setpts={1/speed}*PTS")
            
            # Kombiniere Filter
            filter_str = ",".join(filter_complex) if filter_complex else "null"
            
            cmd = [
                "ffmpeg", "-i", clip_path,
                "-ss", f"{clip_start:.3f}",
                "-t", f"{extract_duration:.3f}",
                "-vf", filter_str,
                "-c:v", "libx264", "-preset", "medium", "-crf", "17",  # Higher quality video
                "-profile:v", "high", "-level", "4.2",  # Higher profile and level
                "-pix_fmt", "yuv420p",  # Ensure compatibility
                "-an",
                segment_file, "-y"
            ]
            
            print(f"Debug: FFmpeg command: {' '.join(cmd)}")
            
            # Ausgabe mit Rückwärts-Information
            rev_info = " (rückwärts)" if is_reverse else ""
            print(f"Erstelle Segment {i+1}/{min(len(self.beat_times)-1, max_segments)}: {os.path.basename(clip_path)}{rev_info} "
                  f"von {clip_start:.2f}s für {extract_duration:.2f}s")
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Segmentinfo speichern
                segment_info = {
                    "file": segment_file,
                    "start_time": start_time,
                    "end_time": end_time,
                    "base_duration": base_duration,
                    "extract_duration": extract_duration,
                    "index": i
                }
                raw_segments.append(segment_info)
                print(f"Debug: Successfully created segment {i+1}")
            except subprocess.CalledProcessError as e:
                print(f"Fehler beim Erstellen von Segment {i}: {str(e)}")
                print(f"Fehlerausgabe: {e.stderr.decode() if e.stderr else 'Keine Fehlerausgabe'}")
                continue
        
        print(f"\nDebug: Total segments created: {len(raw_segments)}")
        
        # Prüfen, ob Segmente erstellt wurden
        if not raw_segments:
            raise ValueError("Keine Segmente wurden erstellt!")
        
        print(f"\nErstelle Übergänge zwischen {len(raw_segments)} Segmenten...")
        
        # Erstelle die Filterkomplexe für Übergänge
        if transition_duration > 0:
            # Zwei-Pass-Ansatz: Zuerst alle Segmente mit xfade-Komplexfilter kombinieren
            xfade_complex_file = os.path.join(self.temp_dir, "xfade_script.txt")
            
            # Erstelle liste mit allen Segmentdateien für Eingabe
            inputs = []
            for segment in raw_segments:
                inputs.extend(["-i", segment["file"]])
            
            # Erstelle die Filterkomplexe für Crossfades
            filter_complex = []
            
            # Setze das erste Segment als Ausgangspunkt
            if len(raw_segments) == 1:
                # Wenn nur ein Segment, verwende es direkt
                filter_complex.append(f"[0:v]copy[v]")
            else:
                # Vorbereitung für Crossfades zwischen allen Segmenten
                for i in range(len(raw_segments) - 1):
                    if i == 0:
                        # Erstes Segment
                        filter_complex.append(f"[0:v]setpts=PTS-STARTPTS[v0];")
                        # Zweites Segment
                        filter_complex.append(f"[1:v]setpts=PTS-STARTPTS[v1];")
                        # Crossfade zwischen erstem und zweitem
                        filter_complex.append(f"[v0][v1]xfade=transition=fade:duration={transition_duration}:offset={raw_segments[0]['base_duration'] - transition_duration}[v01];")
                    else:
                        # Nächstes Segment vorbereiten
                        filter_complex.append(f"[{i+1}:v]setpts=PTS-STARTPTS[v{i+1}];")
                        # Crossfade zwischen vorherigem Ergebnis und aktuellem Segment
                        accumulated_duration = sum(s['base_duration'] for s in raw_segments[:i+1]) - i * transition_duration
                        filter_complex.append(f"[v0{i}][v{i+1}]xfade=transition=fade:duration={transition_duration}:offset={accumulated_duration - transition_duration}[v0{i+1}];")
                
                # Letztes Ergebnis als Ausgabe markieren
                filter_complex[-1] = filter_complex[-1].replace(f"[v0{len(raw_segments)-1}];", f"[v];")
            
            # Finale Filterkomplexe zusammenfügen
            filter_complex_str = "".join(filter_complex)
            
            # Zusammenführen mit Crossfades
            silent_output = os.path.join(self.temp_dir, "silent_output.mp4")
            xfade_cmd = ["ffmpeg"]
            xfade_cmd.extend(inputs)
            xfade_cmd.extend([
                "-filter_complex", filter_complex_str,
                "-map", "[v]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "17",  # Higher quality video
                "-profile:v", "high", "-level", "4.2",  # Higher profile and level
                "-pix_fmt", "yuv420p",  # Ensure compatibility
                silent_output, "-y"
            ])
            
            print("\nFüge Videosegmente mit Übergängen zusammen...")
            try:
                subprocess.run(xfade_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Fehler beim Erstellen der Übergänge: {str(e)}")
                
                # Fallback-Methode: Einfache Verkettung ohne Übergänge
                print("\nFallback: Einfache Verkettung ohne Übergänge...")
                
                # Erstelle einfache Segmentliste
                segments_file = os.path.join(self.temp_dir, "segments.txt")
                with open(segments_file, 'w') as f:
                    for segment in raw_segments:
                        f.write(f"file '{segment['file']}'\n")
                
                concat_cmd = [
                    "ffmpeg", "-f", "concat",
                    "-safe", "0",
                    "-i", segments_file,
                    "-c:v", "copy",
                    silent_output, "-y"
                ]
                subprocess.run(concat_cmd, check=True)
        else:
            # Wenn keine Übergänge gewünscht sind, einfach verketten
            segments_file = os.path.join(self.temp_dir, "segments.txt")
            with open(segments_file, 'w') as f:
                for segment in raw_segments:
                    f.write(f"file '{segment['file']}'\n")
                    
            silent_output = os.path.join(self.temp_dir, "silent_output.mp4")
            concat_cmd = [
                "ffmpeg", "-f", "concat",
                "-safe", "0",
                "-i", segments_file,
                "-c:v", "copy",
                silent_output, "-y"
            ]
            print("\nFüge Videosegmente zusammen...")
            subprocess.run(concat_cmd, check=True)
        
        # Finale Ausgabe mit originaler Audiospur
        cmd = [
            "ffmpeg", "-i", silent_output,
            "-i", self.audio_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "320k",  # Higher quality audio
            "-af", "aresample=48000:first_pts=0",  # Ensure consistent audio sample rate
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