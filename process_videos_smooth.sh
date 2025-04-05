#!/usr/bin/env python3
import subprocess
import os
import numpy as np
import argparse
import random
from glob import glob
import shutil
import sys

class SimpleVideoEditor:
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
    
    def get_audio_duration(self):
        """Ermittelt die Dauer der Audiodatei"""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", self.audio_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            print(f"Warnung: Konnte Dauer für {self.audio_path} nicht ermitteln")
            return 180.0  # Standard-Dauer von 3 Minuten zurückgeben
    
    def create_music_synchronized_video(self, segment_duration=3.0, transition_duration=0.5, reverse_probability=0.3):
        """Erstellt ein Video mit regelmäßigen Abschnitten, synchronisiert zur Musik"""
        if not self.video_clips:
            raise ValueError("Keine Videoclips gefunden!")
        
        # Bestimme Musik-Dauer
        audio_duration = self.get_audio_duration()
        print(f"Audiodauer: {audio_duration:.2f} Sekunden")
        
        # Erstelle eine Liste mit erweiterten Clip-Informationen
        enhanced_clips = []
        for clip in self.video_clips:
            enhanced_clips.append({"path": clip, "reverse": False})
            enhanced_clips.append({"path": clip, "reverse": True})
        
        # Liste mehrmals mischen für bessere Durchmischung
        for _ in range(3):
            random.shuffle(enhanced_clips)
        
        # Tracking-Variablen
        last_clip_path = None
        segments_list = []
        segments_file = os.path.join(self.temp_dir, "segments.txt")
        
        # Bestimme, wie viele Segmente erstellt werden sollen
        # Segmentlänge um 20% verringern, um Platz für Übergänge zu haben
        effective_segment_duration = segment_duration * 0.8
        num_segments = int(audio_duration / effective_segment_duration) + 1
        print(f"Erstelle {num_segments} Segmente mit je {segment_duration:.1f} Sekunden")
        
        with open(segments_file, 'w') as f:
            for i in range(num_segments):
                # Wähle einen Clip, der nicht der letzte war
                valid_clips = [clip for clip in enhanced_clips if clip["path"] != last_clip_path]
                if not valid_clips:
                    valid_clips = enhanced_clips
                
                chosen_clip = random.choice(valid_clips)
                clip_path = chosen_clip["path"]
                is_reverse = chosen_clip["reverse"]
                
                # Update last clip
                last_clip_path = clip_path
                
                # Bestimme einen zufälligen Startpunkt im Clip
                clip_duration = self.get_clip_duration(clip_path)
                if clip_duration <= segment_duration:
                    clip_start = 0
                else:
                    max_start = clip_duration - segment_duration
                    clip_start = random.uniform(0, max_start)
                
                # Ausgabedatei für das Segment
                segment_file = os.path.join(self.temp_dir, f"segment_{i:03d}.mp4")
                
                # FFmpeg-Befehl zum Extrahieren des Segments
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
                print(f"Erstelle Segment {i+1}/{num_segments}: {os.path.basename(clip_path)}{rev_info} "
                      f"von {clip_start:.2f}s für {segment_duration:.2f}s")
                
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    # Überprüfen, ob die Datei existiert und nicht leer ist
                    if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                        # Füge zur Segmentliste hinzu
                        f.write(f"file '{segment_file}'\n")
                        segments_list.append(segment_file)
                    else:
                        print(f"  Fehler: Segment wurde nicht erstellt oder ist leer")
                except subprocess.CalledProcessError as e:
                    print(f"  Fehler beim Erstellen von Segment {i}: {str(e)}")
                    continue
        
        # Prüfen, ob Segmente erstellt wurden
        if not segments_list:
            raise ValueError("Keine Segmente wurden erstellt!")
        
        print(f"\nErstelle Video aus {len(segments_list)} Segmenten...")
        
        # Einfache Verkettung der Segmente
        silent_output = os.path.join(self.temp_dir, "silent_output.mp4")
        concat_cmd = [
            "ffmpeg", "-f", "concat",
            "-safe", "0",
            "-i", segments_file,
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            silent_output, "-y"
        ]
        
        try:
            subprocess.run(concat_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Fehler beim Zusammenfügen der Segmente: {str(e)}")
            # Notfall-Fallback - kopiere erstes Segment als Ausgabe
            if segments_list:
                shutil.copy(segments_list[0], silent_output)
                print("Fallback: Verwende nur das erste Segment")
            else:
                raise ValueError("Keine Segmente konnten erstellt werden")
        
        # Hinzufügen der Audio
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
    parser = argparse.ArgumentParser(description="Erstellt ein Video aus Clips, synchronisiert zur Musik")
    parser.add_argument("--segment", "-s", type=float, default=3.0, 
                       help="Dauer der Segmente in Sekunden")
    parser.add_argument("--transition", "-t", type=float, default=0.5,
                       help="Übergangsdauer in Sekunden (Crossfade)")
    parser.add_argument("--keep-temp", "-k", action="store_true",
                       help="Temporäre Dateien nicht löschen")
    
    args = parser.parse_args()
    
    # Editor initialisieren
    editor = SimpleVideoEditor(
        clips_dir=input_dir,
        audio_path=audio_file,
        output_file=output_file,
        temp_dir=temp_dir
    )
    
    try:
        # Prozess ausführen
        editor.find_video_clips()
        output_file = editor.create_music_synchronized_video(
            segment_duration=args.segment,
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