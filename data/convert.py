import os
import torch
import csv
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
from datetime import timedelta

def format_time(ms):
    """Convert milliseconds to SRT time format"""
    td = timedelta(milliseconds=ms)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{td.microseconds // 1000:03}"

def split_audio(input_path, output_wavs_dir):
    """Split audio into LJ Speech-compatible WAV chunks with minimum duration"""
    audio = AudioSegment.from_file(input_path, format="ogg")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Convert to LJ Speech format
    audio = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2)
    
    # Split with conservative parameters
    chunks = split_on_silence(
        audio,
        min_silence_len=700,
        silence_thresh=-35,
        keep_silence=400,
        seek_step=50
    )
    
    # Merge short chunks and ensure minimum duration
    processed_chunks = []
    current_chunk = AudioSegment.empty()
    
    for chunk in chunks:
        current_chunk += chunk
        if len(current_chunk) >= 3000:  # 3 seconds minimum
            processed_chunks.append(current_chunk)
            current_chunk = AudioSegment.empty()
    
    # Handle remaining audio
    if len(current_chunk) > 0:
        if processed_chunks:
            processed_chunks[-1] += current_chunk
        else:
            processed_chunks.append(current_chunk)

    # Split large chunks into 25MB parts (≈10 minutes at LJ specs)
    final_chunks = []
    file_counter = 1
    for chunk in processed_chunks:
        chunk_duration = len(chunk)
        max_duration = 10 * 60 * 1000  # 10 minutes in ms
        
        for i in range(0, chunk_duration, max_duration):
            sub_chunk = chunk[i:i+max_duration]
            chunk_name = f"{base_name}_{file_counter:04d}"
            chunk_path = os.path.join(output_wavs_dir, f"{chunk_name}.wav")
            
            sub_chunk.export(chunk_path, format="wav", 
                           parameters=["-ac", "1", "-ar", "22050"])
            final_chunks.append(chunk_path)
            file_counter += 1

    return final_chunks

def transcribe_chunks(chunks, output_srt, metadata_entries):
    """Transcribe chunks and create SRT + metadata"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("turbo", device=device)
    
    srt_entries = []
    
    for chunk_path in chunks:
        result = model.transcribe(
            chunk_path,
            language="ru",
            fp16=(device == "cuda"),
            verbose=False
        )
        
        chunk_id = os.path.splitext(os.path.basename(chunk_path))[0]
        full_text = " ".join([s["text"].strip() for s in result["segments"]])
        metadata_entries.append((chunk_id, full_text))
        
        for segment in result["segments"]:
            start = int(segment["start"] * 1000)
            end = int(segment["end"] * 1000)
            srt_entries.append((start, end, segment["text"].strip()))
    
    # Write SRT file
    srt_entries.sort(key=lambda x: x[0])
    with open(output_srt, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(srt_entries, 1):
            f.write(f"{idx}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n")

def process_files():
    """Main processing function"""
    # Create output directories
    os.makedirs("wavs", exist_ok=True)
    
    metadata_entries = []
    input_dir = "audio"
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found!")
        return
    
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.ogg')]
    
    if not audio_files:
        print(f"No OGG files found in '{input_dir}' directory")
        return
    
    for file in audio_files:
        input_path = os.path.join(input_dir, file)
        base_name = os.path.splitext(file)[0]
        
        print(f"Processing {file}...")
        chunks = split_audio(input_path, "wavs")
        transcribe_chunks(chunks, f"{base_name}.srt", metadata_entries)
    
    # Write metadata
    with open("metadata.csv", "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter='|', quotechar='"', 
                          quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["id", "transcription"])
        writer.writerows(metadata_entries)
    
    print("\nProcessing complete. Output files:")
    print(f"- Audio chunks: wavs/ directory ({len(metadata_entries)} files)")
    print(f"- Subtitles: {len(audio_files)} .srt files")
    print(f"- Metadata: metadata.csv")

if __name__ == "__main__":
    process_files()