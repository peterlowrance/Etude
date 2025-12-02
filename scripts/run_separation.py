# scripts/run_separation.py

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model

def separate_and_extract_features(input_path: str, output_path: str):
    """
    Performs 6-stem source separation using Demucs (htdemucs_6s), maps the stems
    to the expected 5-channel format (Vocals, Drums, Bass, Piano, Other+Guitar),
    converts each stem into a dB-scaled Mel spectrogram, and saves the stacked 
    features as a NumPy array.

    Args:
        input_path (str): Path to the source audio file.
        output_path (str): Path to save the resulting feature array as a .npy file.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"[ERROR] Input audio file not found at {input_file}", file=sys.stderr)
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    > Using device: {device}")

    try:
        print("    > Loading Demucs model (htdemucs_6s)...")
        # htdemucs_6s sources: ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
        model = get_model('htdemucs_6s')
        model.to(device)
        
        print(f"    > Loading audio: {input_file.name}")
            
        # Demucs expects audio as (channels, length) tensor.
        # It handles resampling internally if needed, but we'll load with torchaudio.
        wav, sr = torchaudio.load(str(input_file))
        
        # Convert to stereo if mono
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        else:
            wav = wav[:2] # Ensure stereo
            
        # Add batch dimension: (1, channels, length)
        wav = wav.unsqueeze(0).to(device)

        print("    > Separating audio into stems...")
        # apply_model returns (batch, sources, channels, time)
        # sources order for htdemucs_6s: drums, bass, other, vocals, guitar, piano
        ref = model.sources
        # ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
        
        sources = apply_model(model, wav, device=device, shifts=1, split=True, overlap=0.25, progress=True)[0]
        # sources shape: (6, 2, time) - (sources, channels, time)
        
        # Map to expected 5 stems: Vocals, Drums, Bass, Piano, Other
        # Expected order by downstream: 0:Vocals, 1:Drums, 2:Bass, 3:Piano, 4:Other
        
        # Demucs indices:
        idx_drums = ref.index('drums')
        idx_bass = ref.index('bass')
        idx_other = ref.index('other')
        idx_vocals = ref.index('vocals')
        idx_guitar = ref.index('guitar')
        idx_piano = ref.index('piano')
        
        mapped_stems = {}
        mapped_stems['vocals'] = sources[idx_vocals]
        mapped_stems['drums'] = sources[idx_drums]
        mapped_stems['bass'] = sources[idx_bass]
        mapped_stems['piano'] = sources[idx_piano]
        mapped_stems['other'] = sources[idx_other] + sources[idx_guitar] # Sum other and guitar
        
        ordered_keys = ['vocals', 'drums', 'bass', 'piano', 'other']
        
        print("    > Converting stems to dB Mel Spectrograms...")
        
        # Define Mel filter bank (using librosa to match original parameters)
        # sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000
        mel_filter_bank = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000).T
        
        processed_spectrograms = []
        
        for key in ordered_keys:
            stem_tensor = mapped_stems[key] # (2, time)
            
            # Convert to mono: (time,)
            stem_mono = stem_tensor.mean(dim=0).cpu().numpy()
            
            # STFT
            stft_matrix = librosa.stft(stem_mono, n_fft=4096, hop_length=1024)
            
            # Power Spectrum
            power_spec = np.abs(stft_matrix)**2 # (freq, time)
            
            # Mel Spectrogram
            mel_spec = np.dot(power_spec.T, mel_filter_bank) # (time, n_mels)
            
            processed_spectrograms.append(mel_spec)

        # Stack: (5, time, 128)
        stacked_mel_specs = np.stack(processed_spectrograms)
        
        # Convert to dB
        db_specs = librosa.power_to_db(stacked_mel_specs, ref=np.max)
        
        final_features = db_specs
        
        print(f"    > Saving final feature array to {output_file.name}...")
        np.save(output_file, final_features)
        
        print("    > Feature extraction complete.")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during Demucs processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Audio feature extraction via Demucs 6-stem separation and Mel spectrogram conversion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Path to the input audio file.")
    parser.add_argument("--output", required=True, help="Path for the output .npy feature file.")
    args = parser.parse_args()
    
    separate_and_extract_features(args.input, args.output)

if __name__ == "__main__":
    main()
