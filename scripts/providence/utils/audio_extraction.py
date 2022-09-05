import subprocess


def cut_wave_file(wav_file, out_file, onset, duration):
    """
    This function cut 'audio_file' from 'onset' to 'onset+duration'
    Stores the chunk in the .wav format in the 'out_file' path.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['sox', wav_file, out_file,
           'trim', str(onset/1000), str(duration/1000)]
    subprocess.call(cmd)