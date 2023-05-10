import subprocess

def playTTS(text, voice):
    tts_command = f"edge-playback --voice {voice} --text \"{text}\""
    subprocess.run(tts_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
