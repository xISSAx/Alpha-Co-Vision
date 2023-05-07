import subprocess

def playTTS(text, voice):
    tts_command = f"edge-playback --voice {voice} --text \"{text}\""
    subprocess.run(tts_command, shell=True)

# playTTS("Hello, world!", config_module.config.edge_tts_voice)
