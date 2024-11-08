import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
from tflite_runtime.interpreter import Interpreter

# Paramètres
debug_time = 1
debug_acc = 1  # Mettre à 1 pour afficher les valeurs de confiance
word_threshold = 0.5
rec_duration = 0.5
window_stride = 0.5
sample_rate = 48000
resample_rate = 8000
num_channels = 1
num_mfcc = 16
model_path = 'wake_word_stop_lite.tflite'  # Chemin vers votre modèle .tflite

# Fenêtre de glissement pour l'audio
window = np.zeros(int(rec_duration * resample_rate) * 2)

# Charger le modèle
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Fonction de décimation pour réduire la fréquence d'échantillonnage
def decimate(signal, old_fs, new_fs):
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Erreur : la décimation doit être un facteur entier")
        return signal, old_fs
    return scipy.signal.decimate(signal, int(dec_factor)), new_fs

# Fonction de rappel pour traiter chaque segment audio
def sd_callback(rec, frames, time, status):
    # Démarrer le chronométrage pour le débogage
    start = timeit.default_timer()
    
    # Afficher l'erreur si une erreur est rencontrée lors de l'acquisition
    if status:
        print('Erreur:', status)

    # Supprimer la 2e dimension de l'échantillon audio
    rec = np.squeeze(rec)
    
    # Décimer pour réduire la fréquence d'échantillonnage
    rec, new_fs = decimate(rec, sample_rate, resample_rate)

    # Enregistrer l'audio dans la fenêtre de glissement
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Calculer les MFCC (caractéristiques)
    mfccs = python_speech_features.base.mfcc(
        window, samplerate=new_fs, winlen=0.256, winstep=0.050,
        numcep=num_mfcc, nfilt=26, nfft=2048, preemph=0.0, ceplifter=0,
        appendEnergy=False, winfunc=np.hanning
    ).transpose()

    # Prédiction avec le modèle
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0][0]

    # Vérifier si le mot "stop" est détecté
    if val > word_threshold:
        print('stop détecté')

    # Afficher la confiance si debug_acc est activé
    if debug_acc:
        print("Confiance:", val)

    # Afficher le temps de traitement si debug_time est activé
    if debug_time:
        print("Temps de traitement:", timeit.default_timer() - start)

# Démarrer le flux audio pour capturer en continu
print("Dites 'stop' pour tester la détection...")
with sd.InputStream(channels=num_channels, samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration), callback=sd_callback):
    while True:
        pass
