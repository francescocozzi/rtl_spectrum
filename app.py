from flask import Flask, render_template, jsonify, request
from rtlsdr import RtlSdr
import numpy as np
from threading import Thread, Lock, Event
import time
import json
import atexit

app = Flask(__name__)

class SDRHandler:
    def __init__(self):
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        self.sdr = None
        self.data_lock = Lock()
        self.sdr_lock = Lock()
        self.running = True
        self.recovery_event = Event()
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        
        self.data = {
            'frequencies': [],
            'powers': [],
            'iq_data': {
                'i': [],
                'q': [],
                'time': []
            },
            'center_freq': 100e6,
            'sample_rate': 2.4e6,
            'gain': 20
        }
        
        try:
            self.sdr = RtlSdr()
            self._configure_sdr()
            
            self.update_thread = Thread(target=self.update_spectrum)
            self.update_thread.daemon = True
            self.update_thread.start()
            
            self.initialized = True
        except Exception as e:
            print(f"Errore nell'inizializzazione SDR: {e}")
            if self.sdr:
                self.sdr.close()
            raise

    def _configure_sdr(self):
        if not self.sdr:
            return False
        
        try:
            with self.sdr_lock:
                self.sdr.sample_rate = self.data['sample_rate']
                self.sdr.center_freq = self.data['center_freq']
                
                # Gestione gain dinamica basata sulla frequenza
                if self.data['gain'] == 'auto':
                    if self.data['center_freq'] < 300e6:  # Per VHF
                        self.sdr.gain = 15  # Gain più basso per VHF
                    else:  # Per UHF
                        self.sdr.gain = 20
                else:
                    gain_value = float(self.data['gain'])
                    # Limita il gain massimo in VHF
                    if self.data['center_freq'] < 300e6:
                        gain_value = min(gain_value, 30)
                    self.sdr.gain = gain_value
                
                time.sleep(0.1)
            return True
        except Exception as e:
            print(f"Errore nella configurazione SDR: {e}")
            return False

    def _recover_device(self):
        """Tenta di recuperare il dispositivo in caso di blocco"""
        print("Tentativo di recupero del dispositivo...")
        try:
            with self.sdr_lock:
                self.sdr.close()
                time.sleep(1)
                self.sdr = RtlSdr()
                self._configure_sdr()
                self.consecutive_errors = 0
                print("Dispositivo recuperato con successo")
                return True
        except Exception as e:
            print(f"Errore nel recupero del dispositivo: {e}")
            return False

    def update_spectrum(self):
        samples_size = 256 if self.data['center_freq'] < 300e6 else 512
        time_axis = np.arange(samples_size) / self.data['sample_rate']
        
        while self.running:
            if not self.sdr:
                time.sleep(0.1)
                continue
                
            try:
                if self.recovery_event.is_set():
                    time.sleep(0.1)
                    continue
                    
                with self.sdr_lock:
                    samples = self.sdr.read_samples(samples_size)
                
                self.consecutive_errors = 0
                
                # Preparazione dati IQ
                i_data = samples.real
                q_data = samples.imag
                
                # Normalizzazione dei dati IQ
                max_iq = max(np.max(np.abs(i_data)), np.max(np.abs(q_data)))
                i_data = i_data / (max_iq + 1e-10)
                q_data = q_data / (max_iq + 1e-10)
                
                # Processo FFT per lo spettro
                # Normalizzazione migliorata per VHF
                samples_fft = samples - np.mean(samples)
                if self.data['center_freq'] < 300e6:
                    max_val = np.percentile(np.abs(samples_fft), 95)
                    samples_fft = samples_fft / (max_val + 1e-10)
                else:
                    samples_fft = samples_fft / (np.std(samples_fft) + 1e-10)
                
                # FFT con dimensione variabile
                nfft = 512 if self.data['center_freq'] < 300e6 else 1024
                
                # Finestra variabile
                if self.data['center_freq'] < 300e6:
                    window = np.hamming(len(samples_fft))
                else:
                    window = np.blackman(len(samples_fft))
                    
                samples_fft = samples_fft * window
                pxx = np.fft.fftshift(np.abs(np.fft.fft(samples_fft, n=nfft)))
                
                # Range dinamico adattativo
                if self.data['center_freq'] < 300e6:
                    pxx_db = 20 * np.log10(pxx + 1e-10)
                    pxx_db = pxx_db - np.max(pxx_db)
                    pxx_db = np.clip(pxx_db, -60, 0)
                else:
                    pxx_db = 20 * np.log10(pxx + 1e-10)
                    pxx_db = pxx_db - np.max(pxx_db)
                    pxx_db = np.clip(pxx_db, -70, 0)
                
                freqs = self.sdr.center_freq + np.fft.fftshift(
                    np.fft.fftfreq(nfft, 1/self.sdr.sample_rate)
                )
                
                with self.data_lock:
                    # Aggiornamento dati IQ
                    self.data['iq_data'] = {
                        'i': i_data.tolist()[:100],  # Limitiamo a 100 campioni per l'IQ
                        'q': q_data.tolist()[:100],
                        'time': time_axis[:100].tolist()
                    }
                    
                    # Aggiornamento spettro
                    if not hasattr(self, 'avg_buffer'):
                        self.avg_buffer = []
                    
                    max_avg_samples = 2 if self.data['center_freq'] < 300e6 else 3
                    
                    self.avg_buffer.append(pxx_db)
                    if len(self.avg_buffer) > max_avg_samples:
                        self.avg_buffer.pop(0)
                    
                    pxx_db_avg = np.mean(self.avg_buffer, axis=0)
                    
                    self.data['frequencies'] = (freqs / 1e6).tolist()
                    self.data['powers'] = pxx_db_avg.tolist()
                    
            except Exception as e:
                print(f"Errore nell'acquisizione: {e}")
                self.consecutive_errors += 1
                
                if self.consecutive_errors >= self.max_consecutive_errors:
                    self.recovery_event.set()
                    if self._recover_device():
                        self.recovery_event.clear()
                    else:
                        print("Recupero fallito, attendo prima di riprovare...")
                        time.sleep(1)
                        
                time.sleep(0.1)
                continue
            
            time.sleep(0.005 if self.data['center_freq'] < 300e6 else 0.01)

    def get_data(self):
        with self.data_lock:
            return {
                'frequencies': self.data['frequencies'],
                'powers': self.data['powers'],
                'iq_data': self.data['iq_data'],
                'current_settings': {
                    'center_freq': self.sdr.center_freq / 1e6 if self.sdr else 0,
                    'sample_rate': self.sdr.sample_rate / 1e6 if self.sdr else 0,
                    'gain': self.sdr.gain if self.sdr else 'auto'
                }
            }


    def cleanup(self):
        self.running = False
        if hasattr(self, 'update_thread'):
            self.update_thread.join()
        if self.sdr:
            self.sdr.close()

# Creazione singola istanza
try:
    sdr_handler = SDRHandler()
except Exception as e:
    print(f"Errore nella creazione dell'handler SDR: {e}")
    sdr_handler = None

atexit.register(lambda: sdr_handler.cleanup() if sdr_handler else None)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_spectrum')
def get_spectrum():
    if not sdr_handler:
        return jsonify({'error': 'SDR non inizializzato'}), 500
    return jsonify(sdr_handler.get_data())

@app.route('/update_params', methods=['POST'])
def update_params():
    if not sdr_handler:
        return jsonify({'status': 'error', 'message': 'SDR non inizializzato'}), 500
        
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Content-Type deve essere application/json'}), 400
    
    params = request.get_json()
    success = sdr_handler.update_params(params)
    
    if success:
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Errore nell\'aggiornamento dei parametri'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)