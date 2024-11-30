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
                if self.data['gain'] == 'auto':
                    self.sdr.gain = 20
                else:
                    self.sdr.gain = float(self.data['gain'])
                
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
        samples_size = 512  # Ridotto il numero di campioni
        while self.running:
            if not self.sdr:
                time.sleep(0.1)
                continue
                
            try:
                # Se è in corso un recupero, aspetta
                if self.recovery_event.is_set():
                    time.sleep(0.1)
                    continue
                    
                with self.sdr_lock:
                    samples = self.sdr.read_samples(samples_size)
                
                # Resetta il contatore errori se l'acquisizione è riuscita
                self.consecutive_errors = 0
                
                # Normalizzazione e rimozione DC
                samples = samples - np.mean(samples)
                samples = samples / (np.std(samples) + 1e-10)
                
                # FFT con finestra Blackman
                window = np.blackman(len(samples))
                samples = samples * window
                
                nfft = 1024  # Ridotto per maggiore velocità
                pxx = np.fft.fftshift(np.abs(np.fft.fft(samples, n=nfft)))
                
                # Conversione in dB e calibrazione
                pxx_db = 20 * np.log10(pxx + 1e-10)
                pxx_db = pxx_db - np.max(pxx_db)
                pxx_db = np.clip(pxx_db, -70, 0)
                
                freqs = self.sdr.center_freq + np.fft.fftshift(
                    np.fft.fftfreq(nfft, 1/self.sdr.sample_rate)
                )
                
                with self.data_lock:
                    if not hasattr(self, 'avg_buffer'):
                        self.avg_buffer = []
                    
                    self.avg_buffer.append(pxx_db)
                    if len(self.avg_buffer) > 3:
                        self.avg_buffer.pop(0)
                    
                    pxx_db_avg = np.mean(self.avg_buffer, axis=0)
                    
                    self.data['frequencies'] = (freqs / 1e6).tolist()
                    self.data['powers'] = pxx_db_avg.tolist()
                    
            except Exception as e:
                print(f"Errore nell'acquisizione: {e}")
                self.consecutive_errors += 1
                
                # Se ci sono troppi errori consecutivi, tenta il recupero
                if self.consecutive_errors >= self.max_consecutive_errors:
                    self.recovery_event.set()
                    if self._recover_device():
                        self.recovery_event.clear()
                    else:
                        print("Recupero fallito, attendo prima di riprovare...")
                        time.sleep(1)
                        
                time.sleep(0.1)
                continue
            
            time.sleep(0.01)

    def update_params(self, params):
        if not self.sdr:
            return False
            
        # Non aggiornare i parametri durante il recupero
        if self.recovery_event.is_set():
            return False
            
        with self.sdr_lock:
            try:
                print(f"Ricevuti parametri: {params}")
                
                old_params = {
                    'center_freq': self.sdr.center_freq,
                    'sample_rate': self.sdr.sample_rate,
                    'gain': self.sdr.gain
                }
                
                if 'center_freq' in params:
                    new_freq = float(params['center_freq'])
                    self.sdr.center_freq = new_freq
                    self.data['center_freq'] = new_freq
                    print(f"Frequenza aggiornata a: {new_freq/1e6} MHz")
                    
                if 'sample_rate' in params:
                    new_rate = float(params['sample_rate'])
                    self.sdr.sample_rate = new_rate
                    self.data['sample_rate'] = new_rate
                    print(f"Sample rate aggiornato a: {new_rate/1e6} MS/s")
                    
                if 'gain' in params:
                    new_gain = params['gain']
                    if new_gain == 'auto':
                        self.sdr.gain = 20
                    else:
                        new_gain = float(new_gain)
                        new_gain = min(max(new_gain, 0), 40)
                        self.sdr.gain = new_gain
                    self.data['gain'] = new_gain
                    print(f"Gain aggiornato a: {new_gain}")
                
                time.sleep(0.1)
                
                with self.data_lock:
                    if hasattr(self, 'avg_buffer'):
                        self.avg_buffer = []
                    
                return True
                    
            except Exception as e:
                try:
                    self.sdr.center_freq = old_params['center_freq']
                    self.sdr.sample_rate = old_params['sample_rate']
                    self.sdr.gain = old_params['gain']
                except:
                    pass
                    
                print(f"Errore nell'aggiornamento dei parametri: {e}")
                return False

    def get_data(self):
        with self.data_lock:
            return {
                'frequencies': self.data['frequencies'],
                'powers': self.data['powers'],
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