from flask import Flask, render_template, jsonify, request
from rtlsdr import RtlSdr
import numpy as np
from threading import Thread, Lock
import time
import json
import atexit

app = Flask(__name__)

class SDRHandler:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SDRHandler, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if self.initialized:
            return
            
        self.sdr = None
        self.lock = Lock()
        self.running = True
        self.data = {
            'frequencies': [],
            'powers': [],
            'center_freq': 100e6,
            'sample_rate': 2.4e6,
            'gain': 'auto'
        }
        self.error_count = 0
        self.max_errors = 3
        
        try:
            self.sdr = RtlSdr()
            # Configurazione iniziale
            self._configure_sdr()
            
            # Avvio thread di acquisizione
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
        """Configurazione base dell'SDR con gestione errori"""
        if not self.sdr:
            return False
        
        try:
            # Imposta un timeout più lungo per le letture
            self.sdr.set_direct_sampling('i')
            self.sdr.reset_buffer()
            
            self.sdr.sample_rate = self.data['sample_rate']
            self.sdr.center_freq = self.data['center_freq']
            if self.data['gain'] == 'auto':
                self.sdr.gain = 'auto'
            else:
                self.sdr.gain = float(self.data['gain'])
            
            # Aggiungi un piccolo ritardo dopo la configurazione
            time.sleep(0.1)
            return True
        except Exception as e:
            print(f"Errore nella configurazione SDR: {e}")
            return False
    
    def _reset_device(self):
        """Resetta il dispositivo in caso di errori persistenti"""
        print("Tentativo di reset del dispositivo...")
        try:
            if self.sdr:
                self.sdr.close()
            time.sleep(1)  # Attendi che il dispositivo si liberi
            self.sdr = RtlSdr()
            self._configure_sdr()
            self.error_count = 0
            return True
        except Exception as e:
            print(f"Errore nel reset del dispositivo: {e}")
            return False
    
    def update_spectrum(self):
        while self.running:
            if not self.sdr:
                time.sleep(1)
                continue
                
            try:
                with self.lock:
                    # Aumenta il numero di campioni per una migliore risoluzione
                    samples = self.sdr.read_samples(2048)
                    
                    # Applica una finestra di Hanning per ridurre il leakage spettrale
                    window = np.hanning(len(samples))
                    samples = samples * window
                    
                    # Calcolo FFT con zero-padding per migliore interpolazione
                    nfft = 4096
                    pxx = np.fft.fftshift(np.abs(np.fft.fft(samples, n=nfft)))
                    pxx_db = 20 * np.log10(pxx + 1e-10)  # Evita log(0)
                    
                    # Calcolo frequenze
                    freqs = self.sdr.center_freq + np.fft.fftshift(
                        np.fft.fftfreq(nfft, 1/self.sdr.sample_rate)
                    )
                    
                    # Media mobile per ridurre il rumore
                    if not hasattr(self, 'avg_buffer'):
                        self.avg_buffer = []
                    
                    self.avg_buffer.append(pxx_db)
                    if len(self.avg_buffer) > 5:  # Media su 5 campioni
                        self.avg_buffer.pop(0)
                    
                    pxx_db_avg = np.mean(self.avg_buffer, axis=0)
                    
                    # Aggiornamento dati
                    self.data['frequencies'] = (freqs / 1e6).tolist()  # MHz
                    self.data['powers'] = pxx_db_avg.tolist()
                    
                    self.error_count = 0  # Reset contatore errori se successo
                    
            except Exception as e:
                print(f"Errore nell'acquisizione: {e}")
                self.error_count += 1
                
                if self.error_count >= self.max_errors:
                    print("Troppi errori consecutivi, tentativo di reset...")
                    if not self._reset_device():
                        time.sleep(5)  # Attendi più a lungo se il reset fallisce
                    continue
                    
                time.sleep(0.5)
                continue
            
            time.sleep(0.05)  # Ridotto il tempo di sleep per aggiornamenti più frequenti
    
    def update_params(self, params):
        if not self.sdr:
            return False
            
        with self.lock:
            try:
                changed = False
                if 'center_freq' in params:
                    new_freq = float(params['center_freq'])
                    if new_freq != self.data['center_freq']:
                        self.data['center_freq'] = new_freq
                        changed = True
                        
                if 'sample_rate' in params:
                    new_rate = float(params['sample_rate'])
                    if new_rate != self.data['sample_rate']:
                        self.data['sample_rate'] = new_rate
                        changed = True
                        
                if 'gain' in params:
                    if params['gain'] != self.data['gain']:
                        self.data['gain'] = params['gain']
                        changed = True
                
                if changed:
                    # Reset buffer media se cambiamo parametri
                    self.avg_buffer = []
                    success = self._configure_sdr()
                    if not success:
                        self._reset_device()
                    return success
                return True
                
            except Exception as e:
                print(f"Errore nell'aggiornamento dei parametri: {e}")
                return False
    
    def get_data(self):
        with self.lock:
            data = self.data.copy()
            if self.sdr:
                data['current_settings'] = {
                    'center_freq': self.sdr.center_freq / 1e6,
                    'sample_rate': self.sdr.sample_rate / 1e6,
                    'gain': self.sdr.gain
                }
            return data
    
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