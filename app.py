from flask import Flask, render_template, jsonify, request
from rtlsdr import RtlSdr
import numpy as np
from threading import Thread, Lock
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
        self.data = {
            'frequencies': [],
            'powers': [],
            'center_freq': 100e6,
            'sample_rate': 2.4e6,
            'gain': 20  # Impostato un gain iniziale più basso
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
                # Configurazione più conservativa
                self.sdr.sample_rate = self.data['sample_rate']
                self.sdr.center_freq = self.data['center_freq']
                if self.data['gain'] == 'auto':
                    self.sdr.gain = 20  # Default gain se in auto
                else:
                    self.sdr.gain = float(self.data['gain'])
                
                # Configurazione aggiuntiva per migliorare la stabilità
                self.sdr.freq_correction = 0
                time.sleep(0.1)
            return True
        except Exception as e:
            print(f"Errore nella configurazione SDR: {e}")
            return False

    def update_spectrum(self):
        samples_size = 1024  # Ridotto il numero di campioni
        while self.running:
            if not self.sdr:
                time.sleep(0.1)
                continue
                
            try:
                # Acquisizione con timeout più breve
                with self.sdr_lock:
                    samples = self.sdr.read_samples(samples_size)
                
                # Normalizzazione dei campioni
                samples = samples / np.max(np.abs(samples))
                
                # Applicazione finestra
                window = np.hamming(len(samples))  # Cambiato da hanning a hamming
                samples = samples * window
                
                # FFT con parametri ottimizzati
                nfft = 2048  # Ridotto per maggiore velocità
                pxx = np.fft.fftshift(np.abs(np.fft.fft(samples, n=nfft)))
                
                # Conversione in dB con range dinamico limitato
                pxx_db = 20 * np.log10(pxx + 1e-10)
                pxx_db = np.clip(pxx_db, -50, 30)  # Limita il range dinamico
                
                freqs = self.sdr.center_freq + np.fft.fftshift(
                    np.fft.fftfreq(nfft, 1/self.sdr.sample_rate)
                )
                
                # Media mobile con meno campioni
                with self.data_lock:
                    if not hasattr(self, 'avg_buffer'):
                        self.avg_buffer = []
                    
                    self.avg_buffer.append(pxx_db)
                    if len(self.avg_buffer) > 3:  # Ridotto da 5 a 3
                        self.avg_buffer.pop(0)
                    
                    pxx_db_avg = np.mean(self.avg_buffer, axis=0)
                    
                    self.data['frequencies'] = (freqs / 1e6).tolist()
                    self.data['powers'] = pxx_db_avg.tolist()
                    
            except Exception as e:
                print(f"Errore nell'acquisizione: {e}")
                time.sleep(0.1)
                continue
            
            time.sleep(0.01)

    def update_params(self, params):
        if not self.sdr:
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
                        self.sdr.gain = 20  # Default gain se in auto
                    else:
                        new_gain = float(new_gain)
                        # Limita il gain a un range più sicuro
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