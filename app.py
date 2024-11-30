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
            self.sdr.sample_rate = self.data['sample_rate']
            self.sdr.center_freq = self.data['center_freq']
            if self.data['gain'] == 'auto':
                self.sdr.gain = 'auto'
            else:
                self.sdr.gain = float(self.data['gain'])
            
            time.sleep(0.5)  # Pausa più lunga per stabilizzazione
            return True
        except Exception as e:
            print(f"Errore nella configurazione SDR: {e}")
            return False

    def update_spectrum(self):
        while self.running:
            if not self.sdr:
                time.sleep(1)
                continue
                
            try:
                with self.lock:
                    samples = self.sdr.read_samples(2048)
                    
                    window = np.hanning(len(samples))
                    samples = samples * window
                    
                    nfft = 4096
                    pxx = np.fft.fftshift(np.abs(np.fft.fft(samples, n=nfft)))
                    pxx_db = 20 * np.log10(pxx + 1e-10)
                    
                    freqs = self.sdr.center_freq + np.fft.fftshift(
                        np.fft.fftfreq(nfft, 1/self.sdr.sample_rate)
                    )
                    
                    if not hasattr(self, 'avg_buffer'):
                        self.avg_buffer = []
                    
                    self.avg_buffer.append(pxx_db)
                    if len(self.avg_buffer) > 5:
                        self.avg_buffer.pop(0)
                    
                    pxx_db_avg = np.mean(self.avg_buffer, axis=0)
                    
                    self.data['frequencies'] = (freqs / 1e6).tolist()
                    self.data['powers'] = pxx_db_avg.tolist()
                    
            except Exception as e:
                print(f"Errore nell'acquisizione: {e}")
                time.sleep(0.5)
                continue
            
            time.sleep(0.05)

    def update_params(self, params):
        if not self.sdr:
            return False
            
        with self.lock:
            try:
                print(f"Ricevuti parametri: {params}")
                
                # Salva i parametri vecchi in caso di errore
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
                        self.sdr.gain = 'auto'
                    else:
                        self.sdr.gain = float(new_gain)
                    self.data['gain'] = new_gain
                    print(f"Gain aggiornato a: {new_gain}")
                
                # Pausa più lunga per la stabilizzazione
                time.sleep(0.5)
                
                # Pulisci il buffer della media mobile
                if hasattr(self, 'avg_buffer'):
                    self.avg_buffer = []
                    
                return True
                    
            except Exception as e:
                # In caso di errore, prova a ripristinare i parametri precedenti
                try:
                    self.sdr.center_freq = old_params['center_freq']
                    self.sdr.sample_rate = old_params['sample_rate']
                    self.sdr.gain = old_params['gain']
                except:
                    pass  # Se il ripristino fallisce, continua comunque
                    
                print(f"Errore nell'aggiornamento dei parametri: {e}")
                return False

    def get_data(self):
        """Restituisce i dati correnti dello spettro"""
        with self.lock:
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