# app.py
from flask import Flask, render_template, jsonify, request
from rtlsdr import RtlSdr
import numpy as np
from threading import Thread, Lock
import time
import json

app = Flask(__name__)

class SDRHandler:
    def __init__(self):
        self.sdr = RtlSdr()
        self.lock = Lock()
        self.running = True
        self.data = {
            'frequencies': [],
            'powers': [],
            'center_freq': 100e6,
            'sample_rate': 2.4e6,
            'gain': 'auto'
        }
        
        # Configurazione iniziale
        self.update_sdr_settings()
        
        # Avvio thread di acquisizione
        self.update_thread = Thread(target=self.update_spectrum)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def update_sdr_settings(self):
        try:
            self.sdr.sample_rate = self.data['sample_rate']
            self.sdr.center_freq = self.data['center_freq']
            if self.data['gain'] == 'auto':
                self.sdr.gain = 'auto'
            else:
                self.sdr.gain = float(self.data['gain'])
            return True
        except Exception as e:
            print(f"Errore nell'aggiornamento delle impostazioni SDR: {e}")
            return False
    
    def update_spectrum(self):
        while self.running:
            try:
                with self.lock:
                    samples = self.sdr.read_samples(1024)
                    # Calcolo FFT
                    pxx = np.fft.fftshift(np.abs(np.fft.fft(samples)))
                    pxx_db = 20 * np.log10(pxx)
                    
                    # Calcolo frequenze
                    freqs = self.sdr.center_freq + np.fft.fftshift(
                        np.fft.fftfreq(len(samples), 1/self.sdr.sample_rate)
                    )
                    
                    # Aggiornamento dati
                    self.data['frequencies'] = (freqs / 1e6).tolist()  # MHz
                    self.data['powers'] = pxx_db.tolist()
            except Exception as e:
                print(f"Errore nell'acquisizione: {e}")
                time.sleep(1)
                continue
            
            time.sleep(0.1)
    
    def update_params(self, params):
        with self.lock:
            try:
                if 'center_freq' in params:
                    self.data['center_freq'] = float(params['center_freq'])
                if 'sample_rate' in params:
                    self.data['sample_rate'] = float(params['sample_rate'])
                if 'gain' in params:
                    self.data['gain'] = params['gain']
                
                success = self.update_sdr_settings()
                return success
            except Exception as e:
                print(f"Errore nell'aggiornamento dei parametri: {e}")
                return False
    
    def get_data(self):
        with self.lock:
            data = self.data.copy()
            # Aggiungi i parametri correnti
            data['current_settings'] = {
                'center_freq': self.sdr.center_freq / 1e6,  # Converti in MHz
                'sample_rate': self.sdr.sample_rate / 1e6,  # Converti in MHz
                'gain': self.sdr.gain
            }
            return data
    
    def cleanup(self):
        self.running = False
        self.update_thread.join()
        self.sdr.close()

sdr_handler = SDRHandler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_spectrum')
def get_spectrum():
    return jsonify(sdr_handler.get_data())

@app.route('/update_params', methods=['POST'])
def update_params():
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Content-Type deve essere application/json'}), 400
    
    params = request.get_json()
    success = sdr_handler.update_params(params)
    
    if success:
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Errore nell\'aggiornamento dei parametri'}), 400

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        sdr_handler.cleanup()