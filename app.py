# app.py
from flask import Flask, render_template, jsonify
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
        self.sdr.sample_rate = self.data['sample_rate']
        self.sdr.center_freq = self.data['center_freq']
        self.sdr.gain = self.data['gain']
        
        # Avvio thread di acquisizione
        self.update_thread = Thread(target=self.update_spectrum)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def update_spectrum(self):
        while self.running:
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
            
            time.sleep(0.1)  # Limita aggiornamenti
    
    def update_params(self, params):
        with self.lock:
            if 'center_freq' in params:
                self.sdr.center_freq = float(params['center_freq'])
                self.data['center_freq'] = float(params['center_freq'])
            if 'sample_rate' in params:
                self.sdr.sample_rate = float(params['sample_rate'])
                self.data['sample_rate'] = float(params['sample_rate'])
            if 'gain' in params:
                self.sdr.gain = params['gain']
                self.data['gain'] = params['gain']
    
    def get_data(self):
        with self.lock:
            return self.data.copy()
    
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
    params = request.get_json()
    sdr_handler.update_params(params)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        sdr_handler.cleanup()