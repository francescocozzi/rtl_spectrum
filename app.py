from flask import Flask, render_template, jsonify, request
from rtlsdr import RtlSdr
import numpy as np
from threading import Thread, Lock, Event
import time
from dataclasses import dataclass
from scipy import signal

app = Flask(__name__)

@dataclass
class SignalFeatures:
    bandwidth: float
    peak_power: float
    modulation_type: str
    confidence: float

class SignalClassifier:
    def __init__(self):
        self.modulation_patterns = {
            'AM': {'bandwidth_ratio': 0.1, 'iq_variance_ratio': 0.2},
            'FM': {'bandwidth_ratio': 0.2, 'iq_variance_ratio': 0.8},
            'FSK': {'bandwidth_ratio': 0.15, 'iq_phase_var': 0.5},
            'PSK': {'bandwidth_ratio': 0.1, 'iq_phase_var': 0.3},
            'QAM': {'bandwidth_ratio': 0.12, 'iq_constellation': 'square'}
        }

    def _identify_modulation(self, bw_ratio, iq_var_ratio, phase_var):
        """
        Identifica il tipo di modulazione basandosi sui parametri del segnale
        
        Args:
            bw_ratio (float): Rapporto di banda
            iq_var_ratio (float): Rapporto di varianza I/Q
            phase_var (float): Varianza della fase
            
        Returns:
            tuple: (tipo_modulazione, confidenza)
        """
        scores = {}
        
        # Calcola un punteggio per ogni tipo di modulazione
        for mod_type, pattern in self.modulation_patterns.items():
            score = 0
            
            # Valuta il rapporto di banda
            if 'bandwidth_ratio' in pattern:
                bandwidth_score = 1 - min(abs(pattern['bandwidth_ratio'] - bw_ratio) / 0.1, 1)
                score += 0.4 * bandwidth_score
            
            # Valuta il rapporto di varianza I/Q
            if 'iq_variance_ratio' in pattern:
                variance_score = 1 - min(abs(pattern['iq_variance_ratio'] - iq_var_ratio) / 0.2, 1)
                score += 0.3 * variance_score
            
            # Valuta la varianza della fase
            if 'iq_phase_var' in pattern:
                phase_score = 1 - min(abs(pattern['iq_phase_var'] - phase_var) / 0.2, 1)
                score += 0.3 * phase_score
                
            scores[mod_type] = max(0, min(score, 1))  # Normalizza tra 0 e 1
        
        # Trova la modulazione con il punteggio più alto
        if not scores:
            return 'Unknown', 0.0
            
        best_mod = max(scores.items(), key=lambda x: x[1])
        return best_mod[0], best_mod[1]

    def analyze_signal(self, frequencies, powers, iq_data):
        try:
            powers_array = np.array(powers)
            frequencies_array = np.array(frequencies)
            
            # Trova picchi significativi
            peak_indices = signal.find_peaks(powers_array, 
                                           height=np.mean(powers_array) + np.std(powers_array),
                                           distance=int(len(powers_array) * 0.05))[0]
            
            if len(peak_indices) == 0:
                return None
            
            peak_powers = powers_array[peak_indices]
            max_peak_idx = peak_indices[int(np.argmax(peak_powers))]
            
            # Calcola larghezza di banda
            bandwidth = self._estimate_bandwidth(frequencies_array, powers_array, max_peak_idx)
            
            # Analisi I/Q
            i_data = np.array(iq_data.get('i', []))
            q_data = np.array(iq_data.get('q', []))
            
            if len(i_data) == 0 or len(q_data) == 0:
                return SignalFeatures(
                    bandwidth=float(bandwidth),
                    peak_power=float(powers_array[max_peak_idx]),
                    modulation_type="Unknown",
                    confidence=0.0
                )
            
            # Calcolo caratteristiche I/Q
            complex_signal = i_data + 1j*q_data
            phase_var = np.var(np.angle(complex_signal))
            iq_var_ratio = np.var(i_data) / (np.var(q_data) + 1e-10)
            
            # Identifica modulazione
            mod_type, confidence = self._identify_modulation(
                bandwidth/(frequencies_array[1] - frequencies_array[0]), 
                iq_var_ratio,
                phase_var
            )
            
            return SignalFeatures(
                bandwidth=float(bandwidth),
                peak_power=float(powers_array[max_peak_idx]),
                modulation_type=str(mod_type),
                confidence=float(confidence)
            )
            
        except Exception as e:
            print(f"Error in signal analysis: {e}")
            return None

    def _estimate_bandwidth(self, freqs, powers, peak_idx):
        try:
            # Calcola soglia dinamica
            noise_floor = np.median(powers)
            threshold = powers[peak_idx] - 3  # -3dB threshold
            threshold = max(threshold, noise_floor + 6)  # almeno 6dB sopra il rumore
            
            left_idx = peak_idx
            right_idx = peak_idx
            
            while left_idx > 0 and powers[left_idx] > threshold:
                left_idx -= 1
            while right_idx < len(powers)-1 and powers[right_idx] > threshold:
                right_idx += 1
                
            return abs(freqs[right_idx] - freqs[left_idx])
            
        except Exception as e:
            print(f"Error in bandwidth estimation: {e}")
            return 0.0

class SDRHandler:
    def __init__(self):
        self.sdr = None
        self.data_lock = Lock()
        self.sdr_lock = Lock()
        self.running = True
        self.recovery_event = Event()
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        self.classifier = SignalClassifier()
        self.initialized = False
        
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
            print(f"SDR initialization error: {e}")
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
                self.sdr.gain = self._calculate_gain()
                time.sleep(0.1)
            return True
        except Exception as e:
            print(f"SDR configuration error: {e}")
            return False

    def _calculate_gain(self):
        if self.data['gain'] == 'auto':
            return 20 if self.data['center_freq'] >= 300e6 else 15
        gain_value = float(self.data['gain'])
        return min(gain_value, 30) if self.data['center_freq'] < 300e6 else min(gain_value, 40)

    def update_spectrum(self):
        while self.running:
            if not self.sdr or self.recovery_event.is_set():
                time.sleep(0.1)
                continue
                
            try:
                samples = self._get_samples()
                if samples is None:
                    continue
                    
                spectrum_data = self._process_spectrum(samples)
                iq_data = self._process_iq_data(samples)
                
                with self.data_lock:
                    self.data['frequencies'] = spectrum_data['frequencies']
                    self.data['powers'] = spectrum_data['powers']
                    self.data['iq_data'] = iq_data
                    
            except Exception as e:
                self._handle_error(e)
                
            time.sleep(0.01)

    def _get_samples(self):
        try:
            with self.sdr_lock:
                return self.sdr.read_samples(1024)
        except Exception as e:
            print(f"Sample reading error: {e}")
            return None

    def _process_spectrum(self, samples):
        # Normalizzazione e FFT
        samples_norm = (samples - np.mean(samples)) / (np.std(samples) + 1e-10)
        window = np.blackman(len(samples_norm))
        samples_windowed = samples_norm * window
        
        # FFT con dimensione ottimizzata
        nfft = 2048  # Aumentato per migliore risoluzione
        pxx = np.fft.fftshift(np.abs(np.fft.fft(samples_windowed, n=nfft)))
        
        # Conversione in dB con range dinamico ottimizzato
        pxx_db = 20 * np.log10(pxx + 1e-10)
        pxx_db = pxx_db - np.max(pxx_db)
        pxx_db = np.clip(pxx_db, -80, 0)  # Range dinamico aumentato
        
        freqs = self.sdr.center_freq + np.fft.fftshift(
            np.fft.fftfreq(nfft, 1/self.sdr.sample_rate)
        )
        
        return {
            'frequencies': (freqs / 1e6).tolist(),
            'powers': pxx_db.tolist()
        }

    def _process_iq_data(self, samples):
        i_data = samples.real[:100]  # Limitato a 100 campioni per performance
        q_data = samples.imag[:100]
        
        # Normalizzazione robusta
        max_iq = max(np.max(np.abs(i_data)), np.max(np.abs(q_data)))
        i_data = i_data / (max_iq + 1e-10)
        q_data = q_data / (max_iq + 1e-10)
        
        return {
            'i': i_data.tolist(),
            'q': q_data.tolist(),
            'time': np.arange(len(i_data)).tolist()
        }

    def _handle_error(self, error):
        print(f"Processing error: {error}")
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.max_consecutive_errors:
            self.recovery_event.set()
            if not self._recover_device():
                time.sleep(1)

    def _recover_device(self):
        try:
            with self.sdr_lock:
                self.sdr.close()
                time.sleep(1)
                self.sdr = RtlSdr()
                success = self._configure_sdr()
                if success:
                    self.consecutive_errors = 0
                    self.recovery_event.clear()
                return success
        except Exception as e:
            print(f"Device recovery error: {e}")
            return False

    def get_data(self):
        try:
            with self.data_lock:
                data = {
                    'frequencies': self.data['frequencies'],
                    'powers': self.data['powers'],
                    'iq_data': self.data['iq_data'],
                    'current_settings': {
                        'center_freq': self.sdr.center_freq / 1e6 if self.sdr else 0,
                        'sample_rate': self.sdr.sample_rate / 1e6 if self.sdr else 0,
                        'gain': self.sdr.gain if self.sdr else 'auto'
                    }
                }
                
                signal_info = self.classifier.analyze_signal(
                    self.data['frequencies'],
                    self.data['powers'],
                    self.data['iq_data']
                )
                
                if signal_info:
                    data['signal_info'] = {
                        'modulation': signal_info.modulation_type,
                        'bandwidth': signal_info.bandwidth,
                        'peak_power': signal_info.peak_power,
                        'confidence': signal_info.confidence
                    }
                
                return data
                
        except Exception as e:
            print(f"Data retrieval error: {e}")
            return {'error': str(e)}

    def cleanup(self):
        self.running = False
        if self.sdr:
            self.sdr.close()

# Inizializzazione e route
sdr_handler = None
try:
    sdr_handler = SDRHandler()
except Exception as e:
    print(f"SDR handler creation error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_spectrum')
def get_spectrum():
    if not sdr_handler or not sdr_handler.initialized:
        return jsonify({'error': 'SDR not initialized'}), 503
    return jsonify(sdr_handler.get_data())

@app.route('/update_params', methods=['POST'])
def update_params():
    if not sdr_handler or not sdr_handler.initialized:
        return jsonify({'error': 'SDR not initialized'}), 503
        
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
        
    try:
        params = request.get_json()
        success = sdr_handler.update_params(params)
        return jsonify({'status': 'success' if success else 'error'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)