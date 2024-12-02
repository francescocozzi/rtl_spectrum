from flask import Flask, render_template, jsonify, request
from rtlsdr import RtlSdr
import numpy as np
from threading import Thread, Lock, Event
import time
import json
import atexit
import numpy as np
from scipy import signal
from dataclasses import dataclass

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

    def analyze_signal(self, frequencies, powers, iq_data):
        # Converti i dati in array numpy
        powers_array = np.array(powers)
        frequencies_array = np.array(frequencies)
        
        # Trova picchi significativi nello spettro
        peak_indices = signal.find_peaks(powers_array, height=-40, distance=10)[0]
        if len(peak_indices) == 0:
            return None
        
        # Trova l'indice del picco più forte
        peak_powers = powers_array[peak_indices]
        max_peak_idx = peak_indices[int(np.argmax(peak_powers))]
        
        # Calcola i parametri del segnale
        peak_freq = frequencies_array[max_peak_idx]
        peak_power = powers_array[max_peak_idx]
        
        # Calcola larghezza di banda
        bandwidth = self._estimate_bandwidth(frequencies_array, powers_array, max_peak_idx)
        
        # Analizza caratteristiche IQ
        i_data = np.array(iq_data['i'])
        q_data = np.array(iq_data['q'])
        phase_var = np.var(np.angle(i_data + 1j*q_data))
        iq_var_ratio = np.var(i_data) / (np.var(q_data) + 1e-10)
        
        # Identifica modulazione
        mod_type, confidence = self._identify_modulation(
            bandwidth/abs(frequencies_array[1] - frequencies_array[0]), 
            iq_var_ratio,
            phase_var
        )
        
        return SignalFeatures(
            bandwidth=float(bandwidth),
            peak_power=float(peak_power),
            modulation_type=str(mod_type),
            confidence=float(confidence)
        )

    def _estimate_bandwidth(self, freqs, powers, peak_idx):
        threshold = powers[peak_idx] - 3  # -3dB threshold
        left_idx = peak_idx
        right_idx = peak_idx
        
        while left_idx > 0 and powers[left_idx] > threshold:
            left_idx -= 1
        while right_idx < len(powers)-1 and powers[right_idx] > threshold:
            right_idx += 1
            
        return freqs[right_idx] - freqs[left_idx]
    
    def _identify_modulation(self, bw_ratio, iq_var_ratio, phase_var):
        scores = {}
        for mod_type, pattern in self.modulation_patterns.items():
            score = 0
            if abs(pattern['bandwidth_ratio'] - bw_ratio) < 0.05:
                score += 0.4
            if 'iq_variance_ratio' in pattern and abs(pattern['iq_variance_ratio'] - iq_var_ratio) < 0.1:
                score += 0.3
            if 'iq_phase_var' in pattern and abs(pattern['iq_phase_var'] - phase_var) < 0.1:
                score += 0.3
            scores[mod_type] = score
        
        best_mod = max(scores.items(), key=lambda x: x[1])
        return best_mod[0], best_mod[1]

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
        self.classifier = SignalClassifier()

        
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

    def update_params(self, params):
        if not self.sdr:
            return False
            
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
                        if self.data['center_freq'] < 300e6:
                            self.sdr.gain = 15  # Gain più basso per VHF
                        else:
                            self.sdr.gain = 20  # Gain standard per UHF
                    else:
                        new_gain = float(new_gain)
                        # Limita il gain in base alla banda
                        if self.data['center_freq'] < 300e6:
                            new_gain = min(new_gain, 30)
                        else:
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
            
                print("Dati base recuperati")
                print(f"Lunghezza frequencies: {len(self.data['frequencies'])}")
                print(f"Lunghezza powers: {len(self.data['powers'])}")
            
                try:
                    signal_info = self.classifier.analyze_signal(
                        self.data['frequencies'],
                        self.data['powers'],
                        self.data['iq_data']
                    )
                    print("Analisi del segnale completata")
                
                    if signal_info:
                        data['signal_info'] = {
                            'modulation': signal_info.modulation_type,
                            'bandwidth': float(signal_info.bandwidth),
                            'peak_power': float(signal_info.peak_power),
                            'confidence': float(signal_info.confidence)
                        }
                        print("Info segnale aggiunto ai dati")
                except Exception as e:
                    print(f"Errore nell'analisi del segnale: {e}")
                    data['signal_info'] = None
            
                return data
            
        except Exception as e:
            print(f"Errore generale in get_data: {e}")
            return {
                'error': f'Errore nel recupero dei dati: {str(e)}'
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
@app.route('/analyze_signal', methods=['POST'])
def analyze_specific_signal():
    try:
        data = request.get_json()
        start_freq = float(data['start_freq'])
        end_freq = float(data['end_freq'])
        
        with sdr_handler.data_lock:
            # Troviamo gli indici corrispondenti all'intervallo di frequenza
            freqs = np.array(sdr_handler.data['frequencies'])
            powers = np.array(sdr_handler.data['powers'])
            
            mask = (freqs >= start_freq) & (freqs <= end_freq)
            
            if not any(mask):
                return jsonify({'error': 'Nessun dato nella selezione'}), 400
            
            # Analizziamo solo la porzione selezionata
            signal_info = sdr_handler.classifier.analyze_signal(
                freqs[mask],
                powers[mask],
                sdr_handler.data['iq_data']
            )
            
            if signal_info:
                return jsonify({
                    'modulation': signal_info.modulation_type,
                    'bandwidth': float(signal_info.bandwidth),
                    'peak_power': float(signal_info.peak_power),
                    'confidence': float(signal_info.confidence),
                    'center_freq': (start_freq + end_freq) / 2
                })
            else:
                return jsonify({'error': 'Nessun segnale rilevato nella selezione'}), 400
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500