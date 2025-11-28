import paho.mqtt.client as mqtt
import threading
import time
from flask_socketio import SocketIO

class MQTTService:
    def __init__(self, broker, port, socketio=None):
        self.broker = broker
        self.port = port
        self.socketio = socketio
        
        # MQTT Topics
        self.TOPIC_PIR = "ex/pir"
        self.TOPIC_ADC = "ex/accl"
        self.TOPIC_LEFT_IND = "ex/LI"
        self.TOPIC_RIGHT_IND = "ex/RI"
        
        # Current state
        self.state = {
            'speed': 0,
            'pir_alert': 0,
            'left_indicator': 0,
            'right_indicator': 0
        }
        
        self.client = None
        self.connected = False
        self.running = False
        self.thread = None
        self._stop_event = threading.Event()  # Thread-safe stop signal
        
    def on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback when connected to MQTT broker (API v2 signature)"""
        if rc == 0:
            print("✓ MQTT Connected. Subscribing to topics...")
            self.connected = True
            
            # Subscribe to all topics
            client.subscribe(self.TOPIC_PIR)
            client.subscribe(self.TOPIC_ADC)
            client.subscribe(self.TOPIC_LEFT_IND)
            client.subscribe(self.TOPIC_RIGHT_IND)
        else:
            print(f"✗ MQTT Connection failed with code {rc}")
            self.connected = False
    
    def on_disconnect(self, client, userdata, flags, rc, properties=None):
        """Callback when disconnected from MQTT broker (API v2 signature)"""
        print(f"✗ MQTT Disconnected (code: {rc})")
        self.connected = False
    
    def on_message(self, client, userdata, msg):
        """Callback when a message is received"""
        # CRITICAL: Check if service is still running before processing
        if not self.running:
            return  # Ignore messages when service is stopped
            
        try:
            payload = msg.payload.decode().strip()
            topic = msg.topic
            
            # Parse payload as integer
            value = int(payload)
            
            # Update state based on topic
            if topic == self.TOPIC_PIR:
                self.state['pir_alert'] = value
                print(f"PIR Alert: {value}")
            elif topic == self.TOPIC_ADC:
                # Map ADC value to speed (0-180 km/h range)
                # Adjust this mapping based on your ADC sensor range
                self.state['speed'] = min(180, max(0, value))
                print(f"Speed (ADC): {self.state['speed']}")
            elif topic == self.TOPIC_LEFT_IND:
                self.state['left_indicator'] = value
                print(f"Left Indicator: {value}")
            elif topic == self.TOPIC_RIGHT_IND:
                self.state['right_indicator'] = value
                print(f"Right Indicator: {value}")
            
            # Emit state update via SocketIO if available and running
            if self.socketio and self.connected and self.running:
                self.socketio.emit('mqtt_update', self.state, namespace='/mqtt')
                
        except ValueError as e:
            print(f"Error parsing MQTT message: {e}")
        except Exception as e:
            print(f"Error handling MQTT message: {e}")
    
    def start(self):
        """Start MQTT client in a background thread"""
        if self.running:
            print("MQTT service already running")
            return True
        
        try:
            # Clear stop event
            self._stop_event.clear()
            
            # Create MQTT client (using callback API version 2)
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            
            # Start connection in background thread
            self.running = True
            self.thread = threading.Thread(target=self._run_mqtt_loop, daemon=True)
            self.thread.start()
            
            print(f"✓ MQTT service started (connecting to {self.broker}:{self.port})")
            return True
            
        except Exception as e:
            print(f"✗ Failed to start MQTT service: {e}")
            self.running = False
            return False
    
    def _run_mqtt_loop(self):
        """Run MQTT client loop in background thread"""
        try:
            self.client.connect(self.broker, self.port, 60)
            
            # Use loop_start() instead of loop_forever() for better control
            self.client.loop_start()
            
            # Wait until stop event is set
            while not self._stop_event.is_set():
                time.sleep(0.1)
            
            # Clean shutdown
            self.client.loop_stop()
            
        except Exception as e:
            print(f"✗ MQTT loop error: {e}")
        finally:
            self.running = False
            self.connected = False
    
    def stop(self):
        """Stop MQTT client completely"""
        print("🛑 Stopping MQTT service...")
        
        # Signal the thread to stop
        self._stop_event.set()
        self.running = False
        
        if self.client:
            try:
                # Unsubscribe from all topics first
                self.client.unsubscribe(self.TOPIC_PIR)
                self.client.unsubscribe(self.TOPIC_ADC)
                self.client.unsubscribe(self.TOPIC_LEFT_IND)
                self.client.unsubscribe(self.TOPIC_RIGHT_IND)
            except:
                pass
            
            try:
                # Disconnect client
                self.client.disconnect()
            except:
                pass
            
            try:
                # Stop the loop if it's running
                self.client.loop_stop()
            except:
                pass
        
        # Reset state to defaults when stopped
        self.state = {
            'speed': 0,
            'pir_alert': 0,
            'left_indicator': 0,
            'right_indicator': 0
        }
        
        self.connected = False
        self.client = None
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        
        print("✓ MQTT service stopped and state reset")
    
    def get_state(self):
        """Get current MQTT state"""
        return {
            **self.state,
            'connected': self.connected
        }
    
    def is_connected(self):
        """Check if MQTT client is connected"""
        return self.connected and self.running


# Global MQTT service instance
_mqtt_service = None
_mqtt_lock = threading.Lock()

def get_mqtt_service(broker="10.42.0.1", port=1883, socketio=None):
    """Get or create global MQTT service instance"""
    global _mqtt_service
    with _mqtt_lock:
        if _mqtt_service is None:
            _mqtt_service = MQTTService(broker, port, socketio)
    return _mqtt_service

def reset_mqtt_service():
    """Reset the global MQTT service instance (for clean restart)"""
    global _mqtt_service
    with _mqtt_lock:
        if _mqtt_service is not None:
            _mqtt_service.stop()
            _mqtt_service = None
