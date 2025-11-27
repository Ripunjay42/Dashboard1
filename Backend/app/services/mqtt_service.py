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
            
            # Emit state update via SocketIO if available
            if self.socketio and self.connected:
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
            self.client.loop_forever()
        except Exception as e:
            print(f"✗ MQTT loop error: {e}")
            self.running = False
            self.connected = False
    
    def stop(self):
        """Stop MQTT client"""
        if self.client and self.running:
            self.running = False
            self.client.disconnect()
            print("✓ MQTT service stopped")
    
    def get_state(self):
        """Get current MQTT state"""
        return {
            **self.state,
            'connected': self.connected
        }
    
    def is_connected(self):
        """Check if MQTT client is connected"""
        return self.connected


# Global MQTT service instance
_mqtt_service = None

def get_mqtt_service(broker="10.42.0.1", port=1883, socketio=None):
    """Get or create global MQTT service instance"""
    global _mqtt_service
    if _mqtt_service is None:
        _mqtt_service = MQTTService(broker, port, socketio)
    return _mqtt_service
