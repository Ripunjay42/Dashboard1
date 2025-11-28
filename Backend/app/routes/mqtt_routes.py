from flask import Blueprint, jsonify, request
from app.services.mqtt_service import get_mqtt_service, reset_mqtt_service

mqtt_bp = Blueprint('mqtt', __name__)


@mqtt_bp.route('/start', methods=['POST'])
def start_mqtt():
    """Start MQTT service"""
    try:
        data = request.get_json() or {}
        broker = data.get('broker', '10.42.0.1')
        port = data.get('port', 1883)
        
        mqtt_service = get_mqtt_service(broker, port)
        
        if mqtt_service.is_connected():
            return jsonify({
                'status': 'success',
                'message': 'MQTT service already running',
                'data': mqtt_service.get_state()
            })
        
        if mqtt_service.start():
            return jsonify({
                'status': 'success',
                'message': 'MQTT service started',
                'data': mqtt_service.get_state()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to start MQTT service'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@mqtt_bp.route('/stop', methods=['POST'])
def stop_mqtt():
    """Stop MQTT service completely and reset state"""
    try:
        mqtt_service = get_mqtt_service()
        mqtt_service.stop()
        
        # Return the reset state (all zeros)
        return jsonify({
            'status': 'success',
            'message': 'MQTT service stopped',
            'data': mqtt_service.get_state()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@mqtt_bp.route('/status', methods=['GET'])
def get_status():
    """Get current MQTT state"""
    try:
        mqtt_service = get_mqtt_service()
        return jsonify({
            'status': 'success',
            'data': mqtt_service.get_state()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@mqtt_bp.route('/state', methods=['GET'])
def get_state():
    """Get current MQTT state (alias for /status)"""
    return get_status()
