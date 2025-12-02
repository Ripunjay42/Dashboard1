"""
WebRTC Routes for Video Streaming
"""

from flask import Blueprint, jsonify, request

# Try to import WebRTC service - gracefully handle if aiortc not installed
try:
    from app.services.webrtc_service import get_webrtc_manager, is_webrtc_available
    WEBRTC_IMPORT_OK = True
except ImportError as e:
    WEBRTC_IMPORT_OK = False
    print(f"WebRTC service import failed: {e}")
    
    def is_webrtc_available():
        return False
    
    def get_webrtc_manager():
        return None

webrtc_bp = Blueprint('webrtc', __name__)


@webrtc_bp.route('/status', methods=['GET'])
def get_status():
    """Check WebRTC availability and stats"""
    if not is_webrtc_available():
        return jsonify({
            'available': False,
            'message': 'aiortc not installed'
        })
    
    manager = get_webrtc_manager()
    stats = manager.get_stats()
    return jsonify({
        'available': True,
        'stats': stats
    })


@webrtc_bp.route('/offer/<track_id>', methods=['POST'])
def create_offer(track_id):
    """Create a WebRTC offer for the specified track."""
    if not is_webrtc_available():
        return jsonify({
            'status': 'error',
            'message': 'WebRTC not available'
        }), 503
    
    try:
        manager = get_webrtc_manager()
        offer = manager.create_offer(track_id)
        
        if offer:
            return jsonify({
                'status': 'success',
                **offer
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to create offer'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@webrtc_bp.route('/answer/<connection_id>', methods=['POST'])
def handle_answer(connection_id):
    """Handle WebRTC answer from client."""
    if not is_webrtc_available():
        return jsonify({
            'status': 'error',
            'message': 'WebRTC not available'
        }), 503
    
    try:
        data = request.get_json()
        sdp = data.get('sdp')
        
        if not sdp:
            return jsonify({
                'status': 'error',
                'message': 'Missing SDP'
            }), 400
        
        manager = get_webrtc_manager()
        success = manager.handle_answer(connection_id, sdp)
        
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({
                'status': 'error',
                'message': 'Connection not found'
            }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@webrtc_bp.route('/close/<connection_id>', methods=['POST'])
def close_connection(connection_id):
    """Close a WebRTC connection."""
    if not is_webrtc_available():
        return jsonify({'status': 'ok'})
    
    try:
        manager = get_webrtc_manager()
        manager.close_connection(connection_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
