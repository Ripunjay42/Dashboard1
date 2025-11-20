from flask import Blueprint, jsonify

hello_bp = Blueprint('hello', __name__)


@hello_bp.route('/hello', methods=['GET'])
def hello():
    """Simple hello endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Hello from Experience Centre Dashboard API!'
    })


@hello_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Experience Centre Backend'
    })
