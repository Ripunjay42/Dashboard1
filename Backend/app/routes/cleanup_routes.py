from flask import Blueprint, jsonify
import gc

cleanup_bp = Blueprint('cleanup', __name__)


@cleanup_bp.route('/force', methods=['POST'])
def force_cleanup():
    """
    Force garbage collection and camera cleanup.
    Called when user clicks home button to free resources.
    """
    try:
        # Import camera manager for emergency cleanup
        from app.services.camera_manager import cleanup_all_cameras
        
        # Import controllers directly
        from app.controllers import pothole_controller
        from app.controllers import blindspot_controller
        from app.controllers import dms_controller
        
        # Stop pothole detection
        try:
            pothole_controller.stop_detection()
        except Exception as e:
            print(f"Pothole cleanup error: {e}")
        
        # Stop blindspot detection
        try:
            blindspot_controller.stop_detection()
        except Exception as e:
            print(f"Blindspot cleanup error: {e}")
        
        # Stop DMS detection
        try:
            dms_controller.stop_detection()
        except Exception as e:
            print(f"DMS cleanup error: {e}")
        
        # Emergency camera cleanup
        cleanup_all_cameras()
        
        # Force garbage collection (3 passes for thorough cleanup)
        collected = []
        for i in range(3):
            collected.append(gc.collect())
        
        # Try to clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        
        return jsonify({
            'status': 'success',
            'message': 'All cameras stopped and garbage collected',
            'garbage_collected': sum(collected)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
