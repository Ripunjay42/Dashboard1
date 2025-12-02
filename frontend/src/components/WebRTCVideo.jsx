/**
 * WebRTCVideo Component - Video player with WebRTC/MJPEG fallback
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';

const WebRTCVideo = ({ 
  trackId,
  mjpegUrl,
  className = '',
  style = {},
  showStatus = true,
  preferWebRTC = true,
  enabled = true,
}) => {
  const [mjpegKey, setMjpegKey] = useState(0);

  // Refresh MJPEG stream
  const refreshMjpeg = useCallback(() => {
    setMjpegKey(k => k + 1);
  }, []);

  // For now, just use MJPEG (WebRTC can be added later when aiortc is installed)
  return (
    <div style={{ position: 'relative', ...style }} className={className}>
      {!enabled ? (
        <div
          style={{
            width: '100%',
            height: '100%',
            backgroundColor: '#000',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <span style={{ color: '#666' }}>Video Paused</span>
        </div>
      ) : (
        <img
          key={mjpegKey}
          src={mjpegUrl}
          alt={`${trackId} stream`}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
          }}
          onError={() => {
            console.warn(`[WebRTCVideo ${trackId}] MJPEG error, refreshing...`);
            setTimeout(refreshMjpeg, 1000);
          }}
        />
      )}

      {/* Status indicator */}
      {showStatus && enabled && (
        <div
          style={{
            position: 'absolute',
            top: '8px',
            right: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            backgroundColor: 'rgba(0, 0, 0, 0.6)',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            color: '#fff',
          }}
        >
          <div
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: '#4CAF50',
              boxShadow: '0 0 4px #4CAF50',
            }}
          />
          MJPEG
        </div>
      )}
    </div>
  );
};

export default WebRTCVideo;
