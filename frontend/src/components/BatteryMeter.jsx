import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const BatteryMeter = ({ value, max = 100 }) => {
  const svgRef = useRef();
  const needleRef = useRef();

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 380;
    const height = 380;
    const radius = 170;
    const centerX = width / 2;
    const centerY = height / 2;

    // Arc spans from bottom-left (135°) clockwise to bottom-right (45°)
    const startAngle = Math.PI * 0.75; // 135° (bottom left)
    const endAngle = Math.PI * 2.25; // 405° = 45° (bottom right)
    const totalAngle = endAngle - startAngle; // 270° arc

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${centerX}, ${centerY})`);

    // Gradient definitions
    const defs = svg.append('defs');
    
    // Background gradient
    const bgGradient = defs.append('radialGradient')
      .attr('id', 'battery-bg-gradient');
    bgGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#0f172a');
    bgGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#000000');

    // Battery level gradient (green to red)
    const batteryGradient = defs.append('linearGradient')
      .attr('id', 'battery-level-gradient');
    batteryGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#ef4444'); // Red (low battery)
    batteryGradient.append('stop')
      .attr('offset', '50%')
      .attr('stop-color', '#eab308'); // Yellow (medium battery)
    batteryGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#22c55e'); // Green (high battery)

    // Outer decorative circles
    g.append('circle')
      .attr('r', radius + 5)
      .attr('fill', 'none')
      .attr('stroke', '#1e293b')
      .attr('stroke-width', 3);

    g.append('circle')
      .attr('r', radius)
      .attr('fill', 'url(#battery-bg-gradient)')
      .attr('stroke', '#334155')
      .attr('stroke-width', 2);

    // Battery level color based on percentage
    const getBatteryColor = (percentage) => {
      if (percentage <= 20) return '#ef4444'; // Red
      if (percentage <= 50) return '#eab308'; // Yellow
      return '#22c55e'; // Green
    };

    // Dotted progress line
    const segmentCount = 40;
    const progress = value / max;
    for (let i = 0; i < segmentCount; i++) {
      const segProgress = i / segmentCount;
      const angle = startAngle + segProgress * totalAngle;
      const dotRadius = radius - 16;
      const dotX = dotRadius * Math.cos(angle);
      const dotY = dotRadius * Math.sin(angle);
      
      const isActive = i <= Math.floor(progress * segmentCount);
      const currentProgress = (i / segmentCount) * 100;
      
      g.append('circle')
        .attr('cx', dotX)
        .attr('cy', dotY)
        .attr('r', 4)
        .attr('fill', isActive ? getBatteryColor(currentProgress) : '#1e293b')
        .attr('opacity', isActive ? 0.9 : 0.3);
    }

    // Tick marks (every 10%)
    for (let i = 0; i <= max; i += 10) {
      const angle = startAngle + (i / max) * totalAngle;
      const isMajor = i % 20 === 0;
      const innerR = radius - 35;
      const outerR = isMajor ? radius - 23 : radius - 28;
      g.append('line')
        .attr('x1', innerR * Math.cos(angle))
        .attr('y1', innerR * Math.sin(angle))
        .attr('x2', outerR * Math.cos(angle))
        .attr('y2', outerR * Math.sin(angle))
        .attr('stroke', '#475569')
        .attr('stroke-width', isMajor ? 2 : 1)
        .attr('stroke-linecap', 'round');
    }

    // Numbers (0-100 for battery percentage)
    const numberPositions = [0, 20, 40, 60, 80, 100];
    numberPositions.forEach(num => {
      const angle = startAngle + (num / max) * totalAngle;
      const textRadius = radius - 50;
      g.append('text')
        .attr('x', textRadius * Math.cos(angle))
        .attr('y', textRadius * Math.sin(angle))
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', num <= 20 ? '#ef4444' : '#94a3b8')
        .attr('font-size', '14px')
        .attr('font-weight', '600')
        .text(num);
    });

    // Needle group
    const needleGroup = g.append('g')
      .attr('class', 'needle-group');
    needleRef.current = needleGroup;
    const initialNeedleAngle = startAngle + (value / max) * totalAngle;
    const initialNeedleAngleDegrees = (initialNeedleAngle * 180 / Math.PI) + 90;
    needleGroup.attr('transform', `rotate(${initialNeedleAngleDegrees})`);
    needleGroup.append('path')
      .attr('d', 'M -2,0 L -1,-90 L 0,-95 L 1,-90 L 2,0 Z')
      .attr('fill', getBatteryColor(value))
      .attr('filter', `drop-shadow(0 0 6px ${getBatteryColor(value)}80)`);

  }, [max, value]);

  // Animate needle when value changes
  useEffect(() => {
    const startAngle = Math.PI * 0.75; // 135 degrees
    const endAngle = Math.PI * 2.25; // 405 degrees (45 degrees)
    const totalAngle = endAngle - startAngle; // 270 degrees total

    if (needleRef.current) {
      // Calculate needle angle in radians
      const needleAngle = startAngle + (value / max) * totalAngle;
      // Convert to degrees for rotation, add 90° to convert from our coordinate system
      const needleAngleDegrees = (needleAngle * 180 / Math.PI) + 90;
      
      d3.select(needleRef.current.node())
        .transition()
        .duration(500)
        .ease(d3.easeCubicOut)
        .attr('transform', `rotate(${needleAngleDegrees})`);
    }
  }, [value, max]);

  return (
    <div className="relative w-[380px] h-[380px]">
      <svg
        ref={svgRef}
        width="380"
        height="380"
        className="absolute inset-0"
      />
      
      {/* Center display */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none" style={{ zIndex: 10 }}>
        <div className="text-center mt-8">
          <div className="text-7xl font-bold text-white drop-shadow-lg">
            {Math.round(value)}
          </div>
          <div className="text-xs text-gray-400 font-semibold tracking-wider mt-2">BATTERY %</div>
        </div>
      </div>
    </div>
  );
};

export default BatteryMeter;
