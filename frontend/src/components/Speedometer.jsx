import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const Speedometer = ({ value, max = 180 }) => {
  const svgRef = useRef();
  const needleRef = useRef();
  const progressArcRef = useRef();

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
      .attr('id', 'speedometer-bg-gradient');
    bgGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#0f172a');
    bgGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#000000');

    // Outer decorative circles
    g.append('circle')
      .attr('r', radius + 5)
      .attr('fill', 'none')
      .attr('stroke', '#1e293b')
      .attr('stroke-width', 3);

    g.append('circle')
      .attr('r', radius)
      .attr('fill', 'url(#speedometer-bg-gradient)')
      .attr('stroke', '#334155')
      .attr('stroke-width', 2);

    // Dotted progress line
    const segmentCount = 40;
    const progress = value / max;
    for (let i = 0; i < segmentCount; i++) {
      const segProgress = i / segmentCount;
      const angle = startAngle + segProgress * totalAngle;
      const dotRadius = radius - 16;
      const dotX = dotRadius * Math.cos(angle);
      const dotY = dotRadius * Math.sin(angle);
      g.append('circle')
        .attr('cx', dotX)
        .attr('cy', dotY)
        .attr('r', 4)
        .attr('fill', i < Math.floor(progress * segmentCount) ? '#3b82f6' : '#1e293b')
        .attr('opacity', i < Math.floor(progress * segmentCount) ? 0.9 : 0.3);
    }

    // Tick marks (every 10)
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

    // Numbers
    const numberPositions = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180];
    numberPositions.forEach(num => {
      const angle = startAngle + (num / max) * totalAngle;
      const textRadius = radius - 50;
      g.append('text')
        .attr('x', textRadius * Math.cos(angle))
        .attr('y', textRadius * Math.sin(angle))
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', num >= 140 ? '#ef4444' : '#94a3b8')
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
      .attr('fill', '#ef4444')
      .attr('filter', 'drop-shadow(0 0 6px rgba(239, 68, 68, 0.8))');
    // needleGroup.append('circle')
    //   .attr('r', 10)
    //   .attr('fill', '#1e293b')
    //   .attr('stroke', '#ef4444')
    //   .attr('stroke-width', 3);
    // needleGroup.append('circle')
    //   .attr('r', 4)
    //   .attr('fill', '#ef4444');

    // Progress arc
    const arc = d3.arc()
      .innerRadius(radius - 20)
      .outerRadius(radius - 10)
      .startAngle(startAngle)
      .endAngle(startAngle); // Initially, end angle is the same as start angle

    const path = g.append('path')
      .attr('d', arc)
      .attr('fill', '#3b82f6')
      .attr('opacity', value > 0 ? 1 : 0); // Hide if value is 0

    progressArcRef.current = { path, arc, startAngle };

  }, [max, value]); // Add value dependency

  // Animate needle and progress arc when value changes
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

    // if (progressArcRef.current) {
    //   const { path, arc, startAngle } = progressArcRef.current;
    //   const currentEndAngle = startAngle + (value / max) * totalAngle;
      
    //   if (value > 0) {
    //     path.transition()
    //       .duration(500)
    //       .ease(d3.easeCubicOut)
    //       .attr('d', arc.endAngle(currentEndAngle)());
    //   } else {
    //     // Hide arc when value is 0
    //     path.transition()
    //       .duration(500)
    //       .ease(d3.easeCubicOut)
    //       .attr('d', null);
    //   }
    // }
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
            {value}
          </div>
          <div className="text-xs text-gray-400 font-semibold tracking-wider mt-2">KM/H</div>
        </div>
      </div>
    </div>
  );
};

export default Speedometer;
