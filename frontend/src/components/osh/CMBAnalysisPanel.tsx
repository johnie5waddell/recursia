/**
 * Cosmic Microwave Background Analysis Panel
 * Simplified version without MUI dependencies
 */

import React from 'react';

interface CMBAnalysisPanelProps {
  className?: string;
  style?: React.CSSProperties;
}

const CMBAnalysisPanel: React.FC<CMBAnalysisPanelProps> = ({ className, style }) => {
  return (
    <div className={className} style={style}>
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <h3>CMB Analysis Panel</h3>
        <p>CMB Analysis functionality is temporarily disabled due to missing dependencies.</p>
        <p>Please install @mui/material and @mui/icons-material to enable this feature.</p>
      </div>
    </div>
  );
};

export default CMBAnalysisPanel;