/**
 * Visualization Controls
 * Enterprise-grade color and intensity controls for OSH Universe visualization elements
 * Styled consistently with modern UI standards and responsive design
 */

import React from 'react';
import { Palette, Sun, Eye, Layers, GitBranch, Waves, Box } from 'lucide-react';

export interface VisualizationSettings {
  // Lattice Node Settings
  latticeNodeColor: string;
  latticeNodeIntensity: number;
  latticeNodeGlow: number;
  latticeVisible: boolean;
  
  // Memory Field Settings
  memoryFieldColor1: string;
  memoryFieldColor2: string;
  memoryFieldOpacity: number;
  memoryFieldDistortion: number;
  memoryFieldVisible: boolean;
  
  // Entanglement Thread Settings
  entanglementColor: string;
  entanglementIntensity: number;
  entanglementWidth: number;
  entanglementVisible: boolean;
  
  // Simulation Boundary Settings
  boundaryColor: string;
  boundaryOpacity: number;
  boundaryDistortion: number;
  boundaryVisible: boolean;
  
  // Wavefront Settings
  wavefrontColor: string;
  wavefrontIntensity: number;
  wavefrontSpeed: number;
  wavefrontVisible: boolean;
  
  // Post-processing Settings
  bloomIntensity: number;
  chromaticAberration: number;
  depthOfFieldFocus: number;
  vignetteIntensity: number;
}

export const defaultVisualizationSettings: VisualizationSettings = {
  latticeNodeColor: '#6366f1',
  latticeNodeIntensity: 1.0,
  latticeNodeGlow: 0.5,
  latticeVisible: true,
  
  memoryFieldColor1: '#10b981',
  memoryFieldColor2: '#f59e0b',
  memoryFieldOpacity: 0.3,
  memoryFieldDistortion: 0.5,
  memoryFieldVisible: true,
  
  entanglementColor: '#3b82f6',
  entanglementIntensity: 0.8,
  entanglementWidth: 0.02,
  entanglementVisible: true,
  
  boundaryColor: '#ef4444',
  boundaryOpacity: 0.2,
  boundaryDistortion: 0.3,
  boundaryVisible: true,
  
  wavefrontColor: '#a855f7',
  wavefrontIntensity: 0.6,
  wavefrontSpeed: 1.0,
  wavefrontVisible: true,
  
  bloomIntensity: 0.8,
  chromaticAberration: 0.002,
  depthOfFieldFocus: 10,
  vignetteIntensity: 0.3
};

interface VisualizationControlsProps {
  settings: VisualizationSettings;
  onSettingsChange: (settings: VisualizationSettings) => void;
  collapsed?: boolean;
  primaryColor?: string;
}

/**
 * Reusable styled input component for consistent design
 */
const StyledRangeInput: React.FC<{
  label: string;
  value: number;
  min: string;
  max: string;
  step: string;
  onChange: (value: number) => void;
  primaryColor: string;
  formatValue?: (value: number) => string;
}> = ({ label, value, min, max, step, onChange, primaryColor, formatValue }) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  }}>
    <label style={{
      fontSize: '11px',
      color: '#999',
      minWidth: '60px',
      fontWeight: '500'
    }}>{label}</label>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      style={{
        flex: 1,
        accentColor: primaryColor,
        height: '4px'
      }}
    />
    <span style={{
      fontSize: '10px',
      color: '#666',
      minWidth: '28px',
      textAlign: 'right',
      fontFamily: 'monospace'
    }}>
      {formatValue ? formatValue(value) : value.toFixed(1)}
    </span>
  </div>
);

/**
 * Reusable styled color input component
 */
const StyledColorInput: React.FC<{
  label: string;
  value: string;
  onChange: (value: string) => void;
}> = ({ label, value, onChange }) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  }}>
    <label style={{
      fontSize: '11px',
      color: '#999',
      minWidth: '60px',
      fontWeight: '500'
    }}>{label}</label>
    <input
      type="color"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        width: '32px',
        height: '20px',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer',
        backgroundColor: 'transparent'
      }}
    />
    <span style={{
      fontSize: '10px',
      color: '#666',
      fontFamily: 'monospace'
    }}>{value}</span>
  </div>
);

/**
 * Reusable section header with visibility toggle
 */
const SectionHeader: React.FC<{
  title: string;
  icon: React.ReactNode;
  visible: boolean;
  onToggle: (visible: boolean) => void;
  primaryColor: string;
}> = ({ title, icon, visible, onToggle, primaryColor }) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '12px'
  }}>
    <h4 style={{
      color: '#ddd',
      fontSize: '13px',
      fontWeight: '500',
      display: 'flex',
      alignItems: 'center',
      gap: '6px'
    }}>
      {icon}
      {title}
    </h4>
    <label style={{
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
      cursor: 'pointer'
    }}>
      <input
        type="checkbox"
        checked={visible}
        onChange={(e) => onToggle(e.target.checked)}
        style={{
          accentColor: primaryColor,
          transform: 'scale(1.1)'
        }}
      />
      <Eye size={12} style={{ color: '#888' }} />
    </label>
  </div>
);

export const VisualizationControls: React.FC<VisualizationControlsProps> = ({
  settings,
  onSettingsChange,
  collapsed = false,
  primaryColor = '#4fc3f7'
}) => {
  const updateSetting = (key: keyof VisualizationSettings, value: any) => {
    onSettingsChange({
      ...settings,
      [key]: value
    });
  };

  if (collapsed) {
    return (
      <div style={{
        background: 'rgba(0, 0, 0, 0.85)',
        backdropFilter: 'blur(12px)',
        padding: '8px',
        borderRadius: '8px',
        border: `1px solid ${primaryColor}30`
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          color: '#aaa',
          fontSize: '12px'
        }}>
          <Palette size={16} />
          <span>Visualization Controls</span>
        </div>
      </div>
    );
  }

  const sectionStyle = {
    paddingLeft: '16px',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '8px'
  };

  return (
    <div style={{
      background: 'rgba(0, 0, 0, 0.92)',
      backdropFilter: 'blur(20px)',
      padding: '16px',
      borderRadius: '12px',
      border: `2px solid ${primaryColor}40`,
      maxHeight: '70vh',
      overflowY: 'auto',
      minWidth: '280px',
      maxWidth: '320px',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
    }}>
      <h3 style={{
        color: primaryColor,
        fontWeight: '600',
        fontSize: '14px',
        marginBottom: '16px',
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>
        <Palette size={16} />
        Visualization Controls
      </h3>

      {/* Lattice Node Controls */}
      <div style={{ marginBottom: '20px' }}>
        <SectionHeader
          title="Lattice Nodes"
          icon={<Box size={14} />}
          visible={settings.latticeVisible}
          onToggle={(visible) => updateSetting('latticeVisible', visible)}
          primaryColor={primaryColor}
        />
        
        <div style={sectionStyle}>
          <StyledColorInput
            label="Color"
            value={settings.latticeNodeColor}
            onChange={(value) => updateSetting('latticeNodeColor', value)}
          />
          
          <StyledRangeInput
            label="Intensity"
            value={settings.latticeNodeIntensity}
            min="0"
            max="2"
            step="0.1"
            onChange={(value) => updateSetting('latticeNodeIntensity', value)}
            primaryColor={primaryColor}
          />
          
          <StyledRangeInput
            label="Glow"
            value={settings.latticeNodeGlow}
            min="0"
            max="1"
            step="0.05"
            onChange={(value) => updateSetting('latticeNodeGlow', value)}
            primaryColor={primaryColor}
            formatValue={(v) => v.toFixed(2)}
          />
        </div>
      </div>

      {/* Memory Field Controls */}
      <div style={{ marginBottom: '20px' }}>
        <SectionHeader
          title="Memory Field"
          icon={<Layers size={14} />}
          visible={settings.memoryFieldVisible}
          onToggle={(visible) => updateSetting('memoryFieldVisible', visible)}
          primaryColor={primaryColor}
        />
        
        <div style={sectionStyle}>
          <StyledColorInput
            label="Color 1"
            value={settings.memoryFieldColor1}
            onChange={(value) => updateSetting('memoryFieldColor1', value)}
          />
          
          <StyledColorInput
            label="Color 2"
            value={settings.memoryFieldColor2}
            onChange={(value) => updateSetting('memoryFieldColor2', value)}
          />
          
          <StyledRangeInput
            label="Opacity"
            value={settings.memoryFieldOpacity}
            min="0"
            max="1"
            step="0.05"
            onChange={(value) => updateSetting('memoryFieldOpacity', value)}
            primaryColor={primaryColor}
            formatValue={(v) => v.toFixed(2)}
          />
          
          <StyledRangeInput
            label="Distortion"
            value={settings.memoryFieldDistortion}
            min="0"
            max="2"
            step="0.1"
            onChange={(value) => updateSetting('memoryFieldDistortion', value)}
            primaryColor={primaryColor}
          />
        </div>
      </div>

      {/* Entanglement Thread Controls */}
      <div style={{ marginBottom: '20px' }}>
        <SectionHeader
          title="Entanglement"
          icon={<GitBranch size={14} />}
          visible={settings.entanglementVisible}
          onToggle={(visible) => updateSetting('entanglementVisible', visible)}
          primaryColor={primaryColor}
        />
        
        <div style={sectionStyle}>
          <StyledColorInput
            label="Color"
            value={settings.entanglementColor}
            onChange={(value) => updateSetting('entanglementColor', value)}
          />
          
          <StyledRangeInput
            label="Intensity"
            value={settings.entanglementIntensity}
            min="0"
            max="2"
            step="0.1"
            onChange={(value) => updateSetting('entanglementIntensity', value)}
            primaryColor={primaryColor}
          />
          
          <StyledRangeInput
            label="Width"
            value={settings.entanglementWidth}
            min="0.01"
            max="0.1"
            step="0.01"
            onChange={(value) => updateSetting('entanglementWidth', value)}
            primaryColor={primaryColor}
            formatValue={(v) => v.toFixed(2)}
          />
        </div>
      </div>

      {/* Wavefront Controls */}
      <div style={{ marginBottom: '20px' }}>
        <SectionHeader
          title="Wavefronts"
          icon={<Waves size={14} />}
          visible={settings.wavefrontVisible}
          onToggle={(visible) => updateSetting('wavefrontVisible', visible)}
          primaryColor={primaryColor}
        />
        
        <div style={sectionStyle}>
          <StyledColorInput
            label="Color"
            value={settings.wavefrontColor}
            onChange={(value) => updateSetting('wavefrontColor', value)}
          />
          
          <StyledRangeInput
            label="Intensity"
            value={settings.wavefrontIntensity}
            min="0"
            max="2"
            step="0.1"
            onChange={(value) => updateSetting('wavefrontIntensity', value)}
            primaryColor={primaryColor}
          />
          
          <StyledRangeInput
            label="Speed"
            value={settings.wavefrontSpeed}
            min="0.1"
            max="3"
            step="0.1"
            onChange={(value) => updateSetting('wavefrontSpeed', value)}
            primaryColor={primaryColor}
          />
        </div>
      </div>

      {/* Post-processing Controls */}
      <div style={{ marginBottom: '20px' }}>
        <h4 style={{
          color: '#ddd',
          fontSize: '13px',
          fontWeight: '500',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          marginBottom: '12px'
        }}>
          <Sun size={14} />
          Post-processing
        </h4>
        
        <div style={sectionStyle}>
          <StyledRangeInput
            label="Bloom"
            value={settings.bloomIntensity}
            min="0"
            max="2"
            step="0.1"
            onChange={(value) => updateSetting('bloomIntensity', value)}
            primaryColor={primaryColor}
          />
          
          <StyledRangeInput
            label="Chromatic"
            value={settings.chromaticAberration}
            min="0"
            max="0.01"
            step="0.001"
            onChange={(value) => updateSetting('chromaticAberration', value)}
            primaryColor={primaryColor}
            formatValue={(v) => (v * 1000).toFixed(0)}
          />
          
          <StyledRangeInput
            label="Vignette"
            value={settings.vignetteIntensity}
            min="0"
            max="1"
            step="0.05"
            onChange={(value) => updateSetting('vignetteIntensity', value)}
            primaryColor={primaryColor}
            formatValue={(v) => v.toFixed(2)}
          />
        </div>
      </div>
    </div>
  );
};