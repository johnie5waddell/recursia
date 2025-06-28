/**
 * Enterprise Loading States Components
 * Provides consistent loading UI across the application
 */

import React from 'react';
import { motion } from 'framer-motion';
import { 
  Loader2, 
  Brain, 
  Atom, 
  Sparkles, 
  Activity,
  Waves,
  Network,
  Zap
} from 'lucide-react';

interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  color?: string;
  message?: string;
}

/**
 * Simple loading spinner with optional message
 */
export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  size = 'medium', 
  color = '#4fc3f7',
  message 
}) => {
  const sizes = {
    small: 16,
    medium: 32,
    large: 64
  };

  return (
    <div className="loading-spinner-container">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
      >
        <Loader2 size={sizes[size]} color={color} />
      </motion.div>
      {message && <p className="loading-message">{message}</p>}
    </div>
  );
};

interface QuantumLoadingProps {
  primaryColor?: string;
  message?: string;
}

/**
 * Quantum-themed loading animation
 */
export const QuantumLoading: React.FC<QuantumLoadingProps> = ({ 
  primaryColor = '#4fc3f7',
  message = 'Initializing quantum engine...'
}) => {
  return (
    <div className="quantum-loading-container">
      <div className="quantum-loading-animation">
        <motion.div
          className="quantum-orbit orbit-1"
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        >
          <div className="quantum-particle" style={{ background: primaryColor }} />
        </motion.div>
        <motion.div
          className="quantum-orbit orbit-2"
          animate={{ rotate: -360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        >
          <div className="quantum-particle" style={{ background: primaryColor }} />
        </motion.div>
        <motion.div
          className="quantum-orbit orbit-3"
          animate={{ rotate: 360 }}
          transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
        >
          <div className="quantum-particle" style={{ background: primaryColor }} />
        </motion.div>
        <div className="quantum-core">
          <Atom size={24} color={primaryColor} />
        </div>
      </div>
      <p className="quantum-loading-message">{message}</p>
    </div>
  );
};

interface DataLoadingProps {
  items?: string[];
  primaryColor?: string;
}

/**
 * Data loading with progress indicators
 */
export const DataLoading: React.FC<DataLoadingProps> = ({ 
  items = ['Fetching metrics', 'Initializing engines', 'Rendering visualization'],
  primaryColor = '#4fc3f7'
}) => {
  const [currentIndex, setCurrentIndex] = React.useState(0);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % items.length);
    }, 1500);
    return () => clearInterval(interval);
  }, [items.length]);

  return (
    <div className="data-loading-container">
      <div className="data-loading-icon">
        <Activity size={48} color={primaryColor} />
      </div>
      <div className="data-loading-items">
        {items.map((item, index) => (
          <motion.div
            key={item}
            className={`data-loading-item ${index === currentIndex ? 'active' : ''}`}
            initial={{ opacity: 0.3 }}
            animate={{ opacity: index === currentIndex ? 1 : 0.3 }}
            transition={{ duration: 0.3 }}
          >
            <div className="loading-dot" style={{ background: index === currentIndex ? primaryColor : '#666' }} />
            <span>{item}</span>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  variant?: 'text' | 'circular' | 'rectangular';
  animation?: 'pulse' | 'wave';
}

/**
 * Skeleton loader for content placeholders
 */
export const Skeleton: React.FC<SkeletonProps> = ({ 
  width = '100%', 
  height = 20,
  variant = 'rectangular',
  animation = 'pulse'
}) => {
  const className = `skeleton skeleton-${variant} skeleton-${animation}`;
  
  const style: React.CSSProperties = {
    width,
    height,
    borderRadius: variant === 'circular' ? '50%' : variant === 'text' ? '4px' : '8px'
  };

  return <div className={className} style={style} />;
};

interface ComponentLoadingProps {
  componentName: string;
  primaryColor?: string;
  showProgress?: boolean;
  progress?: number;
}

/**
 * Component-specific loading state
 */
export const ComponentLoading: React.FC<ComponentLoadingProps> = ({ 
  componentName,
  primaryColor = '#4fc3f7',
  showProgress = false,
  progress = 0
}) => {
  const icons = [Brain, Atom, Sparkles, Waves, Network, Zap];
  const Icon = icons[Math.floor(Math.random() * icons.length)];

  return (
    <div className="component-loading-container">
      <motion.div
        className="component-loading-icon"
        animate={{ 
          scale: [1, 1.1, 1],
          rotate: [0, 180, 360]
        }}
        transition={{ 
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      >
        <Icon size={48} color={primaryColor} />
      </motion.div>
      
      <h3 className="component-loading-title">Loading {componentName}</h3>
      
      {showProgress && (
        <div className="loading-progress-container">
          <div className="loading-progress-bar">
            <motion.div 
              className="loading-progress-fill"
              style={{ background: primaryColor }}
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <span className="loading-progress-text">{Math.round(progress)}%</span>
        </div>
      )}
    </div>
  );
};

interface EmptyStateProps {
  icon?: React.ComponentType<{ size?: string | number; color?: string }>;
  title: string;
  message: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  primaryColor?: string;
}

/**
 * Empty state component for when no data is available
 */
export const EmptyState: React.FC<EmptyStateProps> = ({ 
  icon: Icon = Activity,
  title,
  message,
  action,
  primaryColor = '#4fc3f7'
}) => {
  return (
    <div className="empty-state-container">
      <div className="empty-state-icon">
        <Icon size={64} color={primaryColor} />
      </div>
      <h3 className="empty-state-title">{title}</h3>
      <p className="empty-state-message">{message}</p>
      {action && (
        <button 
          className="empty-state-action"
          style={{ borderColor: primaryColor, color: primaryColor }}
          onClick={action.onClick}
        >
          {action.label}
        </button>
      )}
    </div>
  );
};