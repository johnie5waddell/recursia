import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * Global tooltip manager to ensure only one tooltip is visible at a time
 * This prevents tooltip persistence issues when multiple tooltips try to show
 */
class TooltipManager {
  private static instance: TooltipManager;
  private activeTooltipId: string | null = null;
  private listeners: Map<string, () => void> = new Map();

  static getInstance(): TooltipManager {
    if (!TooltipManager.instance) {
      TooltipManager.instance = new TooltipManager();
    }
    return TooltipManager.instance;
  }

  register(id: string, hideCallback: () => void): void {
    this.listeners.set(id, hideCallback);
  }

  unregister(id: string): void {
    this.listeners.delete(id);
    if (this.activeTooltipId === id) {
      this.activeTooltipId = null;
    }
  }

  show(id: string): void {
    // Hide any other active tooltip
    if (this.activeTooltipId && this.activeTooltipId !== id) {
      const hideCallback = this.listeners.get(this.activeTooltipId);
      if (hideCallback) {
        hideCallback();
      }
    }
    this.activeTooltipId = id;
  }

  hide(id: string): void {
    if (this.activeTooltipId === id) {
      this.activeTooltipId = null;
    }
  }

  hideAll(): void {
    this.listeners.forEach((hideCallback) => hideCallback());
    this.activeTooltipId = null;
  }
}

interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right' | 'auto';
  delay?: number;
  className?: string;
  primaryColor?: string;
  disabled?: boolean;
}

interface TooltipPosition {
  x: number;
  y: number;
  actualPosition: 'top' | 'bottom' | 'left' | 'right';
}

/**
 * Enterprise-grade tooltip component with edge detection and smart positioning
 * Provides consistent styling and behavior across the application
 */
// Export TooltipManager for external use if needed
export { TooltipManager };

export const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  position = 'auto',
  delay = 300,
  className = '',
  primaryColor = '#4fc3f7',
  disabled = false
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState<TooltipPosition>({ 
    x: 0, 
    y: 0, 
    actualPosition: 'top' 
  });
  const triggerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout>();
  const tooltipIdRef = useRef<string>(`tooltip-${Math.random().toString(36).substr(2, 9)}`);

  /**
   * Calculate optimal tooltip position with edge detection
   * Ensures tooltip remains fully visible within viewport boundaries
   */
  const calculatePosition = useCallback(() => {
    if (!triggerRef.current || !tooltipRef.current) return;

    const triggerRect = triggerRef.current.getBoundingClientRect();
    const tooltipRect = tooltipRef.current.getBoundingClientRect();
    const viewport = {
      width: window.innerWidth,
      height: window.innerHeight
    };

    // Spacing between trigger and tooltip
    const offset = 8;
    
    // Calculate available space in each direction
    const space = {
      top: triggerRect.top,
      bottom: viewport.height - triggerRect.bottom,
      left: triggerRect.left,
      right: viewport.width - triggerRect.right
    };

    // Determine best position based on available space
    let bestPosition: 'top' | 'bottom' | 'left' | 'right' = position === 'auto' ? 'top' : position;
    
    if (position === 'auto') {
      // Auto-positioning logic: prefer top/bottom over left/right
      if (space.top >= tooltipRect.height + offset) {
        bestPosition = 'top';
      } else if (space.bottom >= tooltipRect.height + offset) {
        bestPosition = 'bottom';
      } else if (space.left >= tooltipRect.width + offset) {
        bestPosition = 'left';
      } else if (space.right >= tooltipRect.width + offset) {
        bestPosition = 'right';
      } else {
        // Default to top if no ideal position
        bestPosition = 'top';
      }
    }

    // Calculate exact coordinates based on position
    let x = 0;
    let y = 0;

    switch (bestPosition) {
      case 'top':
        x = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
        y = triggerRect.top - tooltipRect.height - offset;
        break;
      case 'bottom':
        x = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
        y = triggerRect.bottom + offset;
        break;
      case 'left':
        x = triggerRect.left - tooltipRect.width - offset;
        y = triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2;
        break;
      case 'right':
        x = triggerRect.right + offset;
        y = triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2;
        break;
    }

    // Ensure tooltip stays within viewport bounds
    const padding = 8;
    x = Math.max(padding, Math.min(x, viewport.width - tooltipRect.width - padding));
    y = Math.max(padding, Math.min(y, viewport.height - tooltipRect.height - padding));

    setTooltipPosition({ x, y, actualPosition: bestPosition });
  }, [position]);

  /**
   * Hide tooltip handler used by TooltipManager
   */
  const hideTooltip = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = undefined;
    }
    setIsVisible(false);
    TooltipManager.getInstance().hide(tooltipIdRef.current);
  }, []);

  /**
   * Show tooltip after delay
   */
  const handleMouseEnter = useCallback(() => {
    if (disabled || !content) return;

    // Clear any existing timeout to prevent multiple tooltips
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = undefined;
    }

    timeoutRef.current = setTimeout(() => {
      TooltipManager.getInstance().show(tooltipIdRef.current);
      setIsVisible(true);
    }, delay);
  }, [disabled, content, delay]);

  /**
   * Hide tooltip immediately
   */
  const handleMouseLeave = useCallback(() => {
    hideTooltip();
  }, [hideTooltip]);

  /**
   * Force hide tooltip on click anywhere
   */
  const handleClick = useCallback(() => {
    hideTooltip();
  }, [hideTooltip]);

  /**
   * Update position when tooltip becomes visible
   */
  useEffect(() => {
    if (isVisible) {
      // Use requestAnimationFrame to ensure DOM has updated
      requestAnimationFrame(() => {
        calculatePosition();
      });
    }
  }, [isVisible, calculatePosition]);

  /**
   * Register with TooltipManager
   */
  useEffect(() => {
    const tooltipId = tooltipIdRef.current;
    TooltipManager.getInstance().register(tooltipId, hideTooltip);

    return () => {
      TooltipManager.getInstance().unregister(tooltipId);
    };
  }, [hideTooltip]);

  /**
   * Cleanup timeout on unmount
   */
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = undefined;
      }
      // Force hide tooltip on unmount
      setIsVisible(false);
      TooltipManager.getInstance().hide(tooltipIdRef.current);
    };
  }, []);

  /**
   * Handle window events that should hide tooltip
   */
  useEffect(() => {
    const handleWindowEvents = () => {
      if (isVisible) {
        setIsVisible(false);
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
          timeoutRef.current = undefined;
        }
      }
    };

    // Hide tooltip on any click outside
    const handleGlobalClick = (e: MouseEvent) => {
      // Don't hide if clicking on the trigger element
      if (triggerRef.current && triggerRef.current.contains(e.target as Node)) {
        return;
      }
      handleWindowEvents();
    };

    // Hide tooltip on scroll, resize, or click
    window.addEventListener('scroll', handleWindowEvents, true);
    window.addEventListener('resize', handleWindowEvents);
    document.addEventListener('click', handleGlobalClick, true);
    document.addEventListener('touchstart', handleGlobalClick, true);
    
    return () => {
      window.removeEventListener('scroll', handleWindowEvents, true);
      window.removeEventListener('resize', handleWindowEvents);
      document.removeEventListener('click', handleGlobalClick, true);
      document.removeEventListener('touchstart', handleGlobalClick, true);
    };
  }, [isVisible]);

  // Don't render if disabled or no content
  if (disabled || !content) {
    return <>{children}</>;
  }

  // Portal container for tooltips
  const portalRoot = typeof document !== 'undefined' ? document.body : null;

  return (
    <>
      <div
        ref={triggerRef}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        onTouchEnd={handleClick}
        style={{ display: 'inline-block', cursor: 'inherit' }}
      >
        {children}
      </div>
      {portalRoot && createPortal(
        <AnimatePresence mode="wait">
          {isVisible && (
            <motion.div
            ref={tooltipRef}
            className={`tooltip tooltip-${tooltipPosition.actualPosition} ${className}`}
            initial={{ opacity: 0, scale: 0.95, y: tooltipPosition.actualPosition === 'top' ? 4 : tooltipPosition.actualPosition === 'bottom' ? -4 : 0 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
            onAnimationComplete={() => {
              // Ensure tooltip is properly hidden after exit animation
              if (!isVisible && tooltipRef.current) {
                tooltipRef.current.style.display = 'none';
              }
            }}
            style={{
              position: 'fixed',
              left: tooltipPosition.x,
              top: tooltipPosition.y,
              zIndex: 2147483647, // Maximum z-index to ensure tooltip is always on top
              pointerEvents: 'none',
              maxWidth: '300px',
              // Built-in styles for consistent appearance
              background: 'rgba(20, 20, 20, 0.95)',
              backdropFilter: 'blur(8px)',
              border: `1px solid ${primaryColor}30`,
              borderRadius: '6px',
              padding: '8px 12px',
              boxShadow: `0 4px 12px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.05) inset`,
              // Force tooltip to be visible
              display: isVisible ? 'block' : 'none',
              visibility: isVisible ? 'visible' : 'hidden',
            }}
          >
            <div 
              className="tooltip-content"
              style={{
                color: primaryColor,
                fontSize: '12px',
                fontFamily: '"JetBrains Mono", "Fira Code", monospace',
                fontWeight: 500,
                lineHeight: 1.4,
                textAlign: 'center',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word'
              }}
            >
              {content}
            </div>
            {/* Arrow pointer */}
            <div
              className={`tooltip-arrow tooltip-arrow-${tooltipPosition.actualPosition}`}
              style={{
                position: 'absolute',
                width: 0,
                height: 0,
                borderStyle: 'solid',
                ...(tooltipPosition.actualPosition === 'top' && {
                  bottom: '-6px',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  borderWidth: '6px 6px 0 6px',
                  borderColor: `rgba(20, 20, 20, 0.95) transparent transparent transparent`,
                }),
                ...(tooltipPosition.actualPosition === 'bottom' && {
                  top: '-6px',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  borderWidth: '0 6px 6px 6px',
                  borderColor: `transparent transparent rgba(20, 20, 20, 0.95) transparent`,
                }),
                ...(tooltipPosition.actualPosition === 'left' && {
                  right: '-6px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  borderWidth: '6px 0 6px 6px',
                  borderColor: `transparent transparent transparent rgba(20, 20, 20, 0.95)`,
                }),
                ...(tooltipPosition.actualPosition === 'right' && {
                  left: '-6px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  borderWidth: '6px 6px 6px 0',
                  borderColor: `transparent rgba(20, 20, 20, 0.95) transparent transparent`,
                }),
              }}
            />
            </motion.div>
          )}
        </AnimatePresence>,
        portalRoot
      )}
    </>
  );
};