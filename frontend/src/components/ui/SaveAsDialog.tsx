/**
 * Save As Dialog Component
 * Enterprise-grade dialog for saving quantum programs
 */

import React, { useState, useEffect } from 'react';
import { Save, FileText, AlertTriangle } from 'lucide-react';
import { programNameExists, generateUniqueProgramName } from '../../utils/programStorage';

interface SaveAsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (name: string, description: string) => void;
  currentName?: string;
  currentDescription?: string;
  primaryColor: string;
  isUpdate?: boolean;
}

export const SaveAsDialog: React.FC<SaveAsDialogProps> = ({
  isOpen,
  onClose,
  onSave,
  currentName = '',
  currentDescription = '',
  primaryColor,
  isUpdate = false
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (isOpen) {
      // If updating, use current values
      if (isUpdate && currentName) {
        setName(currentName);
        setDescription(currentDescription);
      } else {
        // For new save, generate unique name
        const baseName = currentName || 'New Quantum Program';
        setName(generateUniqueProgramName(baseName));
        setDescription(currentDescription || '');
      }
      setError('');
    }
  }, [isOpen, currentName, currentDescription, isUpdate]);

  const handleSave = () => {
    const trimmedName = name.trim();
    
    if (!trimmedName) {
      setError('Program name is required');
      return;
    }
    
    // Check for duplicate names (exclude current if updating)
    if (!isUpdate && programNameExists(trimmedName)) {
      setError('A program with this name already exists');
      return;
    }
    
    onSave(trimmedName, description.trim());
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleSave();
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="save-dialog-backdrop"
        onClick={onClose}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          backdropFilter: 'blur(4px)'
        }}
      />
      
      {/* Dialog */}
      <div 
        className="save-dialog"
        style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: '#1a1a1a',
          border: `1px solid ${primaryColor}40`,
          borderRadius: '8px',
          padding: '2rem',
          minWidth: '450px',
          maxWidth: '600px',
          zIndex: 10000,
          boxShadow: `0 0 40px ${primaryColor}40`
        }}
        onKeyDown={handleKeyDown}
      >
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1.5rem' }}>
          <FileText size={24} color={primaryColor} style={{ marginRight: '0.75rem' }} />
          <h3 style={{ margin: 0, color: primaryColor }}>
            {isUpdate ? 'Update Program' : 'Save Program As'}
          </h3>
        </div>
        
        <div style={{ marginBottom: '1.5rem' }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '0.5rem', 
            color: '#ccc',
            fontSize: '0.875rem'
          }}>
            Program Name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              setError('');
            }}
            placeholder="Enter program name..."
            autoFocus
            style={{
              width: '100%',
              padding: '0.75rem',
              backgroundColor: '#0a0a0a',
              border: `1px solid ${error ? '#ff6b6b' : '#333'}`,
              borderRadius: '4px',
              color: '#fff',
              fontSize: '0.875rem',
              outline: 'none',
              transition: 'border-color 0.2s'
            }}
            onFocus={(e) => {
              e.target.style.borderColor = primaryColor;
            }}
            onBlur={(e) => {
              e.target.style.borderColor = error ? '#ff6b6b' : '#333';
            }}
          />
          {error && (
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              marginTop: '0.5rem',
              color: '#ff6b6b',
              fontSize: '0.75rem'
            }}>
              <AlertTriangle size={12} style={{ marginRight: '0.25rem' }} />
              {error}
            </div>
          )}
        </div>
        
        <div style={{ marginBottom: '2rem' }}>
          <label style={{ 
            display: 'block', 
            marginBottom: '0.5rem', 
            color: '#ccc',
            fontSize: '0.875rem'
          }}>
            Description (optional)
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Enter program description..."
            rows={3}
            style={{
              width: '100%',
              padding: '0.75rem',
              backgroundColor: '#0a0a0a',
              border: '1px solid #333',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '0.875rem',
              outline: 'none',
              resize: 'vertical',
              transition: 'border-color 0.2s'
            }}
            onFocus={(e) => {
              e.target.style.borderColor = primaryColor;
            }}
            onBlur={(e) => {
              e.target.style.borderColor = '#333';
            }}
          />
        </div>
        
        <div style={{ 
          display: 'flex', 
          gap: '0.75rem', 
          justifyContent: 'flex-end' 
        }}>
          <button
            onClick={onClose}
            style={{
              padding: '0.5rem 1.25rem',
              backgroundColor: '#333',
              border: '1px solid #666',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.875rem',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#444';
              e.currentTarget.style.borderColor = '#888';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#333';
              e.currentTarget.style.borderColor = '#666';
            }}
          >
            Cancel
          </button>
          
          <button
            onClick={handleSave}
            disabled={!name.trim()}
            style={{
              padding: '0.5rem 1.25rem',
              backgroundColor: name.trim() ? primaryColor : '#444',
              border: `1px solid ${name.trim() ? primaryColor : '#666'}`,
              borderRadius: '4px',
              color: name.trim() ? '#000' : '#888',
              cursor: name.trim() ? 'pointer' : 'not-allowed',
              fontSize: '0.875rem',
              fontWeight: '500',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              if (name.trim()) {
                e.currentTarget.style.backgroundColor = primaryColor + 'cc';
                e.currentTarget.style.transform = 'translateY(-1px)';
              }
            }}
            onMouseLeave={(e) => {
              if (name.trim()) {
                e.currentTarget.style.backgroundColor = primaryColor;
                e.currentTarget.style.transform = 'translateY(0)';
              }
            }}
          >
            <Save size={16} />
            {isUpdate ? 'Update' : 'Save'}
          </button>
        </div>
        
        <div style={{
          marginTop: '1rem',
          paddingTop: '1rem',
          borderTop: '1px solid #333',
          fontSize: '0.75rem',
          color: '#666',
          textAlign: 'center'
        }}>
          Press Ctrl+Enter to save • Press Esc to cancel
        </div>
      </div>
    </>
  );
};