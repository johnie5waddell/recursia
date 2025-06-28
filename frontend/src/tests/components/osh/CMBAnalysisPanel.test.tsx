import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CMBAnalysisPanel from './CMBAnalysisPanel';
import { generateSampleCSV, generateSampleJSON } from './sampleData';

// Mock file for testing
const createMockFile = (content: string, name: string, type: string): File => {
  const blob = new Blob([content], { type });
  return new File([blob], name, { type });
};

describe('CMBAnalysisPanel', () => {
  it('renders without crashing', () => {
    render(<CMBAnalysisPanel />);
    expect(screen.getByText('CMB Complexity Analysis')).toBeInTheDocument();
  });

  it('displays file upload section', () => {
    render(<CMBAnalysisPanel />);
    expect(screen.getByText('Upload CMB Data')).toBeInTheDocument();
    expect(screen.getByText('Choose File')).toBeInTheDocument();
  });

  it('handles CSV file upload', async () => {
    const user = userEvent.setup();
    render(<CMBAnalysisPanel />);
    
    const csvContent = generateSampleCSV();
    const file = createMockFile(csvContent, 'test-cmb.csv', 'text/csv');
    
    const input = screen.getByLabelText('Choose File').parentElement?.querySelector('input[type="file"]');
    if (input) {
      await user.upload(input, file);
      
      await waitFor(() => {
        expect(screen.getByText('test-cmb.csv')).toBeInTheDocument();
        expect(screen.getByText('Loaded Dataset')).toBeInTheDocument();
      });
    }
  });

  it('handles JSON file upload', async () => {
    const user = userEvent.setup();
    render(<CMBAnalysisPanel />);
    
    const jsonContent = generateSampleJSON();
    const file = createMockFile(jsonContent, 'test-cmb.json', 'application/json');
    
    const input = screen.getByLabelText('Choose File').parentElement?.querySelector('input[type="file"]');
    if (input) {
      await user.upload(input, file);
      
      await waitFor(() => {
        expect(screen.getByText('test-cmb.json')).toBeInTheDocument();
        expect(screen.getByText('Loaded Dataset')).toBeInTheDocument();
      });
    }
  });

  it('shows error for invalid file format', async () => {
    const user = userEvent.setup();
    render(<CMBAnalysisPanel />);
    
    const file = createMockFile('invalid content', 'test.txt', 'text/plain');
    
    const input = screen.getByLabelText('Choose File').parentElement?.querySelector('input[type="file"]');
    if (input) {
      await user.upload(input, file);
      
      await waitFor(() => {
        expect(screen.getByText(/Unsupported file format/)).toBeInTheDocument();
      });
    }
  });

  it('performs analysis when button is clicked', async () => {
    const user = userEvent.setup();
    render(<CMBAnalysisPanel />);
    
    // Upload a file first
    const csvContent = generateSampleCSV();
    const file = createMockFile(csvContent, 'test-cmb.csv', 'text/csv');
    
    const input = screen.getByLabelText('Choose File').parentElement?.querySelector('input[type="file"]');
    if (input) {
      await user.upload(input, file);
      
      await waitFor(() => {
        expect(screen.getByText('Loaded Dataset')).toBeInTheDocument();
      });
      
      // Click analyze button
      const analyzeButton = screen.getByText('Analyze Complexity');
      await user.click(analyzeButton);
      
      await waitFor(() => {
        expect(screen.getByText('Lempel-Ziv Complexity')).toBeInTheDocument();
        expect(screen.getByText('Statistical Analysis')).toBeInTheDocument();
        expect(screen.getByText('Power Spectrum Analysis')).toBeInTheDocument();
      });
    }
  });

  it('clears analysis when clear button is clicked', async () => {
    const user = userEvent.setup();
    render(<CMBAnalysisPanel />);
    
    // Upload a file
    const csvContent = generateSampleCSV();
    const file = createMockFile(csvContent, 'test-cmb.csv', 'text/csv');
    
    const input = screen.getByLabelText('Choose File').parentElement?.querySelector('input[type="file"]');
    if (input) {
      await user.upload(input, file);
      
      await waitFor(() => {
        expect(screen.getByText('Loaded Dataset')).toBeInTheDocument();
      });
      
      // Click clear button
      const clearButton = screen.getByText('Clear');
      await user.click(clearButton);
      
      await waitFor(() => {
        expect(screen.queryByText('Loaded Dataset')).not.toBeInTheDocument();
        expect(screen.queryByText('test-cmb.csv')).not.toBeInTheDocument();
      });
    }
  });

  it('exports results when export button is clicked', async () => {
    const user = userEvent.setup();
    render(<CMBAnalysisPanel />);
    
    // Mock URL.createObjectURL
    global.URL.createObjectURL = jest.fn(() => 'mock-url');
    
    // Upload and analyze
    const csvContent = generateSampleCSV();
    const file = createMockFile(csvContent, 'test-cmb.csv', 'text/csv');
    
    const input = screen.getByLabelText('Choose File').parentElement?.querySelector('input[type="file"]');
    if (input) {
      await user.upload(input, file);
      
      await waitFor(() => {
        expect(screen.getByText('Loaded Dataset')).toBeInTheDocument();
      });
      
      const analyzeButton = screen.getByText('Analyze Complexity');
      await user.click(analyzeButton);
      
      await waitFor(() => {
        expect(screen.getByText('Lempel-Ziv Complexity')).toBeInTheDocument();
      });
      
      // Click export
      const exportButton = screen.getByText('Export Results');
      await user.click(exportButton);
      
      expect(global.URL.createObjectURL).toHaveBeenCalled();
    }
  });

  it('toggles advanced analysis section', async () => {
    const user = userEvent.setup();
    render(<CMBAnalysisPanel />);
    
    // Upload and analyze first
    const csvContent = generateSampleCSV();
    const file = createMockFile(csvContent, 'test-cmb.csv', 'text/csv');
    
    const input = screen.getByLabelText('Choose File').parentElement?.querySelector('input[type="file"]');
    if (input) {
      await user.upload(input, file);
      await waitFor(() => expect(screen.getByText('Loaded Dataset')).toBeInTheDocument());
      
      const analyzeButton = screen.getByText('Analyze Complexity');
      await user.click(analyzeButton);
      
      await waitFor(() => expect(screen.getByText('Lempel-Ziv Complexity')).toBeInTheDocument());
      
      // Toggle advanced section
      const advancedButton = screen.getByText('Show Advanced Analysis');
      await user.click(advancedButton);
      
      await waitFor(() => {
        expect(screen.getByText('Advanced Analysis')).toBeInTheDocument();
        expect(screen.getByText('Multi-Scale Decomposition')).toBeInTheDocument();
      });
      
      // Toggle again to hide
      const hideButton = screen.getByText('Hide Advanced Analysis');
      await user.click(hideButton);
      
      await waitFor(() => {
        expect(screen.queryByText('Multi-Scale Decomposition')).not.toBeInTheDocument();
      });
    }
  });
});