
// Recursia Dashboard JavaScript
class RecursiaDashboard {
    constructor() {
        this.websocket = null;
        this.connected = false;
        this.metrics = {};
        this.experiments = [];
        this.currentVisualization = null;
        
        this.initializeWebSocket();
        this.setupEventHandlers();
        this.loadExperiments();
    }
    
    initializeWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.connected = true;
            this.updateConnectionStatus(true);
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.connected = false;
            this.updateConnectionStatus(false);
            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.initializeWebSocket(), 5000);
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    handleWebSocketMessage(data) {
        switch(data.type) {
            case 'metrics_update':
                this.updateMetrics(data.metrics);
                break;
            case 'visualization_update':
                this.updateVisualization(data.visualization);
                break;
            case 'experiment_result':
                this.handleExperimentResult(data);
                break;
            case 'system_alert':
                this.showAlert(data.message, data.level);
                break;
        }
    }
    
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('connection-status');
        if (indicator) {
            indicator.className = `status-indicator ${connected ? 'status-online' : 'status-offline'}`;
        }
    }
    
    updateMetrics(metrics) {
        this.metrics = metrics;
        
        // Update metric displays
        this.updateMetricDisplay('coherence', metrics.coherence);
        this.updateMetricDisplay('entropy', metrics.entropy);
        this.updateMetricDisplay('strain', metrics.strain);
        this.updateMetricDisplay('quantum-states', metrics.quantum_states_count);
        this.updateMetricDisplay('observers', metrics.observer_count);
        this.updateMetricDisplay('render-fps', metrics.render_fps);
    }
    
    updateMetricDisplay(id, value) {
        const element = document.getElementById(`metric-${id}`);
        if (element) {
            if (typeof value === 'number') {
                element.textContent = value.toFixed(3);
            } else {
                element.textContent = value;
            }
        }
    }
    
    updateVisualization(visualizationData) {
        const canvas = document.getElementById('visualization-canvas');
        if (canvas && visualizationData.image_data) {
            // Update visualization
            const img = new Image();
            img.onload = () => {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = `data:image/png;base64,${visualizationData.image_data}`;
        }
    }
    
    async loadExperiments() {
        try {
            const response = await fetch('/api/experiments');
            this.experiments = await response.json();
            this.updateExperimentList();
        } catch (error) {
            console.error('Failed to load experiments:', error);
        }
    }
    
    updateExperimentList() {
        const container = document.getElementById('experiment-list');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.experiments.forEach(exp => {
            const item = document.createElement('div');
            item.className = 'experiment-item';
            item.innerHTML = `
                <h4>${exp.name}</h4>
                <p>${exp.description}</p>
                <div class="experiment-tags">
                    ${exp.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
                <button onclick="dashboard.runExperiment('${exp.id}')" class="btn btn-success">Run</button>
                <button onclick="dashboard.editExperiment('${exp.id}')" class="btn">Edit</button>
            `;
            container.appendChild(item);
        });
    }
    
    async runExperiment(experimentId) {
        try {
            this.showLoading('Running experiment...');
            
            const response = await fetch(`/api/experiments/${experimentId}/execute`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            const result = await response.json();
            this.hideLoading();
            
            if (result.success) {
                this.showAlert('Experiment completed successfully!', 'success');
                // Trigger visualization update
                this.requestVisualizationUpdate();
            } else {
                this.showAlert(`Experiment failed: ${result.error}`, 'error');
            }
        } catch (error) {
            this.hideLoading();
            this.showAlert(`Failed to run experiment: ${error.message}`, 'error');
        }
    }
    
    async requestVisualizationUpdate() {
        try {
            const response = await fetch('/api/visualization/render', {
                method: 'POST'
            });
            const data = await response.json();
            if (data.success) {
                this.updateVisualization(data);
            }
        } catch (error) {
            console.error('Failed to update visualization:', error);
        }
    }
    
    showLoading(message = 'Loading...') {
        const loader = document.getElementById('loading-overlay');
        if (loader) {
            loader.style.display = 'flex';
            const messageEl = loader.querySelector('.loading-message');
            if (messageEl) messageEl.textContent = message;
        }
    }
    
    hideLoading() {
        const loader = document.getElementById('loading-overlay');
        if (loader) {
            loader.style.display = 'none';
        }
    }
    
    showAlert(message, level = 'info') {
        // Create alert element
        const alert = document.createElement('div');
        alert.className = `alert alert-${level}`;
        alert.textContent = message;
        
        const container = document.getElementById('alerts-container');
        if (container) {
            container.appendChild(alert);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.parentNode.removeChild(alert);
                }
            }, 5000);
        }
    }
    
    setupEventHandlers() {
        // Setup form handlers, button clicks, etc.
        document.addEventListener('DOMContentLoaded', () => {
            // Initialize UI components
            this.initializeCodeEditor();
            this.setupExperimentBuilder();
        });
    }
    
    initializeCodeEditor() {
        const editor = document.getElementById('code-editor');
        if (editor) {
            // Basic syntax highlighting could be added here
            editor.addEventListener('input', () => {
                // Auto-save draft
                localStorage.setItem('experiment-draft', editor.value);
            });
            
            // Load draft
            const draft = localStorage.getItem('experiment-draft');
            if (draft) {
                editor.value = draft;
            }
        }
    }
    
    setupExperimentBuilder() {
        const form = document.getElementById('experiment-form');
        if (form) {
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this.createExperiment();
            });
        }
    }
    
    async createExperiment() {
        const name = document.getElementById('exp-name').value;
        const description = document.getElementById('exp-description').value;
        const code = document.getElementById('code-editor').value;
        
        try {
            const response = await fetch('/api/experiments', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name,
                    description,
                    code,
                    parameters: {}
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showAlert('Experiment created successfully!', 'success');
                this.loadExperiments();
                
                // Clear form
                document.getElementById('experiment-form').reset();
                localStorage.removeItem('experiment-draft');
            } else {
                this.showAlert(`Failed to create experiment: ${result.error}`, 'error');
            }
        } catch (error) {
            this.showAlert(`Error: ${error.message}`, 'error');
        }
    }
}

// Initialize dashboard when page loads
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new RecursiaDashboard();
});
